"""Бот для рамен-мини-игры в Wanted: Dead (PS5, Remote Play).

Играет две дорожки нот сам и берёт трофей «Ramen World Legend» — 250 попаданий
подряд без единого промаха. Этим ботом взято комбо 268.

Как это работает:
  * кадр берётся из потока Remote Play (pyremoteplay), 720p / 60 fps;
  * в поле нот ищутся кружки, внутри них распознаётся значок по шаблонам;
  * нота ведётся треком, нажатие уходит за `--lead-ms` до линии попадания —
    это компенсирует задержку Remote Play;
  * удержания (бирюзовые полосы) жмутся на левом конце и отпускаются на правом;
  * в трофейном режиме (`--trophy`) бот различает экраны GAMEPLAY / INTRO /
    RESULTS, читает результаты через OCR (если установлен rapidocr) и жмёт Retry,
    чтобы сразу начать следующую порцию.

Запуск:
    python ramen_bot.py --run 192.168.1.50 --trophy

Подробная инструкция — в README.md.
"""

from __future__ import annotations

import argparse
import asyncio
import heapq
import json
import socket
import sys
import tempfile
import threading
import time
import types
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Консоль Windows по умолчанию в cp866/cp1251 и падает на «—» и стрелках,
# поэтому вывод принудительно в UTF-8 (Windows Terminal и cmd с chcp 65001 это
# понимают, а на старых консолях лучше заменить символ, чем упасть).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass



def install_netifaces_compat() -> None:
    """Replace pyremoteplay's old netifaces dependency with psutil on Windows."""
    if "netifaces" in sys.modules:
        return
    import psutil

    compat = types.ModuleType("netifaces")
    compat.AF_INET = socket.AF_INET

    def interfaces():
        return list(psutil.net_if_addrs().keys())

    def ifaddresses(name):
        found = []
        for address in psutil.net_if_addrs().get(name, []):
            if address.family == socket.AF_INET:
                found.append({
                    "addr": address.address,
                    "netmask": address.netmask,
                    "broadcast": address.broadcast,
                })
        return {socket.AF_INET: found}

    compat.interfaces = interfaces
    compat.ifaddresses = ifaddresses
    sys.modules["netifaces"] = compat


install_netifaces_compat()

try:
    import cv2
    import numpy as np
    import av
    from av.codec.context import Flags, Flags2

    # pyremoteplay 0.7.6 uses the older uppercase PyAV flag names.
    # PyAV 17 renamed these enum members to lowercase.
    if not hasattr(Flags, "LOW_DELAY") and hasattr(Flags, "low_delay"):
        Flags.LOW_DELAY = Flags.low_delay
    if not hasattr(Flags2, "FAST") and hasattr(Flags2, "fast"):
        Flags2.FAST = Flags2.fast

    from pyremoteplay import RPDevice
    from pyremoteplay.__main__ import main as pyremoteplay_cli
    from pyremoteplay.controller import Controller
    from pyremoteplay.receiver import QueueReceiver
except ImportError as exc:
    print(f"Не хватает библиотек: {exc}")
    print("Запусти setup_ramen_bot.bat из папки бота.")
    raise SystemExit(1)

try:  # optional, used only for reading the results screen (not for MASH)
    from rapidocr_onnxruntime import RapidOCR
    # MEASURED on a real 720p results screenshot: the default engine (all 24
    # cores) takes 4.5-6.0 s per screen and starves the main loop for ~5 s
    # (visible in the log as "299.4 мс/кадр" right after the screenshot). Two
    # threads take 1.8-2.8 s and leave the rest of the CPU to the bot.
    _OCR_ENGINE = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)
except Exception:
    _OCR_ENGINE = None


NOTE_BUTTONS = {"X": "CROSS", "O": "CIRCLE", "SQUARE": "SQUARE",
                "TRIANGLE": "TRIANGLE", "L2": "L2", "R2": "R2"}

# --- live combo counter (OCR) --------------------------------------------
# Everything measured so far says HOW MANY notes were missed but never WHEN the
# streak actually broke, so every fix was a guess. The user located the counter
# on screen: it sits in the top-right corner under the OPTIONS overlay -- which
# is why those two clickable overlays were moved to the centre of the window.
# Reading the number turns each break into a timestamped event that can be
# matched against the note journal.
# MEASURED LEAK -- DO NOT RE-ENABLE WITHOUT FIXING THE ENGINE:
# RapidOCR/onnxruntime creates a thread pool per call and never releases it. Called
# once per song (the results screen) that was harmless; called every 2 s it made the
# bot process grow by ~18 threads and ~160 MB per 10 s and burn 420% of a core, which
# is exactly the "everything stutters" the user reported. The counter is therefore
# OFF; the break snapshot is now triggered by a cluster of missed notes instead
# MEASURED: one reading of that band costs ~400 ms of CPU on 2 threads whatever the
# crop scale (the model has a fixed cost), so reading it once a second would eat a
# third of the machine and add jitter to the press loop. Every 2 s keeps a break
# inside a 2 s window. MEASURED BUG in the first version: a FAILED reading (the
# counter is unreadable on dark frames) waited the whole period again, so readings
# could be 12 s apart and the break snapshot landed 12 s after the real break --
# useless. A failure now retries after 0.6 s.
# Region of the frame to read, as fractions (x1, y1, x2, y2). Verified on 30 saved
# 720p frames: this band reads "NN | Combo" in 24-26 of 30 (the misses are the dark

# How long to wait between connection attempts. Was 15 s / 30 s, which the user
# found too slow when the console is busy with another session: after a restart the
# whole point is to be playing again within seconds.
RETRY_CONNECT_WAIT_S = 5.0
RETRY_CONNECT_WAIT_LATE_S = 10.0

# Which ends of a detected hold pair leave the TAP loop, i.e. how the hold logic and
# the ordinary note logic share those notes. Settable from farm_trophies_auto.bat via
# --hold-mode / HOLD_MODE, so the two behaviours the user compared can be switched
# without touching the code:
#   "record"    -> both ends are excluded the moment a pair is detected (the behaviour
#                  of the 243-combo run, v99R18)
#   "on_engage" -> the tail is excluded always, the head only once the hold is really
#                  pressed (swallows fewer notes, but the user saw BADs with it)
HOLD_ENDPOINT_MODE = "record"
HOLD_ENDPOINT_MODES = ("record", "on_engage")

# --- snapshot of a combo break -------------------------------------------
# The counter tells us THAT the streak died; to see WHY, the frames around it are
# needed. MEASURED COST PROBLEM of the first version: 47 breaks per 48 minutes wrote
# 30 folders of 20 MB (603 MB total) as PNG -- encoding a 1280x720 PNG takes 50-150
# ms, so eight of them per break stalled the machine about once a minute. Now:
#   * frames are written as JPEG (10-20x faster to encode, ~10x smaller);
#   * the ring is shorter (1.5 s of lane crops, 3 full frames every 16th frame);
#   * a snapshot is taken only when a MEANINGFUL streak died (>= COMBO_BREAK_MIN_WAS)
#     and at most once per COMBO_BREAK_MIN_GAP_S -- a 21->3 drop is noise, not the
#     thing worth 20 MB of disk.











KEY_BUTTONS = {
    ord("x"): "CROSS",
    ord("o"): "CIRCLE",
    ord("q"): "SQUARE",
    ord("t"): "TRIANGLE",
    ord("l"): "L2",
    ord("b"): "R2",
    ord("i"): "UP",
    ord("j"): "LEFT",
    ord("k"): "DOWN",
    ord("m"): "RIGHT",
    ord("p"): "OPTIONS",
}

VIDEO_DIR = Path.home() / "Videos" / "WantedDeadRamen"
BOT_LOG = VIDEO_DIR / "bot.log"
ROI_STATE = VIDEO_DIR / "roi_state.json"
LOCK_FILE = Path(tempfile.gettempdir()) / "ramen_trophy_bot.lock"




def load_roi_state():
    """Разметка поля нот: сначала из рабочей папки, потом из папки со скриптом.

    Возвращает (rx, ry, rw, rh, target_lines) или None, если разметки нет.
    """
    for candidate in (ROI_STATE, Path(__file__).with_name("roi_state.json")):
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
            rx, ry, rw, rh = (int(data[k]) for k in ("rx", "ry", "rw", "rh"))
            lines = [(int(a), int(b)) for a, b in data.get("target_lines", [])]
            if rw < 100 or rh < 20:
                continue
            return rx, ry, rw, rh, lines
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def save_roi_state(rx, ry, rw, rh, target_lines) -> None:
    try:
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        ROI_STATE.write_text(json.dumps({
            "rx": int(rx), "ry": int(ry), "rw": int(rw), "rh": int(rh),
            "target_lines": [[int(a), int(b)] for a, b in target_lines],
        }, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def acquire_lock() -> bool:
    """Prevent two bot instances from fighting over the PS5 session.

    PID reuse is a real trap on Windows: a stale lock PID can be handed to an
    unrelated process (seen: it became "smartscreen"), pid_exists() then says
    True forever and the bot refuses to start with no window open. Only honour
    the lock when the live process is actually a Python interpreter.
    """
    import os
    try:
        if LOCK_FILE.exists():
            old = LOCK_FILE.read_text(encoding="utf-8").strip()
            try:
                if old.isdigit() and int(old) != os.getpid():
                    import psutil
                    try:
                        proc = psutil.Process(int(old))
                        pname = (proc.name() or "").lower()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pname = ""
                    if "python" in pname:
                        print(f"Другой экземпляр бота уже запущен (PID {old}). "
                              "Закрой его окно (ESC) и запусти заново.")
                        return False
                    # stale lock (PID reused by another program) -> take it over
            except Exception:
                pass
        LOCK_FILE.write_text(str(os.getpid()), encoding="utf-8")
        return True
    except OSError:
        return True


def release_lock() -> None:
    import os
    try:
        if LOCK_FILE.exists() and LOCK_FILE.read_text(encoding="utf-8").strip() == str(os.getpid()):
            LOCK_FILE.unlink()
    except OSError:
        pass






def log_line(message: str) -> None:
    """Print and append a line to the trophy log."""
    try:
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        with BOT_LOG.open("a", encoding="utf-8") as handle:
            handle.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}\n")
    except OSError:
        pass
    print(message)


SONG_EVENTS: list = []      # per-song diagnostic events, flushed to CSV
SONG_TAG: list = [""]       # wall-clock label of the current song
# A combo-counter OCR thread also appends events, so the journal needs a lock:
# a flush running at the same moment must not lose or duplicate the entry.


def _best_of(track) -> float:
    """Best real template score of a track (internal keys ignored)."""
    scores = getattr(track, "scores_max", None) or {}
    return max((v for k, v in scores.items() if not k.startswith("_")), default=0.0)






@dataclass
class NoteTrack:
    """One moving note, kept distinct from nearby notes and other rows."""
    track_id: int
    x: float
    y: float
    radius: int
    symbol: str
    last_seen: float
    speed: float = 0.0
    fired: bool = False
    press_time: float | None = None
    seen_skull: bool = False   # this note was ever classified as a skull
    frames_seen: int = 1       # how many frames this note was matched in
    moved: bool = False        # seen at a second, closer-to-the-line position
    last_x: float = 0.0        # previous measured x, for the movement test
    counted_lost: bool = False # already counted as a dropped note (once only)
    counted_blocked: bool = False  # already counted as blocked by a press gate
    recorded: bool = False     # already written into the learned pattern
    miss_logged: bool = False  # already logged as a missed note (time analysis)
    hold_blocked: bool = False # skipped at least once inside a detected Hold bar
    stale_logged: bool = False # already journalled as skipped for a stale reading
    endpoint_logged: bool = False  # already journalled as skipped (Hold endpoint)
    # Best template score per symbol over the note's whole lifetime. A note at
    # the press line is often blurred by the hit flash, so the decision must use
    # what the note looked like while it was still approaching.
    scores_max: dict = field(default_factory=dict)


class ButtonDispatcher:
    """Single-threaded button scheduler (no Timer threads).

    Every press/release is queued with a wall-clock time, and the main loop
    calls pump(now) each frame to fire due actions. All physical sends go
    through loop.call_soon_threadsafe, so pyremoteplay's UDP transport is
    only ever written from its own asyncio thread — no concurrent-write
    crash and no multi-thread timing races.
    """
    def __init__(self, controller, loop):
        self.controller = controller
        self.loop = loop
        self.heap: list[tuple[float, int, str, str]] = []  # (when, seq, button, action)
        self._seq = 0
        self.down: set[str] = set()      # buttons physically pressed now
        self.held: set[str] = set()      # hold-pressed (repeatable)
        self.next_free: dict[str, float] = {}  # earliest next press per button
        self.muted = False               # True once the stream is gone: drop sends

    def mute(self) -> None:
        """Stop all physical sends once the stream is dead."""
        self.muted = True
        self.heap.clear()

    def _send(self, button: str, action: str) -> None:
        if self.muted:
            return
        # The controller.button call runs on the asyncio loop thread. If the
        # Remote Play stream just dropped, its UDP transport is already closed
        # (transport._buffer is None) and button() raises AttributeError from
        # inside the callback. Wrap the call so a dropped stream degrades to a
        # silent no-op instead of a traceback storm + lost press.
        def _safe():
            try:
                self.controller.button(button, action)
            except Exception:
                pass
        try:
            self.loop.call_soon_threadsafe(_safe)
        except RuntimeError:
            pass

    def _push(self, when: float, button: str, action: str) -> None:
        self._seq += 1
        heapq.heappush(self.heap, (when, self._seq, button, action))

    def tap(self, button: str, delay: float = .022) -> None:
        # RECORD CONFIGURATION: 22 ms press + 10 ms gap.
        if button in self.held:
            return
        now = time.perf_counter()
        when = max(now, self.next_free.get(button, -1e18))
        self._push(when, button, "press")
        self._push(when + delay, button, "release")
        self.next_free[button] = when + delay + .010

    def press(self, button: str) -> None:
        if button not in self.held:
            self._push(time.perf_counter(), button, "press")
        self.held.add(button)

    def release(self, button: str) -> None:
        if button in self.held:
            self.held.discard(button)
            self._push(time.perf_counter(), button, "release")

    def release_all(self) -> None:
        for button in list(self.down):
            self._send(button, "release")
        self.down.clear()
        self.held.clear()
        self.heap.clear()
        self.next_free.clear()

    def pump(self, now: float) -> None:
        while self.heap and self.heap[0][0] <= now:
            _when, _seq, button, action = heapq.heappop(self.heap)
            if action == "press":
                self.down.add(button)
            elif action == "release":
                self.down.discard(button)
            self._send(button, action)


def associate_note_tracks(
    tracks: list[NoteTrack],
    notes: list[tuple[int, int, int, str]],
    now: float,
    fallback_speed: float,
    lane_width: int,
    next_track_id: int,
    skip_fired: bool = False,
) -> tuple[list[NoteTrack], dict[int, NoteTrack], int, list[float]]:
    """Associate note detections independently across every vertical row."""
    # MEASURED problem: on one 12-minute song the bot created 7236 tracks for
    # ~1194 real notes -- one note kept dying and reappearing as 2-4 tracks.
    # Every such fragment either fires again (a duplicate press, 1986 of them
    # were suppressed) or fires only after the note has already passed the line
    # (949 "late" presses). Two causes, both fixed below:
    #   * a track expired after only 0.8 s of missing detections;
    #   * a track was refused continuation because the glyph read differently in
    #     one frame -- even when that read was weak noise (1029 such splits).
    tracks = [track for track in tracks if now - track.last_seen <= 1.20]
    proposals = []
    # Detections that are simply the SAME note still visible after it was
    # pressed. They must not become new tracks: a fresh track has fired=False,
    # so it fires again a frame later once the note has already passed the line
    # (measured: 885 "late" presses out of 1264 in one song, and 5595 tracks for
    # 601 real notes).
    served_notes: set[int] = set()
    for note_index, (x, y, radius, symbol) in enumerate(notes):
        note_scores = _FRAME_SCORES.get((x, y, radius)) or {}
        # A symbol read counts as CONFIDENT from 0.68 (the same bar the
        # classifier uses for a press). Weak reads must never re-label a track
        # or split it: they are the frames where the glyph is blurred.
        confident = bool(symbol and symbol != "?" and note_scores.get(symbol, 0.0) >= 0.68)
        for track in tracks:
            # A track that already fired (past the hit line) must not swallow
            # the NEXT identical note arriving close behind it: that follower
            # would latch onto a finished track and never press. Skip fired
            # tracks so the follower opens its own track and fires separately.
            # Only for tap notes: holds mark BOTH endpoints fired at press
            # time but still need the tail track for the release, so the hold
            # tracker keeps associating fired tracks.
            if skip_fired and track.fired:
                # ...but if this detection sits exactly where the fired track is,
                # it IS that note, already pressed. Remember it so no new track
                # is opened for it below.
                dt_fired = max(.001, now - track.last_seen)
                speed_fired = track.speed or fallback_speed
                if (abs(x - (track.x - speed_fired * dt_fired)) <= 14
                        and abs(y - track.y) <= max(8, .45 * (radius + track.radius))):
                    served_notes.add(note_index)
                continue
            # A CONFIDENT different symbol means a different note: never latch
            # onto that track (prevents cross-matching in dense runs).
            if confident and track.symbol != "?" and symbol != track.symbol:
                continue
            dt = max(.001, now - track.last_seen)
            speed = track.speed or fallback_speed
            predicted_x = track.x - speed * dt
            dx = abs(x - predicted_x)
            dy = abs(y - track.y)
            # Gate: predicted_x already accounts for the note's motion, so the
            # gate only needs to cover detection jitter and the error of that
            # prediction. The prediction error grows with the gap, so the factor
            # is generous (0.75): a note that flickered out of the detector for a
            # few frames must be picked up again instead of starting a new track.
            x_gate = max(12, speed * dt * .75 + .3 * (radius + track.radius))
            # Two note rows sit ~26px apart vertically. A wide y_gate merges a
            # note of the upper row into the lower row's track, so the second
            # one is never pressed (a Miss). Keep y_gate below the inter-row
            # distance while still absorbing the few-pixel vertical jitter.
            y_gate = max(8, .45 * (radius + track.radius))
            if dx <= x_gate and dy <= y_gate:
                proposals.append((dx + 1.4 * dy, note_index, track.track_id))
    proposals.sort(key=lambda item: item[0])

    by_id = {track.track_id: track for track in tracks}
    matched_notes: set[int] = set()
    matched_tracks: set[int] = set()
    current: dict[int, NoteTrack] = {}
    measured_speeds = []
    for _cost, note_index, track_id in proposals:
        if note_index in matched_notes or track_id in matched_tracks:
            continue
        x, y, radius, symbol = notes[note_index]
        track = by_id[track_id]
        dt = now - track.last_seen
        dx = track.x - x
        if dt > .001 and 0 < dx < lane_width * .12:
            measured = dx / dt
            if 0 < measured < lane_width * 5:
                # speed is fixed; keep per-track speed unset so the constant
                # DEFAULT_SPEED is used everywhere (no fire-line jitter)
                measured_speeds.append(measured)
        track.x, track.y, track.radius = float(x), float(y), int(radius)
        # A real note always travels towards the hit line. Bright static blobs in
        # the lane art (they sit left of the line and are detected again and
        # again at the same x) are the biggest source of junk presses, and they
        # never pass this test.
        if track.frames_seen <= 1:
            track.last_x = float(x)
        elif track.last_x - float(x) >= 4.0:
            track.moved = True
            track.last_x = float(x)
        elif float(x) - track.last_x >= 4.0:
            track.last_x = float(x)   # jumping about: not a note in flight
        note_scores = _FRAME_SCORES.get((x, y, radius))
        # Only a CONFIDENT read may re-label a track. Letting a weak read rename
        # it was what produced duplicate presses: the same note flipped between
        # two buttons, the track symbol changed with it, and the follower frame
        # then looked like a second note.
        if symbol != "?" and note_scores and note_scores.get(symbol, 0.0) >= 0.68:
            track.symbol = symbol
        elif track.symbol == "?" and symbol != "?":
            track.symbol = symbol
        # Accumulate the best score each symbol ever reached for this note.
        if note_scores:
            track.frames_seen += 1
            for name, value in note_scores.items():
                if name.startswith("_"):
                    continue
                if value > track.scores_max.get(name, -1.0):
                    track.scores_max[name] = value
        # Remember a skull forever: a skull misread as a button glyph far from
        # the line must never be pressed later (that ends the combo).
        if symbol == "SKULL" or track.symbol == "SKULL":
            track.seen_skull = True
        # ...and remember WHERE it was seen. Note tracking deliberately starts a
        # new track when a note's symbol changes (dense clusters), so a skull
        # that reads as SQUARE on its last frames becomes a fresh track with a
        # clean `seen_skull` flag -- and that is how a skull used to get pressed.
        # A short-lived spatial memory survives that fragmentation.
        # RECORD BEHAVIOUR: no confidence bar here (adding one was one of the
        # rolled-back experiments).
        if symbol == "SKULL":
            skull_sightings.append((now, float(x), float(y)))
        track.last_seen = now
        current[note_index] = track
        matched_notes.add(note_index)
        matched_tracks.add(track_id)

    for note_index, (x, y, radius, symbol) in enumerate(notes):
        if note_index in matched_notes:
            continue
        if note_index in served_notes:
            # Same physical note as an already-pressed track: drop it silently.
            TRACK_STATS["served_skip"] += 1
            continue
        # Instrumentation only: why did this detection open a NEW track instead
        # of continuing an existing one? Two reasons are possible -- the symbol
        # rule split a continuous note in two, or the note is genuinely new.
        # One real note that keeps being re-created as several tracks is a
        # pressed-twice/Missed-twice source, so the split count matters.
        split = False
        for track in tracks:
            if skip_fired and track.fired:
                continue
            dt = max(.001, now - track.last_seen)
            speed = track.speed or fallback_speed
            x_gate = max(10, speed * dt * .5 + .3 * (radius + track.radius))
            y_gate = max(8, .45 * (radius + track.radius))
            if (abs(x - (track.x - speed * dt)) <= x_gate
                    and abs(y - track.y) <= y_gate):
                split = True
                break
        TRACK_STATS["symbol_split" if split else "new_clean"] += 1
        note_scores = _FRAME_SCORES.get((x, y, radius)) or {}
        track = NoteTrack(next_track_id, float(x), float(y), int(radius),
                          symbol, now, fallback_speed,
                          seen_skull=(symbol == "SKULL"),
                          scores_max={name: value for name, value in note_scores.items()
                                      if not name.startswith("_")})
        next_track_id += 1
        tracks.append(track)
        current[note_index] = track
    return tracks, current, next_track_id, measured_speeds


def make_templates() -> dict[str, np.ndarray]:
    """Load a small set of clean note crops, including saved unknown samples."""
    path = Path(__file__).with_name("ramen_templates.npz")
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл шаблонов {path.name} рядом со скриптом")
    data = np.load(path, allow_pickle=False)
    # RECORD CONFIGURATION: full set used by v32/v33.
    limits = {"O": 4, "L2": 6, "R2": 8, "X": 5, "SQUARE": 6}
    return {name: data[name][:limits.get(name, 2)] for name in data.files}


def is_yellow_note(lane: np.ndarray, x: int, y: int, radius: int) -> bool:
    """Detect the rare golden/yellow note (always single). Its glyph is a
    yellow-green (or orange when pulsing) circle -> mapped to the CIRCLE button.

    IMPORTANT: the note field itself is orange-red (hue ~13). A wide mask (h 8-75)
    therefore matched ORDINARY notes and turned them into CIRCLE -- that alone
    produced a flood of misses. The window is kept narrow and high (20-70), i.e.
    clearly above the lane background, and needs a big share of the patch (18%).
    """
    height, width = lane.shape[:2]
    half = max(5, round(radius * 1.1))
    patch = lane[max(0, y - half):min(height, y + half + 1),
                 max(0, x - half):min(width, x + half + 1)]
    if patch.size == 0:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = (h >= 20) & (h <= 70) & (s > 90) & (v > 110)
    return float(mask.mean()) > 0.18


def classify_note(gray: np.ndarray, x: int, y: int, radius: int,
                  templates: dict[str, np.ndarray],
                  lane: np.ndarray | None = None,
                  scores_out: dict | None = None) -> str:
    height, width = gray.shape
    scores_by_name = {name: -1.0 for name in templates}
    margin = 2
    # RECORD CONFIGURATION (v32/v33: Score 660550, Max Combo 122):
    # three scales, full template set, thresholds 0.58 / 0.68.
    contrast = 0.0
    for ratio in (.72, .95, 1.10):
        half = max(4, round(radius * ratio))
        crop = gray[max(0, y - half - margin):min(height, y + half + margin + 1),
                    max(0, x - half - margin):min(width, x + half + margin + 1)]
        if crop.size == 0:
            continue
        search = cv2.resize(crop, (40, 40), interpolation=cv2.INTER_CUBIC)
        contrast = max(contrast, float(search.std()))
        for name, examples in templates.items():
            score = max(float(cv2.matchTemplate(search, example, cv2.TM_CCOEFF_NORMED).max())
                        for example in examples)
            scores_by_name[name] = max(scores_by_name[name], score)
    scores = sorted(((score, name) for name, score in scores_by_name.items()), reverse=True)
    if scores_out is not None:
        scores_out.clear()
        scores_out.update(scores_by_name)
        scores_out["_std"] = contrast
    if not scores or scores[0][0] < 0.50:
        # Very weak matches are usually reflections/background circles. The golden
        # note is the exception: its colour lowers the template score.
        # NB: this path must never return SKULL.
        if lane is not None and is_yellow_note(lane, x, y, radius) and scores:
            yellow_buttons = [name for _score, name in scores if name != "SKULL"]
            if yellow_buttons:
                return yellow_buttons[0]
        return ""
    top_score, top_name = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else -1.0
    margin = top_score - second_score
    # Flat, featureless crops are not glyphs at all: they come from Hough
    # circles found on a bright blob (the plate gauge, a bloomed nota) and
    # their normalised correlation is meaningless. Real note crops measured
    # std 50-90, blobs 23-30. Dropping them keeps the "?" bucket meaningful.
    if contrast and contrast < 40.0:
        return ""
    # MEASURED on a live song (140 classified notes): real notes peak at
    # 0.55-0.67 with a 0.15-0.30 margin over the runner-up, so the old rule
    # ">= 0.68 or nothing" threw away roughly a third of perfectly readable
    # notes as "?" -- and a skipped note is a Miss, the very thing the full
    # combo trophy forbids. A clear winner from a lower score is accepted;
    # only genuinely torn classifications stay "?".
    #
    # SKULL is special-cased twice over. Measured on 25 refused notes, real
    # skulls score 0.76-0.91 with a huge margin, while every false skull was a
    # blurred frame where SKULL "won" with 0.30-0.56 and no margin at all. An
    # unconditional "ever saw SKULL -> never press" rule therefore skipped real
    # notes (each one a Miss). A skull is only trusted above SKULL_MIN_SCORE;
    # below that the skull template is ignored and the buttons compete normally.
    button_scores = sorted(((score, name) for score, name in scores
                            if name != "SKULL"), reverse=True)
    # MEASURED from the saved miss frames: two CROSS/CIRCLE notes were refused as
    # skulls with SKULL=0.91, and the very same crops re-read as "O" with a high
    # score. So a high skull score alone is NOT enough -- the skull must also beat
    # the best button, otherwise a circular note is thrown away (a real Miss).
    best_button_score = button_scores[0][0] if button_scores else -1.0
    if top_name == "SKULL" and top_score >= SKULL_MIN_SCORE \
            and best_button_score < top_score - SKULL_BUTTON_MARGIN:
        return "SKULL"
    if not button_scores:
        return "?"
    best_button, button_name = button_scores[0]
    second_button = button_scores[1][0] if len(button_scores) > 1 else -1.0
    button_margin = best_button - second_button
    if best_button >= 0.68 and button_margin >= 0.08:
        return button_name
    if best_button >= 0.55 and button_margin >= 0.13:
        return button_name
    # LAST RESORT: still press the best guess. For the full-combo trophy a wrong
    # button (Bad) and a skipped note (Miss) are equally fatal, so guessing is
    # never worse and is right maybe half the time. Only a genuine tie is left
    # as "?" -- the bot has no preference there.
    if best_button >= 0.50 and button_margin >= 0.03:
        return button_name
    return "?"


def detect_notes_by_blob(lane: np.ndarray,
                         templates: dict[str, np.ndarray]) -> list[tuple[int, int, int, str]]:
    """Find notes as bright connected components (white notes on a dark lane).

    HoughCircles merges/misses two adjacent notes (their glows touch), but
    connected components keep them separate as long as there is a dark gap
    between them -- which is exactly the "two/three in a row" case. Each kept
    component is re-classified with the same template matcher, so the symbol
    contract is identical to detect_notes().
    """
    gray = cv2.cvtColor(lane, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    # Adaptive-ish threshold: notes are bright; the lane is dark. Use a fixed
    # high threshold so only the note glyphs (not the dim lane texture) pass.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu may pick a low level if the lane has bright artwork; bias upward so
    # only near-white note pixels remain.
    mean_bright = float(gray.mean())
    thresh = max(160.0, min(235.0, mean_bright + 40.0))
    binary = (gray >= thresh).astype(np.uint8) * 255
    # Close small gaps inside a single glyph (the letter stroke), without
    # bridging the dark gap between two adjacent notes.
    k = max(2, width // 300)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    found: list[tuple[int, int, int, str]] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        # A note is a compact roundish blob. Drop tiny specks and huge bars
        # (the MASH bar / hold bars are wide; notes are roughly square).
        if area < 60:
            continue
        if w < 10 or h < 10:
            continue
        if w > h * 3 or h > w * 3:
            continue
        cx = x + w // 2
        cy = y + h // 2
        radius = max(10, (w + h) // 4)
        symbol = classify_note(gray, cx, cy, radius, templates, lane)
        if not symbol:
            continue
        found.append((cx, cy, radius, symbol))
    # merge overlapping detections, keep distinct notes separate
    unique: list[tuple[int, int, int, str]] = []
    for note in sorted(found, key=lambda item: (item[3] != "?", item[2]), reverse=True):
        dup = None
        for index, prev in enumerate(unique):
            d2 = (note[0] - prev[0]) ** 2 + (note[1] - prev[1]) ** 2
            if d2 <= max(6, round(.5 * (note[2] + prev[2]))) ** 2:
                dup = index
                break
        if dup is None:
            unique.append(note)
    unique.sort(key=lambda note: note[0])
    return unique


def detect_notes(lane: np.ndarray, templates: dict[str, np.ndarray]) -> list[tuple[int, int, int, str]]:
    _FRAME_SCORES.clear()
    gray = cv2.cvtColor(lane, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
    height = gray.shape[0]
    min_radius = max(10, height // 20)
    max_radius = min(45, max(22, height // 5))
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, 1.0, max(8, height // 45),
        param1=100, param2=24, minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return []
    hsv = cv2.cvtColor(lane, cv2.COLOR_BGR2HSV)
    pink = cv2.inRange(hsv, np.array([130, 35, 120]), np.array([179, 255, 255]))
    found: list[tuple[int, int, int, str]] = []
    for x, y, radius in np.round(circles[0]).astype(int):
        # The selected ROI may contain one or several note rows. Keep a small
        # border margin, then let note matching and temporal tracking reject
        # circles from the artwork.
        if y < height * .12 or y > height * .88:
            continue
        # Exclude the magenta spiral hit marker and the pink gauge, but NOT a
        # white note that merely passes next to them. Notes are bright and
        # unsaturated; the marker/gauge are magenta. Checking the circle's
        # CENTRE keeps notes alive all the way to the hit line -- the old
        # ">1.2% pink anywhere nearby" test deleted every note approaching the
        # marker, which is exactly where it must be pressed.
        c = max(2, radius // 2)
        center = lane[max(0, y - c):min(height, y + c + 1),
                      max(0, x - c):min(lane.shape[1], x + c + 1)]
        pad = max(4, radius)
        marker_patch = pink[max(0, y - pad):min(height, y + pad + 1),
                            max(0, x - pad):min(pink.shape[1], x + pad + 1)]
        pink_frac = (cv2.countNonZero(marker_patch) / float(marker_patch.size)
                     if marker_patch.size else 0.0)
        is_white_note = False
        if center.size:
            chsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
            v_mean = float(chsv[:, :, 2].mean())
            s_mean = float(chsv[:, :, 1].mean())
            is_white_note = v_mean > 150 and s_mean < 110
        if not is_white_note:
            # A golden note is saturated yellow, so it is not "white" either;
            # keep it alive near the marker just like the white notes.
            if not is_yellow_note(lane, x, y, radius) and pink_frac > .012:
                continue
        note_scores: dict = {}
        symbol = classify_note(gray, x, y, radius, templates, lane, note_scores)
        _FRAME_SCORES[(x, y, radius)] = note_scores
        if symbol:
            if symbol == "?":
                pass
            found.append((x, y, radius, symbol))
    unique: list[tuple[int, int, int, str]] = []
    # RECORD CONFIGURATION: merge distance used by v32/v33.
    for note in sorted(found, key=lambda item: (item[3] != "?", item[2]), reverse=True):
        duplicate = None
        for index, previous in enumerate(unique):
            distance2 = (note[0] - previous[0]) ** 2 + (note[1] - previous[1]) ** 2
            merge_radius = max(8, round(.30 * (note[2] + previous[2])))
            if distance2 <= merge_radius ** 2:
                duplicate = index
                break
        if duplicate is None:
            unique.append(note)
    unique.sort(key=lambda note: note[0])
    return unique


def merge_notes(hough: list[tuple[int, int, int, str]],
                nn: list[tuple[int, int, int, str]],
                gray: np.ndarray | None = None,
                templates: dict[str, np.ndarray] | None = None,
                lane: np.ndarray | None = None) -> list[tuple[int, int, int, str]]:
    """Combine NN detection (positions) with template-matching classification.

    The NN finds individual notes in dense clusters (where Hough merges/misses
    them), but its own symbol classification is unreliable on real frames
    (heavy L2/R2 confusion). So every NN note is re-classified with template
    matching at the NN's centre; a note whose crop matches no template (empty
    result) is dropped as a likely false positive. A Hough note is added only
    when no NN note is near it, so a Hough "merged" circle overlapping two NN
    notes is dropped in favour of the two separate NN boxes.
    """
    merged: list[tuple[int, int, int, str]] = []
    for nx, ny, nradius, nsymbol in nn:
        symbol = nsymbol  # NN's own class is the fallback
        if gray is not None and templates is not None:
            cls = classify_note(gray, nx, ny, nradius, templates, lane)
            # Trust the template only when it actually recognizes the glyph.
            # A "" (no match) or "?" (ambiguous) result must NOT drop the note
            # to Miss — keep the NN class and press anyway (a wrong button is
            # a Bad, which is always better than a Miss).
            if cls not in ("", "?"):
                symbol = cls
        merged.append((nx, ny, nradius, symbol))
    for hx, hy, hradius, hsymbol in hough:
        if any(abs(hx - nx) <= 12 and abs(hy - ny) <= 9
               for nx, ny, _nr, _ns in nn):
            continue
        merged.append((hx, hy, hradius, hsymbol))
    merged.sort(key=lambda note: note[0])
    return merged


def merge_hough_blob(hough: list[tuple[int, int, int, str]],
                     blob: list[tuple[int, int, int, str]]) -> list[tuple[int, int, int, str]]:
    """Combine Hough (single notes) with blob (dense clusters) detections.

    The blob detector finds the two/three-in-a-row notes that Hough merges.
    Keep a Hough note only when no blob note is near it (a blob note is more
    precise for dense clusters); otherwise the blob note wins.
    """
    merged: list[tuple[int, int, int, str]] = list(blob)
    for hx, hy, hradius, hsymbol in hough:
        if any(abs(hx - bx) <= max(12, round(.6 * (hradius + br)))
               and abs(hy - by) <= max(10, round(.8 * (hradius + br)))
               for bx, by, br, _bs in blob):
            continue
        merged.append((hx, hy, hradius, hsymbol))
    merged.sort(key=lambda note: note[0])
    return merged


# Guards for a cyan contour to count as a hold bar (see detect_hold_spans).
HOLD_MAX_HEIGHT = 34          # MEASURED on saved frames: a real cyan bar is
                              # 26-28 px tall, so 26 would have rejected every
                              # genuine bar; a merge with the HUD is much taller
HOLD_ROW_TOL = 16             # its centre must sit on a known note row


def detect_hold_spans(lane: np.ndarray,
                      rows: list[int] | None = None) -> list[tuple[int, int, int]]:
    """Find cyan Hold badges/bars; results are (left, right, row) in lane pixels.

    MEASURED asymmetry that this fixes: across four songs the UPPER row (y=44)
    produced 252 hold pairs of which 42% never pressed, while the LOWER row (y=84)
    produced 118 pairs with only 19% failures -- twice as many "holds" detected on
    the upper row and twice the failure rate. The upper row sits just 44 px below
    the top of the 110 px lane strip with the game HUD above it, so cyan HUD
    elements land in the mask as contours that satisfy the old test (width >= 50
    and w > 2h). When such a contour touches a real bar the centre row drifts up and
    the pair is built from the wrong notes.

    Two guards, both cheap:
      * the contour's centre must lie within HOLD_ROW_TOL px of a KNOWN row;
      * the contour must be thin (height <= HOLD_MAX_HEIGHT): a real bar is a thin
        strip, a HUD blob or a merge with it is tall.
    """
    hsv = cv2.cvtColor(lane, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([72, 70, 70]), np.array([108, 255, 255]))
    width = lane.shape[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, width // 45), 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    found = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]
    spans = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # On the live stream, only the bright cyan Hold badge may pass the
        # color threshold; the thin connecting bar can be nearly gray.
        if w >= max(50, int(width * .045)) and w > h * 2:
            spans.append((x, x + w, y + h // 2))
    return spans


def detect_mash_note(lane: np.ndarray) -> bool:
    """Detect the red MASH bar (the strip the bot must not touch at all).

    The MASH prompt is a wide PURE-RED bar (hue < 10). The note field itself is
    orange-red (hue ~13), so the threshold must be tight to avoid false
    positives on the ordinary lane. Measured on the screenshot: MASH ~26% pure
    red, ordinary lane < 1%.

    Eagerly tuned: the bar fades in and out, and any frame where it is visible
    must count as "inside the strip", because a press on such a frame is exactly
    what the user saw ("мэш пробился"). The coverage bar for the "wide" test is
    therefore a fifth of the lane, not a quarter.
    """
    hsv = cv2.cvtColor(lane, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red = (h < 10) & (s > 100) & (v > 100)
    if red.mean() < 0.02:
        return False
    col_red = red.sum(axis=0)
    wide = int((col_red > 8).sum())
    return wide > lane.shape[1] * 0.15


def pair_hold_notes(hold_spans: list[tuple[int, int, int]],
                    notes: list[tuple[int, int, int, str]]) -> list[tuple[tuple[int, int, int, str], tuple[int, int, int, str]]]:
    """Pair the note endpoints immediately outside each cyan Hold span."""
    pairs = []
    for left, right, row in hold_spans:
        aligned = [note for note in notes
                   if abs(note[1] - row) < max(18, note[2])]
        before = [note for note in aligned if note[0] < left]
        after = [note for note in aligned if note[0] > right]
        if not before or not after:
            continue
        left_note = max(before, key=lambda note: note[0])
        right_note = min(after, key=lambda note: note[0])
        max_gap = max(55, 4 * max(left_note[2], right_note[2]))
        if left - left_note[0] > max_gap or right_note[0] - right > max_gap:
            continue
        pairs.append((left_note, right_note))
    return pairs


def note_index_for_position(notes: list[tuple[int, int, int, str]],
                            x: float, y: float, radius: int) -> int | None:
    """Find the index in `notes` nearest to (x, y) within a small gate.

    Hold endpoints are paired on Hough-only notes, but the note tracks are built
    from the merged (Hough+NN) list; this maps a hold endpoint back to its
    merged-note index so the firing code can reach its track.
    """
    best_i = None
    best_d = float("inf")
    for i, (nx, ny, _nr, _ns) in enumerate(notes):
        d = (nx - x) ** 2 + (ny - y) ** 2
        if d < best_d:
            best_d = d
            best_i = i
    if best_i is None or best_d > max(20, radius) ** 2:
        return None
    return best_i



# Score vector of the CURRENT frame's classifications, keyed by (x, y, radius).
# Used to explain presses: a skull that is pressed must show up here as
# "SKULL was the second best match".
# A skull classification is only trusted above this template score. Real skulls
# measured 0.76-0.91; blurred junk frames that "looked" like skulls scored
# 0.30-0.56 and only ever caused real notes to be skipped (a Miss).
# MEASURED with 0.68: of 81 refusals in one song only 12 were >= 0.76 and 67 sat
# in the 0.68-0.76 band -- i.e. ordinary notes thrown away, and the song had 112
# Misses in total. The bar is therefore set at the bottom of the real-skull band.
SKULL_MIN_SCORE = 0.76
# ...and the skull must beat the best button by this much. Saved miss frames showed
# a CIRCLE note refused as a skull at 0.91 while the same crop read as O, so a
# straight tie is a button, not a skull.
# Set to -1 to DISABLE the margin, which restores the v99 behaviour (that build
# set the record: Score 1 689 950, 4 misses up to the fifth plate).
SKULL_BUTTON_MARGIN = -1.0

# Skip the first N notes of every song, then press non-stop. Community advice
# for "Ramen World Legend": the opening of the first plate is the messiest part
# (notes come out of the intro transition), and the combo that counts for the
# trophy is the uninterrupted chain that follows -- with a clean chain the
# trophy is reportedly awarded as early as the 4th plate. 0 disables the skip.
SKIP_FIRST_NOTES = 10

# Notes ride FIXED rows (the marker row and the second row ~40 px below it).
# Offline bench on the reference full-combo video: pressing only what sits on a
# row cut the surplus presses from 90 to 73 per 30 s of video (53 real notes)
# and removed every single-frame press. Junk from the lane artwork is scattered
# vertically, so it never passes this test. Widened a little for safety.
ROW_TOLERANCE_PX = 15
ROW_SPACING_PX = 40

# Distance (px) inside which a second press counts as a duplicate of the same
# note. 11 px ~ 45 ms of scroll at 240 px/s: a duplicate crosses within a few
# pixels, a real second note of a dense pair is 20-40 px away.
# Duplicate-press window, in seconds. A second track for the SAME note crosses
# within 1-2 frames (17-35 ms at 30 fps); two different notes are >=80 ms apart.
# Positions cannot be used: every press happens at the hit line, so positions are
# all identical. MEASURED at 80 ms: 65 real notes were still blocked here, so the
# window was narrowed to 50 ms -- still above a true duplicate, below a real pair.
DEDUPE_S = 0.03
# Rows are 40 px apart, so this only decides "same row or not" for the duplicate
# guard. Kept separate from ROW_TOLERANCE_PX so the two knobs stay independent.
DEDUPE_ROW_PX = 10
# MEASURED: 68 times per song the SAME note is pressed twice, ~232 ms apart, with
# the same button, the same row and almost the same x (39.72 CIRCLE x=244 ->
# 39.95 CIRCLE x=244). Real notes in one row are 270-300 px apart (~1.4 s), so the
# second press cannot be a new note: the game keeps the note visible briefly after
# the hit, the tracker opens a fresh track for it, and the second press lands with
# nothing in front of it -- which the game counts as a Miss. A genuine duplicate
# track is BORN at the line (a handful of frames, x already near the target), while
# a real note travels in from the right and is matched in dozens of frames.
DEDUPE_LATE_S = 0.32          # how far back to look for that second press
DEDUPE_BORN_FRAMES = 6        # a track matched this few times was born late
DEDUPE_BORN_PX = 40           # ...and this close to the hit line

# A note whose detection is older than this is not pressed: the prediction is
# unreliable and the press would land past the line (a stray press).
# MEASURED on the 858 s run of v99: the game counted 154 Misses while only 28 of
# our notes were "visible misses", so the rest are presses landing too late. At
# 211-330 px/s a 30 px margin is 90-140 ms -- past any judgement window. Hence
# the tighter limits: fresher prediction, and a much smaller late margin.
STALE_TRACK_S = 0.15
# Do not press a note that is already this many pixels PAST the hit line.
LATE_MARGIN_PX = 30

# Press gates derived from the reference full-combo video. Profiling presses
# there gave a perfect separation:
#   real note  -> tracked for a median of 111 frames, moving in 92% of them,
#                 best template score 0.89
#   junk press -> 1 frame, not moving in 97% of them, best score 0.53
# Requiring a score of 0.65 there cut presses from 158 to 80 with a 100% hit
# rate (orphan presses 78 -> 5). The score bar is lower here because the live
# 720p stream is blurrier: live real notes measure 0.55-0.68 while junk sits
# around 0.53, so 0.55 keeps real notes and still drops the weakest noise.
MIN_TRACK_FRAMES = 1
# MEASURED on a 742 s run: of 676 notes that reached the hit line unpressed, 476
# were blocked by THIS score gate ("weak"). The burst detector then showed WHY:
# every burst happened while the screen was DARK (brightness 39-49, below the 55
# darkness threshold) -- on those frames the notes are faded, their template
# score collapses and the gate throws them away. For the trophy a wrong button
# (Bad) and a skipped note (Miss) are equally fatal, so the bar is now low enough
# to press dimmed notes instead of discarding them.
# Tried 0.15 as well and it made things WORSE (Miss 8.7% -> 11.1%): notes read
# that badly are mostly junk, and pressing junk steals the next real note.
MIN_NOTE_SCORE = 0.30

# Replay a recorded note pattern by timing instead of pressing each note as it is
# detected at the hit line.
# MEASURED AND REJECTED for now: the recorded pattern is tied to ONE run. The
# ramen round is endless until you lose -- run lengths here were 127, 344 and 824
# seconds -- so the note sequence differs in length and position between runs and
# the replay fired 195 presses for ZERO hits. Kept behind the flag: it can be
# retried once the plate/section alignment is known (the community says patterns
# repeat from plate 5 on, so a plate-aware sync would be needed).

# Remote Play stream bitrate. "very_high" is 15 Mbit/s (the default is 10).
# Notes are ~24 px with a ~10 px glyph, so a sharper stream directly improves
# template scores -- the live data shows only ~22% of tracks read confidently,
# against ~97% symbol stability on the (high quality) reference video.
SESSION_QUALITY = "very_high"

# MASH presses are switched OFF by request. The bot used to mash CROSS every
# 55 ms while the red MASH bar was on screen -- the only completely blind button
# path in the bot, and the only way a skull could be pressed without the bot
# ever looking at it. With this False the bot ignores the bar entirely and keeps
# pressing the ordinary notes that scroll by. Set to True to bring it back.
MASH_PRESS_ENABLED = False

# While the red MASH strip is on screen, keep pressing the ORDINARY notes
# (never a mash tap). Measured: with notes suppressed inside the strip, 79 of
# 120 misses in a single song happened there -- notes keep scrolling under the
# bar. Only confidently read notes are pressed, so the strip's own phantom
# detections cannot turn into presses.
PRESS_NOTES_IN_MASH = True
# USER DECISION (final): while the red MASH strip is on screen NOTHING is pressed at
# all -- no mash taps (never) and no ordinary notes either. Watching the game the user
# saw the strip make the bot press notes at its edges, i.e. presses landing with
# nothing in front of them. The strip is therefore a FULL skip for as long as it is
# visible: notes and holds are dropped, and the frame is still measured
# (mash_on / mash_off in song_events.csv) so its cost stays visible in the numbers.

# IGNORE_MASH: the strip must not change ANYTHING about how notes are handled.
# MEASURED cause of the periodic collapse: every burst of misses in a run lined
# up with the red strip being on screen (bursts came every ~52 s, Max Combo
# stopped at 88/113/202, i.e. at exactly one burst). While the strip was up the
# bot dropped every weakly-read note ("weak" dominates the burst reasons even
# though those notes sit right on the hit line), so the strip was effectively
# blinding the bot for a few seconds at a time. The user's verdict: the trophy
# does not need the mash at all -- ignore the prompt completely and press notes
# exactly as on any other frame. The strip is still MEASURED (mash_on/mash_off
# events in song_events.csv) so its cost stays visible in the numbers.
IGNORE_MASH = True

# Split notes that HoughCircles merged into one circle (two touching notes).
# Running the connected-component detector on EVERY frame re-classifies every
# blob and cost ~20 ms/frame (loop fell from 30 ms to 60 ms, i.e. 16 fps, and
# Miss jumped to 28%). It is therefore applied only on frames where a Hough
# circle actually looks like a merged pair, which is cheap to detect.
USE_BLOB_DETECTOR = True


def has_merged_pair(lane: np.ndarray,
                    notes: list[tuple[int, int, int, str]]) -> bool:
    """True when a detected circle is much wider than tall (two notes merged)."""
    if not notes:
        return False
    gray = cv2.cvtColor(lane, cv2.COLOR_BGR2GRAY)
    thresh = max(160.0, min(235.0, float(gray.mean()) + 40.0))
    bright = (gray >= thresh)
    height, width = bright.shape
    for x, y, radius, _symbol in notes:
        band = bright[max(0, y - 2):min(height, y + 3), :]
        if band.size == 0:
            continue
        columns = band.any(axis=0)
        lo = max(0, x - 3 * radius)
        hi = min(width, x + 3 * radius)
        run = 0
        best = 0
        for value in columns[lo:hi]:
            run = run + 1 if value else 0
            best = max(best, run)
        if best > 3.4 * radius:
            return True
    return False

_FRAME_SCORES: dict[tuple[int, int, int], dict] = {}
# Track-continuity counters (diagnostics for Miss analysis):
#   symbol_split -- a detection that would have continued an existing track but
#                   was rejected because its symbol read differently (so one
#                   real note becomes several tracks)
#   new_clean    -- a detection with no positional match at all
TRACK_STATS: dict[str, int] = {"symbol_split": 0, "new_clean": 0, "served_skip": 0}
# Skull refusals split by how sure the skull reading was (see _log_skull_skip).
SKULL_STATS: dict[str, int] = {"strong": 0, "weak": 0}
# (time, x, y) of every confident skull sighting, in lane coordinates. Consumed
# by the press gate so a re-tracked skull cannot slip through as a button.
skull_sightings: list[tuple[float, float, float]] = []
SKULL_MEMORY_S = 1.2


def _fmt_scores(scores: dict) -> str:
    items = sorted(((value, name) for name, value in scores.items()
                    if not name.startswith("_")), reverse=True)
    text = " ".join(f"{name}={value:.2f}" for value, name in items[:4])
    if "_std" in scores:
        text += f" std={scores['_std']:.0f}"
    return text


























def skull_nearby(now: float, x: float, y: float, speed: float) -> bool:
    """True if a confident skull was seen at this spot a moment ago.

    Each sighting is extrapolated with the scroll speed, so a skull seen at
    x=400 half a second ago still matches the note now sitting at x=280.
    """
    if not skull_sightings:
        return False
    keep_from = now - SKULL_MEMORY_S
    skull_sightings[:] = [item for item in skull_sightings if item[0] >= keep_from]
    for seen_at, seen_x, seen_y in skull_sightings:
        predicted = seen_x - speed * (now - seen_at)
        if abs(predicted - x) <= 22 and abs(seen_y - y) <= 12:
            return True
    return False








def discover() -> None:
    try:
        devices = RPDevice.search()
    except Exception as exc:  # UDP-широковещание ломается в части сетей и на VPN
        print(f"Автопоиск не удался ({exc}).")
        print("IP консоли можно узнать на PS5: Настройки → Система → Сеть → "
              "Состояние соединения → IP-адрес.")
        return
    if not devices:
        print("PS5 не обнаружена. Проверь, что ПК и PS5 в одной локальной сети.")
        return
    for device in devices:
        print(f"{device.host}: {device.host_name or 'PlayStation'} [{device.host_type}]")


def register(host: str) -> None:
    """Run the library's own PSN and console-link setup prompts."""
    sys.argv = ["pyremoteplay", host, "--register"]
    pyremoteplay_cli()


# ---------------------------------------------------------------------------
# Trophy helpers: results-screen OCR
# ---------------------------------------------------------------------------

STAT_ROWS = ("Perfect", "Great", "Good", "Bad", "Miss", "Max Combo", "Refills")


def parse_results_stats(ocr_lines: list[tuple[int, int, int, int, str]]) -> dict:
    """Parse RapidOCR output into a stats dict.

    ocr_lines: list of (x1, y1, x2, y2, text).
    """
    stats = {key: None for key in STAT_ROWS}
    stats["total"] = None
    stats["raw"] = []

    def number_near(cx1, cy1, cx2, cy2):
        """Find the closest number box right of the row label (same line)."""
        label_cy = (cy1 + cy2) / 2.0
        best, best_dist, best_x = None, 1e9, 1e9
        for x1, y1, x2, y2, text in ocr_lines:
            if x1 <= cx1 + 10 or x1 > cx1 + 480:
                continue
            digits = "".join(ch for ch in text.upper().replace("O", "0")
                             if ch.isdigit())
            if not digits:
                continue
            dist = abs((y1 + y2) / 2.0 - label_cy)
            if dist < 40 and (dist < best_dist or
                              (dist == best_dist and x1 < best_x)):
                best, best_dist, best_x = int(digits), dist, x1
        return best

    for x1, y1, x2, y2, text in ocr_lines:
        compact = "".join(ch for ch in text.upper() if ch.isalnum())
        stats["raw"].append(text)
        for row in STAT_ROWS:
            key = "".join(ch for ch in row.upper() if ch.isalnum())
            names = {key, key + "TIMES"}
            if row == "Max Combo":
                names.update(("MAXCOMBO", "MAXCOMBOTIMES", "MAXCOMBOCOMBOTIMES"))
            if compact in names:
                stats[row] = number_near(x2, y1, x2, y2)
    # total score line sits below the rows, roughly y 500-600 in a 720p frame
    for x1, y1, x2, y2, text in ocr_lines:
        digits = "".join(ch for ch in text if ch.isdigit())
        if digits and 480 < (y1 + y2) / 2 < 620 and 400 < x1 < 800:
            stats["total"] = int(digits)
            break
    return stats


def ocr_frame_stats(image) -> dict | None:
    """OCR the results screen; returns the stats dict or None."""
    if _OCR_ENGINE is None:
        return None
    try:
        result, _ = _OCR_ENGINE(image)
    except Exception:
        return None
    if not result:
        return None
    lines = []
    for box, text, _score in result:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        lines.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)), text))
    stats = parse_results_stats(lines)
    stats["raw"] = [line[4] for line in lines]
    return stats


def format_stats(stats: dict) -> str:
    parts = []
    for row in STAT_ROWS:
        if stats.get(row) is not None:
            parts.append(f"{row}={stats[row]}")
    if stats.get("total") is not None:
        parts.append(f"Score={stats['total']}")
    return ", ".join(parts)






# ---------------------------------------------------------------------------
# Main bot loop
# ---------------------------------------------------------------------------



def run_bot(host: str, lead_ms: float, record_enabled: bool = True,
            trophy_mode: bool = False, auto_retry_delay: float = 8.0,
            auto_menu: bool = False, adaptive_lead: bool = False,
            lead_step: float = 10.0, song_end_wait: float = 12.0,
            replay_chart: str | None = None,
            learn_file: str | None = None,
            assist_mode: bool = False,
            nn_model: str | None = None) -> None:
    if not acquire_lock():
        raise SystemExit(1)
    def _connect_once():
        """Create a device+session and connect. Raises RuntimeError on failure."""
        dev = RPDevice(host)
        status = dev.get_status()
        if not status:
            raise RuntimeError("PS5 не отвечает по Remote Play. Проверь IP и сеть.")
        users = dev.get_users()
        if not users:
            raise RuntimeError("Нет связанного профиля pyremoteplay. Сначала запусти --register.")
        recv = QueueReceiver(max_frames=3)
        # 60 fps, 720p. The bitrate is a real lever for detection quality: notes
        # are tiny (about 24 px) and the glyph inside them is ~10 px, so every
        # compression artefact directly eats the template score. VERY_HIGH is
        # 15 Mbit/s instead of the 10 Mbit/s default (the enum also allows 2/4/6
        # Mbit, and arbitrary numbers are rejected by pyremoteplay).
        if not dev.create_session(users[0], receiver=recv, resolution="720p",
                                  fps="high", quality=SESSION_QUALITY):
            raise RuntimeError("Не удалось создать сессию Remote Play.")
        lp = asyncio.new_event_loop()
        res: list[bool] = []
        done_ev = threading.Event()

        def worker():
            asyncio.set_event_loop(lp)
            try:
                res.append(lp.run_until_complete(dev.connect()))
                done_ev.set()
                if res[-1]:
                    lp.run_forever()
            except Exception as exc:  # surface connection errors in the main window
                res.append(False)
                print(f"Ошибка Remote Play: {exc}")
                done_ev.set()
            finally:
                if not lp.is_closed():
                    try:
                        lp.close()
                    except Exception:
                        pass

        threading.Thread(target=worker, daemon=True).start()
        if not done_ev.wait(45):
            dev.disconnect()
            try:
                lp.call_soon_threadsafe(lp.stop)
            except Exception:
                pass
            raise RuntimeError("Remote Play не подключился за 45 секунд.")
        if not res or not res[0]:
            reason = dev.session.error if dev.session else "неизвестная ошибка"
            dev.disconnect()
            raise RuntimeError(f"Remote Play не подключился: {reason}")
        return dev, recv, lp, users[0]

    # Retry the connection instead of dying: right after a crash the PS5 still
    # holds the previous Remote Play session ("Another Remote Play session is
    # connected") and it may briefly be in standby. A hard crash here just makes
    # the batch loop restart into the same failure.
    device = receiver = loop = None
    user = None
    for attempt in range(1, 6):
        try:
            device, receiver, loop, user = _connect_once()
            break
        except RuntimeError as exc:
            if trophy_mode:
                log_line(f"Подключение не удалось (попытка {attempt}/5): {exc}")
            print(f"Подключение не удалось (попытка {attempt}/5): {exc}")
            if attempt >= 5:
                raise
            wait = RETRY_CONNECT_WAIT_S if attempt < 3 else RETRY_CONNECT_WAIT_LATE_S
            print(f"Жду {wait} с и пробую снова…")
            time.sleep(wait)
    print(f"Подключено к профилю {user}. Ожидаю видеокадры; после первого кадра откроется окно выделения поля нот…")
    if trophy_mode:
        log_line(f"Подключено к профилю {user}.")
    deadline = time.time() + 25
    first_frame = None
    while time.time() < deadline:
        frame = receiver.get_latest_video_frame()
        if frame is not None:
            first_frame = frame.to_ndarray(format="bgr24")
            break
        time.sleep(.05)
    if first_frame is None:
        device.disconnect()
        try:
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            pass
        raise RuntimeError("Сессия открылась, но видеокадр не пришёл.")

    # Start with the saved region if we have one; otherwise show the live
    # video first and let the user draw the ROI (press C) when notes appear.
    saved_roi = load_roi_state()
    if saved_roi is not None:
        rx, ry, rw, rh, _saved_lines = saved_roi
        print("Загружена сохранённая разметка поля нот "
              f"({rw}x{rh}). Нажми C, чтобы перерисовать.")
    else:
        rx = ry = rw = rh = 0
        print("Видео запущено. Когда увидишь поле с нотами, нажми C и выдели его мышью.")

    def detect_hit_markers(frame) -> list[tuple[int, int]]:
        """Find the game's pink spiral hit markers inside the ROI.

        Returns [(row_y, target_x)] in ROI coordinates. The markers are the
        only compact magenta blobs in the note field before the bot draws
        its own line, so this removes manual line-placement mistakes.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        pink = cv2.inRange(hsv, np.array([130, 35, 120]), np.array([179, 255, 255]))
        pink = cv2.morphologyEx(pink, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, _labels, stats, _ = cv2.connectedComponentsWithStats(pink, 8)
        found = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if not (250 < area < 20000):
                continue
            if not (rx <= x and ry <= y and x + w <= rx + rw and y + h <= ry + rh):
                continue
            ratio = w / max(1, h)
            if not (0.2 < ratio < 4.0):
                continue
            found.append((y + h // 2 - ry, x + w // 2 - rx, area))
        found.sort(key=lambda item: -item[2])
        markers = []
        for my, mx, _area in found:
            if all(abs(my - oy) > 30 or abs(mx - ox) > 30 for oy, ox in markers):
                markers.append((my, mx))
        markers.sort()
        return markers

    # Each click can calibrate a row independently. One click is a shared
    # target for all rows; additional clicks add row-specific hit lines.
    target_lines: list[tuple[int, int]] = []  # (row y, target x), ROI coordinates
    auto = False
    if saved_roi is not None:
        target_lines.extend(_saved_lines)
        if target_lines:
            auto = True
            print(f"Восстановлено меток попадания: {target_lines}. Автонажатия включены.")
    note_tracks: list[NoteTrack] = []
    note_track_map: dict[int, NoteTrack] = {}
    next_track_id = 1
    # Holds are tracked separately from regular notes: the NN (used for regular
    # notes) adds extra detections near holds that fragment the shared tracker,
    # so holds keep their own Hough-only track list and never see the NN.
    hold_tracks: list[NoteTrack] = []
    hold_track_map: dict[int, NoteTrack] = {}
    next_hold_track_id = 1
    # Notes travel ~240 px/s on the 720p Remote Play stream (measured: 340 px/s
    # at 1080p in the trophy video, scaled by 720/1080). The old 770 constant
    # was 3x too fast and made every early press ~140 ms early.
    DEFAULT_SPEED = 240.0  # measured note scroll speed on the 720p stream
    speed_px_s = DEFAULT_SPEED
    last_frame_token = None
    nn_frame_counter = 0  # run the NN every 2nd frame to keep the loop fast
    active_holds: dict[int, tuple[str, int]] = {}
    hold_timing: dict[int, tuple[float, float]] = {}   # tail id -> (start, expected)
    hold_endpoint_track_ids: set[int] = set()
    # Which merged notes each detected hold pair has claimed. If the pair can never
    # fire (its tail died), those notes are handed back to the tap loop.
    assist_held: set[str] = set()
    # MASH prompt: a "MASH" line appears between notes; mash CROSS until it ends.
    mash_ocr_frame = None          # latest frame handed to the OCR worker
    mash_ocr_text = ""             # latest OCR text (set by the worker thread)
    mash_ocr_stamp = 0.0           # perf_counter when the OCR finished
    mash_last_press = 0.0          # last mash tap time
    mash_until = 0.0               # red MASH strip: skip presses until this time
    mash_was_active = [False]      # previous MASH visibility, for the event log
    events_flush_at = [0.0]        # last partial write of song_events.csv
    frame_dt = [0.04]              # measured interval between new video frames
    last_frame_at = [None]         # perf_counter of the previous new frame
    fake_songs = [0]               # consecutive "songs" that were really a menu
    mash_stop = threading.Event()  # stop the OCR worker on exit
    templates = make_templates()
    button_dispatcher = ButtonDispatcher(device.controller, loop)

    # --- trophy state machine -------------------------------------------
    state = "INTRO"            # GAMEPLAY | INTRO | RESULTS
    state_since = time.perf_counter()
    dark_since = None
    bright_since = None
    no_notes_since = None
    last_marker_scan = None
    # --- replay mode (chart-driven) + learn mode (record & grow) --------
    new_events: list[dict] = []   # recorded during this song, absolute times
    results_ocr_done = False
    results_screenshot_done = False

    def enter_state(new_state: str, now: float) -> None:
        """Переключить состояние и сбросить всё, что относится к прошлому экрану."""
        nonlocal state, state_since, results_ocr_done, results_screenshot_done
        nonlocal next_retry_press, intro_warned, intro_menu_press
        nonlocal no_notes_since
        if state == new_state:
            return
        if trophy_mode:
            print(f"Состояние: {state} -> {new_state}")
        state = new_state
        state_since = now
        results_ocr_done = False
        results_screenshot_done = False
        next_retry_press = None
        intro_warned = False
        intro_menu_press = None
        no_notes_since = None
        if new_state == "RESULTS":
            button_dispatcher.release_all()
            active_holds.clear()
            hold_endpoint_track_ids.clear()


    next_retry_press = None
    intro_warned = False
    intro_menu_press = None
    song_fired = 0
    song_unknown = 0
    song_skull = 0
    song_holds = 0
    song_notes_seen = 0
    song_tracks_lost = 0   # tracks that expired without ever firing (a Miss)
    song_ambiguous = 0      # notes classified "?" (skipped, no button)
    song_confident = 0      # tracks that ever reached a confident button score
    song_late_fired = 0     # presses fired well past the hit line (wasted)
    song_double_fired = 0   # same button pressed again within 90 ms
    song_mash_dodged = 0    # mash taps skipped because a skull sat on the line
    song_mash_skipped = 0   # frames inside the red MASH strip (all presses off)
    song_static_skipped = 0 # detections that never moved (static blobs, not notes)
    song_line_notes = 0     # notes that reached the hit line this song
    song_skipped_start = 0  # of those, the opening notes deliberately skipped
    song_mash_notes = 0     # notes that crossed the line inside the MASH strip
    song_mash_pressed = 0   # of those, how many the bot actually pressed
    song_blob_frames = 0    # frames where Hough had merged a pair of notes
    song_lost_confident = 0 # dropped notes that WERE read confidently (real Misses)
    song_lost_x = [0] * 11  # ...and where in the lane they died (100 px buckets)
    song_lost_in_mash = 0   # ...how many inside the red MASH strip
    song_lost_near_hold = 0 # ...how many inside a Hold bar
    song_off_row = 0        # detections between/outside the note rows (artwork)
    song_low_confidence = 0 # presses held back: too few frames or too weak a read
    song_hold_errors: list[float] = []   # actual hold duration minus expected, s
    song_hold_lost = 0      # holds released because the tail track disappeared
    song_dark_boost = 0     # frames whose lane was brightened before detection
    # [count, reasons-dict]: unpressed notes read CONFIDENTLY (real misses). This
    # is the number to trust -- it matched the game's own Miss count (114 vs 101).
    miss_strong: list = [0, {}]
    song_started = None
    # Pattern recording: every note that reaches the hit line, with the time it
    # crossed and the button the whole track agreed on. The community method for
    # this trophy is to learn the repeating note pattern from plate 5 on; this
    # list is that pattern, recorded automatically. Pressing from a recorded
    # pattern does not depend on detecting the note at press time, which is where
    # ~13% of the presses are currently lost.
    # (seconds from the run start, reason) for every note that reached the hit
    # line and was not pressed. Bursts here have a single cause and can be fixed.
    miss_log: list[tuple[float, str]] = []
    miss_log_len = 0            # last seen len(miss_log), to spot new misses
    _ignored_key = [255]        # last stray key we refused (log once per key)
    servings = 0
    last_press_button = ""
    last_press_time = 0.0
    recent_presses: list[tuple[float, str, float, float]] = []
    streak = 0
    last_song_duration = None
    # --- recorded note pattern (timing replay) -------------------------------
    # --- closed-loop latency measurement (the "watcher") -----------------
    # Every press is timestamped; the judgement popup flash near the line
    # gives the game's register time, so latency = flash - press, and the
    # lead is set to that latency. It self-tunes every song.
    pending_presses: list[tuple[float, float, float]] = []  # (t_press, row_y, target_x)
    popup_bg: dict[float, float] = {}      # row_y -> rolling brightness
    latency_samples: list[float] = []      # seconds, collected during the song
    DARK_LEVEL = 55.0
    BRIGHT_EMA = 0.85
    bright = None  # initialized from the first video frame


    preview_width = first_frame.shape[1]
    preview_height = first_frame.shape[0]
    # The two clickable overlays used to sit in the TOP-RIGHT corner -- which is
    # exactly where the game shows the live combo counter, so the counter was
    # covered by our own buttons and could not be read. They now sit HORIZONTALLY
    # CENTRED AT THE TOP of the window (the user's choice): the counter in the
    # top-right stays visible and the lane in the middle is not covered either.
    _btn_w, _btn_h = 210, 48
    _btn_x1 = max(0, (preview_width - _btn_w) // 2)
    _btn_x2 = min(preview_width, _btn_x1 + _btn_w)
    _btn_y1 = 96
    replay_button = (_btn_x1, _btn_y1, _btn_x2, _btn_y1 + _btn_h)
    options_button = (_btn_x1, _btn_y1 + _btn_h + 6, _btn_x2,
                      min(preview_height, _btn_y1 + 2 * _btn_h + 6))

    def click_target(event, x, y, flags, param):
        nonlocal auto
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        bx1, by1, bx2, by2 = replay_button
        obx1, oby1, obx2, oby2 = options_button
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            button_dispatcher.tap("CROSS")
            print("Отправлен X: повтор/подтверждение.")
        elif obx1 <= x <= obx2 and oby1 <= y <= oby2:
            button_dispatcher.tap("OPTIONS")
            print("Отправлен OPTIONS: пауза.")
        elif rx <= x <= rx + rw and ry <= y <= ry + rh:
            local_y, local_x = y - ry, x - rx
            replace_within = max(22, int(rh * .08))
            for index, (row_y, _row_x) in enumerate(target_lines):
                if abs(row_y - local_y) <= replace_within:
                    target_lines[index] = (local_y, local_x)
                    print(f"Линия попадания обновлена: ряд y={local_y}.")
                    break
            else:
                target_lines.append((local_y, local_x))
                if len(target_lines) == 1:
                    print("Общая линия попадания задана. Для разных линий щёлкни метку в каждом ряду.")
                else:
                    print(f"Линия попадания добавлена для ряда y={local_y}.")
                if trophy_mode and not auto:
                    auto = True
                    enter_state("INTRO", time.perf_counter())
                    print("Автонажатия включены автоматически (трофейный режим).")
            save_roi_state(rx, ry, rw, rh, target_lines)

    def target_x_for_y(note_y: float) -> int | None:
        if not target_lines:
            return None
        if len(target_lines) == 1:
            return target_lines[0][1]
        return min(target_lines, key=lambda item: abs(item[0] - note_y))[1]

    window = "Wanted: Dead — бот рамен-мини-игры (C — разметить поле, SPACE — пуск/пауза, ESC — выход)"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, click_target)
    print("Выдели все ряды нот. Розовая метка попадания ищется автоматически.")
    print("Когда бот найдёт её, на метке появится жёлтый кружок, а фиолетовая")
    print("линия встанет чуть правее (это линия нажатия). Если кружка нет —")
    print("кликни по розовой метке сам, клик заменит авто-метку.")
    print("Если линия общая для всех рядов, достаточно одной метки. SPACE включает/пауза автонажатий.")
    print("X или ENTER, либо кнопка X/RETRY в окне — отправить крест (повтор/подтверждение).")
    print("Другие клавиши: O круг, Q квадрат, T треугольник, L=L2, B=R2; I/J/K/M — крестовина.")
    print("P — OPTIONS (пауза на геймпаде).")
    print("[ и ] — подстроить упреждение lead на 10 мс: ] жмёт раньше, [ жмёт позже.")
    print("R — начать/остановить запись видео; ESC — выход.")
    if trophy_mode:
        print("ТРОФЕЙНЫЙ РЕЖИМ: авто-Retry на экране результатов, OCR статистики, лог серии порций.")
        log_line(f"Сессия началась: {host}, lead={lead_ms:.0f} мс, пропуск первых {SKIP_FIRST_NOTES} нот, hold={HOLD_ENDPOINT_MODE}, трофейный режим, ВЕРСИЯ hybrid-v99R42 (ОТСЕВ ДУБЛЕЙ СУЖЕН ПО ЗАМЕРУ: блокировка «трек родился у линии» теперь срабатывает только если предыдущее нажатие в том же ряду было ТОЙ ЖЕ кнопкой. Замер по двум ночным песням: в 295 блокировках из 295 значок был ДРУГОЙ, то есть отсекались настоящие ноты - и число блокировок (86 и 107) совпало со счётом Miss игры (117 и 180) и с разбором уверенных промахов (dupe 121 и 168). Это была главная причина промахов; УДЕРЖАНИЯ = РЕКОРДНАЯ ЛОГИКА ЦЕЛИКОМ: убраны зазор до полосы, возврат нот, проверки ряда и значков, фильтры полос; оставлен только журнал hold_pair/hold_take/hold_end/hold_miss; удержание нажимается по НАЧАЛУ полосы, а не по ноте-голове: замер показал зазор 17 px = 60-99 мс раннего нажатия, из-за чего игра ставила BAD и удержание не начиналось; журнал: длительность каждого удержания и причина отпускания (line / track_gone) - чтобы увидеть раннее отпускание в миллисекундах; удержания: пара строится только на одном ряду (кросс-рядовые пары падали в 3 из 4 случаев) и не строится при уверенном расхождении значков; полоса ищется только на известных рядах и тонкой (высота <=34 px); ноты удержания возвращаются обычной логике, если пара не может сработать: 39% пар умирали вместе с хвостом и уносили уверенную ноту; УДЕРЖАНИЯ ВОЗВРАЩЕНЫ К РЕКОРДНОМУ ПОВЕДЕНИЮ: убраны все мои правки пар/кнопки/черепа, SKIP_FIRST снова только для тапов, MASH ничего не меняет; добавлен только журнал hold_pair/hold_take/hold_miss/hold_skull) и сохраняет кадр полосы и весь экран в Videos\WantedDeadRamen\hold_miss; кнопка удержания выбирается по более уверенно прочитанному концу: у удержания значок на концах всегда одинаков, расхождение = ошибка чтения; во время MASH пропускаются только обычные ноты, удержания берутся: раньше полоса гасила и их, отсюда полный пропуск удержаний; откат «коротких удержаний»: он ломал настоящие удержания; SKIP_FIRST пропускает и удержания; MASH = полный пропуск; SKIP_FIRST пропускает и удержания; MASH полностью пропускается всегда - во время полосы не жмётся ничего; короткие удержания: если голову уже нажали обычным порядком и удержание начаться не может, хвост больше не съедается, а жмётся сам; пауза между попытками подключения 5 с вместо 15; перезапуск каждые 6 мин делает батник; OCR счётчика ВЫКЛЮЧЕН: RapidOCR течёт по потокам и памяти; снимок теперь по залпу промахов; кадры в JPEG; лимит 1 снимок в 20 с; снимок обрыва берётся по ПЕРВОМУ подозрению и хранит ~3 с кадров, а чтение счётчика повторяется через 0.6 с после неудачи - раньше снимок опаздывал на 12 с; повторы {DEDUPE_S*1000:.0f} мс, устаревшие {STALE_TRACK_S*1000:.0f} мс, опоздание {LATE_MARGIN_PX} px)")
    elif _OCR_ENGINE is None:
        print("Подсказка: RapidOCR не установлен — статистика экрана результатов не читается.")

    try:
        last_new_frame_time = time.perf_counter()
        stream_warned = False
        # Performance probe: if the loop cannot keep up with the 60 fps stream the
        # Remote Play buffer backs up and the video "freezes". Log the real
        # per-frame cost so stalls can be told apart from network problems.
        perf_accum = 0.0
        perf_count = 0
        perf_since = time.perf_counter()
        detect_tick = 0
        display_tick = 0
        prev_notes: list[tuple[int, int, int, str]] = []
        prev_notes_t = time.perf_counter()
        # --- Диагностика потока -----------------------------------------
        # Когда видео "зависает", важно знать, ГДЕ именно оно встало:
        #   * пакеты перестали приходить  -> сеть / кодер PS5;
        #   * пакеты идут, кадров нет     -> декодер/поток pyremoteplay умер;
        #   * кадры есть, но "пауза" большая -> завис наш цикл (OCR и т.п.).
        # Одна строка раз в 2 секунды отвечает на этот вопрос.
        diag_since = time.perf_counter()
        diag_frames = 0
        diag_gap = 0.0
        diag_last_iter = time.perf_counter()
        diag_prev_rx = None
        diag_prev_lost = None
        send_err_seen = 0
        frame_shape_warned: tuple[int, int] | None = None
        while device.connected:
            _iter_start = time.perf_counter()
            # Loop-period watchdog + stream diagnostic. Deliberately FIRST in the
            # iteration, so it also runs while no frames are arriving at all --
            # that is exactly the case that needs explaining.
            _t = time.perf_counter()
            _loop_gap = _t - diag_last_iter
            if _loop_gap > diag_gap:
                diag_gap = _loop_gap
            diag_last_iter = _t
            if _t - diag_since >= 2.0:
                diag_since = _t
                # NB: never reuse rx/ry/rw/rh here -- those are the ROI values.
                n_packets = n_lost = n_queue = -1
                stream_state = "?"
                stopped = None
                sess_error = ""
                send_err = recv_err = -1
                try:
                    _stream = receiver._session.stream
                    if _stream is not None:
                        _avh = _stream._av_handler
                        n_packets = _avh.received
                        n_lost = _avh.lost
                        n_queue = len(_avh._queue)
                        stream_state = _stream.state
                        _proto = getattr(_stream, "_protocol", None)
                        if _proto is not None:
                            send_err = getattr(_proto, "send_errors", -1)
                            recv_err = getattr(_proto, "recv_errors", -1)
                    stopped = receiver._session.is_stopped
                    sess_error = receiver._session.error or ""
                except Exception as exc:  # pylint: disable=broad-except
                    stream_state = f"ошибка чтения ({exc})"
                d_packets = 0 if diag_prev_rx is None else n_packets - diag_prev_rx
                d_lost = 0 if diag_prev_lost is None else n_lost - diag_prev_lost
                diag_prev_rx, diag_prev_lost = n_packets, n_lost
                if send_err > 0:
                    send_err_seen = send_err
                if trophy_mode:
                    log_line(
                        f"Поток[2с]: пакетов {n_packets} (+{d_packets}), потеряно {n_lost} (+{d_lost}), "
                        f"очередь {n_queue}, кадров +{diag_frames}, пауза {diag_gap * 1000:.0f} мс, "
                        f"ошибок отправки {send_err}/приёма {recv_err}, "
                        f"стрим={stream_state}, стоп={stopped}"
                        + (f" | {sess_error}" if sess_error else "")
                    )
                diag_frames = 0
                diag_gap = 0.0
            video = receiver.get_latest_video_frame()
            if video is None:
                button_dispatcher.pump(time.perf_counter())
                time.sleep(.005)
                continue
            image = video.to_ndarray(format="bgr24")
            # One grayscale conversion of the full frame, reused by everything
            # that needs brightness (quick_bright, frame_bright, the popup
            # watcher). Doing it three times cost ~2 ms each and helped starve
            # the 60 fps loop.
            gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            quick_bright = float(gray_full.mean())
            # The ROI is stored in stream pixels, so it must be re-checked
            # against the CURRENT frame: if the host ever sends a smaller
            # stream (resolution fallback), the slice below comes out empty and
            # cv2.cvtColor() raises, killing the whole session. Report it once
            # instead of crashing.
            frame_h, frame_w = image.shape[:2]
            roi_fits = rw >= 100 and rh >= 20 and frame_h > ry + 20 and frame_w > rx + 20
            if not roi_fits and rw >= 100 and rh >= 20:
                if frame_shape_warned != (frame_w, frame_h):
                    frame_shape_warned = (frame_w, frame_h)
                    msg = (f"Кадр {frame_w}x{frame_h} меньше сохранённой разметки "
                           f"(x={rx} y={ry} {rw}x{rh}) — перерисуй поле нот клавишей C.")
                    print(msg)
                    if trophy_mode:
                        log_line(msg)
            if roi_fits:
                lane = image[ry:ry + rh, rx:rx + rw]
                # Brightness normalisation. The burst detector proved the combo
                # breaks on DARK stretches (brightness 39-49 while the darkness
                # threshold is 55): there the notes are faded, their template
                # score collapses and they are rejected as "weak" or not detected
                # at all. Stretching a dim lane back to a normal level before
                # detection makes them readable again; a normal frame is untouched.
                lane_mean = float(gray_full[ry:ry + rh, rx:rx + rw].mean())
                if lane_mean < 70.0:
                    lane = cv2.convertScaleAbs(
                        lane, alpha=min(3.0, 110.0 / max(25.0, lane_mean)), beta=0)
                    song_dark_boost += 1
                # The red MASH strip: the bot NEVER mashes a button (that is the
                # blind press that hit skulls). What it does with the ordinary
                # notes scrolling under the strip is controlled by
                # PRESS_NOTES_IN_MASH. Measured: skipping them entirely cost 79
                # of the 120 Misses in one song -- two thirds of all misses --
                # because notes keep coming while the bar is up. The strip fades
                # in and out, so the flag is held for a moment after the last
                # detection to close both edges.
                if detect_mash_note(lane):
                    mash_until = time.perf_counter() + .35
                mash_active = time.perf_counter() < mash_until
                if mash_active != mash_was_active[0]:
                    mash_was_active[0] = mash_active
                # With IGNORE_MASH the strip is measured but filters nothing.
                mash_filter = mash_active and not IGNORE_MASH
                hough_notes = detect_notes(lane, templates)
                if USE_BLOB_DETECTOR and has_merged_pair(lane, hough_notes):
                    # Only on frames that actually contain a merged pair: the
                    # component detector is what separates the two notes, but it
                    # is far too expensive to run all the time.
                    blob_notes = detect_notes_by_blob(lane, templates)
                    hough_notes = merge_hough_blob(hough_notes, blob_notes)
                    song_blob_frames += 1
                # Known note rows: the configured row(s) plus the row ROW_SPACING_PX
                # below each of them. The hold-bar detector needs them, otherwise cyan
                # HUD elements above the upper row pass as bars (measured: twice as
                # many "holds" on the upper row and twice the failure rate there).
                _known_rows = [row_y for row_y, _row_x in target_lines]
                _known_rows += [row_y + ROW_SPACING_PX for row_y, _row_x in target_lines]
                hold_spans = detect_hold_spans(lane, _known_rows)
                if mash_filter:
                    song_mash_skipped += 1
                    if PRESS_NOTES_IN_MASH:
                        # Keep pressing ordinary notes, but only ones that are
                        # confidently read -- the strip itself produces weak
                        # phantom detections, and those must not become presses.
                        notes = [note for note in hough_notes
                                 if note[3] not in ("?", "", "SKULL")]
                        hold_spans = []
                    else:
                        # USER DECISION: during the strip no ORDINARY note is pressed
                        # (that was pressing note-shaped junk at the strip's edges), but
                        # HOLDS are kept. MEASURED: clearing hold_spans here made every
                        # hold that coincided with a strip disappear completely -- no
                        # press at either end, which is exactly what the user reported.
                        # A hold is a long unmistakable bar, not edge junk.
                        notes = []
                        # hold_spans keeps what detect_hold_spans found above.
                        # Diagnostics for the "notes suppressed" variant: every note
                        # that reaches the line inside the strip is a Miss.
                        for _nx, _ny, _nr, _nsym in hough_notes:
                            _line = target_x_for_y(_ny)
                            if _line is not None and abs(_nx - _line) <= 12:
                                song_mash_notes += 1
                else:
                    notes = hough_notes
            else:
                lane = np.zeros((10, 10, 3), np.uint8)
                mash_active = False
                notes = []
                hough_notes = []
                hold_spans = []
            now = time.perf_counter()
            lead_eff = lead_ms   # RECORD CONFIGURATION: no speed-scaled lead
            button_dispatcher.pump(now)
            frame_token = video.pts if video.pts is not None else id(video)
            new_frame = frame_token != last_frame_token
            if new_frame:
                diag_frames += 1

            # Continuous marker auto-search in trophy mode: keep looking until
            # the pink hit marker appears on screen (e.g. the song field was
            # not visible yet when the bot started).
            if trophy_mode and not target_lines and new_frame:
                if last_marker_scan is None or now - last_marker_scan > 1.5:
                    last_marker_scan = now
                    found_markers = detect_hit_markers(image)
                    if found_markers:
                        target_lines.extend(found_markers)
                        auto = True
                        enter_state("INTRO", now)
                        save_roi_state(rx, ry, rw, rh, target_lines)
                        if len(found_markers) == 1:
                            print(f"Розовая метка найдена автоматически: x={found_markers[0][1]}. "
                                  "Автонажатия включены.")
                        else:
                            print(f"Найдено розовых меток: {len(found_markers)}: {found_markers}. "
                                  "Автонажатия включены.")

            if new_frame:
                last_new_frame_time = now
                stream_warned = False
                prev_track_id = next_track_id
                # Count tap tracks that are about to expire without ever firing:
                # each one is a note we saw but never pressed (a Miss), not a
                # classification problem. Skip already-fired tracks.
                if state == "GAMEPLAY":
                    # Count each dropped note ONCE. The old version counted the
                    # same stale track on every frame it stayed stale, which
                    # made the number meaningless (it read 7013 for one song).
                    # Split by whether the note was ever read CONFIDENTLY: a
                    # confidently read note that dies before the line is a real
                    # Miss (detection/timing failure); a never-confident one is
                    # junk from the lane art.
                    for t in note_tracks:
                        if not t.fired and not t.counted_lost \
                                and now - t.last_seen > .80:
                            t.counted_lost = True
                            song_tracks_lost += 1
                            if t.scores_max and max(t.scores_max.values()) >= 0.68:
                                song_lost_confident += 1
                                # A CONFIDENTLY read note that never fired is the
                                # only candidate for a real Miss that no other
                                # counter explains, so journal every one of them
                                # with the position where it died. Offline that
                                # separates ~50 real missing notes per song from
                                # the duplicate-track noise around them.
                                # WHERE does a confidently read note die? A
                                # histogram by lane x separates the cases: dying
                                # near the right edge means it vanishes as it
                                # enters, dying near the line means it is lost in
                                # the busiest part of the lane.
                                song_lost_x[min(10, int(t.x // 100))] += 1
                                if mash_active:
                                    song_lost_in_mash += 1
                                if any(left <= t.x <= right
                                       and abs(t.y - row) < max(18, t.radius)
                                       for left, right, row in hold_spans):
                                    song_lost_near_hold += 1
                note_tracks, note_track_map, next_track_id, measured_speeds = associate_note_tracks(
                    note_tracks, notes, now, speed_px_s, rw, next_track_id, skip_fired=True
                )
                # Separate Hough-only tracker for holds (isolated from the NN).
                hold_tracks, hold_track_map, next_hold_track_id, _hold_speeds = associate_note_tracks(
                    hold_tracks, hough_notes, now, speed_px_s, rw, next_hold_track_id
                )
                if state == "GAMEPLAY":
                    song_notes_seen += next_track_id - prev_track_id
                # Self-calibrate the scroll speed. The calibration used to read
                # ONLY the hold tracker, i.e. only the handful of notes that are
                # part of a Hold -- so on a song with few holds the bot kept
                # predicting with 240 px/s while the song actually scrolled at
                # 200-357 px/s (both values appear in the log). A 25% speed error
                # puts the predicted position ~30 px off after a 0.3 s detection
                # gap, which is exactly where a track gets lost and the note is
                # never pressed. Use the tap tracker's own speed samples, which
                # exist on every note, and fall back to the hold samples.
                _speed_samples = measured_speeds or _hold_speeds
                if _speed_samples:
                    measured = float(np.median(_speed_samples))
                    if 80.0 <= measured <= 2000.0:
                        speed_px_s = 0.9 * speed_px_s + 0.1 * measured
                # How often do NEW video frames actually arrive? MEASURED from the
                # stream log: the interval is not 16 ms but 40 ms on average and it
                # regularly spikes to 300-500 ms (windows with 0-22 frames per 2 s).
                # Every time it spikes, a note moves speed*frame_dt pixels between
                # two evaluations, which is more than the fixed 30 px late margin,
                # so the note jumps from "not there yet" straight to "too late" and
                # is NEVER pressed -- visible in the journal as confidently read
                # notes (score 0.87-0.99) refused with left=45..53 px, i.e. real
                # Misses. The frame interval therefore has to widen the timing
                # windows; the fixed constants stay as the floor for a healthy
                # stream.
                if last_frame_at[0] is not None:
                    _gap = now - last_frame_at[0]
                    if 0.0 < _gap < 1.0:
                        frame_dt[0] = 0.8 * frame_dt[0] + 0.2 * _gap
                last_frame_at[0] = now
                last_frame_token = frame_token
            else:
                # Watchdog: the Remote Play stream can stall; exit cleanly
                # instead of freezing forever on the last frame.
                stall = now - last_new_frame_time
                # Fail fast when the OUTGOING path is broken: the host answers a
                # dead client by stopping the video, so waiting for the 8 s
                # timeout just wastes time. Reconnecting is quicker.
                if stall > 2.0 and send_err_seen > 0:
                    msg = (f"Исходящая связь сломана ({send_err_seen} ошибок отправки) — "
                           f"переподключаюсь вместо ожидания.")
                    print(msg)
                    if trophy_mode:
                        log_line(msg)
                    button_dispatcher.mute()
                    break
                if stall > 6.0 and not stream_warned:
                    stream_warned = True
                    print(f"Поток не обновляется уже {stall:.0f} с (состояние={state}, "
                          f"нот={len(notes)}, авто={auto}) — возможно, зависание Remote Play.")
                    if trophy_mode:
                        log_line(f"Поток не обновляется {stall:.0f} с (состояние={state}, "
                                 f"нот={len(notes)}) — возможное зависание Remote Play.")
                if stall > 8.0:
                    print("Поток завис на 8 секунд. Завершаю работу (запись будет сохранена).")
                    if trophy_mode:
                        log_line("Поток завис на 8 секунд — аварийное завершение.")
                    button_dispatcher.mute()
                    break

            # --- state machine, active whenever auto is on -------------------
            # The song has dark "plate transition" screens mid-song, so screen
            # darkness alone must NOT be treated as the song end. The song is
            # over only when no notes were seen for song_end_wait seconds AND
            # the screen is dark (the results screen). Notes may appear on
            # dark frames too, so note firing is not gated by brightness.
            frame_bright = quick_bright   # same frame: no second cvtColor
            bright = frame_bright if bright is None else BRIGHT_EMA * bright + (1 - BRIGHT_EMA) * frame_bright
            if auto:
                if state == "RESULTS":
                    # Leaving RESULTS used to require a BRIGHT screen. The ramen
                    # play field is often darker than DARK_LEVEL (measured 39-49
                    # against a 55 threshold), so the bot stayed in RESULTS while
                    # the game was already playing: it pressed X instead of notes,
                    # the run failed, the results screen came back -- forever.
                    # Notes on screen are proof enough that play has started.
                    if len(notes) >= 2:
                        enter_state("GAMEPLAY", now)
                    elif bright >= DARK_LEVEL:
                        if bright_since is None:
                            bright_since = now
                        elif now - bright_since > .6:
                            enter_state("INTRO", now)
                    else:
                        bright_since = None
                elif state == "GAMEPLAY":
                    if notes:
                        no_notes_since = None
                    else:
                        if no_notes_since is None:
                            no_notes_since = now
                        elif bright < DARK_LEVEL and now - no_notes_since >= song_end_wait:
                            enter_state("RESULTS", now)
                            # diagnostic only: the flash watcher is unreliable;
                            # the lead is actually tuned by the OCR stats.
                            if latency_samples:
                                med = float(np.median(latency_samples))
                                log_line(f"Замер задержки по песне: {med * 1000:.0f} мс "
                                         f"(n={len(latency_samples)})")
                                latency_samples.clear()
                            if trophy_mode:
                                servings += 1  # one attempt, ended in Retry (defeat)
                                # Retry on the results screen = DEFEAT (the
                                # plate/gauge filled). Report honestly; there
                                # is no success claim here.
                                song_duration = now - song_started if song_started else None
                                last_song_duration = song_duration
                                log_line(
                                    f"Песня закончилась ({song_duration:.0f} с, "
                                    f"скорость {speed_px_s:.0f} px/s): нажато {song_fired}, "
                                    f"распознано {song_confident} из {song_notes_seen} нот, "
                                    f"черепов {song_skull}, удержаний {song_holds}. "
                                    f"Экран Retry (поражение)."
                                )
                                if miss_strong[0]:
                                    log_line(f"Уверенные промахи по причинам: {miss_strong[1]}")
                                # MENU GUARD. MEASURED: when the run ends and the
                                # game sits in a menu or a pause screen, the bot kept
                                # "finishing songs" every 23-25 s with 7-10 presses
                                # and Score=8000, pressing X into nothing and wasting
                                # minutes without anyone noticing. A real song on this
                                # minigame is never that short with that few presses,
                                # so say so loudly and count consecutive cases.
                                if song_duration is not None and song_fired < 30 \
                                        and song_duration < 60.0:
                                    fake_songs[0] += 1
                                    log_line(
                                        f"ПОДОЗРЕНИЕ: это не песня, а меню/пауза "
                                        f"({song_duration:.0f} с, нажатий {song_fired}, "
                                        f"подряд {fake_songs[0]}). Бот жмёт X вслепую — "
                                        f"нужен человек у приставки.")
                                else:
                                    fake_songs[0] = 0
                elif state == "INTRO":
                    if notes:
                        enter_state("GAMEPLAY", now)
                    elif bright < DARK_LEVEL:
                        # Session started on a results screen: switch to
                        # RESULTS so the auto-Retry flow takes over.
                        if no_notes_since is None:
                            no_notes_since = now
                        elif now - no_notes_since >= min(song_end_wait, 6.0):
                            enter_state("RESULTS", now)
                    else:
                        no_notes_since = None

            if state == "RESULTS" and trophy_mode:
                # Read stats once, then keep pressing X to choose Retry.
                elapsed = now - state_since
                if not results_ocr_done and elapsed >= 3.0 and _OCR_ENGINE is not None:
                    results_ocr_done = True
                    stats = ocr_frame_stats(image.copy())
                    if stats is not None:
                        log_line("Результаты (OCR): " + format_stats(stats))
                        print("Результаты: " + format_stats(stats))
                if next_retry_press is None or now >= next_retry_press:
                    next_retry_press = now + 2.5
            elif state == "INTRO" and trophy_mode and auto and auto_menu:
                elapsed = now - state_since
                if not intro_warned and elapsed > 8.0 and not notes:
                    intro_warned = True
                    print("Интро затянулось: нот нет больше 8 секунд. Возможно, нужно нажать X.")
                if intro_warned:
                    if intro_menu_press is None or now >= intro_menu_press:
                        intro_menu_press = now + 4.0
                        button_dispatcher.tap("CROSS")
                        print("Интро/меню: нажат X.")

            hold_pairs = pair_hold_notes(hold_spans, hough_notes)
            hold_note_indices: set[int] = set()
            hold_marker_note_x: set[int] = set()
            for left, right, row in hold_spans:
                for note_index, note in enumerate(notes):
                    if (left <= note[0] <= right
                            and abs(note[1] - row) < max(18, note[2])):
                        hold_note_indices.add(note_index)
                        hold_marker_note_x.add(note[0])
            hold_pair_tracks = []
            for left_note, right_note in hold_pairs:
                # press/release run on the Hough-only tracks (never the NN)
                left_hough_index = hough_notes.index(left_note)
                right_hough_index = hough_notes.index(right_note)
                left_track = hold_track_map.get(left_hough_index)
                right_track = hold_track_map.get(right_hough_index)
                if left_track is None or right_track is None:
                    continue
                # The endpoints of a hold must also be excluded from the TAP loop,
                # otherwise the note would be pressed twice (once as the hold head,
                # once as a tap) -- but only once the hold is REALLY taken.
                #
                # MEASURED (250 pairs in one 330 s song, journal `hold_pair`):
                # 39% of the detected pairs never produced a press at all, and the
                # notes they claimed were excluded from tap pressing for good -- 388
                # confidently read notes per song went through this path and a
                # quarter of them were never pressed by anything. Registering the
                # ids here (at detection time) was the cause, so they are registered
                # in the hold press branch below instead: a pair that never engages
                # now leaves its notes to the normal tap logic.
                li = note_index_for_position(notes, left_note[0], left_note[1], left_note[2])
                ri = note_index_for_position(notes, right_note[0], right_note[1], right_note[2])
                # NB: a failed experiment lived here for two builds -- when the tap logic
                # had already pressed the head, the tail was left pressable ("short holds
                # are skipped"). It made real holds WORSE: the hold press checks its own
                # track's fired flag, not the tap track's, so the hold still fired while
                # the tail was also tapped. The user then saw holds skipped and extra
                # presses on the first plates. Reverted to the record rule.
                head_mt = note_track_map.get(li) if li is not None else None
                head_tapped = bool(head_mt is not None and head_mt.fired)
                endpoint_ids: list[int] = []
                for idx in (li, ri):
                    if idx is None:
                        continue
                    hold_note_indices.add(idx)
                    mt = note_track_map.get(idx)
                    if mt is None:
                        continue
                    if HOLD_ENDPOINT_MODE == "on_engage" and idx != ri:
                        # HEAD in "on_engage": stays pressable until the hold is
                        # really taken, so a pair that never fires cannot swallow it.
                        endpoint_ids.append(mt.track_id)
                    else:
                        # "record" -> both ends leave the tap loop right here; in
                        # "on_engage" only the TAIL does (it is the release end).
                        hold_endpoint_track_ids.add(mt.track_id)
                # Remember which notes this pair claimed, so the claim can be handed
                # back if the pair turns out to be unable to fire (see below).
                # MEASURED (50 bars on saved frames): the head note sits 17 px (6-22)
                # to the left of the bar's own left edge. Pressing when the NOTE reaches
                # the line therefore starts the hold 17 px early -- 99 ms at 171 px/s,
                # 89 ms at 190, 74 ms at 230 -- and the game scores that early press as
                # BAD, so the hold never even starts. The pair keeps the gap and the
                # press point is shifted by it.
                hold_pair_tracks.append((left_track, right_track, left_note, right_note,
                                         tuple(endpoint_ids)))
                # ------------------------------------------------------------------
                # DIAGNOSTIC ONLY (no behaviour change): write every detected hold
                # pair into the note journal once per pair instance. MEASURED reason:
                # the reported "hold bars" are almost always 110-116 px long although
                # the music and the notes differ, which suggests the detector locks
                # onto a repeating screen element and glues two ordinary notes into a
                # fake hold -- and then both notes are handled as hold endpoints
                # instead of being pressed at their own time. One CSV line per pair
                # (tens per song), guarded by a flag so it cannot spam per frame.
                if not getattr(left_track, "pair_logged", False):
                    left_track.pair_logged = True

            # hold_endpoint_track_ids holds MERGED track ids (tap loop); keep it
            # pruned to live merged tracks.
            merged_live_ids = {track.track_id for track in note_tracks}
            hold_endpoint_track_ids.intersection_update(merged_live_ids)
            # the release loop uses the Hough-only tracks
            tracks_by_id = {track.track_id: track for track in hold_tracks}
            observed_track_ids = {track.track_id for track in hold_track_map.values()}

            # --- replay mode: chart sync + timed presses -------------------

            # --- learn mode: fire the loaded sequence by the clock too -----
            # Only on bright gameplay frames: the results screen is dark and
            # must never receive replay presses (they hit Close/Quit).

            if auto and new_frame and target_lines and state == "GAMEPLAY" \
                    :
                for left_track, right_track, left_note, right_note, endpoint_ids \
                        in hold_pair_tracks:
                    left_target = target_x_for_y(left_track.y)
                    if left_target is None:
                        continue
                    left_line = left_target + (left_track.speed or speed_px_s) * lead_eff / 1000.0
                    # DIAGNOSTIC: a pair whose head has ALREADY passed the press point
                    # without the hold being taken. This is the case the user reports as
                    # "the hold is skipped completely": one detection per pair, and the
                    # lane plus the whole frame are saved so the picture can be checked
                    # instead of guessing.
                    if right_track.track_id not in active_holds and not left_track.fired and left_track.x <= left_line:
                        # A hold whose endpoints include a skull must never be
                        # pressed either (same far-misread path as tap notes).
                        if left_track.seen_skull or right_track.seen_skull \
                                or left_track.symbol == "SKULL" or right_track.symbol == "SKULL" \
                                or skull_nearby(now, left_track.x, left_track.y,
                                                left_track.speed or speed_px_s):
                            left_track.fired = True
                            right_track.fired = True
                            song_skull += 1
                            print("Череп в удержании — пропущено.")
                            continue
                        # RECORD BEHAVIOUR: the button comes from the HEAD (the tail only
                        # as a fallback). The "better-read end" rule tried here was rolled
                        # back -- it changed which button 19% of holds pressed and the
                        # user reported holds getting worse everywhere.
                        head_sym = left_track.symbol
                        tail_sym = right_track.symbol
                        head_btn = NOTE_BUTTONS.get(head_sym)
                        tail_btn = NOTE_BUTTONS.get(tail_sym)
                        head_score = _best_of(left_track)
                        tail_score = _best_of(right_track)
                        button = head_btn or tail_btn
                        held_end = "head"
                        if head_sym != tail_sym:
                            # Journal only: it never changes behaviour now.
                            pass
                        if button is None:
                            left_track.fired = True
                            right_track.fired = True
                            song_unknown += 1
                            print("Символ не распознан, нажимаю последним.")
                        else:
                            button_dispatcher.press(button)
                            active_holds[right_track.track_id] = (button, left_track.track_id)
                            # Only "on_engage" collects head ids here; in "record" the
                            # tuple is empty and both ends were already excluded.
                            if endpoint_ids:
                                hold_endpoint_track_ids.update(endpoint_ids)
                            # Instrumentation: how long SHOULD this hold last
                            # (the tail note has to travel from the head to the
                            # line) versus how long it actually lasted. A hold is
                            # two notes out of one press, so a systematic timing
                            # error here would show up as a steady Miss rate.
                            hold_timing[right_track.track_id] = (
                                now,
                                max(.05, (right_track.x - left_track.x)
                                    / max(1.0, right_track.speed or speed_px_s)),
                            )
                            left_track.fired = True
                            right_track.fired = True
                            song_holds += 1
                            # Every TAKEN hold goes into the journal. Without this there
                            # was no way to tell a real hold miss from my own hold_miss
                            # diagnostic firing one frame BEFORE the press in the same
                            # iteration -- which inflated the "skipped holds" numbers.
                            print(f"Hold: удерживаю {left_track.symbol} до второй ноты {right_track.symbol}")
                            log_line(f"HOLD+ {button} left_x={left_track.x:.0f} right_x={right_track.x:.0f}")

                for right_id, (button, _left_id) in list(active_holds.items()):
                    right_track = tracks_by_id.get(right_id)
                    if right_track is None:
                        button_dispatcher.release(button)
                        active_holds.pop(right_id, None)
                        started = hold_timing.pop(right_id, None)
                        if started is not None:
                            song_hold_errors.append(now - started[0] - started[1])
                            song_hold_lost += 1
                        print("Hold: отпущено (дорожка пропала).")
                    elif (right_id in observed_track_ids
                          and (right_target := target_x_for_y(right_track.y)) is not None
                          and right_track.x <= right_target + (right_track.speed or speed_px_s) * lead_ms / 1000.0):
                        button_dispatcher.release(button)
                        active_holds.pop(right_id, None)
                        started = hold_timing.pop(right_id, None)
                        if started is not None:
                            song_hold_errors.append(now - started[0] - started[1])
                        print("Hold: отпущено")
                        log_line(f"HOLD- {button}")

                if auto:
                    # --- pattern replay: press by recorded timing ----------
                    # The pattern times were recorded at the moment the bot would
                    # have pressed that note, so they already include the lead;
                    # the replay adds only the sync offset, never the lead again.
                    # Fire from LIVE TRACKS, not from this frame's detections.
                    # Before this, a note was pressed only if it was DETECTED in
                    # the same frame in which it crossed the line; whenever the
                    # detector flickered exactly there, no press happened at all
                    # and the note was only pressed once re-detected -- as one of
                    # the 400-900 "late" presses measured per song, or missed
                    # entirely. Predicting the crossing from the track fires each
                    # note exactly once whether or not it was detected this frame.
                    for track in list(note_tracks):
                        if track.fired:
                            continue
                        if track.track_id in hold_endpoint_track_ids:
                            # LAST silent gate. A note whose track was paired as the
                            # END of a Hold bar is dropped from the tap loop and left
                            # to the hold logic. If the pairing is wrong (two ordinary
                            # notes read as one bar) that note is never pressed at all
                            # and the game counts a Miss -- which is exactly the shape
                            # of the reported "miss on an ordinary white triangle with
                            # the combo already past 100". Journalled once per track.
                            if not track.endpoint_logged:
                                track.endpoint_logged = True
                            continue
                        note = (int(track.x), int(track.y), int(track.radius), track.symbol)
                        note_target = target_x_for_y(track.y)
                        if note_target is None:
                            continue
                        speed = track.speed or speed_px_s
                        dt = now - track.last_seen
                        # A stale track's prediction is not trustworthy: after
                        # 150 ms of silence the note may already be past the line,
                        # and a press on it is a stray press. Numbers from a long
                        # run: 727 presses were fired 30+ px PAST the line while
                        # the game counted 132 Misses -- the game appears to punish
                        # a press with no note in front of it, which is exactly what
                        # a late press on a vanished note is.
                        # The limit follows the MEASURED frame interval: with frames
                        # every 40 ms a 150 ms stale reading is suspicious, but when
                        # the stream hiccups to 300-500 ms per frame every track is
                        # legitimately that old, and the fixed limit silently threw
                        # away notes that were tracked perfectly.
                        stale_limit = max(STALE_TRACK_S, 2.5 * frame_dt[0])
                        if dt > stale_limit:
                            if not track.stale_logged:
                                track.stale_logged = True
                            continue
                        predicted_x = track.x - speed * dt
                        note_line = note_target + speed * lead_eff / 1000.0
                        if predicted_x > note_line:
                            continue
                        # Record the pattern once per note, BEFORE any press gate:
                        # the learned pattern must contain the notes we fail to
                        # press too, otherwise replaying it reproduces our misses.
                        # Notes that belong to a Hold bar are handled by the hold
                        # logic (press at the left end, release at the right end).
                        if any(left <= predicted_x <= right
                               and abs(track.y - row) < max(18, track.radius)
                               for left, right, row in hold_spans):
                            # A tap note inside a detected Hold bar is left to the
                            # hold logic. This branch is silent and it is the prime
                            # suspect for notes that are tracked for a hundred
                            # frames, read confidently and then simply never get
                            # pressed, so record it.
                            track.hold_blocked = True
                            continue
                        # Never press something that has sat still for several
                        # frames: static highlights in the lane art sit left of
                        # the line, so they used to fire immediately. A real
                        # note that appears late (it is already at the line) has
                        # only 1-2 frames before it passes, so it is still
                        # pressed -- those late presses are what rescues a note
                        # from becoming a Miss.
                        if not track.moved and track.frames_seen >= 3:
                            # NOT a miss and NOT part of miss_log: MEASURED that
                            # this branch fires on a lane-edge highlight that sits
                            # still at x~156, y~24 (score 0.32-0.42, i.e. never a
                            # confident glyph read) and logged it again and again.
                            # Those entries made the burst detector cry "8+ misses
                            # in 3 s" on frames where the game saw nothing wrong,
                            # so every burst conclusion drawn from the text log was
                            # built on junk. Counted, journalled, never a Miss.
                            if not track.counted_blocked:
                                track.counted_blocked = True
                                song_static_skipped += 1
                            continue
                        # Notes only ride the known rows; anything between or
                        # outside them is artwork, not a note.
                        if not any(abs(track.y - row_y) <= ROW_TOLERANCE_PX
                                   for row_y, _row_x in target_lines) \
                                and not any(abs(track.y - (row_y + ROW_SPACING_PX))
                                            <= ROW_TOLERANCE_PX
                                            for row_y, _row_x in target_lines):
                            # Same reasoning as the static branch above: this is
                            # the lane's own artwork between the rows (measured
                            # score 0.32-0.42 at y~24-28), not a note, so it must
                            # not enter miss_log and fake a burst.
                            if not track.counted_blocked:
                                track.counted_blocked = True
                                song_off_row += 1
                            continue
                        # Junk separation measured on the reference video: a real
                        # note is tracked for many frames and reads well; junk
                        # lives one frame and reads weakly. Pressing junk steals
                        # the timing of the next real note.
                        _acc = [v for k, v in track.scores_max.items()
                                if not k.startswith("_")]
                        if track.frames_seen < MIN_TRACK_FRAMES \
                                or (not _acc or max(_acc) < MIN_NOTE_SCORE):
                            # Counted ONCE per note: the gate leaves the track
                            # unfired so it can be pressed later if it gains
                            # frames/score, and the old counter re-counted the
                            # same note on every frame it sat at the line.
                            if not track.counted_blocked:
                                track.counted_blocked = True
                                song_low_confidence += 1
                            continue
                        track.fired = True
                        # Never press a note that was ever classified as a
                        # skull: a far-away misread (skull -> CIRCLE) used to
                        # stick in the track and the bot pressed it at the line,
                        # breaking the combo.
                        if track.seen_skull or note[3] == "SKULL" \
                                or skull_nearby(now, predicted_x, track.y, speed):
                            song_skull += 1
                            print("Череп пропущен.")
                            continue
                        symbol = track.symbol if track.symbol else note[3]
                        button = NOTE_BUTTONS.get(symbol)
                        if button is None:
                            if symbol == "SKULL":
                                song_skull += 1
                                print("Череп пропущен.")
                            else:
                                song_unknown += 1
                                if symbol == "?":
                                    song_ambiguous += 1
                                print("Символ не распознан, нажимаю последним.")
                        else:
                            # in learn mode, the loaded part is replayed by
                            # the timed block; press only the NEW notes here
                            # ONE press per physical note. The tracker starts a
                            # new track whenever a note's symbol changes and it
                            # skips tracks that already fired, so the same note
                            # got pressed again one or two frames later at the
                            # same spot: measured 748 repeats within 90 ms and
                            # 767 presses fired 30+ px PAST the line in one
                            # song. Those repeats are not just wasted -- they
                            # collide with the press of the following note.
                            # Community recipe for the full-combo trophy: skip the
                            # opening notes of the first plate, then press every
                            # note without a single break. Only notes that WOULD
                            # have been pressed consume the budget (skulls and
                            # static blobs already returned above).
                            if song_line_notes < SKIP_FIRST_NOTES:
                                song_line_notes += 1
                                song_skipped_start += 1
                                continue
                            song_line_notes += 1
                            # ONE press per physical SPOT, whichever button the
                            # glyph was read as. Offline bench on the reference
                            # full-combo video: one note often produces TWO tracks
                            # (the two readings differ), both travel to the line
                            # and both fire -- 62 presses for 20 real notes. The
                            # old guard only caught repeats of the SAME button, so
                            # the second track slipped through with a wrong button.
                            # Ignoring the button here cut the presses to 42 and
                            # raised the median tracked length of a fired note from
                            # 1 frame to 110.
                            recent_presses = [item for item in recent_presses
                                              if now - item[0] <= DEDUPE_LATE_S]
                            # TIME-ONLY window, and a small one. MEASURED bug: the
                            # old guard compared the press POSITION, but every
                            # press happens at the hit line, so all recorded
                            # positions were the same value (~247 px) and the test
                            # degenerated into "same row within 0.28 s" -- which
                            # blocked every second note of a normal run. The
                            # confident-miss breakdown proved it: of 140 real
                            # misses, 122 were blocked HERE ("dupe").
                            _same_row = [item for item in recent_presses
                                         if abs(item[3] - track.y) <= DEDUPE_ROW_PX]
                            _duplicate = False
                            _gap_ms = -1.0
                            _why = ""
                            if _same_row:
                                _newest = max(item[0] for item in _same_row)
                                _gap_ms = (now - _newest) * 1000.0
                                if _gap_ms <= DEDUPE_S * 1000.0:
                                    _duplicate = True
                                    _why = "same-frame"
                                elif (track.frames_seen <= DEDUPE_BORN_FRAMES
                                      and abs(predicted_x - note_target) <= DEDUPE_BORN_PX
                                      and any(item[1] == button for item in _same_row)):
                                    # MEASURED on two night songs (05:02 and 04:46):
                                    # this branch blocked 86 and 107 notes, and in ALL
                                    # 295 dupe blocks of both songs the blocked note had a
                                    # DIFFERENT button from the press that closed the
                                    # window -- so it was never "the same note pressed
                                    # twice", it was the next real note. Those numbers
                                    # match the game's own Miss count (117 and 180) and
                                    # the confident-miss breakdown (dupe 121 / 168), i.e.
                                    # this one rule was the single biggest source of
                                    # misses: a note first read at the line (median glyph
                                    # score 0.52) followed a press in its row by less than
                                    # DEDUPE_LATE_S and was thrown away.
                                    # The same note re-detected keeps its button, so
                                    # requiring the same button keeps that protection
                                    # and releases the real notes behind it.
                                    _duplicate = True
                                    _why = "born-at-line"
                            if _duplicate:
                                song_double_fired += 1
                                if _why == "born-at-line":
                                    # MEASURED 68 times per song: the same note pressed
                                    # again ~232 ms later (same button, same row, same
                                    # x). The second track is born right at the hit line
                                    # -- the game keeps the note on screen a moment
                                    # after the hit -- while a real note travels in from
                                    # the far right and is matched in dozens of frames.
                                    # That second press hits nothing, and the game
                                    # counts a press with no note in front of it as a
                                    # Miss. Journalled separately so the effect of this
                                    # rule is visible in the numbers.
                                    pass
                                continue
                            # NB: the press is recorded as "recent" only AFTER the
                            # late gate lets it through. Before, a note refused as
                            # too late still occupied the duplicate window and
                            # blocked the note behind it on the same row.
                            # A note already well past the line: pressing it is a
                            # stray press, and the game seems to punish a press
                            # with no note in front of it as a Miss.
                            late_limit = max(LATE_MARGIN_PX, speed * frame_dt[0] * 1.5)
                            if note_line - predicted_x > late_limit:
                                song_late_fired += 1
                                track.fired = True
                                continue
                            recent_presses.append((now, button, predicted_x, track.y))
                            if mash_active:
                                song_mash_pressed += 1
                            song_fired += 1
                            button_dispatcher.tap(button)
                            if track.scores_max:
                                _best_conf = max(track.scores_max.values())
                                if _best_conf >= 0.68:
                                    song_confident += 1
                            pending_presses.append((time.perf_counter(),
                                                    track.y, note_target))
                            print(f"{symbol}  |  оценка скорости {speed_px_s:.0f} px/s")

            # --- MASH prompt: DISABLED -------------------------------------
            # MASH presses are off (MASH_PRESS_ENABLED = False): the bot does not
            # mash CROSS at all, so the red bar cannot cause a blind press on a
            # skull. The code is kept behind the flag for reference.
            if MASH_PRESS_ENABLED and auto and state == "GAMEPLAY" \
                    and mash_active:
                mash_blocked = any(
                    symbol == "SKULL"
                    and (line_x := target_x_for_y(y)) is not None
                    and abs(x - line_x) <= 45
                    for x, y, _r, symbol in hough_notes
                )
                if mash_blocked:
                    song_mash_dodged += 1
                elif now - mash_last_press > 0.055:
                    button_dispatcher.tap("CROSS")
                    mash_last_press = now
                    print("MASH: жму CROSS")

            # Burst detector: the miss trace showed spikes of 80-100 notes missed
            # back to back -- that is what actually kills the combo, so record what
            # the bot was doing when such a spike starts.
            if len(miss_log) != miss_log_len:
                miss_log_len = len(miss_log)

            # Flush any press/release scheduled this frame IMMEDIATELY instead
            # of waiting for the next loop iteration (~47 ms). This removes a
            # full loop of input latency, so the lead can be much smaller.
            button_dispatcher.pump(time.perf_counter())

            # Write the event journal while the song is still running: a restart
            # in the middle of a song used to throw the whole song's evidence
            # away, which is exactly when it is needed.
            if song_started is not None and now - events_flush_at[0] > 30.0:
                events_flush_at[0] = now
                # Frame interval at this moment: the whole adaptive timing scheme
                # only makes sense while this stays visible in the data.

            # --- watcher: measure latency from judgement popups ----------
            # The game flashes "Perfect/Great/..." at the hit line when a
            # press registers. flash_time - press_time = input latency, so
            # the lead is set to that latency and self-tunes every note.
            if new_frame and target_lines and state == "GAMEPLAY" and rw >= 100:
                gray_w = gray_full   # reuse the one conversion already done
                pending_presses[:] = [p for p in pending_presses if now - p[0] <= 0.8]
                for row_y, target_x in target_lines:
                    fx = rx + int(target_x)
                    fy = ry + int(row_y)
                    x0 = max(0, fx - 110)
                    x1 = min(gray_w.shape[1], fx + 140)
                    y0 = max(0, fy - 35)
                    y1 = min(gray_w.shape[0], fy + 50)
                    region = gray_w[y0:y1, x0:x1]
                    if region.size == 0:
                        continue
                    m = float(region.mean())
                    bg = popup_bg.get(row_y)
                    if bg is None:
                        popup_bg[row_y] = m
                        continue
                    popup_bg[row_y] = 0.92 * bg + 0.08 * m
                    if m > bg + 9.0:
                        best = None
                        for idx, (t_press, pr, _px) in enumerate(pending_presses):
                            if abs(pr - row_y) > 45:
                                continue
                            lat = now - t_press
                            if 0.02 <= lat <= 0.5:
                                best = idx
                                break
                        if best is not None:
                            t_press, _pr, _px = pending_presses.pop(best)
                            lat = now - t_press
                            latency_samples.append(lat)
                            print(f"Замер задержки {lat * 1000:.0f} мс (применю после Retry)")

            # --- live combo counter: read it about once a second -----------
            # Runs in its own thread: RapidOCR takes tens of milliseconds and the
            # press loop must never wait for it. Every reading lands in the note
            # journal, and a drop is printed with the events around it, so the
            # moment a streak dies can be matched against what the bot was doing.


            if ry >= 0 and rx >= 0:
                cv2.rectangle(image, (rx, ry), (rx + rw, ry + rh), (0, 230, 255), 1)
            if len(target_lines) == 1:
                _row_y, row_x = target_lines[0]
                display_x = int(row_x + speed_px_s * lead_ms / 1000.0)
                cv2.line(image, (rx + display_x, ry), (rx + display_x, ry + rh), (255, 0, 255), 2)
            elif target_lines:
                ordered_targets = sorted(target_lines)
                for index, (row_y, row_x) in enumerate(ordered_targets):
                    top = 0 if index == 0 else (ordered_targets[index - 1][0] + row_y) // 2
                    bottom = rh if index == len(ordered_targets) - 1 else (row_y + ordered_targets[index + 1][0]) // 2
                    display_x = int(row_x + speed_px_s * lead_ms / 1000.0)
                    cv2.line(image, (rx + display_x, ry + top), (rx + display_x, ry + bottom), (255, 0, 255), 2)
            # Yellow dots show where the hit marker was auto-detected (the
            # pink line itself is the press trigger, shifted by the lead).
            for row_y, row_x in target_lines:
                cv2.circle(image, (rx + int(row_x), ry + int(row_y)), 7, (0, 255, 255), 2)
                cv2.circle(image, (rx + int(row_x), ry + int(row_y)), 2, (0, 255, 255), -1)
            visible_notes = []
            for x, y, radius, symbol in notes:
                if x in hold_marker_note_x:
                    continue
                color = (0, 220, 0) if symbol != "?" else (0, 0, 255)
                cv2.circle(image, (rx + x, ry + y), radius, color, 2)
                visible_notes.append((x, y, radius, symbol))
            placed_label_rects = []
            display_names = {"SQUARE": "SQ", "TRIANGLE": "TRI", "SKULL": "SKULL"}
            for note_index, (x, y, radius, symbol) in enumerate(visible_notes):
                label = display_names.get(symbol, symbol)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale, thickness = .42, 1
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, scale, thickness
                )
                center_x, center_y = rx + x, ry + y
                label_x = max(rx + 2, min(rx + rw - text_width - 2,
                                          center_x - text_width // 2))
                base_above = center_y - radius - 5
                base_below = center_y + radius + text_height + 5
                positions = [
                    base_above,
                    base_below,
                    base_above - text_height - 4,
                    base_below + text_height + 4,
                ]
                if note_index % 2:
                    positions = [positions[i] for i in (1, 0, 3, 2)]
                choices = []
                for slot, text_y in enumerate(positions):
                    top, bottom = text_y - text_height, text_y + baseline
                    if top < ry + 1 or bottom > ry + rh - 1:
                        continue
                    rect = (label_x - 2, top - 2, label_x + text_width + 2, bottom + 2)
                    overlap = 0
                    for ox1, oy1, ox2, oy2 in placed_label_rects:
                        dx = max(0, min(rect[2], ox2) - max(rect[0], ox1))
                        dy = max(0, min(rect[3], oy2) - max(rect[1], oy1))
                        overlap += dx * dy
                    choices.append((overlap, slot, text_y, rect))
                if choices:
                    _overlap, _slot, text_y, rect = min(choices)
                    cv2.putText(image, label, (label_x, text_y), font, scale,
                                (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(image, label, (label_x, text_y), font, scale,
                                (0, 220, 0) if symbol != "?" else (0, 0, 255),
                                thickness, cv2.LINE_AA)
                    placed_label_rects.append(rect)
            for left, right, row in hold_spans:
                hold_y = ry + row
                cv2.rectangle(image, (rx + left, hold_y - 10), (rx + right, hold_y + 10),
                              (255, 170, 0), 1)
            state_label = "AUTO" if auto else "PAUSE"
            if trophy_mode:
                # NB: no "lead" here -- it is printed once, in the line below. It used
                # to appear twice on screen ("lead 51 | ... lead=51 ms").
                state_label += f" | {state}"
            if rw < 100:
                cv2.rectangle(image, (8, 210), (620, 280), (40, 40, 40), -1)
                cv2.putText(image, "Нажми C и выдели поле нот", (20, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 255, 255), 2)
                cv2.putText(image, "(когда ноты уже видны)", (20, 270),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
            cv2.putText(image, f"{state_label} | lead={lead_ms:.0f} ms | skip={SKIP_FIRST_NOTES} | hold={HOLD_ENDPOINT_MODE} | ESC exit",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
            # --- всё на экран: те же счётчики, что уходят в журнал -----------
            # Пользователь смотрит трансляцию с приставки, а не консоль бота, поэтому
            # все показатели печатаются поверх окна: по ним видно, что происходит в
            # конкретной тарелке, без чтения логов.
            if trophy_mode and rw >= 100:
                _song_t = (now - song_started) if song_started is not None else 0.0
                _bright = bright if bright is not None else 0.0
                _dt_ms = frame_dt[0] * 1000.0
                _fps = 1.0 / max(frame_dt[0], 1e-3)
                _status = [
                    f"песня {_song_t:6.1f} с   нажато {song_fired}  (увер {song_confident})",
                    f"повтор {song_double_fired}   слабых {song_low_confidence}   поздно {song_late_fired}",
                    f"края hold {len(hold_endpoint_track_ids)}   череп {song_skull}   "
                    f"потеряно {song_tracks_lost} (увер {song_lost_confident})",
                    f"мимо рядов {song_off_row}   блики {song_static_skipped}   "
                    f"дорожек {song_notes_seen}   удержаний {song_holds}",
                    f"в поле {len(note_tracks)} нот   скорость {speed_px_s:.0f} px/s   "
                    f"mash {'ДА' if mash_active else 'нет'}",
                    f"кадр {_dt_ms:.0f} мс ({_fps:.0f} fps)   яркость {_bright:.0f}   "
                    f"состояние {state}",
                ]
                _line_h = 17
                _top = 268
                cv2.rectangle(image, (8, _top - 15),
                              (740, _top + _line_h * len(_status) - 3), (25, 25, 25), -1)
                for _index, _text in enumerate(_status):
                    cv2.putText(image, _text, (14, _top + _index * _line_h),
                                cv2.FONT_HERSHEY_SIMPLEX, .52, (180, 255, 180), 1)
            bx1, by1, bx2, by2 = replay_button
            cv2.rectangle(image, (bx1, by1), (bx2, by2), (35, 35, 35), -1)
            cv2.rectangle(image, (bx1, by1), (bx2, by2), (0, 210, 255), 2)
            cv2.putText(image, "X / RETRY", (bx1 + 18, by1 + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 2)
            obx1, oby1, obx2, oby2 = options_button
            cv2.rectangle(image, (obx1, oby1), (obx2, oby2), (35, 35, 35), -1)
            cv2.rectangle(image, (obx1, oby1), (obx2, oby2), (255, 210, 0), 2)
            cv2.putText(image, "OPTIONS", (obx1 + 18, oby1 + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 2)
            # RECORD CONFIGURATION: show every frame.
            cv2.imshow(window, image)
            key = cv2.waitKey(1) & 0xFF
            # A keyboard shared between two machines (KVM / Flow) drops stray keys
            # into this window. SPACE pauses the presses, C re-draws the note field
            # (and switches auto OFF when it cannot find the pink marker), R toggles
            # recording, X sends a click to the console. Symptom on the console: the
            # bot keeps closing the results screen with X but presses no notes at
            # all (measured: 162 X presses, 0 note presses). In trophy/farm mode
            # only ESC is honoured.
            if trophy_mode and key != 255 and key != 27:
                if key != _ignored_key[0]:
                    _ignored_key[0] = key
                    msg = (f"Клавиша {key} проигнорирована (трофейный режим: "
                           f"работает только ESC).")
                    print(msg)
                    log_line(msg)
                key = 255
            elif key == 255:
                _ignored_key[0] = 255
            # Perf probe (once every ~5 s): real loop cost vs the 16.7 ms budget.
            if new_frame:
                perf_accum += time.perf_counter() - _iter_start
                perf_count += 1
                if time.perf_counter() - perf_since >= 5.0:
                    avg_ms = perf_accum / max(1, perf_count) * 1000.0
                    if trophy_mode:
                        log_line(f"Производительность цикла: {avg_ms:.1f} мс/кадр "
                                 f"(бюджет 16.7; {perf_count} кадров)")
                    perf_accum = 0.0
                    perf_count = 0
                    perf_since = time.perf_counter()
            if key == 27:
                break
            if key == 32:
                auto = not auto
                if not auto:
                    button_dispatcher.release_all()
                    active_holds.clear()
                    hold_endpoint_track_ids.clear()
                    print("Удерживаемые кнопки отпущены при паузе автонажатий.")
                else:
                    state_since = time.perf_counter()
                    dark_since = bright_since = None
                    if state == "RESULTS":
                        enter_state("INTRO", state_since)  # re-detect, avoid instant Retry
                print("Автонажатия включены." if auto else "Пауза.")
            if key == ord("r"):
                pass
            if key in KEY_BUTTONS:
                button_dispatcher.tap(KEY_BUTTONS[key])
            # live lead tuning: [ = press later, ] = press earlier
            if key == ord("["):
                lead_ms = max(0.0, lead_ms - 10.0)
                print(f"lead = {lead_ms:.0f} мс (жмёт ПОЗЖЕ)")
            if key == ord("]"):
                lead_ms = min(300.0, lead_ms + 10.0)
                print(f"lead = {lead_ms:.0f} мс (жмёт РАНЬШЕ)")
            if key in (10, 13):
                button_dispatcher.tap("CROSS")
                print("Отправлен X: повтор/подтверждение.")
            if key == ord("c"):
                auto = False
                button_dispatcher.release_all()
                active_holds.clear()
                hold_endpoint_track_ids.clear()
                cv2.destroyWindow(window)
                fresh = receiver.get_latest_video_frame()
                clean = fresh.to_ndarray(format="bgr24") if fresh is not None else image
                roi = cv2.selectROI("Выдели всё поле нот, включая оба ряда, и нажми ENTER", clean,
                                    showCrosshair=True, fromCenter=False)
                cv2.destroyWindow("Выдели всё поле нот, включая оба ряда, и нажми ENTER")
                new_rx, new_ry, new_rw, new_rh = [int(value) for value in roi]
                if new_rw >= 100 and new_rh >= 20:
                    rx, ry, rw, rh = new_rx, new_ry, new_rw, new_rh
                    target_lines.clear()
                    note_tracks.clear()
                    note_track_map.clear()
                    hold_tracks.clear()
                    hold_track_map.clear()
                    next_track_id = 1
                    next_hold_track_id = 1
                    speed_px_s = DEFAULT_SPEED
                    if trophy_mode:
                        auto_markers = detect_hit_markers(clean)
                        if auto_markers:
                            target_lines.extend(auto_markers)
                            auto = True
                            print(f"Метки найдены автоматически: {auto_markers}. "
                                  "Автонажатия включены.")
                        else:
                            auto = False
                            print("Метки не найдены — кликни по розовой метке мышью.")
                    save_roi_state(rx, ry, rw, rh, target_lines)
                    print("Область нот обновлена.")
                else:
                    print("Выбор отменён; оставил прежнюю область.")
                cv2.namedWindow(window)
                cv2.setMouseCallback(window, click_target)
            if key == ord("s"):
                cv2.imwrite("ramen_remoteplay_debug.png", image)
                print("Снимок сохранён: ramen_remoteplay_debug.png")
    finally:
        mash_stop.set()
        cv2.destroyAllWindows()
        button_dispatcher.release_all()
        device.disconnect()
        try:
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            pass
        if trophy_mode:
            log_line(f"Сессия завершена. Попыток (Retry): {servings}.")
        release_lock()




def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wanted: Dead — бот для рамен-мини-игры (трофей Ramen World Legend)")
    parser.add_argument("--discover", action="store_true",
                        help="найти консоль PS5 в локальной сети")
    parser.add_argument("--register", metavar="PS5_IP",
                        help="привязать консоль: вход в PSN и PIN Remote Play")
    parser.add_argument("--run", metavar="PS5_IP",
                        help="подключиться к консоли и играть")
    parser.add_argument("--lead-ms", type=float, default=51.0,
                        help="опережение нажатия в миллисекундах (по умолчанию 51)")
    parser.add_argument("--trophy", action="store_true",
                        help="трофейный режим: сам жмёт Retry и читает экран результатов")
    parser.add_argument("--auto-retry-delay", type=float, default=8.0,
                        help="сколько ждать на экране результатов перед Retry (с)")
    parser.add_argument("--auto-menu", action="store_true",
                        help="если в интро нет нот, прокликивать X каждые 4 секунды")
    parser.add_argument("--song-end-wait", type=float, default=12.0,
                        help="сколько секунд без нот считать концом песни (с)")
    parser.add_argument("--skip-first", type=int, default=20, metavar="N",
                        help="не нажимать первые N нот песни (по умолчанию 20)")
    args = parser.parse_args()

    global SKIP_FIRST_NOTES
    if args.skip_first is not None:
        SKIP_FIRST_NOTES = max(0, int(args.skip_first))

    try:
        if args.discover:
            discover()
        elif args.register:
            register(args.register)
        elif args.run:
            run_bot(args.run, args.lead_ms, trophy_mode=args.trophy,
                    auto_retry_delay=args.auto_retry_delay,
                    auto_menu=args.auto_menu, song_end_wait=args.song_end_wait)
        else:
            parser.print_help()
    except KeyboardInterrupt:
        print("Остановлено пользователем.")
    except Exception as exc:
        import traceback
        print(f"Ошибка: {exc}")
        print(traceback.format_exc())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
