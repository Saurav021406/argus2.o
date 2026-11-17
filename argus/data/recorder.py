"""Human demonstration recorder: screen frames + mouse/keyboard events.

Run on Windows:
    python -m argus.data record --instruction "Open Notepad" --task open_app --app notepad

Press F10 to stop. The F10 press itself is not recorded as an action.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime
from typing import List, Optional

from argus.actions.action_space import GRID
from argus.actions.platform import ensure_dpi_aware
from argus.data.episode import Episode
from argus.data.labeler import build_steps

_SPECIAL = {
    "enter": "enter", "backspace": "backspace", "tab": "tab", "esc": "esc",
    "space": "space", "delete": "delete", "up": "up", "down": "down",
    "left": "left", "right": "right", "home": "home", "end": "end",
    "page_up": "pageup", "page_down": "pagedown",
    "ctrl": "ctrl", "ctrl_l": "ctrl", "ctrl_r": "ctrl",
    "alt": "alt", "alt_l": "alt", "alt_r": "alt", "alt_gr": "alt",
    "shift": "shift", "shift_l": "shift", "shift_r": "shift",
    "cmd": "win", "cmd_l": "win", "cmd_r": "win",
}


def _key_name(key) -> Optional[str]:
    """Normalise a pynput key to a string. Returns None for keys we ignore."""
    ch = getattr(key, "char", None)
    if ch is not None:
        # Ctrl+letter arrives as a control character; recover the letter
        if len(ch) == 1 and ord(ch) < 32:
            return chr(ord(ch) + 96)
        return ch
    name = getattr(key, "name", str(key)).lower()
    if name in _SPECIAL:
        return _SPECIAL[name]
    if name.startswith("f") and name[1:].isdigit():
        return name
    return name


class Recorder:
    def __init__(self, out_root: str, instruction: str, task: str = "", app: str = "",
                 fps: float = 5.0, max_width: int = 1280, monitor: int = 1,
                 stop_key: str = "f10", source: str = "human", grid: int = GRID):
        self.out_root, self.instruction, self.task, self.app = out_root, instruction, task, app
        self.fps, self.max_width, self.monitor = fps, max_width, monitor
        self.stop_key, self.source, self.grid = stop_key, source, grid
        self.events: List[dict] = []
        self.frames: List[dict] = []
        self._stop = threading.Event()
        self._t0 = 0.0
        self.ep_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:4]
        self.dir = os.path.join(out_root, self.ep_id)

    def _now(self) -> float:
        return time.perf_counter() - self._t0

    # ---- input listeners -------------------------------------------------
    def _start_listeners(self):
        from pynput import keyboard, mouse

        def on_click(x, y, button, pressed):
            self.events.append({"t": self._now(), "kind": "press" if pressed else "release",
                                "x": int(x), "y": int(y), "button": button.name})

        def on_scroll(x, y, dx, dy):
            self.events.append({"t": self._now(), "kind": "scroll", "x": int(x), "y": int(y), "dy": int(dy)})

        def on_down(key):
            name = _key_name(key)
            if name is None:
                return
            if name == self.stop_key:
                self._stop.set()
                return
            self.events.append({"t": self._now(), "kind": "key_down", "key": name})

        def on_up(key):
            name = _key_name(key)
            if name is None or name == self.stop_key:
                return
            self.events.append({"t": self._now(), "kind": "key_up", "key": name})

        self._ml = mouse.Listener(on_click=on_click, on_scroll=on_scroll)
        self._kl = keyboard.Listener(on_press=on_down, on_release=on_up)
        self._ml.start()
        self._kl.start()

    # ---- capture loop ----------------------------------------------------
    def _capture_loop(self):
        import mss
        from PIL import Image

        os.makedirs(os.path.join(self.dir, "frames"), exist_ok=True)
        period = 1.0 / self.fps
        with mss.mss() as sct:
            mon = sct.monitors[self.monitor]
            self.screen_w, self.screen_h = mon["width"], mon["height"]
            scale = min(1.0, self.max_width / self.screen_w)
            self.frame_w, self.frame_h = int(self.screen_w * scale), int(self.screen_h * scale)
            next_t = 0.0
            i = 0
            while not self._stop.is_set():
                now = self._now()
                if now < next_t:
                    time.sleep(min(0.005, next_t - now))
                    continue
                shot = sct.grab(mon)
                img = Image.frombytes("RGB", shot.size, shot.rgb)
                if scale < 1.0:
                    img = img.resize((self.frame_w, self.frame_h), Image.BILINEAR)
                rel = f"frames/{i:06d}.jpg"
                img.save(os.path.join(self.dir, rel), quality=85)
                self.frames.append({"i": i, "t": round(now, 4), "file": rel})
                i += 1
                next_t += period
                if next_t < now - period:      # fell behind; do not burst
                    next_t = now

    # ---- public API ------------------------------------------------------
    def run(self, countdown: int = 3) -> Episode:
        ensure_dpi_aware()
        for n in range(countdown, 0, -1):
            print(f"Recording starts in {n}...", flush=True)
            time.sleep(1)
        print(f"RECORDING. Do the task: {self.instruction!r}. Press {self.stop_key.upper()} to stop.", flush=True)
        self._t0 = time.perf_counter()
        self._start_listeners()
        try:
            self._capture_loop()
        finally:
            self._ml.stop()
            self._kl.stop()
        return self.finish()

    def finish(self) -> Episode:
        steps = build_steps(self.events, [f["t"] for f in self.frames],
                            self.screen_w, self.screen_h, self.grid)
        meta = {
            "id": self.ep_id, "instruction": self.instruction, "task": self.task, "app": self.app,
            "screen_w": self.screen_w, "screen_h": self.screen_h,
            "frame_w": self.frame_w, "frame_h": self.frame_h,
            "fps": self.fps, "grid": self.grid, "source": self.source,
            "created": datetime.now().isoformat(timespec="seconds"), "success": None,
        }
        ep = Episode(self.dir, meta, self.frames, steps)
        with open(os.path.join(self.dir, "events.jsonl"), "w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(e) + "\n")
        ep.save_labels()
        print(f"Saved {len(self.frames)} frames, {len(steps)} actions -> {self.dir}")
        return ep
