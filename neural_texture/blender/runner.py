"""Non-blocking job runner (V4): progress bar + cancel without freezing Blender.

A job is a list of steps:
  ("main",   callable -> iterator of (fraction, message) | None)  bpy work, one
             iteration per timer tick on the main thread (renders, bakes, materials)
  ("thread", callable(progress, cancel) -> result)                 pure-Python work
             (ComfyUI generation, synthesis, map building) in a worker thread;
             must NOT touch bpy.
Ticks are driven by bpy.app.timers, so it works from operators, the live
agent bridge and even background mode (run_blocking()).
"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Callable, List, Optional, Tuple

import bpy

from ..core.synth import CancelToken, Cancelled

Step = Tuple[str, str, Callable]   # (kind, label, fn)


class JobRunner:
    def __init__(self, name: str, steps: List[Step], on_done: Optional[Callable] = None):
        self.name = name
        self.steps = steps
        self.on_done = on_done
        self.cancel_token = CancelToken()
        self.status = "idle"         # idle | running | done | cancelled | error
        self.progress = 0.0
        self.message = ""
        self.error = ""
        self.result = None
        self.started = 0.0
        self._i = 0
        self._iter = None
        self._thread: Optional[threading.Thread] = None
        self._thread_out = None
        self._thread_err: Optional[BaseException] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------ control
    def start(self):
        global CURRENT
        if CURRENT is not None and CURRENT.running:
            raise RuntimeError(f"Another Neural Texture job is running: {CURRENT.name}")
        CURRENT = self
        self.status = "running"
        self.started = time.time()
        bpy.app.timers.register(self._tick, first_interval=0.01)
        return self

    def cancel(self):
        self.cancel_token.cancel()
        self.message = "cancelling..."

    @property
    def running(self) -> bool:
        return self.status == "running"

    def _set(self, frac: float, msg: str):
        with self._lock:
            n = max(1, len(self.steps))
            self.progress = min(1.0, (self._i + max(0.0, min(1.0, frac))) / n)
            self.message = msg

    # ------------------------------------------------------------ ticking
    def _finish(self, status: str, error: str = ""):
        if self._iter is not None and hasattr(self._iter, "close"):
            try:
                self._iter.close()  # runs finally-blocks (temp scene cleanup)
            except Exception:
                pass
        self._iter = None
        self.status = status
        self.error = error
        if status == "done":
            self.progress = 1.0
        self.message = {"done": "done", "cancelled": "cancelled", "error": f"error: {error}"}.get(status, status)
        if self.on_done:
            try:
                self.on_done(self)
            except Exception:
                traceback.print_exc()
        _redraw()

    def _tick(self):
        try:
            if self.status != "running":
                return None
            if self._i >= len(self.steps):
                self._finish("done")
                return None
            kind, label, fn = self.steps[self._i]
            if kind == "main":
                if self.cancel_token.cancelled:
                    raise Cancelled("cancelled")
                if self._iter is None:
                    out = fn()
                    if out is None or not hasattr(out, "__next__"):
                        self.result = out if out is not None else self.result
                        self._next()
                        return 0.01
                    self._iter = out
                try:
                    frac, msg = next(self._iter)
                    self._set(frac, f"{label}: {msg}")
                except StopIteration:
                    self._iter = None
                    self._next()
                return 0.01
            # thread step
            if self._thread is None:
                self._thread_out, self._thread_err = None, None

                def work():
                    try:
                        self._thread_out = fn(lambda f, m: self._set(f, f"{label}: {m}"), self.cancel_token)
                    except BaseException as e:  # noqa: BLE001 - reported on main thread
                        self._thread_err = e
                        self._thread_tb = traceback.format_exc()
                self._thread = threading.Thread(target=work, name=f"NT-{label}", daemon=True)
                self._thread.start()
                return 0.1
            if self._thread.is_alive():
                _redraw()
                return 0.2
            self._thread = None
            if self._thread_err is not None:
                raise self._thread_err
            self.result = self._thread_out
            self._next()
            return 0.01
        except Cancelled:
            self._finish("cancelled")
            return None
        except Exception as e:  # noqa: BLE001
            print(getattr(self, "_thread_tb", "") or traceback.format_exc())
            self._finish("error", str(e))
            return None

    def _next(self):
        self._i += 1
        self._set(0.0, self.message)
        _redraw()

    # ------------------------------------------------------------ headless
    def run_blocking(self, poll: float = 0.05):
        """Background mode / CLI: drive ticks inline (timers don't run without UI)."""
        global CURRENT
        CURRENT = self
        self.status = "running"
        self.started = time.time()
        while self.status == "running":
            delay = self._tick()
            if delay is None:
                break
            time.sleep(min(delay, poll))
        if self.status == "error":
            raise RuntimeError(self.error)
        if self.status == "cancelled":
            raise Cancelled("cancelled")
        return self.result


CURRENT: Optional[JobRunner] = None


def current() -> Optional[JobRunner]:
    return CURRENT


def _redraw():
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type in ("VIEW_3D", "PROPERTIES", "IMAGE_EDITOR"):
                area.tag_redraw()
