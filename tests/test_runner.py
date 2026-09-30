"""JobRunner semantics (main-thread generator steps, worker threads, cancel, errors)
with a stub bpy, in a subprocess so the stub never leaks."""
import os
import subprocess
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPT = textwrap.dedent(r'''
import sys, types, time
sys.path.insert(0, ROOT)
class Any_:
    def __getattr__(self, n): return Any_()
    def __call__(self, *a, **k): return Any_()
bpy = types.ModuleType("bpy")
bpy.app = types.SimpleNamespace(timers=types.SimpleNamespace(register=lambda *a, **k: None))
bpy.context = types.SimpleNamespace(window_manager=None)
sys.modules["bpy"] = bpy
from neural_texture.blender import runner
from neural_texture.core.synth import Cancelled

log = []
def gen():
    for i in range(3):
        log.append(f"main{i}")
        yield (i + 1) / 3, f"step {i}"
def work(progress, cancel):
    for i in range(5):
        cancel.check(); progress(i / 5, "w"); time.sleep(0.01)
    return {"answer": 42}
job = runner.JobRunner("t", [("main", "a", gen), ("thread", "b", work), ("main", "c", lambda: "final")])
res = job.run_blocking()
assert job.status == "done" and res == "final" and log == ["main0", "main1", "main2"], (job.status, res, log)
assert job.progress == 1.0

# cancel during a thread step
def slow(progress, cancel):
    for i in range(200):
        cancel.check(); time.sleep(0.01)
job2 = runner.JobRunner("c", [("thread", "slow", slow)])
import threading
threading.Timer(0.1, job2.cancel).start()
try:
    job2.run_blocking(); raise SystemExit("expected cancel")
except Cancelled:
    pass
assert job2.status == "cancelled"

# error in a main step closes the generator (temp-scene cleanup runs)
closed = []
def failing():
    try:
        yield 0.5, "x"
        raise ValueError("boom")
    finally:
        closed.append(True)
job3 = runner.JobRunner("e", [("main", "f", failing)])
try:
    job3.run_blocking(); raise SystemExit("expected error")
except RuntimeError as e:
    assert "boom" in str(e)
assert closed == [True] and job3.status == "error"

# cancel between main steps closes the open generator
closed2 = []
def long():
    try:
        for i in range(100):
            yield i / 100, "r"
    finally:
        closed2.append(True)
job4 = runner.JobRunner("g", [("main", "l", long)])
orig = job4._tick
n = [0]
def tick():
    n[0] += 1
    if n[0] == 5: job4.cancel()
    return orig()
job4._tick = tick
try:
    job4.run_blocking()
except Cancelled:
    pass
assert job4.status == "cancelled" and closed2 == [True]
print("RUNNER OK")
''')


def test_job_runner():
    r = subprocess.run([sys.executable, "-c", f"ROOT={ROOT!r}\n" + SCRIPT], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0 and "RUNNER OK" in r.stdout, r.stdout + r.stderr
