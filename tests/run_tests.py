"""Minimal test runner (works with or without pytest installed).

Usage:  python tests/run_tests.py            (or: python -m pytest tests)
"""
import importlib
import inspect
import os
import sys
import tempfile
import traceback
import types
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

try:
    import pytest  # noqa: F401
except ImportError:
    shim = types.ModuleType("pytest")

    class _Mark:
        @staticmethod
        def parametrize(name, values):
            def deco(fn):
                fn._params = (name, values)
                return fn
            return deco

    shim.mark = _Mark()
    sys.modules["pytest"] = shim


def main():
    files = sorted(f[:-3] for f in os.listdir(HERE) if f.startswith("test_") and f.endswith(".py"))
    passed = failed = 0
    for modname in files:
        mod = importlib.import_module(modname)
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith("test_") or fn.__module__ != mod.__name__:
                continue
            params = getattr(fn, "_params", None)
            cases = [((), {})] if not params else [((), {params[0]: v}) for v in params[1]]
            for _, kw in cases:
                label = f"{modname}::{name}" + (f"[{list(kw.values())[0]!r}]" if kw else "")
                try:
                    if "tmp_path" in inspect.signature(fn).parameters:
                        with tempfile.TemporaryDirectory() as td:
                            fn(tmp_path=Path(td), **kw)
                    else:
                        fn(**kw)
                    passed += 1
                except Exception:
                    failed += 1
                    print(f"FAIL {label}")
                    traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
