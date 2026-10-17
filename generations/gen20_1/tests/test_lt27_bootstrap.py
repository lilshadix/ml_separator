"""LT-27: gen13's output directories exist before gen13sep.paths is imported (it calls mkdir at import), a
missing generations/gen14_direction/cache/bench.pkl fails fast (gen14.dirbench.load would rebuild and write
it), and the bootstrap sets the bytecode and thread guards (IMPLEMENTATION_PLAN.md §8)."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from gen201 import bootstrap


def test_gen13_output_dirs_exist():
    for d in bootstrap.GEN13_OUTPUT_DIRS:
        assert d.is_dir(), d
    bootstrap.require_gen13_output_dirs()


def test_missing_gen13_output_dir_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        bootstrap.require_gen13_output_dirs((tmp_path / "metrics",))
    assert not (tmp_path / "metrics").exists()


def test_missing_bench_cache_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        bootstrap.require_bench_cache(tmp_path / "bench.pkl")
    assert bootstrap.require_bench_cache() == bootstrap.BENCH_PKL


def _run(code: str, env_extra: dict | None = None, drop=(), flags=("-B",)) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update({"PYTHONIOENCODING": "utf-8", **(env_extra or {})})
    if "-B" in flags:
        env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([sys.executable, *flags, "-c", code], capture_output=True, text=True, env=env,
                          cwd=str(bootstrap.ROOT), timeout=300)


def test_bootstrap_import_refuses_a_missing_bench_cache():
    """In a fresh interpreter whose filesystem view hides bench.pkl, importing gen201 raises before any
    frozen package is imported."""
    code = textwrap.dedent(f"""
        import pathlib, sys
        _exists = pathlib.Path.exists
        pathlib.Path.exists = lambda self, *a, **k: False if self.name == "bench.pkl" else _exists(self, *a, **k)
        sys.path.insert(0, {str(bootstrap.G201)!r})
        try:
            import gen201
        except FileNotFoundError as exc:
            print("REFUSED", "gen20k" in sys.modules, "gen13sep" in sys.modules, "gen14" in sys.modules)
            sys.exit(3)
        print("IMPORTED")
    """)
    r = _run(code)
    assert r.returncode == 3, r.stdout + r.stderr
    assert "REFUSED False False False" in r.stdout


def test_bootstrap_sets_bytecode_and_thread_guards(tmp_path):
    """Without -B and PYTHONDONTWRITEBYTECODE (bytecode redirected to a temporary PYTHONPYCACHEPREFIX), the
    bootstrap turns bytecode writing off before any frozen module is imported: only gen201's own two
    modules can have been compiled to bytecode, never gen20k/gen13sep/gen14/gen15 ones."""
    prefix = tmp_path / "pyc"
    code = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(bootstrap.G201)!r})
        before = sys.dont_write_bytecode
        import gen201
        print(before, sys.dont_write_bytecode, os.environ["LOKY_MAX_CPU_COUNT"], os.environ["OMP_NUM_THREADS"],
              "torch" in sys.modules, "gen20k.bootstrap" in sys.modules)
    """)
    r = _run(code, env_extra={"PYTHONPYCACHEPREFIX": str(prefix)},
             drop=("LOKY_MAX_CPU_COUNT", "OMP_NUM_THREADS", "PYTHONDONTWRITEBYTECODE"), flags=())
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["False", "True", "2", "2", "False", "True"]
    written = sorted(p.as_posix() for p in prefix.rglob("*.pyc")) if prefix.exists() else []
    repo = [p for p in written if "/ml_separator_gh/" in p and "/.venv/" not in p]
    assert all("/generations/gen20_1/gen201/" in p for p in repo), repo


def test_sys_path_order():
    """gen20_kshot is importable as gen20k and gen16 is never put on sys.path (two packages share that name)."""
    import gen20k
    assert os.path.normcase(os.path.dirname(os.path.dirname(gen20k.__file__))) == os.path.normcase(str(bootstrap.G20))
    assert not any("gen16" in p for p in sys.path)
