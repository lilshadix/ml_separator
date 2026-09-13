"""``screen.py`` -- the gen15 direction prior as a pre-screen only (DESIGN.md section 10.7).

``direction_prior(smiles, pair)`` runs ``generations/gen15_curve/scripts/g15_predict.py predict``
in a subprocess when the deployed model ``paths.GEN15_DEPLOY`` exists and returns
``{"pair": pair, "sign": +1 | -1, "source": "gen15 deploy_g15.joblib", "note": "pre-screen only;
not a D source"}``; otherwise ``None``.  ``sign`` is the sign of the predicted
``log10 D(pair[0]) - log10 D(pair[1])`` (the gen15 pair table's ``log_SF`` with ``A/B`` order
honoured).  The only consumer is ``scripts/g18_screen.py``; validator V6 refuses any D whose source
mentions gen15, so nothing here can leak into a cascade.  Failures of the subprocess (missing
script, non-zero exit, unparsable table) also return ``None`` -- the prior is optional.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from gen18proc import paths

__all__ = ["direction_prior", "SOURCE_LABEL", "NOTE_LABEL", "gen15_available"]

SOURCE_LABEL = "gen15 deploy_g15.joblib"
NOTE_LABEL = "pre-screen only; not a D source"
_TIMEOUT_S = 600


def gen15_available() -> bool:
    """Whether the deployed gen15 model and its CLI script exist (both gitignored/optional)."""
    return Path(paths.GEN15_DEPLOY).is_file() and Path(paths.GEN15_PREDICT_SCRIPT).is_file()


def _run_predict(smiles: str, out_csv: Path) -> subprocess.CompletedProcess[str] | None:
    cmd = [sys.executable, str(paths.GEN15_PREDICT_SCRIPT), "predict", "--smiles", smiles,
           "--output", str(out_csv), "--top", "1"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        return subprocess.run(cmd, cwd=str(paths.REPO_ROOT), capture_output=True, text=True,
                              timeout=_TIMEOUT_S, env=env, check=False)
    except (OSError, subprocess.SubprocessError):
        return None


def _parse_direction(stdout: str) -> tuple[str | None, float | None]:
    """``direction`` and ``p_heavy`` from the CLI's second line; (None, None) if absent."""
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith("direction"):
            parts = s.split()
            direction = parts[1] if len(parts) > 1 else None
            p_heavy = None
            if "p_heavy" in s:
                try:
                    p_heavy = float(s.split("p_heavy", 1)[1].split(",", 1)[0].strip())
                except ValueError:
                    p_heavy = None
            return direction, p_heavy
    return None, None


def direction_prior(smiles: str, pair: tuple[str, str] | list[str]) -> dict[str, Any] | None:
    """gen15 direction prior for ``pair = (A, B)`` (DESIGN.md section 10.7); ``None`` when the
    deployed model is absent or the prediction cannot be obtained.

    The returned mapping has exactly the four contract keys; the ``note`` carries the predicted
    log SF, its sd and the curve direction after the mandatory ``"pre-screen only; not a D
    source"`` prefix so nothing else needs to be parsed downstream.
    """
    a, b = str(pair[0]), str(pair[1])
    if not gen15_available():
        return None
    with tempfile.TemporaryDirectory(prefix="g18_screen_") as tmp:
        out_csv = Path(tmp) / "pairs.csv"
        proc = _run_predict(smiles, out_csv)
        if proc is None or proc.returncode != 0 or not out_csv.is_file():
            return None
        try:
            table = pd.read_csv(out_csv)
        except (OSError, ValueError):
            return None
        direction, p_heavy = _parse_direction(proc.stdout)
    if not {"A", "B", "log_SF"}.issubset(table.columns):
        return None
    row = table.loc[(table["A"] == a) & (table["B"] == b)]
    flip = 1.0
    if row.empty:
        row = table.loc[(table["A"] == b) & (table["B"] == a)]
        flip = -1.0
    if row.empty:
        return None
    log_sf = flip * float(row["log_SF"].iloc[0])
    sd = float(row["sd"].iloc[0]) if "sd" in row.columns else float("nan")
    sign = 1 if log_sf >= 0 else -1
    note = (f"{NOTE_LABEL}; predicted log10 D({a}) - log10 D({b}) = {log_sf:+.4f} "
            f"(sd {sd:.4f}); curve direction {direction or 'unknown'}"
            + (f", p_heavy {p_heavy:.4f}" if p_heavy is not None else ""))
    return {"pair": (a, b), "sign": sign, "source": SOURCE_LABEL, "note": note}
