"""Script 11 of DESIGN.md section 11: the gen15 direction prior as a pre-screen only.

Reads SMILES (one per line, ``#`` comments allowed) and writes ``results/screen/priors.csv``
with ``gen18proc.screen.direction_prior(smiles, pair)`` per molecule: ``pair``, ``sign``,
``source = "gen15 deploy_g15.joblib"`` and the mandatory note ``"pre-screen only; not a D
source"``.  When the deployed gen15 model is absent the table is empty and carries a note; the
prior never enters a cascade (validator V6 refuses gen15-sourced D).

Usage (from the repository root, one process):
    .venv/Scripts/python.exe generations/gen18_process/scripts/g18_screen.py \\
        --smiles-file <txt> --pair Nd Pr [--out-dir results/screen]

No wall-clock value is written.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen18proc import DEFAULT_SEED, paths  # noqa: E402
from gen18proc.report import write_manifest, write_table  # noqa: E402
from gen18proc.screen import (  # noqa: E402
    NOTE_LABEL,
    SOURCE_LABEL,
    direction_prior,
    gen15_available,
)

COLUMNS = ["smiles", "pair", "sign", "source", "note"]


def resolve_input(spec: str) -> Path:
    """A relative input path, resolved against the working directory, then the generation root.

    Scripts are documented as run from the repository root, but their input files live under
    ``generations/gen18_process/``; both spellings (and an absolute path) therefore work, and the
    error names every place that was tried (integration, 2026-09-13).
    """
    path = Path(spec)
    if path.is_absolute():
        return path
    tried = [Path.cwd() / path, paths.G18_ROOT / path, paths.REPO_ROOT / path]
    for candidate in tried:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{spec!r} not found; tried " + ", ".join(str(t) for t in tried))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smiles-file", required=True)
    ap.add_argument("--pair", nargs=2, default=["Nd", "Pr"], metavar=("A", "B"))
    ap.add_argument("--out-dir", default=str(paths.RESULTS_SCREEN_DIR))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    smiles_path = resolve_input(args.smiles_file)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = paths.G18_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # one SMILES per line; a "#" starts a comment, whole-line or trailing
    smiles = [ln.split("#", 1)[0].strip()
              for ln in smiles_path.read_text(encoding="utf-8").splitlines()]
    smiles = [s for s in smiles if s]
    pair = (str(args.pair[0]), str(args.pair[1]))
    available = gen15_available()
    rows = []
    for s in smiles:
        prior = direction_prior(s, pair) if available else None
        if prior is None:
            rows.append({"smiles": s, "pair": "/".join(pair), "sign": None, "source": SOURCE_LABEL,
                         "note": ("gen15 deploy model absent (%s); no prior" % paths.GEN15_DEPLOY)
                         if not available else f"{NOTE_LABEL}; prediction unavailable"})
        else:
            rows.append({"smiles": s, "pair": "/".join(prior["pair"]), "sign": prior["sign"],
                         "source": prior["source"], "note": prior["note"]})
    table = pd.DataFrame(rows, columns=COLUMNS)
    if not available:
        table = table.iloc[0:0]
    regime = {"cohort": f"{len(smiles)} SMILES from {smiles_path.name}",
              "holdout": "gen15 deploy model (BP hold-out of gen15; not evaluated here)",
              "averaging_unit": "molecule",
              "note": (f"{NOTE_LABEL}; gen15 model absent: empty table"
                       if not available else NOTE_LABEL)}
    out = write_table(table, out_dir / "priors.csv", regime=regime)
    write_manifest(out_dir / "manifest.json", [out], [smiles_path], args.seed,
                   arguments=vars(args), extra={"gen15_available": available,
                                                "n_smiles": len(smiles)})
    print(f"wrote {out} ({len(table)} rows; gen15 available = {available})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
