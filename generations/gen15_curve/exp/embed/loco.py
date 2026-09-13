"""Leave-one-chemotype-out stability of the direction call, TOPO39 vs the embedding.

Protocol point 3 asks for chemotype-level stability rather than one pooled number.  Design B and
design BP already hold a whole chemotype out, so the per-chemotype breakdown of their out-of-fold
predictions *is* the leave-one-chemotype-out curve: for every chemotype, the accuracy of the model
that never saw it.  Reported here as (a) the per-chemotype accuracies, (b) how many chemotypes each
representation wins / loses / ties, and (c) the macro accuracy recomputed with each chemotype
dropped in turn, which says whether the verdict rests on any single chemotype.

Usage:  python loco.py <tag> [designs]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for p in (str(ROOT / "generations" / "gen15_curve"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import features as FEAT                                            # noqa: E402
from direction_acc import run_matrix, run_topo                     # noqa: E402
from gen14 import dirbench as D                                    # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)


def per_chemotype(oof: D.OOF) -> pd.Series:
    """Accuracy inside each chemotype, averaged over extractants first (gen14's unit)."""
    c = oof.cells.copy()
    c["correct"] = ((c["p"] >= 0.5).astype(int) == c["y"]).astype(float)
    by_ext = c.groupby(["chemotype", "extractant"])["correct"].mean().reset_index()
    return by_ext.groupby("chemotype")["correct"].mean()


def jackknife_macro(oof: D.OOF) -> pd.Series:
    """Macro accuracy with each chemotype dropped in turn."""
    pc = per_chemotype(oof)
    return pd.Series({ch: float(pc.drop(ch).mean()) for ch in pc.index})


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "chemberta_mtr__mean"
    designs = sys.argv[2].split(",") if len(sys.argv) > 2 else ["B", "BP"]
    bench = D.load()
    ext = tuple(bench.frame.extractant.astype(str).tolist())
    X = FEAT.cell_block(tag, ext)
    rows, summary = [], []
    for design in designs:
        t = run_topo(bench, design)
        e = run_matrix(bench, tag, X, design, C=1.0)
        pt, pe = per_chemotype(t), per_chemotype(e)
        jt, je = jackknife_macro(t), jackknife_macro(e)
        idx = sorted(set(pt.index) | set(pe.index))
        for ch in idx:
            rows.append({"design": design, "chemotype": ch,
                         "topo39_acc": float(pt.get(ch, np.nan)),
                         "embed_acc": float(pe.get(ch, np.nan)),
                         "delta": float(pe.get(ch, np.nan) - pt.get(ch, np.nan)),
                         "topo39_macro_without": float(jt.get(ch, np.nan)),
                         "embed_macro_without": float(je.get(ch, np.nan))})
        d = np.array([pe.get(ch, np.nan) - pt.get(ch, np.nan) for ch in idx], dtype=float)
        d = d[~np.isnan(d)]
        summary.append({"design": design, "n_chemotypes": len(d),
                        "topo39_macro": float(pt.mean()), "embed_macro": float(pe.mean()),
                        "embed_wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
                        "embed_loses": int((d < 0).sum()),
                        "worst_case_topo39_macro": float(jt.min()),
                        "best_case_embed_macro": float(je.max()),
                        "embed_ever_beats_topo39_after_dropping_one": bool(je.max() > jt.min())})
    R = pd.DataFrame(rows)
    S = pd.DataFrame(summary)
    R.to_csv(OUT / f"loco_{tag.replace(':', '-')}.csv", index=False)
    S.to_csv(OUT / f"loco_summary_{tag.replace(':', '-')}.csv", index=False)
    print("=== leave-one-chemotype-out summary ===")
    print(S.round(4).to_string(index=False))
    print("\n=== per chemotype (design BP) ===")
    b = R[R.design == "BP"].sort_values("delta")
    print(b[["chemotype", "topo39_acc", "embed_acc", "delta"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
