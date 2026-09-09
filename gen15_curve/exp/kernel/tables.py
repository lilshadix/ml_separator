"""Merge the per-design boards of a batch and render the markdown tables the report needs.

    python tables.py <batch> [<batch> ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "gen15_curve"))
from gen15 import valuebench as V     # noqa: E402

OUT = HERE / "results"
DESIGNS = ["B", "BR", "BQ", "A", "BP"]


def load_board(name: str) -> pd.DataFrame:
    parts = [pd.read_csv(OUT / f"board_{name}_{d}.csv")
             for d in DESIGNS if (OUT / f"board_{name}_{d}.csv").exists()]
    if not parts:
        raise SystemExit(f"no boards for batch {name}")
    B = pd.concat(parts, ignore_index=True)
    B.to_csv(OUT / f"board_{name}.csv", index=False)
    return B


def md(w: pd.DataFrame) -> str:
    w = w.round(4)
    head = "| arm | " + " | ".join(w.columns) + " |"
    rule = "|" + "---|" * (len(w.columns) + 1)
    body = ["| %s | %s |" % (i, " | ".join(f"{v:.4f}" for v in r))
            for i, r in zip(w.index, w.to_numpy())]
    return "\n".join([head, rule] + body)


def main() -> None:
    for name in sys.argv[1:]:
        B = load_board(name)
        w = V.wide(B, "macro_mae_extractant")
        key = "BP" if "BP" in w.columns else w.columns[0]
        print(f"\n\n#### `{name}` -- extractant-macro MAE (lower is better), sorted by BP\n")
        print(md(w.sort_values(key)))
        w2 = V.wide(B, "macro_sign_acc_strong")
        print(f"\n\n#### `{name}` -- macro sign accuracy on strong pairs (higher is better)\n")
        print(md(w2.sort_values(key, ascending=False)))
        w3 = V.wide(B, "macro_pair_spearman")
        print(f"\n\n#### `{name}` -- macro pair Spearman (higher is better)\n")
        print(md(w3.sort_values(key, ascending=False)))


if __name__ == "__main__":
    main()
