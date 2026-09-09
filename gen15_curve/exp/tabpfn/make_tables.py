"""Assemble every result CSV this experiment wrote into one markdown block.

Written so that no number in ``REPORT.md`` is ever typed by hand.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K                                   # noqa: E402
from common import V                                 # noqa: E402

OUT = K.OUT
ORDER = ["FLAT", "G14",
         "TP_DIR39", "TP_DIR209", "TP_MAG", "TP_CURV", "TP_ALL",
         "CB_DIR", "CB_DIR209", "CB_DIR_ES", "CB_MAG", "CB_CURV",
         "XGB_MONO", "XGB_FREE", "ISO_PRIOR", "ISO_PICK",
         "GAM_DIR", "GAM_DIR_K", "GAM_MAG", "SYM_DIR",
         "MAG_MED", "CB_MAG_LEVEL", "CB_MAG_RANK"]

SOURCES = (("cheap", False), ("tabpfn", True), ("control", True))


def _order(df: pd.DataFrame) -> pd.DataFrame:
    idx = [a for a in ORDER if a in df.index] + [a for a in df.index if a not in ORDER]
    return df.loc[idx]


def _md(df: pd.DataFrame, floatfmt: str = "%.4f") -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt % v)
    head = "| " + " | ".join([d.index.name or ""] + [str(c) for c in d.columns]) + " |"
    rule = "|" + "|".join(["---"] * (len(d.columns) + 1)) + "|"
    rows = ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |"
            for i, r in zip(d.index, d.to_numpy())]
    return "\n".join([head, rule] + rows)


def boards() -> dict[str, pd.DataFrame]:
    frames = []
    for tag, drop_ref in SOURCES:
        p = OUT / f"{tag}_board.csv"
        if not p.exists():
            continue
        b = pd.read_csv(p)
        if drop_ref:
            b = b[~b.arm.isin(["FLAT", "G14"])]
        frames.append(b)
    B = pd.concat(frames, ignore_index=True)
    # CB_MAG is scored in both the cheap run and the control run on the same folds; keep one.
    B = B.drop_duplicates(subset=["arm", "design"], keep="first")
    return {v: _order(V.wide(B, v)) for v in ("macro_mae_extractant", "macro_mae_chemotype",
                                              "macro_sign_acc_strong", "macro_pair_spearman",
                                              "macro_mae_adjacent", "macro_mae_far")}


def main() -> None:
    out: list[str] = []
    bs = boards()
    out.append("### Extractant-macro MAE of predicted log SF (the endpoint; lower is better)\n")
    out.append(_md(bs["macro_mae_extractant"]))
    out.append("\n### Direction macro accuracy, gen14's yardstick (higher is better)\n")
    dfs = []
    for tag, drop in SOURCES:
        p = OUT / f"{tag}_direction.csv"
        if p.exists():
            d = pd.read_csv(p)
            dfs.append(d[~d.arm.isin(["FLAT", "G14"])] if drop else d)
    if dfs:
        db = pd.concat(dfs, ignore_index=True)
        w = _order(K.wide_dir(db))
        if "FLAT" in w.index:
            heavy = 1.0 - w.loc["FLAT"]      # every cell is either heavy- or light-selective
            w.loc["ALWAYS_HEAVY"] = heavy
            w = w.rename(index={"FLAT": "ALWAYS_LIGHT (= FLAT)"})
        out.append(_md(w))
    out.append("\n### Chemotype-macro MAE (the second unit gen13 reports)\n")
    out.append(_md(bs["macro_mae_chemotype"]))
    out.append("\n### Pairwise sign accuracy on strong pairs (|log SF| >= 0.3)\n")
    out.append(_md(bs["macro_sign_acc_strong"]))
    out.append("\n### Mean per-cell Spearman of predicted vs observed pairwise contrasts\n")
    out.append(_md(bs["macro_pair_spearman"]))

    cons = []
    for f, drop in (("cheap_contrasts.csv", False), ("tabpfn_contrasts.csv", True)):
        p = OUT / f
        if p.exists():
            c = pd.read_csv(p)
            if drop:
                c = c[~c.comparison.isin(["G14_vs_FLAT"])]
            cons.append(c)
    if cons:
        C = pd.concat(cons, ignore_index=True)
        C.to_csv(OUT / "all_contrasts.csv", index=False)
        for tag, sel in (("vs G14", "_vs_G14"), ("vs FLAT", "_vs_FLAT")):
            sub = C[C.comparison.str.endswith(sel)].copy()
            sub["arm"] = sub.comparison.str.replace(sel, "", regex=False)
            w = sub.pivot(index="arm", columns="design", values="point")
            w = w[[d for d in V.DESIGNS if d in w.columns]]
            out.append(f"\n### Paired gain {tag} (chemotype-blocked, positive = candidate better)\n")
            out.append(_md(_order(w)))
        bp = C[(C.design == "BP")][["comparison", "point", "ci95_low", "ci95_high", "bca_low",
                                    "p_two_sided", "seeds_positive", "loco_min", "loco_max",
                                    "loco_sign_stable", "passes_P1"]]
        bp = bp.set_index("comparison")
        out.append("\n### Full BP inference for every contrast\n")
        out.append(_md(bp))

    p = OUT / "symbolic_audit.csv"
    if p.exists():
        a = pd.read_csv(p)
        g = a.groupby("design")[["n_terms", "inner_cv_acc", "outer_acc", "bite_outer_acc",
                                 "oracle_best_outer_acc"]].mean()
        g = g.loc[[d for d in V.DESIGNS if d in g.index]]
        g.index.name = "design"
        out.append("\n### Symbolic search: what the inner CV promised and what came out of fold\n")
        out.append(_md(g))
        top = a.winner.value_counts().head(8).rename("folds").to_frame()
        top.index.name = "winning term"
        out.append("\n### Most frequently selected terms (out of 125 folds)\n")
        out.append(_md(top))

    for f, title in (("confound_magnitude.csv", "Magnitude confound guard"),
                     ("confound_direction_loco.csv", "Direction accuracy, leave-one-chemotype-out")):
        p = OUT / f
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d = d[d.design == "BP"].drop(columns=["design"]).set_index("arm")
        out.append(f"\n### {title} (design BP)\n")
        out.append(_md(_order(d)))

    txt = "\n".join(out) + "\n"
    (OUT / "tables.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
