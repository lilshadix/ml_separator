#!/usr/bin/env python
"""E-LAB: level-matched arm on the pure new-extractant stratum (POPX0: no same-extractant cell in the publication) and
on POPX1 (a same-extractant cell exists).  EXPLORATORY; joins lab_rows.parquet (n_MS) to lab_rows2.parquet.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/lab_popx0.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\scripts")
import pandas as pd  # noqa: E402

from lab_analyze import r2, rel, infer  # noqa: E402

OUT = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\lab_out")
LIGHTP = {"La/Ce", "Ce/Pr", "Pr/Nd"}


def main() -> int:
    a = pd.read_parquet(OUT / "lab_rows.parquet")
    b = pd.read_parquet(OUT / "lab_rows2.parquet")
    key = ["design", "split_seed", "cell_id", "pair"]
    m = b.merge(a[key + ["n_MS", "MX_light", "MX_heavy"]], on=key, how="left", validate="one_to_one")
    for design in ("BPG", "BP"):
        d = m[(m.design == design) & (m.n_MX > 0) & ~m.t_v2drop]
        for nm, s in (("POPX0 (no same-extractant cell)", d[d.n_MS == 0]), ("POPX1 (same-extractant cell exists)", d[d.n_MS > 0])):
            r = infer(s, "MX_LM10", "B1")
            q = infer(s, "MX_LM10", "MG_LM10_all")
            print(f"{design} {nm}: cells {s.cell_id.nunique()}, pubs {s.publication.nunique()}, rows/seed {len(s) / 5:.0f}: "
                  f"B1 {r2(s, 'B1'):.4f}  MX {r2(s, 'MX'):.4f}  MX_LM10 {r2(s, 'MX_LM10'):.4f}  MG_LM10_all {r2(s, 'MG_LM10_all'):.4f}"
                  f"\n    MX_LM10 - B1 rel {r['rel']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] up/down {r['up']}/{r['down']} p {r['p_sf']:.3f};"
                  f"  MX_LM10 - MG_LM10_all rel {q['rel']:+.3f} [{q['lo']:+.3f}, {q['hi']:+.3f}] up/down {q['up']}/{q['down']} p {q['p_sf']:.3f}")
        if design == "BPG":
            d = m[(m.design == design) & (m.n_MX > 0) & ~m.t_v2drop]
            g = (d.B1 - d.y) ** 2 - (d.MX_LM10 - d.y) ** 2
            s = g.groupby(d.pair).sum() / 5
            print("   MX_LM10 - B1 gain share by pair (POPX):", (s / s.sum()).round(3).to_dict())
            print(f"   light-pair share {s[s.index.isin(LIGHTP)].sum() / s.sum():.3f}; Ce pairs "
                  f"{s[s.index.isin({'La/Ce', 'Ce/Pr'})].sum() / s.sum():.3f}; Gd pairs {s[s.index.isin({'Eu/Gd', 'Gd/Tb'})].sum() / s.sum():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
