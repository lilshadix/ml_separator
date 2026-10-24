#!/usr/bin/env python
"""Skeptic B on confirm-2, check 2 (EXPLORATORY, already-seen data): the Gd chord of every V1 bench cell (521 cells, not
only C3) by curated mechanism, to see how many acidic organophosphorus / chelating cells the corpus actually holds and
what their chord is; plus chord vs mean adjacent step (|slope|) to test a proportional-amplitude alternative
(chord = gamma * slope) against the categorical heavy-selective stratum.  Writes only ideas2/confirm2skB_out/.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
OUT = HERE.parents[1] / "confirm2skB_out"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE.parents[3] / "round4_20261022"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as SS  # noqa: E402

from r4 import paths  # noqa: E402
from r4 import donor as DN  # noqa: E402
from r4 import slots_a as SA  # noqa: E402
from r3 import interp  # noqa: E402
from gen20k import data as D  # noqa: E402

CLIP = (-0.40, 0.20)
BETA = -0.0923
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def main() -> int:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 45)
    from rdkit import Chem
    b = D.load_bench()
    f = b.frame.reset_index(drop=True)
    Y = np.asarray(b.Y, float)
    LN = list(D.LANTHANIDES)
    XR = interp.X_R
    ig, ie, it = LN.index("Gd"), LN.index("Eu"), LN.index("Tb")
    t = (XR[ig] - XR[ie]) / (XR[it] - XR[ie])
    c = Y[:, ig] - ((1 - t) * Y[:, ie] + t * Y[:, it])
    sl = np.full(len(Y), np.nan)
    for k in range(len(Y)):
        m = np.isfinite(Y[k])
        if m.sum() >= 5:
            sl[k] = np.polyfit(XR[m], Y[k, m], 1)[0]
    # Eu->Tb local step (log D(Tb) - log D(Eu)), the scale the chord could be proportional to
    step = Y[:, it] - Y[:, ie]
    ref = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    c3 = set(ref.cell_id)
    mech = SA._mechanisms()

    def ik(s):
        try:
            return Chem.MolToInchiKey(Chem.MolFromSmiles(s))
        except Exception:
            return ""
    d = pd.DataFrame({"cell_id": f.cell_id, "c": c, "slope": sl, "step": step, "c3": f.cell_id.isin(c3),
                      "dga": f.extractant.astype(str).map(lambda s: DN.is_dga(s) if s and s != "nan" else False),
                      "pub": f.publication_id.astype(str), "chemotype": f.chemotype.astype(str),
                      "name": f.extractant_name.astype(str)})
    d["mech"] = f.extractant.astype(str).map(ik).map(mech).fillna("?")
    have = np.isfinite(d.c)
    say(f"bench cells {len(d)}; with Eu, Gd, Tb {int(have.sum())}; in C3 {int((have & d.c3).sum())}")
    say("cells with a chord by mechanism x C3:")
    say(d[have].groupby(["mech", "c3"]).size().unstack(fill_value=0).to_string())
    a = d[have & d.mech.isin(["ACIDIC_CATION_EXCHANGE", "CHELATING"])]
    say("\nacidic / chelating cells with a chord (any C3 status):")
    say(a[["pub", "name", "c3", "c", "slope", "step"]].round(3).to_string())
    ua = a.groupby("pub").agg(c=("c", "mean"), n=("c", "size"), name=("name", "first"), step=("step", "mean"))
    say(ua.round(3).to_string())
    # proportional alternative: chord vs Eu->Tb step at unit level, non-DGA and DGA, all bench cells with a chord
    d["unit"] = d.pub + "|" + d.chemotype
    u = d[have].groupby("unit").agg(c=("c", "mean"), step=("step", "mean"), slope=("slope", "mean"), dga=("dga", "mean"),
                                    c3=("c3", "mean"), mech=("mech", "first"))
    u["cw"] = np.clip(u.c, *CLIP)
    say("\nunit level, chord vs Eu->Tb step (log D(Tb) - log D(Eu)):")
    for nm, g in (("all", u), ("DGA", u[u.dga > 0.5]), ("nonDGA", u[u.dga <= 0.5]),
                  ("nonDGA C3", u[(u.dga <= 0.5) & (u.c3 > 0.5)])):
        r = SS.spearmanr(g.step, g.cw)
        # through-origin proportional fit chord = gamma * step
        gam = (g.step * g.cw).sum() / (g.step ** 2).sum()
        say(f"  {nm:10s} units {len(g):3d} spearman {r.statistic:+.3f} (p {r.pvalue:.3f}); gamma (chord/step) {gam:+.3f}")
    # leave-one-unit-out among nonDGA C3 units: categorical HS mean vs proportional gamma*step vs pooled mean
    g = u[(u.dga <= 0.5) & (u.c3 > 0.5) & np.isfinite(u.step) & np.isfinite(u.slope)].copy()
    e = {"pooled": [], "HS_split": [], "prop_step": [], "beta": []}
    for i in g.index:
        tr = g.drop(index=i)
        x = g.loc[i]
        e["beta"].append((x.cw - BETA) ** 2)
        e["pooled"].append((x.cw - tr.cw.mean()) ** 2)
        hs = x.slope > 0
        e["HS_split"].append((x.cw - tr[(tr.slope > 0) == hs].cw.mean()) ** 2)
        gam = (tr.step * tr.cw).sum() / (tr.step ** 2).sum()
        e["prop_step"].append((x.cw - gam * x.step) ** 2)
    say("\nLOUO among nonDGA C3 units (n=%d), mean squared error of the unit chord:" % len(g),
        {k: round(float(np.mean(v)), 5) for k, v in e.items()})
    (OUT / "confirm2skB_acidic.log").write_text("\n".join(LOG), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
