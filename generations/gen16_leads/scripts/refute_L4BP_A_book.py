"""Refuter lens A, step 5: bookkeeping / leakage / statistics audit of L4's BP headline.

No refits.  Everything here is recomputed from the lead's own raw per-extractant parquet
(`results/L4/_pe_keep/_pe_<design>.parquet`) and from the frozen bench, with the aggregation
written from PRE_REGISTRATION.md section 3 (L4) rather than from gen16/l4_acq.py.

Checks
  1  scored-unit identity: are the held-out pairs byte-identical across every (order, draw, budget)?
  2  balance: is every (order, seed, extractant) present at all six budgets?
  3  independent re-derivation of ABC = mean_k [MAE_RANDOM(k) - MAE_AOPT(k)] per extractant,
     digit for digit against the lead's contrasts_abc.csv and per_extractant_abc.parquet
  4  n_metals confound: stratify and partial the per-extractant ABC gain
  5  BH over the registered family of 20, recomputed independently
  6  Kish effective n; per-seed signs; LOCO min/max
  7  percentile-of-random-draws claim, recomputed
  8  seeds= audit over every L4 file; hash() / ordering determinism of the fold plan in a subprocess
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT / "generations" / "gen13_separation", ROOT / "generations" / "gen14_direction", ROOT / "generations" / "gen15_curve"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gen13sep.inference import paired_contrasts  # noqa: E402
from gen13sep.splits import all_folds  # noqa: E402
from gen14.dirbench import load  # noqa: E402
from gen15.valuebench import MIN_METALS  # noqa: E402

OUT = ROOT / "generations" / "gen16_leads" / "results" / "refutation" / "L4BP" / "A"
OUT.mkdir(parents=True, exist_ok=True)
LEAD = ROOT / "generations" / "gen16_leads" / "results" / "L4"
KEEP = LEAD / "_pe_keep"
BUDGETS = (6, 9, 12, 16, 20, 24)
pd.set_option("display.width", 260)
LOG: list[str] = []


def say(m: str = "") -> None:
    print(m, flush=True)
    LOG.append(m)


def bh_independent(p: np.ndarray) -> np.ndarray:
    n = len(p)
    o = np.argsort(p, kind="mergesort")
    adj = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(adj, 1.0)
    return out


def main() -> None:
    bench = load()
    fr = bench.frame

    # ---------------------------------------------------------------- 1 + 2  units and balance
    PE = pd.read_parquet(KEEP / "_pe_BP.parquet")
    say(f"1. lead's BP per-extractant frame: {len(PE)} rows, "
        f"orders {sorted(PE.order.unique())}, budgets {sorted(PE.budget.unique())}")
    key = ["split_seed", "extractant", "budget"]
    # every arm must see the same pair count / cell count for the same (seed, extractant, budget)
    g = PE.groupby(key)[["n_pairs", "n_cells", "n_strong", "y_abs_mean"]].nunique()
    bad = int((g > 1).to_numpy().sum())
    say(f"   scored-unit identity across arms: {bad} (seed, extractant, budget) cells where "
        f"n_pairs/n_cells/n_strong/y_abs_mean differ between arms  [0 = byte-identical]")
    # and across budgets
    g2 = PE.groupby(["split_seed", "extractant"])[["n_pairs", "n_cells"]].nunique()
    say(f"   scored-unit identity across budgets: {int((g2 > 1).to_numpy().sum())} differing")
    cnt = PE[PE.budget != "full"].groupby(["order", "split_seed", "extractant"])["budget"].nunique()
    say(f"   budgets per (order, seed, extractant): min {cnt.min()} max {cnt.max()} "
        f"(expect 6); n units {PE.extractant.nunique()}")
    ABCL = pd.read_parquet(LEAD / "per_extractant_abc.parquet")
    say(f"   lead ABC frame n_budgets: {sorted(ABCL.n_budgets.unique())}; "
        f"rows per design {ABCL.groupby('design').size().to_dict()}")
    nrow = ABCL[ABCL.design == 'BP'].groupby('order').size().to_dict()
    say(f"   BP ABC rows per order: {nrow}")

    # ---------------------------------------------------------------- 3  re-derive the headline
    say()
    rows = []
    for design in ("B", "BR", "BQ", "A", "BP"):
        P = pd.read_parquet(KEEP / f"_pe_{design}.parquet")
        P = P[P.budget != "full"].copy()
        P["budget"] = P["budget"].astype(int)
        # mean over draws, then over budgets, then paired_contrasts averages over seeds
        da = P.groupby(["order", "budget", "split_seed", "extractant", "chemotype"])["mae_all"].mean()
        abc = da.groupby(level=["order", "split_seed", "extractant", "chemotype"]).mean().reset_index()
        abc = abc.rename(columns={"order": "arm"})
        c = paired_contrasts(abc, {"AOPT_vs_RANDOM": ("RANDOM", "AOPT"),
                                   "AOPT_vs_MAXMIN": ("MAXMIN", "AOPT"),
                                   "UNCERT_vs_RANDOM": ("RANDOM", "UNCERT"),
                                   "UNCERT_vs_MAXMIN": ("MAXMIN", "UNCERT")}, value="mae_all")
        c.insert(0, "design", design)
        rows.append(c)
        if design == "BP":
            ABC_BP = abc
    MINE = pd.concat(rows, ignore_index=True)
    LEADC = pd.read_csv(LEAD / "contrasts_abc.csv")
    LEADC = LEADC[LEADC.family == "registered"]
    m = MINE.merge(LEADC[["design", "comparison", "point", "ci95_low", "ci95_high", "bca_low",
                          "p_two_sided", "seeds_positive", "loco_sign_stable", "passes_P1"]],
                   on=["design", "comparison"], suffixes=("_mine", "_lead"))
    m["d_point"] = m.point_mine - m.point_lead
    say("3. independent re-derivation of the registered ABC contrasts (mine vs lead):")
    say(m[["design", "comparison", "point_mine", "point_lead", "d_point", "ci95_low_mine",
           "ci95_high_mine", "p_two_sided_mine", "seeds_positive_mine",
           "passes_P1_mine", "passes_P1_lead"]].round(6).to_string(index=False))
    say(f"   max |point_mine - point_lead| = {m.d_point.abs().max():.3e}")
    m.to_csv(OUT / "rederived_registered.csv", index=False)
    hl = m[(m.design == "BP") & (m.comparison == "AOPT_vs_RANDOM")].iloc[0]
    say(f"   BP headline recomputed: {hl.point_mine:.10f} (lead {hl.point_lead:.10f}); "
        f"CI [{hl.ci95_low_mine:.4f}, {hl.ci95_high_mine:.4f}] "
        f"BCa mine [{hl['bca_low_mine']:.4f}, {hl['bca_high']:.4f}] "
        f"BCa lead low {hl['bca_low_lead']:.4f}")

    # ---------------------------------------------------------------- 4  n_metals confound
    say()
    nm = fr.groupby("extractant")["n_metals"].max().rename("n_metals_max")
    nmean = fr.groupby("extractant")["n_metals"].mean().rename("n_metals_mean")
    bp = ABC_BP.merge(nm, left_on="extractant", right_index=True, how="left") \
               .merge(nmean, left_on="extractant", right_index=True, how="left")
    cut = float(nm.median())
    say(f"4. n_metals confound (scoring side).  median max n_metals per extractant = {cut}")
    strat = []
    for lab, sub in (("all", bp),
                     (f"n_metals_max <= {cut:.0f}", bp[bp.n_metals_max <= cut]),
                     (f"n_metals_max >  {cut:.0f}", bp[bp.n_metals_max > cut]),
                     ("well determined (>=5)", bp[bp.n_metals_max >= MIN_METALS]),
                     ("not well determined (<5)", bp[bp.n_metals_max < MIN_METALS])):
        c = paired_contrasts(sub.rename(columns={"arm": "arm"}),
                             {"AOPT_vs_RANDOM": ("RANDOM", "AOPT")}, value="mae_all")
        if len(c):
            r = c.iloc[0]
            say(f"   {lab:26s} point {r.point:+.4f} [{r.ci95_low:+.4f}, {r.ci95_high:+.4f}] "
                f"p {r.p_two_sided:.4f} seeds+ {int(r.seeds_positive)} n {int(r.n_units)} "
                f"P1 {bool(r.passes_P1)}")
            strat.append(dict(stratum=lab, point=float(r.point), ci_low=float(r.ci95_low),
                              ci_high=float(r.ci95_high), p=float(r.p_two_sided),
                              seeds_positive=int(r.seeds_positive), n_units=int(r.n_units),
                              passes_P1=bool(r.passes_P1)))
    pd.DataFrame(strat).to_csv(OUT / "nmetals_strata_BP.csv", index=False)
    from scipy.stats import spearmanr
    per = bp.pivot_table(index=["extractant", "n_metals_max"], columns="arm",
                         values="mae_all").reset_index()
    delta = (per["RANDOM"] - per["AOPT"]).to_numpy(float)
    ok = np.isfinite(delta)
    rho = spearmanr(delta[ok], per.n_metals_max.to_numpy(float)[ok]).statistic
    say(f"   Spearman(per-extractant AOPT gain, its n_metals_max) = {rho:+.3f} (n={int(ok.sum())})")
    # partial the confound out: rank-regress the delta on n_metals, contrast the residual mean
    from scipy.stats import rankdata
    xr = rankdata(per.n_metals_max.to_numpy(float)[ok])
    yr = delta[ok]
    beta = np.polyfit(xr, yr, 1)
    resid_mean = float((yr - (np.polyval(beta, xr) - yr.mean())).mean())
    say(f"   mean gain {yr.mean():+.4f}; mean gain after removing the linear n_metals-rank trend "
        f"{resid_mean:+.4f}")

    # ---------------------------------------------------------------- 5  BH
    say()
    reg = LEADC.reset_index(drop=True)
    mine_bh = bh_independent(reg.p_two_sided.to_numpy(float))
    say(f"5. BH over the registered family of {len(reg)}: max |lead - refuter| "
        f"{np.abs(mine_bh - reg.p_bh_within_lead_family.to_numpy(float)).max():.2e}")
    i = reg.index[(reg.design == "BP") & (reg.comparison == "AOPT_vs_RANDOM")][0]
    say(f"   BP AOPT_vs_RANDOM raw p {reg.p_two_sided[i]:.4g}, BH(registered n=20) {mine_bh[i]:.4g}")
    allp = pd.concat([LEADC.p_two_sided,
                      pd.read_csv(LEAD / "contrasts_abc.csv").p_two_sided,
                      pd.read_csv(LEAD / "contrasts_per_budget.csv").p_two_sided,
                      pd.read_csv(LEAD / "contrasts_cellmatched.csv").p_two_sided,
                      pd.read_csv(LEAD / "contrasts_nodga.csv").p_two_sided]).to_numpy(float)
    say(f"   L4 wrote {len(allp) - len(LEADC)} contrast rows in total")

    # ---------------------------------------------------------------- 6  units / n_eff
    say()
    ext = fr[["extractant", "chemotype"]].drop_duplicates("extractant")
    nb = ext.groupby("chemotype").size().to_numpy(float)
    say(f"6. scoring units {int(nb.sum())} extractants in {len(nb)} chemotype blocks; "
        f"Kish n_eff over blocks {nb.sum() ** 2 / (nb ** 2).sum():.2f}; "
        f"largest block {int(nb.max())} extractants")
    ld = LEADC[(LEADC.design == "BP") & (LEADC.comparison == "AOPT_vs_RANDOM")].iloc[0]
    say(f"   lead BP headline: point {ld.point:.4f} CI [{ld.ci95_low:.4f}, {ld.ci95_high:.4f}] "
        f"BCa [{ld.bca_low:.4f}, {ld.bca_high:.4f}] p {ld.p_two_sided:.4g} "
        f"seeds+ {int(ld.seeds_positive)} LOCO [{ld.loco_min:.4f}, {ld.loco_max:.4f}] "
        f"stable {bool(ld.loco_sign_stable)} P1 {bool(ld.passes_P1)}")

    # ---------------------------------------------------------------- 7  percentile
    say()
    P = pd.read_parquet(KEEP / "_pe_BP.parquet")
    P = P[P.budget != "full"]
    rbar = (P[P.order == "RANDOM"].groupby(["budget", "split_seed", "extractant"])["mae_all"]
            .mean().rename("rbar"))

    def macro_abc(block: pd.DataFrame) -> float:
        j = block.set_index(["budget", "split_seed", "extractant"])["mae_all"].rename("v")
        j = pd.concat([j, rbar], axis=1, join="inner")
        per_ = (j["rbar"] - j["v"]).groupby(level=["split_seed", "extractant"]).mean()
        return float(per_.groupby(level="extractant").mean().mean())

    draws = np.array([macro_abc(P[(P.order == "RANDOM") & (P.draw == i)]) for i in range(20)])
    v = macro_abc(P[P.order == "AOPT"])
    say(f"7. percentile of AOPT's BP ABC among the 20 random draws: "
        f"{100.0 * float((draws < v).mean()):.1f}; AOPT {v:.4f}; "
        f"draws [{draws.min():+.4f}, {draws.max():+.4f}] sd {draws.std(ddof=1):.4f}")
    say(f"   the 20 draw ABCs are centred on {draws.mean():+.2e} by construction "
        f"(each is measured against the mean of all 20)")

    # ---------------------------------------------------------------- 8  seeds= and determinism
    say()
    hits = []
    for f in sorted((ROOT / "generations" / "gen16_leads").rglob("*.py")):
        try:
            txt = f.read_text(encoding="utf-8")
        except Exception:
            continue
        if "seeds=" in txt:
            for ln, line in enumerate(txt.splitlines(), 1):
                if "seeds=" in line:
                    hits.append(f"{f.relative_to(ROOT)}:{ln}: {line.strip()[:110]}")
    say(f"8. 'seeds=' occurrences under gen16_leads/: {len(hits)}")
    for h in hits:
        say(f"   {h}")

    code = (
        "import sys;from pathlib import Path\n"
        f"R=Path(r'{ROOT}')\n"
        "for p in (R/'generations'/'gen13_separation',R/'generations'/'gen14_direction',R/'generations'/'gen15_curve'):sys.path.insert(0,str(p))\n"
        "sys.path.insert(0,str(R/'generations'/'gen16_leads'))\n"
        "import hashlib,numpy as np\n"
        "from gen13sep.splits import all_folds\n"
        "from gen14.dirbench import load\n"
        "from gen16 import l4_acq as L\n"
        "b=load();P=L.prepare(b);h=hashlib.sha256()\n"
        "for f in all_folds(b.frame,design='BP'):\n"
        "    h.update(np.asarray(f.train_index).tobytes());h.update(np.asarray(f.test_index).tobytes())\n"
        "o=L.fold_orders(P,all_folds(b.frame,design='BP')[0],n_draws=3)\n"
        "h2=hashlib.sha256(('|'.join(','.join(v) for k,v in sorted(o.items()))).encode()).hexdigest()\n"
        "print(h.hexdigest()[:16],h2[:16])\n"
    )
    outs = []
    for _ in range(2):
        r = subprocess.run([str(ROOT / ".venv" / "Scripts" / "python.exe"), "-c", code],
                           capture_output=True, text=True, cwd=str(ROOT),
                           env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8",
                                "OMP_NUM_THREADS": "2", "PYTHONHASHSEED": "random"})
        outs.append(r.stdout.strip())
    say(f"   fold-plan + AOPT/MAXMIN/UNCERT order hash, two fresh subprocesses "
        f"(PYTHONHASHSEED random): {outs[0]!r} / {outs[1]!r} -> "
        f"identical {outs[0] == outs[1] and outs[0] != ''}")

    (OUT / "book_audit.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
