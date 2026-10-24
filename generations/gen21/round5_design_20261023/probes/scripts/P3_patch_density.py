"""One-off patch helper for P3_k4_summarise.py (replaces the density section).  Probe P3 only."""
from pathlib import Path

p = Path(__file__).resolve().parent / "P3_k4_summarise.py"
s = p.read_bytes().decode("utf-8")
a = s.index("# ---- density-indexed view ----")
b = s.index('print("\\n=== strata: n_metals class')
new = r'''# ---- density-indexed view ----
print("\n=== density-indexed view (B1): pooled R2 of rows grouped by their cell's reads per measured metal (seed mean) ===")
DBINS = [0, 0.29, 0.32, 0.37, 0.45, 0.55, 0.70, 1.01]
DLAB = ["0.286 (4/14)", "0.29-0.32 (4/13)", "0.32-0.37 (5/14,4/12,4/11)", "0.37-0.45 (5/13,4/10,4/9,5/12)",
        "0.45-0.55 (5/11,4/8,5/10)", "0.55-0.70 (5/9,4/7,5/8,4/6)", ">0.70 (5/7,4/5,5/6)"]
DN = []
for d in ("BPG", "BP"):
    for excl in (False, True):
        srcs = {"4 reads (K3 rows)": F["K3"]}
        for p in MAIN:
            srcs[f"5 reads ({p} k4 rows)"] = F[p][F[p].rung == "k4"]
        srcs["pooled 4+5 (K3 + DOPTQ k4 rows)"] = pd.concat([F["K3"], F["DOPTQ"][F["DOPTQ"].rung == "k4"]])
        for name, fr in srcs.items():
            fr = fr[fr.design == d]
            if excl:
                fr = fr[fr.publication != C5]
            fr = fr.assign(dbin=pd.cut(fr.density, DBINS, labels=DLAB, right=False))
            for b_, g in fr.groupby("dbin", observed=True):
                DN.append({"design": d, "noC5": excl, "source": name, "density_bin": b_, "R2": r2_seeds(g, "B1").mean(),
                           "R2_B0": r2_seeds(g, "B0").mean(), "rows_per_seed": len(g) / 5,
                           "clusters": g.cluster.nunique(), "mean_density": g.density.mean()})
DN = pd.DataFrame(DN)
DN.to_csv(OUT / "density.csv", index=False)
for d in ("BPG", "BP"):
    for excl in (False, True):
        t = DN[(DN.design == d) & (DN.noC5 == excl)]
        print(f"-- {d}{' without C5-BTBP' if excl else ''}: R2 (rows/seed, clusters) --")
        t = t.assign(cell=[f"{r:.3f} ({n:.0f}, {c})" for r, n, c in zip(t.R2, t.rows_per_seed, t.clusters)])
        print(t.pivot(index="density_bin", columns="source", values="cell").reindex(DLAB).fillna("").to_string())
print("\n=== population view: mean reads per measured metal over C3 rows vs pooled R2 (B1) ===")
PV = []
for d in ("BPG", "BP"):
    for excl in (False, True):
        for coh in ("C3", "C4"):
            for p in ["K3"] + MAIN + ["HULL_BISECT", "ORACLE_B1"]:
                fr = F[p][F[p].design == d]
                if excl:
                    fr = fr[fr.publication != C5]
                if coh == "C4":
                    fr = fr[fr.k_max >= 4]
                PV.append({"design": d, "noC5": excl, "cohort": coh, "policy": p, "mean_density_rows": fr.density.mean(),
                           "mean_reads_per_cell": fr.groupby(["split_seed", "cell_id"]).n_read.first().mean(),
                           "hull_out_share": fr.hull_out.mean(), "R2_B1": r2_seeds(fr, "B1").mean()})
PV = pd.DataFrame(PV)
PV.to_csv(OUT / "population_density.csv", index=False)
print(PV.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\n=== saturation (reads = k_max + 1, every other metal read): B1 R2 by k_max stratum ===")
SA = []
for d in ("BPG", "BP"):
    for excl in (False, True):
        parts = {"k_max 3 @ k=3": F["K3"][F["K3"].k_max == 3]}
        for p in MAIN:
            parts[f"k_max 4 @ k=4 {p}"] = F[p][(F[p].k_max == 4)]
            parts[f"k_max 5 @ k=4 {p} (not saturated)"] = F[p][(F[p].k_max == 5)]
            parts[f"k_max 6 @ k=4 {p} (not saturated)"] = F[p][(F[p].k_max == 6)]
        for name, g in parts.items():
            g = g[g.design == d]
            if excl:
                g = g[g.publication != C5]
            if g.empty:
                continue
            SA.append({"design": d, "noC5": excl, "stratum": name, "R2_B1": r2_seeds(g, "B1").mean(), "R2_B0": r2_seeds(g, "B0").mean(),
                       "rows_per_seed": len(g) / 5, "cells_per_seed": g.groupby("split_seed").cell_id.nunique().mean(),
                       "clusters": g.cluster.nunique(), "largest_cluster_share": g.cluster.value_counts(normalize=True).iloc[0],
                       "mean_density": g.density.mean()})
SA = pd.DataFrame(SA)
SA.to_csv(OUT / "saturation.csv", index=False)
print(SA.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

'''
s = s[:a] + new + s[b:]
p.write_bytes(s.encode("utf-8"))
print("patched")
