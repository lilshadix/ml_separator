import sys, time, json
sys.path.insert(0, "D:/ml_separator_gh/generations/gen13_separation/analysis/stage3/verify_direction")
from v_runner import *
import numpy as np, pandas as pd
from gen13sep.splits import all_folds

AMP = COEF[:, 0]; NM = FRAME.n_metals.to_numpy(); Y = (AMP < 0).astype(int)
RICH = NM >= 5
res = []

def record(tag, tab, note=""):
    u = units(tab); ub = const_units(tab, 1)
    g = boot(u, ub, reps=4000); a = boot(u, None, reps=4000)
    r = {"tag": tag, "note": note, "n_ext": len(u), "macro": float(u.hit.mean()),
         "acc_lo": a["ci_low"], "acc_hi": a["ci_high"], "gain_vs_always_heavy": g["value"],
         "gain_lo": g["ci_low"], "gain_hi": g["ci_high"], "p": g["p_two_sided"]}
    res.append(r); print(json.dumps({k:(round(v,4) if isinstance(v,float) else v) for k,v in r.items()}), flush=True)
    return u, tab

t0 = time.time()
# --- constant baselines, computed directly on the unit table -------------------
tab_ref = run(TOPO_COLS, Y, RICH, design="BP", est="et")
u_topo = units(tab_ref)
for v, nm in ((1, "always_heavy"), (0, "always_light")):
    ub = const_units(tab_ref, v)
    print(nm, "macro =", round(float(ub.hit.mean()), 4), flush=True)

# --- fold-wise learned constants (what a constant rule would actually learn) ----
rows_u, rows_w = [], []
for f in all_folds(FRAME, design="BP"):
    tr = f.train_index[RICH[f.train_index]]; te = f.test_index[RICH[f.test_index]]
    if len(tr) < 40 or len(te) < 1: continue
    from gen13sep.amplitude_bench import cell_weights
    w = cell_weights(GROUPS[tr], NOBS[tr])
    p_un = float(Y[tr].mean()); p_w = float(np.average(Y[tr], weights=w))
    for ci in te:
        base = {"split_seed": f.seed, "fold": f.fold, "cell_id": FRAME.cell_id.iat[ci],
                "extractant": FRAME.extractant.iat[ci], "chemotype": FRAME.chemotype.iat[ci], "y": int(Y[ci])}
        rows_u.append(dict(base, p=p_un)); rows_w.append(dict(base, p=p_w))
record("B_train_majority_unweighted", pd.DataFrame(rows_u), "constant, learned per fold")
record("B_train_majority_chemotype_weighted", pd.DataFrame(rows_w), "constant, learned per fold")

# --- stratum majority: training majority within the test cell's chemotype-size tertile ---
csize = FRAME.groupby("chemotype")["cell_id"].transform("size").to_numpy()
strat = pd.qcut(pd.Series(csize), 3, labels=False, duplicates="drop").to_numpy()
rows_s = []
for f in all_folds(FRAME, design="BP"):
    tr = f.train_index[RICH[f.train_index]]; te = f.test_index[RICH[f.test_index]]
    if len(tr) < 40 or len(te) < 1: continue
    for ci in te:
        m = strat[tr] == strat[ci]
        p = float(Y[tr][m].mean()) if m.any() else float(Y[tr].mean())
        rows_s.append({"split_seed": f.seed, "fold": f.fold, "cell_id": FRAME.cell_id.iat[ci],
                       "extractant": FRAME.extractant.iat[ci], "chemotype": FRAME.chemotype.iat[ci],
                       "y": int(Y[ci]), "p": p})
record("B_stratum_majority_chemotype_size", pd.DataFrame(rows_s), "training majority in the cell's chemotype-size tertile")
print("t", round(time.time()-t0), flush=True)

# --- feature-block baselines, identical ExtraTrees estimator -------------------
blocks = {
    "B_DONORS_13": [c for c in FRAMES["DONORS"].columns],
    "B_PHYSCHEM_10": [c for c in FRAMES["PHYSCHEM"].columns],
    "B_MolWt_1": ["MolWt"],
    "B_dentate_1": ["chem__dentate"],
    "B_COND_64": [c for c in FRAMES["COND"].columns],
    "B_COORD_nontopo_75": [c for c in FRAMES["COORD"].columns if c not in TOPO_COLS],
    "B_n_metals_proxy": ["chem__n_fill"],
}
for tag, cols in blocks.items():
    record(tag, run(cols, Y, RICH, design="BP", est="et"), f"{len(cols)} cols, ET BP")
print("t", round(time.time()-t0), flush=True)

# --- the model itself under three estimators ----------------------------------
record("M_topo39_ET", tab_ref, "reference")
record("M_topo39_logit", run(TOPO_COLS, Y, RICH, design="BP", est="lg"), "L2 logistic")
u_lean, tab_lean = record("M_lean209_ET", run(LEAN_COLS, Y, RICH, design="BP", est="et"), "the 'full compact set'")
u_chem, _ = record("M_chem137_ET", run([c for b in ("PHYSCHEM","DONORS","COORD") for c in FRAMES[b].columns],
                                       Y, RICH, design="BP", est="et"), "PHYSCHEM+DONORS+COORD = 137 cols")
print("topo - lean209:", boot(u_topo, u_lean, reps=10000), flush=True)
print("topo - chem137:", boot(u_topo, u_chem, reps=10000), flush=True)

pd.DataFrame(res).to_csv(OUT / "v_baselines.csv", index=False)
tab_ref.to_parquet(OUT / "v_topo_bp_predictions.parquet", index=False)
print("done", round(time.time()-t0))
