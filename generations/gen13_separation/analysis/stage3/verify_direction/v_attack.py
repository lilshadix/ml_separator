import sys, time, json
sys.path.insert(0, "D:/ml_separator_gh/gen13_separation/analysis/stage3/verify_direction")
from v_runner import *
import numpy as np, pandas as pd
from gen13sep.splits import all_folds

AMP = COEF[:, 0]
NM = FRAME.n_metals.to_numpy()
Y_RIDGE = (AMP < 0).astype(int)

# OLS label
centred = Y14 - np.nanmean(Y14, axis=1, keepdims=True)
OLS = np.full(len(Y14), np.nan)
for i in range(len(Y14)):
    m = ~np.isnan(centred[i])
    if m.sum() >= 2:
        OLS[i] = np.linalg.lstsq(BASIS[:, m].T, centred[i][m], rcond=None)[0][0]
Y_OLS = (OLS < 0).astype(int)

TOPOKEY = np.array(["|".join(map(repr, r)) for r in ALLX.loc[:, TOPO_COLS].to_numpy(float)])

res = []
def record(tag, tab, note=""):
    if tab.empty or tab.extractant.nunique() < 5:
        res.append({"tag": tag, "note": note, "macro": np.nan}); return
    u = units(tab); ub = const_units(tab, 1)
    g = boot(u, ub, reps=4000)
    a = boot(u, None, reps=4000)
    r = {"tag": tag, "note": note, "n_cells": tab.cell_id.nunique(), "n_ext": len(u),
         "n_chem": u.chemotype.nunique(), "macro": float(u.hit.mean()),
         "baseline_heavy": float(ub.hit.mean()), "acc_lo": a["ci_low"], "acc_hi": a["ci_high"],
         "gain": g["value"], "gain_lo": g["ci_low"], "gain_hi": g["ci_high"], "p": g["p_two_sided"]}
    res.append(r); print(json.dumps({k: (round(v,4) if isinstance(v,float) else v) for k,v in r.items()}), flush=True)
    return u

t0 = time.time()
# ---- 1. n_metals threshold sweep -------------------------------------------------
for thr in (4, 5, 6, 8, 14):
    keep = NM >= thr
    record(f"thr>={thr}", run(TOPO_COLS, Y_RIDGE, keep, design="BP", est="et"), "topo39 ET BP")
print("t", round(time.time()-t0), flush=True)

# ---- 2. OLS label instead of ridge ----------------------------------------------
record("label=OLS", run(TOPO_COLS, Y_OLS, NM >= 5, design="BP", est="et"), "topo39 ET BP")

# ---- 3. drop the dominant diglycolamide chemotype -------------------------------
keep = (NM >= 5) & (FRAME.chemotype.to_numpy() != "sc009")
record("drop_sc009", run(TOPO_COLS, Y_RIDGE, keep, design="BP", est="et"), "topo39 ET BP")

# ---- 4. margin band: drop cells whose |amp| is inside a band, from BOTH sides ----
for b in (0.05, 0.10, 0.20, 0.30):
    keep = (NM >= 5) & (np.abs(AMP) >= b)
    record(f"|amp|>={b}", run(TOPO_COLS, Y_RIDGE, keep, design="BP", est="et"), "topo39 ET BP, band dropped everywhere")
print("t", round(time.time()-t0), flush=True)
pd.DataFrame(res).to_csv(OUT / "v_attack_target_units.csv", index=False)
print("done", round(time.time()-t0))
