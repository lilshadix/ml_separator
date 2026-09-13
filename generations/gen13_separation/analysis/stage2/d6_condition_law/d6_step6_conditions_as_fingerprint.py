"""D6 step 6: why does a conditions-only model score as well as a structure-aware one?

Hypothesis: the 64 cond__ columns act mostly as a fingerprint of WHICH cell/publication
/extractant this is, not as a physical law about the separation curve.  Test it by
asking how well the condition vector alone identifies the extractant, the chemotype and
the publication of a held-out cell (1-nearest neighbour in standardised condition space).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from d6_common import load_cohort, cond_columns, cell_amplitudes, OUT, MIN_RZ_SPAN

df = load_cohort()
cols = cond_columns(df)
Xc = df[cols].astype(float).copy()
for c in cols:
    Xc[c] = Xc[c].fillna(Xc[c].median())
for c in ["cond__acid_concentration_M", "cond__extractant_concentration_M",
          "cond__metal_concentration_mM"]:
    Xc[c] = np.log10(Xc[c].clip(lower=1e-6))
X = Xc.to_numpy(float)
X = (X - X.mean(0)) / np.where(X.std(0) > 1e-9, X.std(0), 1.0)

# 1-NN leave-one-out in condition space
d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
np.fill_diagonal(d2, np.inf)
nn = d2.argmin(1)
rows = []
for lab in ["extractant", "chemotype", "publication_id", "condition_key"]:
    v = df[lab].to_numpy()
    acc = float((v[nn] == v).mean())
    # chance rate = sum of squared class frequencies
    f = df[lab].value_counts(normalize=True).to_numpy()
    rows.append(dict(target=lab, nn1_loo_accuracy=acc, chance=float((f ** 2).sum()),
                     n_classes=int(df[lab].nunique())))
R = pd.DataFrame(rows)
R.to_csv(OUT + "/d6_step6_conditions_as_fingerprint.csv", index=False)
print("=== 1-NN leave-one-out identification from the 64 condition columns alone "
      f"({len(df)} cells) ===")
print(R.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# how many cells have an exactly duplicated condition vector belonging to another
# extractant vs the same extractant
exact = df.groupby(cols, dropna=False).ngroup()
tmp = pd.DataFrame(dict(g=exact, ext=df.extractant, pub=df.publication_id))
dup = tmp.groupby("g").filter(lambda g: len(g) > 1)
same_ext = dup.groupby("g").ext.nunique().eq(1).mean() if len(dup) else np.nan
print(f"\ncells sharing an exactly identical 64-column condition vector with another "
      f"cell: {len(dup)} of {len(df)}")
print(f"of those duplicate groups, the fraction that contain only ONE extractant: "
      f"{same_ext:.3f} ({dup.g.nunique()} groups)")

# and the same for the amplitude analysis set: does the condition vector predict the
# amplitude better ACROSS extractants than WITHIN one?
amp = df.merge(cell_amplitudes(df), on="cell_id")
amp = amp[amp.rz_span >= MIN_RZ_SPAN]
keep = df.cell_id.isin(set(amp.cell_id)).to_numpy()
Xa = X[keep]
ya = df.loc[keep].merge(amp[["cell_id", "amp"]], on="cell_id").amp.to_numpy()
ea = df.loc[keep, "extractant"].to_numpy()
d2a = ((Xa[:, None, :] - Xa[None, :, :]) ** 2).sum(-1)
np.fill_diagonal(d2a, np.inf)
nna = d2a.argmin(1)
err_all = np.abs(ya[nna] - ya)
# restrict the neighbour search to the SAME extractant
d2s = d2a.copy()
d2s[ea[:, None] != ea[None, :]] = np.inf
has = np.isfinite(d2s).any(1)
nns = np.where(has, d2s.argmin(1), 0)
err_same = np.abs(ya[nns][has] - ya[has])
print(f"\n=== 1-NN amplitude prediction from conditions ({len(ya)} amplitude cells) ===")
print(f"  neighbour anywhere in the cohort : MAE {err_all.mean():.3f} "
      f"(and {float((ea[nna] == ea).mean()):.3f} of those neighbours are the SAME "
      f"extractant)")
print(f"  neighbour forced to the same extractant : MAE {err_same.mean():.3f} "
      f"on {int(has.sum())} cells")
print(f"  predicting the cohort mean amplitude    : MAE "
      f"{np.abs(ya - ya.mean()).mean():.3f}")
pd.DataFrame([dict(nn_anywhere_mae=float(err_all.mean()),
                   frac_neighbour_same_extractant=float((ea[nna] == ea).mean()),
                   nn_same_extractant_mae=float(err_same.mean()),
                   cohort_mean_mae=float(np.abs(ya - ya.mean()).mean()),
                   n_cells=int(len(ya)))]).to_csv(
    OUT + "/d6_step6_nn_amplitude.csv", index=False)
print("\nwrote d6_step6_conditions_as_fingerprint.csv, d6_step6_nn_amplitude.csv")
