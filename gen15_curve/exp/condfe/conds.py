"""Condition variables for the within-publication (fixed-effects) study.

The 64 ``cond__`` columns are one-hot diluent / acid identity plus four continuous numbers.  As a
block they identify a cell's publication with 94 % 1-NN accuracy, which is why every conditions-only
model in this programme has collapsed under design BP.  This module builds a small, *physical*
recoding of them -- the quantities a chemist would say act on the lanthanide-axis slope -- so that
the within-publication variation in them can be looked at separately from the laboratory that
produced them.

Nothing here touches the locked packages; it only reads ``bench.frames['COND']`` and ``bench.frame``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------------
# diluent chemistry:  which of the 44 one-hot diluent columns is what
# ---------------------------------------------------------------------------------------------
_ALIPHATIC = ("n_dodecane", "kerosene", "avkerosene", "sulfonated_kerosene", "isoparl",
              "isopar_l_0_7_exxal_13_0_3", "isopar_l_with_30_vol_exxal_13", "tph", "tphkerosene",
              "total_petroleum_hydrocarbons", "hydrogenated_tetrapropene", "hydrogenated_tetrapropylene",
              "hydrogenated_tetrapropene_with_5_vol_1_octanol",
              "hydrogenated_tetrapropene_with_50_vol_1_octanol", "n_hexane",
              "n_dodecane_with_5_vol_iso_decanol", "n_octane_with_5_1_octanol",
              "kerosene_0_7_1_octanol_0_3", "kerosene_0_95_1_octanol_0_05",
              "kerosene_with_30_vol_1_octanol", "kerosene_with_40_vol_1_octanol",
              "tph_0_95_1_octanol_0_05", "1_octanol_0_5_kerosene_0_5")
_AROMATIC = ("toluene", "benzene", "tert_butylbenzene", "dipb", "1_4_diisopropylbenzene", "tbub",
             "nitrobenzene", "meta_nitrobenzotrifluoride")
_CHLORINATED = ("chcl3", "chloroform", "ch3cl", "dichloromethane", "1_2_dichloroethane")
_POLAR = ("1_octanol", "cyclohexanone", "phenyl_trifluoromethyl_sulfone", "c4mim_tf2n")
#: diluents whose name says an alcohol phase modifier is part of the organic phase
_MODIFIED = ("1_octanol", "octanol", "decanol", "exxal")
#: aqueous complexants -- ligands in the *aqueous* phase that compete for the metal and are the
#: classic way a laboratory reverses or sharpens a lanthanide trend
_COMPLEXANT_ACIDS = ("citric_acid", "lactic_acid", "malonic_acid", "tartaric_acid",
                     "hno3_oxalic_acid")

LOG_FLOOR = 1e-3


def _onehot(cond: pd.DataFrame, prefix: str) -> tuple[list[str], np.ndarray]:
    cols = [c for c in cond.columns if c.startswith(prefix)]
    return cols, cond[cols].to_numpy(dtype=float)


def _name(col: str, prefix: str) -> str:
    return col[len(prefix):]


def build(bench) -> pd.DataFrame:
    """One row per cell: the physical recoding plus the publication and chemotype labels."""
    cond = bench.frames["COND"]
    f = bench.frame
    n = len(f)

    acid_cols, acid_oh = _onehot(cond, "cond__acid__")
    dil_cols, dil_oh = _onehot(cond, "cond__diluent__")
    add_cols, add_oh = _onehot(cond, "cond__additive__")

    acid_name = np.array(["none"] * n, dtype=object)
    for j, c in enumerate(acid_cols):
        acid_name[acid_oh[:, j] > 0.5] = _name(c, "cond__acid__")
    dil_name = np.array(["none"] * n, dtype=object)
    for j, c in enumerate(dil_cols):
        dil_name[dil_oh[:, j] > 0.5] = _name(c, "cond__diluent__")

    def dil_class(nm: str) -> str:
        if nm in _ALIPHATIC:
            return "aliphatic"
        if nm in _AROMATIC:
            return "aromatic"
        if nm in _CHLORINATED:
            return "chlorinated"
        if nm in _POLAR:
            return "polar"
        return "other"

    ca = cond["cond__acid_concentration_M"].to_numpy(dtype=float)
    ce = cond["cond__extractant_concentration_M"].to_numpy(dtype=float)
    cm = cond["cond__metal_concentration_mM"].to_numpy(dtype=float)
    tC = cond["cond__temperature_C"].to_numpy(dtype=float)
    tt = cond["cond__contact_time_min"].to_numpy(dtype=float)

    modifier = np.array([any(m in nm for m in _MODIFIED) for nm in dil_name], dtype=float)
    if add_cols:
        modifier = np.maximum(modifier, (add_oh.sum(axis=1) > 0).astype(float))

    complexant = np.array([nm in _COMPLEXANT_ACIDS for nm in acid_name], dtype=float)

    Y = bench.Y
    with np.errstate(invalid="ignore"):
        maxlogd = np.nanmax(Y, axis=1)
        meanlogd = np.nanmean(Y, axis=1)

    out = pd.DataFrame({
        "publication_id": f["publication_id"].astype(str).to_numpy(),
        "chemotype": f["chemotype"].astype(str).to_numpy(),
        "extractant": f["extractant"].astype(str).to_numpy(),
        "n_metals": f["n_metals"].to_numpy(dtype=float),
        "acid_name": acid_name,
        "diluent_name": dil_name,
        "diluent_class": np.array([dil_class(nm) for nm in dil_name], dtype=object),
        # --- continuous, on the scale a chemist would use ---
        "log_acid_M": np.log10(np.maximum(ca, LOG_FLOOR)),
        "log_extr_M": np.log10(np.maximum(ce, LOG_FLOOR)),
        "log_metal_mM": np.log10(np.maximum(cm, LOG_FLOOR)),
        "temp_C": tC,
        "contact_min": tt,
        "log_contact_min": np.log10(np.maximum(tt, 1.0)),
        # --- categorical, as indicators ---
        "is_hno3": (acid_name == "hno3").astype(float),
        "is_hcl": (acid_name == "hcl").astype(float),
        "is_h2so4": (acid_name == "h2so4").astype(float),
        "complexant": complexant,
        "modifier": modifier,
        "dil_aliphatic": np.array([dil_class(nm) == "aliphatic" for nm in dil_name], dtype=float),
        "dil_aromatic": np.array([dil_class(nm) == "aromatic" for nm in dil_name], dtype=float),
        "dil_chlorinated": np.array([dil_class(nm) == "chlorinated" for nm in dil_name], dtype=float),
        "dil_polar": np.array([dil_class(nm) == "polar" for nm in dil_name], dtype=float),
        # --- level of extraction: NOT deployable, diagnostic only ---
        "max_logD": maxlogd,
        "mean_logD": meanlogd,
    })
    return out


#: the continuous / indicator variables that can enter a linear within-publication model
NUMERIC: tuple[str, ...] = (
    "log_acid_M", "log_extr_M", "log_metal_mM", "temp_C", "log_contact_min",
    "is_hno3", "is_hcl", "is_h2so4", "complexant", "modifier",
    "dil_aliphatic", "dil_aromatic", "dil_chlorinated", "dil_polar",
)

#: the four honest physical columns of the task's hand-picked subset
PHYS4: tuple[str, ...] = ("complexant", "is_hno3", "log_acid_M", "temp_C")
