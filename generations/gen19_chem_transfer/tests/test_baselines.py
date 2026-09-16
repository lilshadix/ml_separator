"""B0-B4 (+ B3x, B3i, B3l, B4x, B4l), B7 and the split-conformal wrapper (``gen19ct/models``).

Synthetic frames unless stated.  The gen18 cross-check reads ``generations/gen18_process/systems/corpus_records.csv``
(read-only) and, for the raw-row comparison, the archive MODEL rows; no test fits or scores anything on an outer fold
of the registered designs, and no test writes a file.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import pytest

from gen19ct import paths
from gen19ct.chemistry import support_graph as SG
from gen19ct.data import leakage as L
from gen19ct.models import baselines as B
from gen19ct.models import interface as I
from gen19ct.models import mass_action as MA

TODGA = "CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC"
TDDGA = "CCCCCCCCCCN(CCCCCCCCCC)C(=O)COCC(=O)N(CCCCCCCCCC)CCCCCCCCCC"
TDDDGA = "CCCCCCCCCCCCN(CCCCCCCCCCCC)C(=O)COCC(=O)N(CCCCCCCCCCCC)CCCCCCCCCCCC"
THDGA = "CCCCCCN(CCCCCC)C(=O)COCC(=O)N(CCCCCC)CCCCCC"
TBP = "CCCCOP(=O)(OCCCC)OCCCC"
DHOA = "CCCCCCCN(CCCCCC)C(=O)CCCCCCC"
SMILES_OF = {"S_TODGA": TODGA, "S_TDDGA": TDDGA, "S_TDDDGA": TDDDGA, "S_THDGA": THDGA, "S_TBP": TBP}


class Rows:
    """Builder of synthetic arm frames (the ``interface.prepare_frame`` layout)."""

    def __init__(self) -> None:
        self.recs: list[dict] = []

    def add(self, state: str | None, system: str, *, y: float = 0.0, acid: float = 0.0, ext: float = -1.0,
            anion: str = "nitrate", pub: str = "g1", elem: str | None = None, family: str = "diglycolamide",
            mech: str = "NEUTRAL_SOLVATING", smiles: str | None = None, cmid: str | None = None,
            label: str | None = None, n: int = 1) -> list[str]:
        out = []
        for _ in range(n):
            k = len(self.recs)
            lab = label if (label is not None and n == 1) else f"r{k:04d}"
            sym = elem if state is None else SG.metal_properties(state)["symbol"]
            self.recs.append({
                "_label": lab, SG.METAL_COL: state, SG.ELEMENT_COL: sym, SG.SYSTEM_COL: system,
                SG.ACID_ANION_COL: anion, SG.LOG_ACID_COL: acid, SG.LOG_EXT_COL: ext, SG.TEMP_COL: 25.0,
                SG.DILUENT_COL: "aliphatic", I.ID_COL: cmid if (cmid is not None and n == 1) else f"ID:{k:05d}",
                I.TARGET_COL: y, I.PUB_GROUP_COL: pub, SG.PUB_COL: f"pub_{pub}", SG.FAMILY_COL: family,
                SG.MECH_COL: mech, SG.SMILES_COL: smiles or SMILES_OF.get(system, system.split("|")[0]),
            })
            out.append(lab)
        return out

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.recs).set_index("_label")


def ctx(**kw) -> I.FitContext:
    return I.FitContext(seed=kw.pop("seed", 7), **kw)


def fit_predict(name: str, train: pd.DataFrame, query: pd.DataFrame, context: I.FitContext | None = None):
    arm = MA.B7MassAction() if name == "B7" else B.BaselineArm(name)
    return arm.fit(train, context or ctx()).predict(query)


def one(pred: pd.DataFrame) -> pd.Series:
    assert len(pred) == 1
    return pred.iloc[0]


# --------------------------------------------------------------------------------------------- #
# B0 / B1 / B2
# --------------------------------------------------------------------------------------------- #

def test_b0_b1_b2_fallback_paths() -> None:
    r = Rows()
    r.add("Nd(III)", "S1", y=1.0, family="fam_a")
    r.add("Nd(III)", "S1", y=3.0, family="fam_a")
    r.add("Eu(III)", "S2", y=10.0, family="fam_b", mech="SOFT_N_DONOR")
    r.add(None, "S2", y=5.0, elem="Sm", family="fam_b", mech="SOFT_N_DONOR")
    r.add("Am(III)", "S3", y=7.0, family="fam_c", mech="MIXED_NEUTRAL")
    r.add("Eu(III)", "S4", y=100.0, family="fam_d", mech="ACIDIC_CATION_EXCHANGE")
    train = r.frame()
    systems = pd.DataFrame({"extractant_system_key": ["S9", "S8", "S7"], "system_family": ["fam_b", "fam_z", "fam_y"],
                            "mechanism": ["NEUTRAL_SOLVATING", "NEUTRAL_SOLVATING", "UNKNOWN"],
                            "primary_extractant_smiles": [TBP, TBP, TBP]}).set_index("extractant_system_key", drop=False)
    q = Rows()
    q.add("Nd(III)", "S1", label="qA")
    q.add("Sm(III)", "S9", label="qB")
    q.add("Gd(III)", "S8", label="qC")
    q.add("Lu(III)", "S7", label="qD")
    query = q.frame()
    c = ctx(systems=systems)
    b0 = fit_predict("B0", train, query, c)
    assert np.allclose(b0["mean_logD"], np.mean([1, 3, 10, 5, 7, 100]))
    assert set(b0["fallback_level"]) == {"B0"}
    b1 = fit_predict("B1", train, query, c).set_index("row_id")
    assert b1.loc["qA", "mean_logD"] == 2.0 and b1.loc["qA", "fallback_level"] == "B1:state"
    assert b1.loc["qB", "mean_logD"] == 5.0 and b1.loc["qB", "fallback_level"] == "B1:element_unknown_state"
    assert b1.loc["qD", "fallback_level"] == "B0" and b1.loc["qD", "mean_logD"] == pytest.approx(21.0)
    b2 = fit_predict("B2", train, query, c).set_index("row_id")
    assert b2.loc["qA", "mean_logD"] == 2.0 and b2.loc["qA", "fallback_level"] == "B2:system"
    assert b2.loc["qB", "mean_logD"] == 7.5 and b2.loc["qB", "fallback_level"] == "B2:family"   # S2 rows incl. X(?)
    assert b2.loc["qC", "mean_logD"] == pytest.approx(5.2) and b2.loc["qC", "fallback_level"] == "B2:expert"  # E2 rows
    assert b2.loc["qD", "mean_logD"] == pytest.approx(21.0) and b2.loc["qD", "fallback_level"] == "B0"  # none_unknown
    counts = I.fallback_counts(b2)
    assert int(counts["n_rows"].sum()) == 4


def test_brief_expert_mapping_matches_the_feasibility_table() -> None:
    f = paths.DATA_AUDIT_DIR / "feasibility_mechanisms.csv"
    if not f.exists():
        pytest.skip("feasibility_mechanisms.csv not built")
    t = pd.read_csv(f)
    lab = t[t["level"] == "mechanism_label"]
    assert dict(zip(lab["label"], lab["brief_expert"])) == {k: I.BRIEF_EXPERT[k] for k in lab["label"]}


# --------------------------------------------------------------------------------------------- #
# B3
# --------------------------------------------------------------------------------------------- #

def test_b3_unstandardised_distance_ties_anion_and_missing_coordinates() -> None:
    r = Rows()
    r.add("Nd(III)", "S1", y=1.0, acid=0.5, ext=0.0, label="p")      # distance 0.5
    r.add("Nd(III)", "S1", y=2.0, acid=0.0, ext=0.6, label="far")    # distance 0.6
    for a in (-8.0, 8.0):                                               # a wide extractant spread: standardising flips it
        r.add("Nd(III)", "S1", y=9.0, acid=0.0, ext=a)
    train = r.frame()
    q = Rows()
    q.add("Nd(III)", "S1", acid=0.0, ext=0.0, label="q")
    p = one(fit_predict("B3", train, q.frame()))
    assert p["fallback_level"] == "B3" and p["nn_row_id"] == "p" and p["nn_distance"] == 0.5 and p["mean_logD"] == 1.0
    sd = train[[SG.LOG_ACID_COL, SG.LOG_EXT_COL]].std(ddof=0).to_numpy()
    z = train[[SG.LOG_ACID_COL, SG.LOG_EXT_COL]].to_numpy() / sd
    assert train.index[int(np.argmin(np.sqrt((z ** 2).sum(axis=1))))] == "far"   # the standardised 1-NN would differ

    # equal distance: smaller |delta log10 acid| wins; equal coordinates: smaller canonical_measurement_id (string)
    r = Rows()
    r.add("Nd(III)", "S1", y=1.0, acid=0.5, ext=0.0, label="dA_big")
    r.add("Nd(III)", "S1", y=2.0, acid=0.0, ext=0.5, label="dA_zero")
    t = one(fit_predict("B3", r.frame(), q.frame()))
    assert t["nn_row_id"] == "dA_zero"
    r = Rows()
    r.add("Nd(III)", "S1", y=5.0, acid=0.0, ext=0.0, cmid="SAE:10", label="id10")
    r.add("Nd(III)", "S1", y=6.0, acid=0.0, ext=0.0, cmid="SAE:9", label="id9")
    t = one(fit_predict("B3", r.frame(), q.frame()))
    assert t["nn_row_id"] == "id10" and t["nn_canonical_measurement_id"] == "SAE:10"   # "SAE:10" < "SAE:9"

    # anion: a same-anion candidate beats a closer other-anion one; with none, the anion is dropped (flagged)
    r = Rows()
    r.add("Nd(III)", "S1", y=1.0, acid=0.0, ext=0.0, anion="nitrate", label="nit")
    r.add("Nd(III)", "S1", y=2.0, acid=3.0, ext=3.0, anion="chloride", label="chl")
    qc = Rows()
    qc.add("Nd(III)", "S1", acid=0.0, ext=0.0, anion="chloride", label="qc")
    qc.add("Nd(III)", "S1", acid=0.0, ext=0.0, anion="sulfate", label="qs")
    out = fit_predict("B3", r.frame(), qc.frame()).set_index("row_id")
    assert out.loc["qc", "nn_row_id"] == "chl" and not out.loc["qc", "anion_dropped"]
    assert out.loc["qs", "nn_row_id"] == "nit" and bool(out.loc["qs", "anion_dropped"])

    # missing coordinates: a query without one -> mean of the candidates; a candidate without one is not used
    r = Rows()
    r.add("Nd(III)", "S1", y=1.0, acid=0.0, ext=0.0, label="full")
    r.add("Nd(III)", "S1", y=3.0, acid=0.0, ext=np.nan, label="noext")
    qm = Rows()
    qm.add("Nd(III)", "S1", acid=0.0, ext=np.nan, label="qm")
    qm.add("Nd(III)", "S1", acid=0.1, ext=0.0, label="qf")
    out = fit_predict("B3", r.frame(), qm.frame()).set_index("row_id")
    assert out.loc["qm", "mean_logD"] == 2.0 and "mean_missing_coordinate" in out.loc["qm", "fallback_reason"]
    assert out.loc["qf", "nn_row_id"] == "full" and out.loc["qf", "n_candidates"] == 1
    r = Rows()
    r.add("Nd(III)", "S1", y=1.0, acid=0.0, ext=np.nan)
    r.add("Nd(III)", "S1", y=4.0, acid=1.0, ext=np.nan)
    assert one(fit_predict("B3", r.frame(), qm.frame().loc[["qf"]]))["mean_logD"] == 2.5   # every candidate misses ext


def test_b3_undefined_pair_moves_down_the_registered_chains() -> None:
    r = Rows()
    r.add("Pr(III)", "S_TODGA", y=1.0)
    r.add("Nd(III)", "S_TDDGA", y=2.0)
    q = Rows()
    q.add("Nd(III)", "S_TODGA", label="cell")          # V5-style: pair absent, system and metal present
    q.add("Nd(III)", "S_NEW", label="sys")              # system absent (static labels from the systems table)
    train, query = r.frame(), q.frame()
    systems = pd.DataFrame({"extractant_system_key": ["S_NEW"], "system_family": ["fam_new"],
                            "mechanism": ["NEUTRAL_SOLVATING"], "primary_extractant_smiles": [TBP]})
    c = ctx(systems=systems.set_index("extractant_system_key", drop=False))
    b3 = fit_predict("B3", train, query, c).set_index("row_id")
    assert b3.loc["cell", "fallback_level"] == "B3x" and b3.loc["cell", "fallback_reason"] == "B3_undefined"
    assert b3.loc["sys", "fallback_level"] == "B3l" and b3.loc["sys", "fallback_reason"] == "B3x_undefined"
    b3x = fit_predict("B3x", train, query, c).set_index("row_id")
    assert b3x.loc["sys", "fallback_level"] == "B2:expert" and b3x.loc["sys", "fallback_reason"].startswith("B3x_undefined")
    b3l = fit_predict("B3l", train, query, c).set_index("row_id")
    assert b3l.loc["cell", "fallback_level"] == "B3l" and b3l.loc["cell", "nearest_ligand_system"] == "S_TDDGA"
    # the exact pair is used when it exists, by every B3 arm
    q2 = Rows()
    q2.add("Pr(III)", "S_TODGA", label="exact")
    for name in ("B3", "B3x", "B3i", "B3l"):
        assert one(fit_predict(name, train, q2.frame()))["fallback_level"] == "B3"


# --------------------------------------------------------------------------------------------- #
# B3x / B3i
# --------------------------------------------------------------------------------------------- #

def test_b3x_pool_order_uranyl_borrows_an_actinyl_and_equals_support_index() -> None:
    r = Rows()
    for st, y in (("Np(VI)", 1.0), ("Pu(VI)", 2.0), ("Pu(IV)", 3.0), ("Zr(IV)", 4.0), ("Am(III)", 5.0)):
        r.add(st, "S_TBP", y=y, family="neutral_organophosphate", n=2)
    train = r.frame()
    q = Rows()
    q.add("U(VI)", "S_TBP", label="u")
    p = one(fit_predict("B3x", train, q.frame()))
    # Zr(IV) is closest on CN8 (0.84 vs U(VI) 0.86) but the pool comes first: same charge and species charge
    assert p["nearest_radius_metal"] == "Np(VI)" and p["nearest_radius_pool"] == "same_species_charge"
    assert p["nearest_radius_basis"] == "CN6" and p["lookup_metal"] == "Np(VI)" and p["mean_logD"] == 1.0
    assert SG.SupportIndex(train).features("U(VI)", "S_TBP")["nearest_radius_metal"] == "Np(VI)"
    no_np = train[train[SG.METAL_COL] != "Np(VI)"]
    assert one(fit_predict("B3x", no_np, q.frame()))["nearest_radius_metal"] == "Pu(VI)"
    assert SG.SupportIndex(no_np).features("U(VI)", "S_TBP")["nearest_radius_metal"] == "Pu(VI)"
    # the query's anion scopes the pool: Np(VI) measured only in chloride is not borrowed for a nitrate query
    r2 = Rows()
    r2.add("Np(VI)", "S_TBP", y=1.0, anion="chloride")
    r2.add("Pu(VI)", "S_TBP", y=2.0, anion="nitrate")
    q2 = Rows()
    q2.add("U(VI)", "S_TBP", anion="nitrate", label="nit")
    q2.add("U(VI)", "S_TBP", anion="sulfate", label="sul")
    out = fit_predict("B3x", r2.frame(), q2.frame()).set_index("row_id")
    assert out.loc["nit", "nearest_radius_metal"] == "Pu(VI)" and not out.loc["nit", "radius_pool_anion_dropped"]
    assert out.loc["sul", "nearest_radius_metal"] == "Np(VI)" and bool(out.loc["sul", "radius_pool_anion_dropped"])
    assert bool(out.loc["sul", "anion_dropped"])


def test_b3x_actinyl_pool_beats_a_closer_same_charge_ion(monkeypatch) -> None:
    """An actinyl borrows an actinyl: a same-charge ion of another species charge is passed over even when closer."""
    real = SG.metal_properties

    def fake(label):
        p = dict(real(label))
        if label == "Mo(VI)":
            p.update(charge=6, species_charge=-2, r_cn6=0.73, r_cn8=np.nan, series="none", category="transition_metal")
        return p
    monkeypatch.setattr(SG, "metal_properties", fake)
    r = Rows()
    r.add("Mo(VI)", "S_TBP", y=1.0)
    r.add("Pu(VI)", "S_TBP", y=2.0)
    q = Rows()
    q.add("U(VI)", "S_TBP", label="u")
    p = one(fit_predict("B3x", r.frame(), q.frame()))
    assert p["nearest_radius_metal"] == "Pu(VI)" and p["nearest_radius_pool"] == "same_species_charge"


def test_b3i_interpolation_arithmetic_and_unbracketed_fallback() -> None:
    r = Rows()
    r.add("Sm(III)", "S_TODGA", y=1.0, acid=0.0, ext=0.0)
    r.add("Sm(III)", "S_TODGA", y=50.0, acid=2.0, ext=2.0)          # the B3 rule picks the nearest condition
    r.add("Pr(III)", "S_TODGA", y=2.0, acid=0.1, ext=0.0)
    r.add("Eu(III)", "S_TODGA", y=-9.0)                             # farther below Sm
    r.add("Ce(III)", "S_TODGA", y=-9.0)                             # farther above Pr
    train = r.frame()
    q = Rows()
    q.add("Nd(III)", "S_TODGA", acid=0.0, ext=0.0, label="nd")
    q.add("La(III)", "S_TODGA", acid=0.0, ext=0.0, label="la")
    out = fit_predict("B3i", train, q.frame()).set_index("row_id")
    r_nd, r_sm, r_pr = (SG.metal_properties(x)["r_cn8"] for x in ("Nd(III)", "Sm(III)", "Pr(III)"))
    frac = (r_nd - r_sm) / (r_pr - r_sm)
    assert out.loc["nd", "fallback_level"] == "B3i"
    assert (out.loc["nd", "bracket_lower_metal"], out.loc["nd", "bracket_upper_metal"]) == ("Sm(III)", "Pr(III)")
    assert out.loc["nd", "interp_fraction"] == pytest.approx(frac, abs=1e-15)
    assert out.loc["nd", "mean_logD"] == pytest.approx(1.0 + frac * (2.0 - 1.0), abs=1e-12)
    b3x = fit_predict("B3x", train, q.frame()).set_index("row_id")
    assert out.loc["la", "fallback_level"] == "B3x" and out.loc["la", "fallback_reason"] == "B3_undefined;not_bracketed"
    assert out.loc["nd", "fallback_reason"] == "B3_undefined"
    assert out.loc["la", "mean_logD"] == b3x.loc["la", "mean_logD"] and b3x.loc["la", "lookup_metal"] == "Ce(III)"


# --------------------------------------------------------------------------------------------- #
# B3l
# --------------------------------------------------------------------------------------------- #

def _components(desc: dict[str, tuple[float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame([{"record_type": "STRUCTURE", "smiles_canonical": s, "mw": v[0], "mol_logp": v[1],
                          "rotatable_bonds": v[2]} for s, v in desc.items()])


def test_b3l_tanimoto_ties_broken_by_d_desc_family_rows_and_key() -> None:
    assert SG.tanimoto(TODGA, TDDGA) == 1.0 and SG.tanimoto(TODGA, TDDDGA) == 1.0 and SG.tanimoto(TODGA, TBP) < 1.0
    q = Rows()
    q.add("Nd(III)", "S_TODGA", label="q")

    def run(desc, fam10="diglycolamide", fam12="diglycolamide", n10=1, n12=1):
        r = Rows()
        r.add("Nd(III)", "S_TDDGA", y=10.0, family=fam10, n=n10)
        r.add("Nd(III)", "S_TDDDGA", y=12.0, family=fam12, n=n12)
        r.add("Nd(III)", "S_TBP", y=99.0, family="neutral_organophosphate")
        r.add("Eu(III)", "S_TODGA", y=0.0)
        c = ctx(components=_components(desc))
        return one(fit_predict("B3l", r.frame(), q.frame(), c))

    base = {TODGA: (100.0, 1.0, 10.0), TBP: (266.0, 3.0, 12.0)}
    p = run({**base, TDDGA: (110.0, 1.0, 10.0), TDDDGA: (130.0, 1.0, 10.0)})
    assert p["nearest_ligand_system"] == "S_TDDGA" and p["n_ligand_ties"] == 2 and p["nearest_ligand_tanimoto"] == 1.0
    mw = np.array([110.0, 130.0, 266.0, 100.0])                    # training systems: TDDGA, TDDDGA, TBP, TODGA
    sd = mw.std()
    assert p["nearest_ligand_d_desc"] == pytest.approx(10.0 / sd, rel=1e-12)
    same = {**base, TDDGA: (120.0, 1.0, 10.0), TDDDGA: (120.0, 1.0, 10.0)}
    assert run(same, fam10="other")["nearest_ligand_system"] == "S_TDDDGA"               # same family wins
    assert run(same, n10=3, n12=2)["nearest_ligand_system"] == "S_TDDGA"                   # more rows of the metal
    assert run(same, n10=2, n12=3)["nearest_ligand_system"] == "S_TDDDGA"
    assert run(same)["nearest_ligand_system"] == min("S_TDDGA", "S_TDDDGA")                # key order
    with pytest.raises(ValueError):                                                        # a tie needs d_desc
        r = Rows()
        r.add("Nd(III)", "S_TDDGA", y=10.0)
        r.add("Nd(III)", "S_TDDDGA", y=12.0)
        r.add("Eu(III)", "S_TODGA", y=0.0)
        fit_predict("B3l", r.frame(), q.frame())


# --------------------------------------------------------------------------------------------- #
# B4
# --------------------------------------------------------------------------------------------- #

def test_b4_never_uses_a_row_of_the_query_publication_group() -> None:
    r = Rows()
    r.add("Nd(III)", "S_TODGA", y=999.0, acid=0.0, ext=0.0, pub="g1", label="same_src")
    r.add("Nd(III)", "S_TODGA", y=1.0, acid=1.0, ext=1.0, pub="g2", label="other_src")
    r.add("Eu(III)", "S_TODGA", y=999.0, pub="g1", label="eu_g1")        # nearest radius to Gd, only in g1
    r.add("Sm(III)", "S_TODGA", y=3.0, pub="g2", label="sm_g2")
    r.add("Gd(III)", "S_TDDGA", y=999.0, pub="g1", label="gd_g1")         # nearest ligand for Gd x TODGA, only in g1
    r.add("Gd(III)", "S_TBP", y=4.0, pub="g2", label="gd_g2", family="neutral_organophosphate")
    train = r.frame()
    q = Rows()
    q.add("Nd(III)", "S_TODGA", acid=0.0, ext=0.0, pub="g1", label="nd")
    q.add("Gd(III)", "S_TODGA", pub="g1", label="gd")
    query = q.frame()
    assert fit_predict("B3", train, query).set_index("row_id").loc["nd", "mean_logD"] == 999.0
    b4 = fit_predict("B4", train, query).set_index("row_id")
    assert b4.loc["nd", "nn_row_id"] == "other_src" and b4.loc["nd", "fallback_level"] == "B4"
    assert b4.loc["nd", "excluded_pub_group"] == "g1"
    assert fit_predict("B3x", train, query).set_index("row_id").loc["gd", "lookup_metal"] == "Eu(III)"
    b4x = fit_predict("B4x", train, query).set_index("row_id")
    assert b4x.loc["gd", "lookup_metal"] == "Sm(III)" and b4x.loc["gd", "fallback_level"] == "B4x"   # re-chosen
    assert fit_predict("B3l", train, query).set_index("row_id").loc["gd", "nearest_ligand_system"] == "S_TDDGA"
    b4l = fit_predict("B4l", train, query).set_index("row_id")
    assert b4l.loc["gd", "nearest_ligand_system"] == "S_TBP" and b4l.loc["gd", "mean_logD"] == 4.0
    for name in ("B4", "B4x", "B4l"):
        assert (fit_predict(name, train, query)["mean_logD"] < 999.0).all()

    # property check on a random frame: no chosen neighbour ever shares the query's group
    rng = np.random.default_rng(3)
    r = Rows()
    states = ["La(III)", "Ce(III)", "Pr(III)", "Nd(III)", "Sm(III)", "Eu(III)"]
    for i in range(300):
        r.add(states[rng.integers(6)], ["S_TODGA", "S_TDDGA", "S_TBP"][rng.integers(3)], y=float(rng.normal()),
              acid=float(rng.uniform(-1, 1)), ext=float(rng.uniform(-2, 0)), pub=f"g{rng.integers(6)}")
    fr = r.frame()
    train, query = fr.iloc[:240], fr.iloc[240:]
    groups = fr[I.PUB_GROUP_COL]
    for name in ("B4", "B4x", "B4l"):
        pred = fit_predict(name, train, query, ctx(components=_components({s: (float(len(s)), 1.0, 1.0)
                                                                            for s in SMILES_OF.values()})))
        used = pred.dropna(subset=["nn_row_id"])
        assert len(used) > 0
        assert (groups.loc[used["nn_row_id"]].to_numpy() != groups.loc[used["row_id"]].to_numpy()).all()


# --------------------------------------------------------------------------------------------- #
# hidden rows
# --------------------------------------------------------------------------------------------- #

def _hidden_fixture():
    r = Rows()
    r.add("Nd(III)", "S_TODGA", y=1e6, acid=0.0, ext=0.0, pub="g1", label="cell_sentinel")
    r.add(None, "S_TODGA", y=1e6, acid=0.0, ext=0.0, elem="Nd", pub="g2", label="alias_sentinel")
    r.add("Nd(III)", f"S_TODGA|{DHOA}", y=1e6, acid=0.0, ext=0.0, smiles=TODGA, pub="g3", label="shared_sentinel")
    for st, y in (("Pr(III)", 1.0), ("Sm(III)", 2.0), ("Eu(III)", 3.0)):
        r.add(st, "S_TODGA", y=y, acid=0.1, ext=0.1, pub="g4", n=3)
        r.add(st, f"S_TODGA|{DHOA}", y=y, acid=0.1, ext=0.1, smiles=TODGA, pub="g4")
    r.add("Nd(III)", "S_TDDGA", y=4.0, pub="g5", n=3)
    r.add("Nd(III)", "S_TBP", y=5.0, pub="g5", family="neutral_organophosphate", mech="NEUTRAL_SOLVATING")
    full = r.frame()
    train = SG.hide_cell(full, "Nd(III)", "S_TODGA", component_aware=True)
    hidden = full.index.difference(train.index)
    return full, train, hidden


def test_hidden_rows_never_enter_a_candidate_pool() -> None:
    full, train, hidden = _hidden_fixture()
    assert set(hidden) == {"cell_sentinel", "alias_sentinel", "shared_sentinel"}
    comps = _components({s: (float(len(s)), 1.0, 1.0) for s in (TODGA, TDDGA, TBP)})
    table = I.RowTable(full, components=comps)
    c = ctx(table=table, hidden_index=hidden, components=comps)
    q = full.loc[["cell_sentinel"]]
    for name in B.ARM_NAMES + ("B7",):
        arm = MA.B7MassAction() if name == "B7" else B.BaselineArm(name)
        pred = arm.fit(train, c).predict(q)
        assert (pred["mean_logD"].abs() < 1e5).all(), name
        assert not set(pred["nn_row_id"].dropna()) & set(hidden), name
        # the fast path with the same mask gives the same answer
        fast = (MA.B7MassAction() if name == "B7" else B.BaselineArm(name)).fit_table(
            table, table.mask_of(train.index), c).predict_positions(table.positions(q.index))
        pd.testing.assert_frame_equal(pred, fast)
    with pytest.raises(AssertionError, match="hidden row"):
        B.BaselineArm("B3").fit(pd.concat([train, full.loc[["shared_sentinel"]]]), c)
    new = full.loc[["cell_sentinel", "shared_sentinel"]].rename(index={"cell_sentinel": "q_cell",
                                                                         "shared_sentinel": "q_shared"})
    arm = B.BaselineArm("B3").fit(train, c)
    arm.engine.mask[table.positions(["cell_sentinel"])] = True                      # a corrupted training mask
    with pytest.raises(AssertionError, match="hidden row"):
        arm.predict(new.loc[["q_cell"]])
    arm = B.BaselineArm("B2").fit(train, c)
    arm.engine.mask[table.positions(["shared_sentinel"])] = True
    with pytest.raises(AssertionError, match="hidden row"):
        arm.predict(new.loc[["q_shared"]])
    with pytest.raises(AssertionError, match="query row"):
        B.BaselineArm("B0").fit(train, c).predict(train.iloc[:1])
    # the context's SupportIndex is built on the training rows only: the hidden cell is invisible to it
    f = c.for_training(train).support_index.features("Nd(III)", "S_TODGA", None)
    assert f["exact_pair_rows"] == 0 and f["alias_unknown_state_rows"] == 0
    with pytest.raises(ValueError):
        _ = c.support_index


def test_hide_cell_mask_equals_support_graph_hide_cell() -> None:
    full, _, _ = _hidden_fixture()
    table = I.RowTable(full)
    all_rows = np.ones(table.n, dtype=bool)
    for st, sy in (("Nd(III)", "S_TODGA"), ("Pr(III)", f"S_TODGA|{DHOA}"), ("Nd(III)", "S_TBP"), ("Gd(III)", "S_TODGA")):
        for aware in (True, False):
            want = SG.hide_cell(full, st, sy, component_aware=aware)
            got, dropped = I.hide_cell_mask(table, all_rows, st, sy, component_aware=aware)
            assert list(table.index[got]) == list(want.index)
            assert set(table.index[dropped]) == set(full.index.difference(want.index))


# --------------------------------------------------------------------------------------------- #
# B7
# --------------------------------------------------------------------------------------------- #

A_TRUE = {"Nd(III)": 1.5, "Eu(III)": 0.7, "Am(III)": -0.2}
DELTA = {"g1": 0.2, "g2": -0.05, "g3": -0.15}


def _mass_action_rows(n_true=2.4, p_true=1.3, lE_levels=(-2.0, -1.5, -1.0, -0.5), system="S_TODGA",
                      mech="NEUTRAL_SOLVATING") -> Rows:
    r = Rows()
    lA_levels = (-1.0, -0.5, 0.0, 0.5)
    block = 0
    for m, a in A_TRUE.items():
        for g, d in DELTA.items():
            for j, lE in enumerate(lE_levels):
                lA = lA_levels[(j * (block % 3 + 1) + block) % 4]          # varying pairing: slopes identifiable
                r.add(m, system, y=a + n_true * lE + p_true * lA + d, acid=lA, ext=lE, pub=g, mech=mech)
            block += 1
    lE0 = lE_levels[0]
    r.add("Nd(III)", system, y=A_TRUE["Nd(III)"] + n_true * lE0 + p_true * 0.25, acid=0.25, ext=lE0, pub="g4",
          mech=mech)   # a single-row group: effect 0
    return r


def test_b7_recovers_slopes_intercepts_and_publication_effects() -> None:
    train = _mass_action_rows().frame()
    arm = MA.B7MassAction().fit(train, ctx())
    u = arm.unit("S_TODGA", "nitrate")
    assert u.slope_status == {"n": "fitted", "p_eff": "fitted"}
    assert abs(u.n - 2.4) < 1e-6 and abs(u.p_eff - 1.3) < 1e-6
    for m, a in A_TRUE.items():
        assert abs(u.intercepts[m] - a) < 1e-6
    for g, d in DELTA.items():
        assert abs(u.pub_effects[g] - d) < 1e-6
    assert u.pub_effects["g4"] == 0.0 and u.n_pub_groups_with_effect == 3
    slopes = arm.slopes_table()
    assert list(slopes.columns) == list(MA.SLOPE_COLUMNS) and len(slopes) == 1
    assert slopes.loc[0, "n_se"] < 1e-6 and slopes.loc[0, "n_status"] == "fitted"

    # the same algebra as gen18's fit_mass_action (every row its own point here), to 1e-9
    paths.add_gen18_to_path()
    from gen18proc import evalproto as EP
    recs = pd.DataFrame({"record_id": train[I.ID_COL].to_numpy(), "system_id": "S", "metal": train[SG.METAL_COL],
                         "log_d": train[I.TARGET_COL], "acid_nominal_M": 10 ** train[SG.LOG_ACID_COL],
                         "ligand_M.L": 10 ** train[SG.LOG_EXT_COL], "publication_id": train[I.PUB_GROUP_COL],
                         "temperature_C": 25.0})
    g18 = EP.fit_mass_action(recs, ligand="L", band=None, reliability=False)
    assert abs(g18.n - u.n) < 1e-9 and abs(g18.p_eff - u.p_eff) < 1e-9
    assert all(abs(g18.intercepts[m] - u.intercepts[m]) < 1e-9 for m in A_TRUE)
    assert all(abs(g18.publication_effects[g] - u.pub_effects[g]) < 1e-9 for g in DELTA)
    assert abs(g18.se["n"] - u.se["n"]) < 1e-9


def test_b7_priors_hidden_metal_new_publication_and_fallbacks() -> None:
    r = _mass_action_rows(n_true=MA.N_PRIOR, lE_levels=(-1.0, -0.5))      # two extractant levels: n assumed
    r.add("Eu(III)", "S_ACID", y=2.0, mech="ACIDIC_CATION_EXCHANGE", family="phosphoric_acid", smiles=TBP)
    r.add("Nd(III)", "S_ACID", y=3.0, mech="ACIDIC_CATION_EXCHANGE", family="phosphoric_acid", smiles=TBP)
    train = r.frame()
    arm = MA.B7MassAction().fit(train, ctx())
    u = arm.unit("S_TODGA", "nitrate")
    assert u.slope_status == {"n": "assumed", "p_eff": "fitted"} and u.n == 3.0 and u.se["n"] == 0.0
    assert abs(u.p_eff - 1.3) < 1e-6 and abs(u.intercepts["Eu(III)"] - 0.7) < 1e-6
    q = Rows()
    q.add("Sm(III)", "S_TODGA", acid=0.0, ext=-1.0, pub="g1", label="hidden_metal")
    q.add("Nd(III)", "S_TODGA", acid=0.5, ext=-0.5, pub="g_new", label="new_pub")
    q.add("Nd(III)", "S_TODGA", acid=0.5, ext=-0.5, pub="g2", label="known_pub")
    q.add("Nd(III)", "S_TODGA", acid=0.0, ext=-1.0, anion="chloride", label="other_anion")
    q.add("Sm(III)", "S_ACID", mech="ACIDIC_CATION_EXCHANGE", smiles=TBP, label="acidic")
    q.add("Nd(III)", "S_ABSENT", smiles=TBP, label="absent")
    pred = arm.predict(q.frame()).set_index("row_id")
    # Sm(III) (CN8 1.079) borrows Am(III) (1.09), the nearest same-charge radius among the unit's metals
    assert pred.loc["hidden_metal", "b7_intercept_metal"] == "Am(III)"
    assert pred.loc["hidden_metal", "fallback_reason"] == "intercept_from_nearest_radius_metal"
    assert pred.loc["hidden_metal", "mean_logD"] == pytest.approx(-0.2 + 3.0 * -1.0 + 1.3 * 0.0 + 0.2, abs=1e-6)
    assert pred.loc["new_pub", "b7_pub_effect"] == 0.0
    assert pred.loc["new_pub", "mean_logD"] == pytest.approx(1.5 + 3.0 * -0.5 + 1.3 * 0.5, abs=1e-6)
    assert pred.loc["known_pub", "mean_logD"] == pytest.approx(1.5 + 3.0 * -0.5 + 1.3 * 0.5 - 0.05, abs=1e-6)
    assert pred.loc["other_anion", "fallback_level"] == "B3"
    assert pred.loc["other_anion", "fallback_reason"].startswith("no_unit_rows_for_anion")
    assert pred.loc["acidic", "fallback_level"] == "B3x"
    assert pred.loc["acidic", "fallback_reason"].startswith("mechanism_ACIDIC_CATION_EXCHANGE")
    assert pred.loc["absent", "fallback_level"].startswith("B2") or pred.loc["absent", "fallback_level"] == "B0"
    assert pred.loc["absent", "fallback_reason"].startswith("system_absent")
    counts = I.fallback_counts(pred.reset_index())
    assert int(counts.loc[counts["fallback_level"] == "B7", "n_rows"].sum()) == 3
    assert int(counts["n_rows"].sum()) == 6


# --------------------------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------------------------- #

def _random_frame(seed: int, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = Rows()
    states = ["La(III)", "Ce(III)", "Pr(III)", "Nd(III)", "Sm(III)", "Eu(III)", "Am(III)", "U(VI)", "Pu(VI)"]
    systems = ["S_TODGA", "S_TDDGA", "S_TDDDGA", "S_THDGA", "S_TBP"]
    for _ in range(n):
        # coarse grids create exact distance and Tanimoto ties, so the tie rules are exercised
        r.add(states[rng.integers(len(states))], systems[rng.integers(len(systems))], y=float(rng.normal()),
              acid=float(rng.integers(-2, 3)) / 2, ext=float(rng.integers(-4, 1)) / 2,
              anion=["nitrate", "chloride"][rng.integers(2)], pub=f"g{rng.integers(8)}")
    return r.frame()


def test_predictions_are_deterministic_and_row_order_free() -> None:
    fr = _random_frame(11)
    comps = _components({s: (float(len(s)), 1.0, float(len(s) % 7)) for s in SMILES_OF.values()})
    train, query = fr.iloc[:320], fr.iloc[320:]
    shuffled = train.sample(frac=1.0, random_state=5)
    for name in B.ARM_NAMES + ("B7",):
        a = fit_predict(name, train, query, ctx(components=comps))
        b = fit_predict(name, train, query, ctx(components=comps))
        c = fit_predict(name, shuffled, query, ctx(components=comps))
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        pd.testing.assert_frame_equal(a, c, check_exact=(name != "B7"), rtol=1e-12, atol=1e-12)
        assert np.isfinite(a["mean_logD"]).all()


# --------------------------------------------------------------------------------------------- #
# conformal
# --------------------------------------------------------------------------------------------- #

def test_conformal_quantile_rank() -> None:
    r = np.arange(1, 10, dtype=float)
    assert I.conformal_quantile(r, 0.8) == 8.0                  # ceil(10 * 0.8) = 8
    assert I.conformal_quantile(r, 0.5) == 5.0
    assert I.conformal_quantile(r, 0.95) == math.inf            # ceil(9.5) = 10 > 9
    assert I.conformal_quantile(np.array([]), 0.5) == math.inf


def _group_guard(frame: pd.DataFrame, calls: list):
    def check(tr: pd.Index, te: pd.Index) -> dict:
        calls.append((len(tr), len(te)))
        ok = not (set(frame.loc[tr, I.PUB_GROUP_COL]) & set(frame.loc[te, I.PUB_GROUP_COL]))
        return {"ok": ok}
    return check


def test_conformal_coverage_on_synthetic_grouped_data() -> None:
    rng = np.random.default_rng(19)
    r = Rows()
    effects = rng.normal(0, 2, size=10)
    for g in range(60):
        for _ in range(30):
            s = int(rng.integers(10))
            r.add("Nd(III)", f"S{s}", y=float(effects[s] + rng.normal()), pub=f"g{g:02d}", smiles=TBP)
    fr = r.frame()
    groups = fr[I.PUB_GROUP_COL]
    train = fr[groups < "g40"]
    test = fr[groups >= "g40"]
    calls: list = []
    c = ctx(table=I.RowTable(fr), v6_mask=pd.Series(False, index=fr.index), isolation_check=_group_guard(fr, calls))
    w = I.ConformalWrapper(B.BaselineArm("B2"), calibration="inner", splitter=I.GroupKFoldCalibration(3)).fit(train, c)
    assert len(calls) == 3 and len(w.residuals) == len(train)
    pred = w.predict(test)
    y = test[I.TARGET_COL].to_numpy()
    for lv, (lo, hi) in {50: (0.44, 0.56), 80: (0.74, 0.86), 95: (0.91, 0.99)}.items():
        cov = np.mean((y >= pred[f"lower_{lv}"]) & (y <= pred[f"upper_{lv}"]))
        assert lo <= cov <= hi, (lv, cov)
    assert (pred["conformal_n_calibration"] == len(train)).all() and pred["std_logD"].isna().all()
    # every inner split is checked; a failing check aborts; the verified splits are reused across arms
    n_before = len(calls)
    I.ConformalWrapper(B.BaselineArm("B0"), splitter=I.GroupKFoldCalibration(3)).fit(train, c)
    assert len(calls) == n_before
    bad = ctx(table=c.table, v6_mask=c.v6_mask, isolation_check=lambda tr, te: {"ok": False})
    with pytest.raises(AssertionError, match="isolation"):
        I.ConformalWrapper(B.BaselineArm("B0"), splitter=I.GroupKFoldCalibration(3)).fit(train, bad)
    with pytest.raises(ValueError):
        I.ConformalWrapper(B.BaselineArm("B0"), splitter=I.GroupKFoldCalibration(3)).fit(
            train, ctx(table=c.table, isolation_check=c.isolation_check))                        # no v6 mask
    with pytest.raises(ValueError):
        I.ConformalWrapper(B.BaselineArm("B0"), calibration="outer", splitter=I.GroupKFoldCalibration(3))


def _cell_grid() -> pd.DataFrame:
    """7 Ln(III) states x 5 systems (+ TODGA|DHOA sharing TODGA), a Pr/Nd V6-like cell, X(?) rows, an Sr(III) cell."""
    r = Rows()
    states = ["La(III)", "Ce(III)", "Pr(III)", "Nd(III)", "Sm(III)", "Eu(III)", "Gd(III)"]
    systems = ["S_TODGA", "S_TDDGA", "S_TDDDGA", "S_THDGA", "S_TBP", f"S_TODGA|{DHOA}"]
    for i, st in enumerate(states):
        for j, sy in enumerate(systems):
            n = 12 if (i + j) % 3 else 4                      # some cells below k = 10
            r.add(st, sy, y=float(i - j), acid=0.0, ext=-1.0, pub=f"g{(i * 7 + j) % 5}", n=n,
                  smiles=TODGA if sy.startswith("S_TODGA") else None)
    r.add(None, "S_TODGA", y=0.0, elem="Nd", pub="g1", n=3)
    r.add("Sr(III)", "S_TBP", y=0.0, pub="g2", n=12)
    return r.frame()


def test_inner_cell_calibration_eligibility_hiding_and_v6_exclusion() -> None:
    fr = _cell_grid()
    table = I.RowTable(fr)
    v6 = fr[SG.ELEMENT_COL].isin(["Pr", "Nd"]) & (fr[SG.SYSTEM_COL] == "S_TDDGA")
    calls: list = []

    def v5_guard(tr: pd.Index, te: pd.Index) -> dict:     # the registered guard, V5 component-aware level
        calls.append((len(tr), len(te)))
        return L.fold_isolation_check(tr, te, fr, "V5", component_aware=True, near_dup_sig=None)
    c = ctx(table=table, v6_mask=v6, isolation_check=v5_guard, seed=104729)
    splitter = I.InnerCellCalibration(k=10, p=1, m=3, max_cells_per_fold=4)
    mask = np.ones(table.n, dtype=bool)
    cells = splitter.eligible_cells(table, mask, c)
    known = fr[fr[SG.METAL_COL].notna()]
    counts = known.groupby([SG.METAL_COL, SG.SYSTEM_COL]).size()
    nsys = known.groupby(SG.METAL_COL)[SG.SYSTEM_COL].nunique()
    nst = known.groupby(SG.SYSTEM_COL)[SG.METAL_COL].nunique()
    G = SG.bipartite_graph(fr)
    want = sorted((st, sy) for (st, sy), n in counts.items()
                  if n >= 10 and nsys[st] - 1 >= 3 and nst[sy] - 1 >= 3 and not SG.cell_is_bridge(G, st, sy)
                  and not v6[(fr[SG.METAL_COL] == st) & (fr[SG.SYSTEM_COL] == sy)].any())
    assert [(st, sy) for st, sy, _ in cells] == want
    assert not any(sy == "S_TDDGA" and st in ("Pr(III)", "Nd(III)") for st, sy, _ in cells)
    splits = splitter.splits(table, mask, c)
    assert 0 < len(splits) <= 3 * 4 and len({s.unit for s in splits}) == len(splits)
    for sp in splits:
        st, sy = sp.unit
        want_train = SG.hide_cell(fr, st, sy, component_aware=True)
        assert list(table.index[sp.train_mask]) == list(want_train.index)
        cal = fr.loc[table.index[sp.cal_positions]]
        assert (cal[SG.METAL_COL] == st).all() and (cal[SG.SYSTEM_COL] == sy).all() and (cal[SG.METAL_COL] != "Sr(III)").all()
    again = splitter.splits(table, mask, c)
    assert [s.unit for s in again] == [s.unit for s in splits]                     # seeded, reproducible
    w = I.ConformalWrapper(B.BaselineArm("B3x"), splitter=splitter).fit_table(table, mask, c)
    assert len(calls) == len(splits) and len(w.residuals) == sum(len(s.cal_positions) for s in splits)
    n_calls = len(calls)
    c2 = ctx(table=table, v6_mask=v6, isolation_check=v5_guard, seed=104729)
    I.ConformalWrapper(B.BaselineArm("B3x"), splitter=splitter, guard="nested_certificate").fit_table(table, mask, c2)
    assert len(calls) - n_calls == len(splits)                                    # one certificate per inner cell
    # a calibration set that would score a V6 row is refused
    v6_bad = v6.copy()
    v6_bad.loc[table.index[splits[0].cal_positions[:1]]] = True
    splitter_bypass = type("S", (), {"splits": lambda self, t, m, cc: splits})()
    with pytest.raises(AssertionError, match="V6_TARGET_ROWS"):
        I.ConformalWrapper(B.BaselineArm("B3x"), splitter=splitter_bypass).fit_table(
            table, mask, ctx(table=table, v6_mask=v6_bad, isolation_check=lambda a, b: {"ok": True}))


def test_inner_metal_calibration_hides_the_element_and_scores_the_state() -> None:
    r = Rows()
    for st in ("Nd(III)", "Eu(III)", "Am(III)", "Pu(IV)", "Pu(VI)"):
        for sy in ("S_TODGA", "S_TDDGA", "S_TDDDGA", "S_THDGA", "S_TBP"):
            r.add(st, sy, y=1.0, pub="g1", n=25)
    r.add(None, "S_TBP", elem="Pu", pub="g2", n=5)
    fr = r.frame()
    table = I.RowTable(fr)
    c = ctx(table=table, v6_mask=pd.Series(False, index=fr.index), seed=5)
    splits = I.InnerMetalCalibration(n_metals=5).splits(table, np.ones(table.n, dtype=bool), c)
    assert sorted(s.unit for s in splits) == ["Am(III)", "Eu(III)", "Nd(III)", "Pu(IV)", "Pu(VI)"]
    for sp in splits:
        el = SG.metal_properties(sp.unit)["symbol"]
        assert not (fr.loc[table.index[sp.train_mask], SG.ELEMENT_COL] == el).any()
        assert (fr.loc[table.index[sp.cal_positions], SG.METAL_COL] == sp.unit).all()


# --------------------------------------------------------------------------------------------- #
# gen18 cross-check
# --------------------------------------------------------------------------------------------- #

GEN18_SYSTEM = "sys_5cb78e5000d40860"     # TODGA, nitrate, 514 records in 28 publications (gen18 corpus)


def _gen18_records():
    f = paths.GEN18_ROOT / "systems" / "corpus_records.csv"
    if not f.exists():
        pytest.skip("gen18 corpus_records.csv absent")
    paths.add_gen18_to_path()
    from gen18proc import evalproto as EP
    rec = pd.read_csv(f)
    sub = rec[rec["system_id"] == GEN18_SYSTEM].copy()
    prep = EP.prepare_records(sub, ligand=str(sub["ligand_name"].iloc[0]), band=EP.DEFAULT_BAND)
    return EP, prep


def _gen18_nearest(EP, train_pts: pd.DataFrame, a: float, e: float, metal: str) -> pd.Series:
    tr_m = train_pts.loc[train_pts["metal"] == metal].reset_index(drop=True)
    return tr_m.iloc[EP._nearest_index(a, e, tr_m)]


def test_b3_reproduces_gen18_nearest_index_on_the_same_partitions() -> None:
    """Exact: on gen18's replicate-aggregated points of one system, B3 picks the same neighbour as
    ``gen18proc.evalproto._nearest_index`` for every leave-one-publication-out partition."""
    EP, prep = _gen18_records()
    pts = EP.aggregate_replicates(prep)
    frame = pd.DataFrame({SG.METAL_COL: pts["metal"].to_numpy(), SG.ELEMENT_COL: pts["metal"].to_numpy(),
                          SG.SYSTEM_COL: GEN18_SYSTEM, SG.ACID_ANION_COL: "nitrate",
                          SG.LOG_ACID_COL: pts["lA"].to_numpy(), SG.LOG_EXT_COL: pts["lE"].to_numpy(),
                          I.ID_COL: pts["record_id"].to_numpy(), I.TARGET_COL: pts["log_d"].to_numpy(),
                          I.PUB_GROUP_COL: pts["publication_id"].to_numpy()},
                         index=pd.Index(pts["record_id"].to_numpy()))
    n = 0
    for p in sorted(set(pts["publication_id"])):
        tr = pts[pts["publication_id"] != p]
        te = pts[(pts["publication_id"] == p) & pts["metal"].isin(set(tr["metal"]))]
        if te.empty:
            continue
        pred = B.BaselineArm("B3").fit(frame.loc[tr["record_id"]], ctx()).predict(frame.loc[te["record_id"]])
        for (_, t), (_, row) in zip(te.iterrows(), pred.iterrows()):
            g = _gen18_nearest(EP, tr, t["lA"], t["lE"], t["metal"])
            assert row["fallback_level"] == "B3"
            assert row["nn_canonical_measurement_id"] == g["record_id"] and row["mean_logD"] == g["log_d"]
            n += 1
    assert n >= 300


def test_b3_on_archive_rows_differs_from_gen18_only_by_replicate_aggregation_and_tie_order() -> None:
    """Raw archive MODEL rows (gen19) against gen18's aggregated points on the same record set and partitions:
    the nearest-neighbour distance is always the same; a differing value is explained by gen18 averaging the
    replicate group of the chosen condition, or by an equidistant neighbour ordered by a different id
    (``canonical_measurement_id`` vs gen18 ``record_id``).

    Record set: gen18's prepared records of the system that are archive MODEL rows with a KNOWN oxidation state.
    gen18 keys a metal by element symbol, so it also uses the rows the archive records without a state; gen19 keys
    by ``g19_metal_state`` and never uses an X(?) row as a B3 candidate or query (prereg section 2).  Those rows are
    removed from both sides so that the candidate sets coincide; that exclusion is the third documented difference.
    """
    EP, prep = _gen18_records()
    from gen19ct.data import load
    t0 = time.perf_counter()
    model = load.load_model_rows()
    prep = prep.assign(exp_id=prep["record_id"].str.split(":").str[-1].astype(int))
    model = model[model["g19_bundle_exp_id"].isin(set(prep["exp_id"])) & model[SG.METAL_COL].notna()]
    n_unknown_state = int(prep["exp_id"].isin(set(load.load_model_rows(copy=False).loc[
        lambda d: d[SG.METAL_COL].isna(), "g19_bundle_exp_id"].dropna().astype(int))).sum())
    prep = prep[prep["exp_id"].isin(set(model["g19_bundle_exp_id"].astype(int)))]
    pts = EP.aggregate_replicates(prep)
    fr = SG.prepare_support_frame(model)
    fr[I.PUB_GROUP_COL] = fr[SG.PUB_COL]
    rid_of = dict(zip(prep["exp_id"], prep["record_id"]))
    fr["gen18_record_id"] = fr["g19_bundle_exp_id"].astype(int).map(rid_of)
    assert fr["gen18_record_id"].notna().all() and fr[SG.METAL_COL].notna().all()
    pub18 = dict(zip(prep["record_id"], prep["publication_id"]))
    assert (fr["gen18_record_id"].map(pub18) == fr[SG.PUB_COL]).all()
    point_of = {rid: i for i, ids in enumerate(pts["record_ids"]) for rid in ids}
    classes = {"identical": 0, "replicate_mean": 0, "equidistant_tie_order": 0, "unexplained": 0}
    for p in sorted(set(pts["publication_id"])):
        tr_pts = pts[pts["publication_id"] != p]
        te_rows = fr[(fr[SG.PUB_COL] == p) & fr[SG.ELEMENT_COL].isin(set(tr_pts["metal"]))]
        if te_rows.empty:
            continue
        pred = B.BaselineArm("B3").fit(fr[fr[SG.PUB_COL] != p], ctx()).predict(te_rows)
        for (_, row), (_, pr) in zip(te_rows.iterrows(), pred.iterrows()):
            t = pts.iloc[point_of[row["gen18_record_id"]]]
            g = _gen18_nearest(EP, tr_pts, t["lA"], t["lE"], t["metal"])
            d18 = math.hypot(g["lA"] - t["lA"], g["lE"] - t["lE"])
            assert pr["fallback_level"] == "B3" and not pr["anion_dropped"]
            assert abs(pr["nn_distance"] - d18) < 1e-9
            nn_rid = fr.loc[pr["nn_row_id"], "gen18_record_id"]
            if abs(pr["mean_logD"] - g["log_d"]) < 1e-12:      # gen18 log_d = log10(d): last-bit differences
                classes["identical"] += 1
            elif nn_rid in g["record_ids"]:
                assert g["n_rep"] > 1                       # gen18 returned the mean of a replicate group
                classes["replicate_mean"] += 1
            elif abs(pr["nn_distance"] - d18) < 1e-12:
                tr_m = tr_pts[tr_pts["metal"] == t["metal"]]
                d_all = np.hypot(tr_m["lA"] - t["lA"], tr_m["lE"] - t["lE"])
                assert int((np.abs(d_all - d18) < 1e-12).sum()) >= 2   # a genuine equidistant alternative exists
                classes["equidistant_tie_order"] += 1
            else:
                classes["unexplained"] += 1
    assert classes["unexplained"] == 0, classes
    assert classes["identical"] > 0 and sum(classes.values()) >= 300, classes
    print(f"gen18 raw-row cross-check: {classes}; unknown-state records excluded {n_unknown_state}; "
          f"{time.perf_counter() - t0:.1f} s")


# --------------------------------------------------------------------------------------------- #
# the prediction record (verification finding VL-05: an explicit shape / column test)
# --------------------------------------------------------------------------------------------- #

def test_prediction_record_shape_columns_and_order_for_every_arm() -> None:
    fr = _cell_grid()
    known = fr[fr[SG.METAL_COL].notna()]
    query = known[(known[SG.METAL_COL] == "Sm(III)") & (known[SG.SYSTEM_COL] == "S_TDDGA")].iloc[::-1]
    train = fr.drop(query.index)
    comps = _components({s: (float(len(s)), 1.0, 1.0) for s in list(SMILES_OF.values()) + [DHOA]})
    for name in tuple(B.ARM_NAMES) + ("B7",):
        pred = fit_predict(name, train, query, ctx(seed=104729, components=comps))
        assert list(pred.columns) == list(I.PREDICTION_COLUMNS), name
        assert len(pred) == len(query) and list(pred["row_id"]) == list(query.index), name       # query order
        for c in I._FLOAT_COLUMNS:
            assert pred[c].dtype == float, (name, c)
        assert np.isfinite(pred["mean_logD"]).all(), name
        assert pred[["lower_50", "upper_50", "lower_80", "upper_80", "lower_95", "upper_95"]].isna().all().all()
        assert pred["fallback_level"].notna().all(), name
    table = I.RowTable(fr, components=comps)
    mask = ~fr.index.isin(query.index)
    c = ctx(table=table, v6_mask=pd.Series(False, index=fr.index), seed=104729, components=comps,
            isolation_check=lambda tr, te: {"ok": True})
    w = I.ConformalWrapper(B.BaselineArm("B3x"), splitter=I.InnerCellCalibration(k=10, p=1, m=3)).fit_table(table, mask, c)
    pw = w.predict_positions(table.positions(query.index))
    assert list(pw.columns) == list(I.PREDICTION_COLUMNS) and len(pw) == len(query)
    for lv in (50, 80, 95):
        q = pw[f"conformal_q{lv}"].to_numpy()
        assert np.allclose(pw[f"upper_{lv}"] - pw["mean_logD"], q) and np.allclose(pw["mean_logD"] - pw[f"lower_{lv}"], q)
    assert (pw["conformal_q50"] <= pw["conformal_q80"]).all() and (pw["conformal_q80"] <= pw["conformal_q95"]).all()
    assert (pw["conformal_n_calibration"] == len(w.residuals)).all()
