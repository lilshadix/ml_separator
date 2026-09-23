"""The deployment prediction step of the process case (``scripts/g19_predict_process_inputs.py``): the configuration
mode / median rule, the corpus-shaped query rows, the frozen fit with inner-fold split-conformal intervals on a synthetic
mini-corpus, the section 13 support of query rows, the adapter's table layout (members = mean, flagged) and the Pr/Nd
residual correlation.  Nothing reads the real corpus, a fold file, a discovery record or the withheld seeds.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

G19 = Path(__file__).resolve().parents[1]
if str(G19) not in sys.path:
    sys.path.insert(0, str(G19))

from gen19ct.chemistry import support_graph as SG  # noqa: E402
from gen19ct.evaluation import discovery as D  # noqa: E402
from gen19ct.folds import io as FI  # noqa: E402
from gen19ct.models import boosted as BO  # noqa: E402
from gen19ct.models import features as F  # noqa: E402
from gen19ct.models import inner_design as ID  # noqa: E402
from gen19ct.models import interface as I  # noqa: E402
from gen19ct.process import gen18_adapter as GA  # noqa: E402


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, G19 / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PP = _load("g19_predict_process_inputs")
RD = _load("g19_run_discovery")

LN = ("La(III)", "Ce(III)", "Pr(III)", "Nd(III)", "Sm(III)", "Eu(III)", "Gd(III)")
AN = ("Am(III)", "Cm(III)")
SYSTEMS = ("S0", "S1", "S2", "S3")
X_COLS = tuple(f"syn_x{i}" for i in range(4))
TINY = BO.CatBoostConfig(4, 3.0, max_iterations=40, early_stopping_rounds=10)


@dataclass
class StubMatrix:
    frame: pd.DataFrame
    categorical_columns: tuple
    state_digest: str
    wide: dict

    @property
    def columns(self) -> tuple:
        return tuple(self.frame.columns)

    def cat_feature_indices(self) -> list[int]:
        return [list(self.frame.columns).index(c) for c in self.categorical_columns]


class StubFeatures:
    """A feature set over the ``syn_x*`` columns plus the metal state and system as categoricals (no condition vector;
    the ``tests/test_boosted.py`` pattern)."""

    def __init__(self):
        self.state_digest = "stub"
        self.n_train_rows = None

    def fit(self, rows, cv=None):
        assert I.TARGET_COL not in rows.columns
        self.n_train_rows = len(rows)
        return self

    def transform(self, rows, cv=None):
        fr = pd.DataFrame({c: rows[c].to_numpy(dtype=float) for c in X_COLS}, index=rows.index)
        fr["state"] = pd.Series(rows[SG.METAL_COL].astype(str).to_numpy(dtype=object), index=rows.index, dtype=object)
        fr["system"] = pd.Series(rows[SG.SYSTEM_COL].astype(str).to_numpy(dtype=object), index=rows.index, dtype=object)
        return StubMatrix(fr, ("state", "system"), self.state_digest, {})


def _frame(n_per_cell: int = 12, seed: int = 5) -> pd.DataFrame:
    """A mini corpus with every column the support graph, the inner design and the query builder need."""
    rng = np.random.default_rng(seed)
    recs = []
    for i, st in enumerate(LN + AN):
        sym = st.split("(")[0]
        for j, sy in enumerate(SYSTEMS):
            for k in range(n_per_cell):
                g = f"g{(i + 2 * j) % 6}" if k < 9 else f"g{(i + 2 * j + 1) % 6}"   # a dominant group per cell
                acid = 0.1 + 0.3 * (k % 5)
                ext = 0.05 + 0.02 * (k % 4)
                x = rng.random(4)
                recs.append({FI.ROW_ID: f"T:{len(recs):05d}", "log_D": float(0.5 * i + 0.3 * j + x[0] + 0.1 * rng.normal()),
                             "duplicate_group_id": f"d{len(recs)}", SG.PUB_COL: f"pub_{g}", SG.METAL_COL: st,
                             SG.ELEMENT_COL: sym, "g19_ox": 3, "metal_symbol": sym, "metal_category":
                             "lanthanide" if st in LN else "actinide", SG.SYSTEM_COL: sy, "system_label": sy,
                             "acid_primary": "HNO3", "acid_signature": "HNO3", SG.ACID_ANION_COL: "nitrate",
                             "acid_concentration_M": acid, SG.LOG_ACID_COL: float(np.log10(acid)),
                             "extractant_primary_concentration_M": ext, SG.LOG_EXT_COL: float(np.log10(ext)),
                             "metal_concentration_M": 1e-4, "log10_metal_M": -4.0, "phase_ratio_org_aq": 1.0,
                             SG.TEMP_COL: 25.0, "contact_time_min": 30.0 + 30.0 * (k % 2), "solvent_primary": "dodecane",
                             "solvent_key": "dodecane:1", "solvent_components": np.array(["dodecane"], dtype=object),
                             SG.DILUENT_COL: "aliphatic", "modifier_name": np.nan, "modifier_concentration_M": np.nan,
                             "complexant_name": np.nan, "acid_concentration_organic_M": np.nan,
                             "nitrate_concentration_M": np.nan, "n_organic_extractants": 1, "shaking_time_min": np.nan, SG.FAMILY_COL: "diglycolamide" if j < 3 else "monoamide",
                             SG.MECH_COL: "NEUTRAL_SOLVATING", SG.SMILES_COL: f"C{j}", I.PUB_GROUP_COL: g, FI.GROUP_COL: g,
                             **{c: float(v) for c, v in zip(X_COLS, x)},
                             "components": np.array([{"role": "organic_extractant", "name": sy, "concentration_M": ext}],
                                                    dtype=object)})
    df = pd.DataFrame(recs)
    df.index = pd.Index(range(len(df)), dtype="int64")
    return df


def _corpus(tmp_path: Path, df: pd.DataFrame):
    halves = {d: pd.Series("S", index=df.index) for d in ("V5", "V5P", "V5PAIR", "V1", "V2")}
    folds_dir = tmp_path / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)
    return RD.Corpus(frame=df, slim=FI.slim_frame(df), table=I.RowTable(df), v6=pd.Series(False, index=df.index),
                     coext=pd.Series(False, index=df.index), systems=None, comps=None, cv=None, pmap=None,
                     row_half=halves, folds_dir=folds_dir)


def _request(n_acid: int = 3, n_ligand: int = 2) -> pd.DataFrame:
    grid = GA.prediction_grid(acid_M={"feed": (0.3, 0.3), "strip": (0.1, 1.5)}, ligand_M=(0.05, 0.11), n_acid=n_acid,
                              n_ligand=n_ligand)
    rows = []
    for m in ("Nd", "Pr"):
        for r in grid.itertuples(index=False):
            rows.append({"metal": m, "system_id": "sys_test", "acid": "HNO3", "anion": "nitrate", "log_acid": r.log_acid,
                         "log_ligand": r.log_ligand, "acid_M": 10 ** r.log_acid, "ligand_M": 10 ** r.log_ligand,
                         "mean_logD": np.nan, "std_logD": np.nan, "lower_95": np.nan, "upper_95": np.nan,
                         "domain_status": "", "support_score": np.nan, "nearest_support": "", "arm": "",
                         **{c: np.nan for c in GA.MEMBER_COLUMNS}, GA.CONFORMAL_Q95_COLUMN: np.nan, "note": "fill"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- #
# the configuration rule
# --------------------------------------------------------------------------------------------- #

def _record(label: str, depth: int, l2: float, iterations: int, path: str = "r.json") -> dict:
    cfg = BO.CatBoostConfig(depth, l2).record()
    return {"selected_config": label, "arm_record": {"frozen": {"config": cfg, "iterations": iterations}},
            "_path": path, "_matched_entry": "superseded"}


def test_deployed_configuration_is_the_mode_and_the_median_iteration_count():
    recs = [_record("depth8_l23", 8, 3.0, 2934), _record("depth8_l23", 8, 3.0, 3000), _record("depth8_l210", 8, 10.0, 1267),
            _record("depth8_l23", 8, 3.0, 2992)]
    c = PP.deployed_configuration(recs)
    assert c["selected_config"] == "depth8_l23" and c["config"].depth == 8 and c["config"].l2_leaf_reg == 3.0
    assert c["iterations"] == int(round(np.median([2934, 3000, 1267, 2992]))) == 2963
    assert c["counts"] == {"depth8_l210": 1, "depth8_l23": 3} and c["tied_modes"] == []
    # a tied mode resolves by the section 7 tie rule: the smaller configuration (lower depth, then larger l2)
    tie = [_record("depth8_l23", 8, 3.0, 100), _record("depth6_l210", 6, 10.0, 200)]
    t = PP.deployed_configuration(tie)
    assert t["selected_config"] == "depth6_l210" and sorted(t["tied_modes"]) == ["depth6_l210", "depth8_l23"]
    assert t["iterations"] == 150
    with pytest.raises(ValueError):
        PP.deployed_configuration([])


# --------------------------------------------------------------------------------------------- #
# query rows
# --------------------------------------------------------------------------------------------- #

def test_query_rows_are_corpus_shaped_and_carry_the_request_conditions():
    df = _frame()
    req = _request()
    templates = PP.template_rows(df, "S1", metals=["Nd", "Pr"])
    assert templates["Nd"][SG.ELEMENT_COL] == "Nd" and templates["Pr"][SG.ELEMENT_COL] == "Pr"
    q = PP.query_rows(req, df, system_key="S1", templates=templates, contact_time_min=45.0)
    assert len(q) == len(req) == 12 and q.index.is_unique and not q.index.isin(df.index).any()
    assert set(q[SG.METAL_COL]) == {"Nd(III)", "Pr(III)"} and (q[SG.SYSTEM_COL] == "S1").all()
    np.testing.assert_allclose(q[SG.LOG_ACID_COL].to_numpy(), req["log_acid"].to_numpy())
    np.testing.assert_allclose(q["acid_concentration_M"].to_numpy(), 10 ** req["log_acid"].to_numpy())
    np.testing.assert_allclose(q["extractant_primary_concentration_M"].to_numpy(), 10 ** req["log_ligand"].to_numpy())
    assert (q["metal_concentration_M"] == PP.TRACER_METAL_M).all() and (q["contact_time_min"] == 45.0).all()
    assert q["log_D"].isna().all() and (q[I.PUB_GROUP_COL] == PP.PROCESS_PUB).all()
    assert q[FI.ROW_ID].is_unique and q[FI.ROW_ID].str.startswith("PROC:").all()
    comps = q["components"].iloc[0]
    assert isinstance(comps, np.ndarray) and comps.ndim == 1 and comps[0]["role"] == "organic_extractant"
    assert math.isclose(float(comps[0]["concentration_M"]), 10 ** float(req["log_ligand"].iloc[0]))
    # the template of an element without a row of its own takes the lanthanide template with the element's identity
    df2 = df[df[SG.ELEMENT_COL] != "Pr"]
    with pytest.raises(ValueError, match="no row of element"):
        PP.template_rows(df2, "S1", metals=["Pr"])
    df3 = pd.concat([df2, df[(df[SG.ELEMENT_COL] == "Pr") & (df[SG.SYSTEM_COL] == "S3")]])
    t3 = PP.template_rows(df3, "S1", metals=["Pr"])
    assert t3["Pr"][SG.ELEMENT_COL] == "Pr" and t3["Pr"][SG.SYSTEM_COL] == "S1"


def test_training_masks_follow_the_section_11_without_rule():
    df = _frame()
    m, n = PP.training_mask(df, "WITHOUT")
    assert n == 2 * len(SYSTEMS) * 12 and not df.loc[m, SG.ELEMENT_COL].isin(["Am", "Cm"]).any()
    m2, n2 = PP.training_mask(df, "WITH")
    assert m2.all() and n2 == 0
    with pytest.raises(ValueError):
        PP.training_mask(df, "PERMUTED")


# --------------------------------------------------------------------------------------------- #
# the fit, the intervals, the support and the table
# --------------------------------------------------------------------------------------------- #

def test_deployment_fit_and_prediction_table_on_a_mini_corpus(tmp_path):
    df = _frame()
    corpus = _corpus(tmp_path, df)
    req = _request()
    templates = PP.template_rows(df, "S1", metals=["Nd", "Pr"])
    q = PP.query_rows(req, df, system_key="S1", templates=templates, contact_time_min=30.0)
    arm_frame = pd.concat([df, q])
    design = ID.SimultaneousInnerCells(4, 1, 2, 3, 12, component_aware=False, require_all_folds=True)
    res = PP.deployment_fit(corpus, q, variant="WITHOUT", config=TINY, iterations=25, seed=104729, design=design,
                            splitter_design=BO.inner_design_for("V5"), guard="every_split",
                            isolation_check=lambda tr, te: {"ok": True}, arm_frame=arm_frame, cv=None,
                            feature_factory=StubFeatures)
    assert res["n_actinide_rows_removed"] == 2 * len(SYSTEMS) * 12 and res["n_train"] == len(LN) * len(SYSTEMS) * 12
    assert res["calibration_folds"] == [0, 1, 2] and res["n_calibration"] > 0
    q50, q80, q95 = (res["quantiles"][k] for k in ("0.5", "0.8", "0.95"))
    assert 0 <= q50 <= q80 <= q95
    pred = res["pred"]
    assert len(pred) == len(q) and np.isfinite(pred["mean_logD"]).all()
    np.testing.assert_allclose(pred["upper_95"] - pred["mean_logD"], q95)
    assert (pred["conformal_n_calibration"] == res["n_calibration"]).all()
    sup = res["support"]
    assert len(sup) == len(q) and set(sup["domain_status"]) <= set(GA.DOMAIN_STATUSES)
    assert (sup["exact_pair_rows"] > 0).all() and sup["nearest_support"].str.contains(r"\|S1").all()
    assert sup["support_score"].between(0, 1).all()
    # the adapter's table: members are the mean, std * q95 / 1.96 reproduces the conformal half-width, rectangular grid
    table = PP.prediction_table(req, pred, sup, q, variant="WITHOUT", system_key="S1")
    assert len(table) == len(req) and (table["arm"] == PP.DEPLOYED).all() and (table["arm_alias"] == "B5").all()
    for c in GA.MEMBER_COLUMNS:
        np.testing.assert_allclose(table[c], table["mean_logD"])
    np.testing.assert_allclose(table["std_logD"] * table[GA.CONFORMAL_Q95_COLUMN], q95, rtol=1e-9)
    np.testing.assert_allclose(table["upper_95"] - table["lower_95"], 2 * q95)
    assert (table["members_note"] == PP.MEMBERS_NOTE).all() and (table["label"] == PP.LABEL).all()
    assert "note" not in table.columns
    pt = GA.PredictionTable.from_frame(table)
    assert pt.has_members and pt.n_members == 5 and pt.shape == (2, 3, 2) and pt.arm == PP.DEPLOYED
    assert pt.meta["n_intervals_repaired"] == 0
    # the process runner flags identical members (single-model arm) and refuses them in registered mode
    RP = _load("g19_run_process")
    assert RP.members_identical(pt) is True
    # a wrong-length support frame is refused
    with pytest.raises(ValueError, match="lengths differ"):
        PP.prediction_table(req, pred, sup.iloc[:-1], q, variant="WITHOUT", system_key="S1")
    # the residual correlation reads the signed inner residuals over comparable pairs, never a V6 row
    rho = PP.residual_correlation(res["calibration_detail"], df, corpus.v6, min_pairs=3)
    assert rho["n_calibration_rows"] == res["n_calibration"] and -1 < rho["rho"] < 1
    assert rho["source"] and (rho["n_pairs_prnd"] >= 0) and isinstance(rho["n_comparable_pairs_all"], int)
    v6 = pd.Series(False, index=df.index)
    v6.iloc[0] = True
    detail = [{"fold": 0, "labels": [df.index[0], df.index[1]], "signed": np.array([0.1, -0.2])}]
    with pytest.raises(AssertionError, match="V6_TARGET_ROWS"):
        PP.residual_correlation(detail, df, v6)
    # too few pairs of every kind -> independent draws, flagged
    none = PP.residual_correlation([{"fold": 0, "labels": [df.index[5]], "signed": np.array([0.1])}], df, corpus.v6)
    assert none["rho"] == 0.0 and "independent" in none["flag"]


def test_readings_and_constants_are_recorded():
    assert PP.DEPLOYED == "M0" and PP.ARM == "B5" and PP.DEPLOYED_VARIANT == "WITHOUT"
    assert PP.TRANSFER_UNSUPPORTED in PP.LABEL and "exploratory" in PP.LABEL
    assert {"training_rows", "configuration", "intervals", "adapter_columns", "members", "query_rows", "support", "rho"} <= set(PP.READINGS)
    assert PP.Z_975 == pytest.approx(1.959963984540054) and PP.SELECTION_SEED == D.PRIMARY_SEED == 104729
    assert PP.SELECTION_DESIGN_DIR == "V5__primary_batched_max4"
    assert all(p.exists() for p in PP.CODE_FILES)
