"""The scoring path on the real, label-free row-manifest keys with synthetic y and synthetic predictions (no real
target value is read: y is drawn from a seeded generator before any scoring code runs).  Checks the population
sizes the sealed documents state, the shared cluster lists, and that the registered-claim machinery runs at the
real scale."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from gen201 import inference as I
from gen201 import report as RP
from gen201 import rules as R

SEEDS = (104729, 130363, 155921, 196613, 262147)


@pytest.fixture(scope="module")
def fake():
    man = RP.load_row_manifests()
    rng = np.random.default_rng(7)
    keys = man["COH-D-ALL"]
    lvl = {c: rng.normal(0, 1.5) for c in keys["ci"].unique()}
    y = np.array([lvl[c] + 0.05 * m + rng.normal(0, 0.2) for c, m in zip(keys["ci"], keys["im"])])
    ylk = {(c, m): v for c, m, v in zip(keys["ci"], keys["im"], y)}
    sf = pd.concat([man["COH-SF-ANY"], man["COH-SF-NDSM"]]).drop_duplicates(["ci", "ia", "ib"])
    d_parts, s_parts = [], []
    for s in SEEDS:
        r = np.random.default_rng(s)
        d = keys.copy()
        d["split_seed"], d["fold"], d["design"], d["variant"] = s, 0, "BP", "primary"
        d["y"] = y
        lev_err = d["ci"].map({c: r.normal(0, 0.5) for c in d["ci"].unique()}).to_numpy()
        d["A"] = y + lev_err + r.normal(0, 0.1, len(d))
        for b, sd in (("BD_MEAN", 2.0), ("BD_METAL", 1.6), ("BD_COND", 1.0), ("BD_NN", 1.2)):
            d[b] = y + r.normal(0, sd, len(d))
        d_parts.append(d)
        q = sf.copy()
        q["split_seed"], q["fold"], q["design"], q["variant"] = s, 0, "BP", "primary"
        q["y"] = [ylk[(c, a)] - ylk[(c, b)] for c, a, b in zip(q["ci"], q["ia"], q["ib"])]
        Al = {(c, m): v for c, m, v in zip(d["ci"], d["im"], d["A"])}
        q["B"] = [Al[(c, a)] - Al[(c, b)] for c, a, b in zip(q["ci"], q["ia"], q["ib"])]
        q["FLAT"] = 0.0
        q["PAIRMEAN"] = q.groupby("pair")["y"].transform("mean") * 0.5
        q["G14_ZERO"] = q["PAIRMEAN"] * 0.8
        s_parts.append(q)
    return man, pd.concat(d_parts, ignore_index=True), pd.concat(s_parts, ignore_index=True)


def test_population_sizes_on_real_keys(fake):
    man, logd, logsf = fake
    sizes = {"COH-D-C3": 2425, "COH-SF-C3": 1802, "COH-D-C3M": 2208, "COH-D-ALL": 3359, "COH-SF-ANY": 2114,
             "COH-SF-C1": 2067, "COH-SF-NDSM": 1984, "COH-SC009-D": 1270, "COH-SC009-SF": 882,
             "COH-NOSC009-D": 1155, "COH-NOSC009-SF": 920, "COH-D-C3@NODMD": 2355, "COH-SF-C3@NODMD": 1742}
    for pop, n in sizes.items():
        t = logsf if "ia" in man[pop].columns else logd
        sel = RP.select_population(t, man[pop], pop)
        assert (sel.groupby("split_seed").size() == n).all(), pop
    Fd = I.Frame(RP.select_population(logd, man["COH-D-C3"]))
    Fs = I.Frame(RP.select_population(logsf, man["COH-SF-C3"]))
    for unit, n in (("chemotype", 38), ("publication", 39)):
        assert Fd.clusters(unit) == Fs.clusters(unit) and len(Fd.clusters(unit)) == n
    assert Fd.L == 79 and Fs.L == 79


def test_registered_machinery_runs_at_real_scale(fake):
    man, logd, logsf = fake
    Fd = I.Frame(RP.select_population(logd, man["COH-D-C3"]), name="COH-D-C3")
    Fs = I.Frame(RP.select_population(logsf, man["COH-SF-C3"]), name="COH-SF-C3")
    t0 = time.time()
    comps = []
    for y in R.H1_COMPARATORS:
        ev = I.contrast_evidence(Fd, "A", y, replicates=1000)
        ev["design_points"] = {d: None for d in R.OTHER_DESIGNS}
        comps.append(R.rule_c(ev))
        assert 0 <= ev["p_wcr"] <= 1 and ev["wcr_weights"] == "rademacher" and ev["chem_n_clusters"] == 38
    v = R.verdict_h(comps)
    assert v["verdict"] in R.VERDICT_ORDER and v["verdict"] not in ("NA", "PASS", "ESTIMAND_INCONSISTENT")
    ev = I.contrast_evidence(Fs, "B", "G14_ZERO", replicates=1000)
    assert np.isfinite(ev["delta_r2_bal"]) and np.isfinite(ev["skill_flat_X"])
    th = I.threshold_evidence(Fd, "A", stat="MET08", replicates=1000)
    assert R.rule_t(th, 0.4)["class"] in R.CLASS_ORDER
    assert time.time() - t0 < 120
