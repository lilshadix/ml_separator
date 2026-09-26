"""``scripts/g19_export_confirmation_views.py`` -- the view files of the finished confirmation run (task X finding,
2026-09-26: the report and F12 read ``evaluation/confirmation/v6_{systems,rows,pairs}.csv``, which no stage wrote).

* the gate: without ``decisions/confirmation.json`` the script refuses (exit 2) and writes nothing -- it is never an
  early look at a confirmation record;
* ``systems_table`` is ``s2.s2a.sign.per_system`` verbatim, with the report's column names, and empty-with-columns
  when S2 has no sign block;
* ``seed_mean_rows`` / ``seed_mean_pairs`` average the prediction columns over the seed indices and keep the observed
  values and the row attributes untouched, counting the seed indices that scored each unit;
* F12 draws the run's own per-system medians when ``systems`` is given and marks the source in its data CSV.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gen19ct import paths
from gen19ct.evaluation import figures as FG

SCRIPTS = paths.G19_ROOT / "scripts"


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


BODY = {"s2": {"status": "COMPLETE", "s2a": {"sign": {"per_system": [
    {"system": "sysA", "observed_median_logsf": 0.31, "predicted_median_logsf": 0.20, "sign_agrees": True},
    {"system": "sysB", "observed_median_logsf": -0.12, "predicted_median_logsf": 0.05, "sign_agrees": False}]}}},
    "seed_indices": [1, 2]}


def test_the_exporter_refuses_before_the_run_has_scored(tmp_path: Path, capsys) -> None:
    ex = _load("g19_export_confirmation_views")
    assert ex.main(["--out-root", str(tmp_path), "--no-manifest"]) == 2
    assert "refused" in capsys.readouterr().out
    assert not (tmp_path / "evaluation").exists()


def test_systems_table_is_the_sign_block_verbatim() -> None:
    ex = _load("g19_export_confirmation_views")
    t = ex.systems_table(BODY)
    assert list(t.columns) == ["system", "observed_median_logsf", "predicted_median_logsf", "sign_agrees", "s2_status", "label"]
    assert t["system"].tolist() == ["sysA", "sysB"] and t["sign_agrees"].tolist() == [True, False]
    assert (t["s2_status"] == "COMPLETE").all() and (t["label"] == ex.LABEL).all()
    empty = ex.systems_table({"s2": {"status": "INCOMPLETE"}})
    assert len(empty) == 0 and list(empty.columns)[:4] == ["system", "observed_median_logsf", "predicted_median_logsf", "sign_agrees"]


def _rows(i: int, shift: float) -> pd.DataFrame:
    return pd.DataFrame({"row_id": ["r1", "r2"], "system": ["sysA", "sysA"], "metal_state": ["Pr(III)", "Nd(III)"],
                         "log_D": [0.5, 0.9], "mean_logD": [0.4 + shift, 1.0 + shift], "lower_80": [0.1 + shift, 0.7 + shift],
                         "upper_80": [0.7 + shift, 1.3 + shift], "seed_index": i})


def _pairs(i: int, shift: float) -> pd.DataFrame:
    return pd.DataFrame({"row_id_a": ["r2"], "row_id_b": ["r1"], "system": ["sysA"], "fold": ["v6_prnd__abc"],
                         "acid_stratum": ["HNO3"], "observed_logsf": [0.4], "predicted_logsf": [0.6 + shift], "seed_index": i})


def test_seed_means_average_predictions_and_keep_observations(tmp_path: Path) -> None:
    ex = _load("g19_export_confirmation_views")
    rows = ex.seed_mean_rows([_rows(1, 0.0), _rows(2, 0.2)])
    assert list(rows.columns) == [*ex.ROW_COLS, "n_seeds", "label"]
    r1 = rows.set_index("row_id").loc["r1"]
    assert r1["log_D"] == 0.5 and r1["metal_state"] == "Pr(III)" and r1["n_seeds"] == 2
    assert np.isclose(r1["mean_logD"], 0.5) and np.isclose(r1["lower_80"], 0.2) and np.isclose(r1["upper_80"], 0.8)
    pairs = ex.seed_mean_pairs([_pairs(1, 0.0), _pairs(2, 0.2)])
    assert list(pairs.columns) == [*ex.PAIR_COLS, "n_seeds", "label"]
    assert pairs.loc[0, "observed_logsf"] == 0.4 and np.isclose(pairs.loc[0, "predicted_logsf"], 0.7) and pairs.loc[0, "n_seeds"] == 2
    assert len(ex.seed_mean_rows([])) == 0 and len(ex.seed_mean_pairs([])) == 0
    # the three view frames feed F12 as written, and the figure records which source its right panel drew
    systems = ex.systems_table(BODY)
    res = FG.fig12_prnd_reconstruction(rows, pairs, tmp_path, inputs=["x"], arm="M2", systems=systems)
    assert res.status == "written" and Path(res.path).exists()
    src = [d for d in res.stats["panels"] if d.get("panel") == "pairs_source"]
    assert src and "v6_systems.csv" in src[0]["source"]
    drawn = [d for d in res.stats["panels"] if d.get("panel") == "pairs"]
    assert {d["system"] for d in drawn} == {"sysA", "sysB"} and [d["sign_agrees"] for d in drawn] == [True, False]
    # without the systems file the panel falls back to medians over the pairs
    res2 = FG.fig12_prnd_reconstruction(rows, pairs, tmp_path, inputs=["x"], arm="M2")
    src2 = [d for d in res2.stats["panels"] if d.get("panel") == "pairs_source"]
    assert src2 and "v6_pairs.csv" in src2[0]["source"]


@pytest.mark.slow
def test_the_exporter_runs_on_a_rehearsed_confirmation_root(tmp_path: Path, monkeypatch) -> None:
    """The whole script on ``test_confirmation_rehearsal``'s synthetic finished run: the records are read through the
    runner's verified reader against the rehearsal's own registry, paired by the run's own builder, and the three files
    come out with the columns F12 / F13 / Q3 read."""
    import json
    import shutil

    from gen19ct.evaluation import registry as REG
    HERE = paths.G19_ROOT / "tests"
    monkeypatch.syspath_prepend(str(HERE))
    RS = importlib.import_module("confirmation_rehearsal_support")
    RC = _load("g19_run_confirmation")
    ex = _load("g19_export_confirmation_views")
    # a finished run: rehearse main() once (the slow rehearsal test does exactly this)
    corpus_dir = RS.make_corpus_dir(tmp_path / "corpus")
    root = tmp_path / "run"
    store_path = tmp_path / "store" / "fake_confirmation_seeds.json"
    RS.write_run_skeleton(root, store_path)
    reg_path = root / "manifests" / "digest_registry.json"
    monkeypatch.setattr(REG, "registry_path", lambda r=None, _p=reg_path: _p)
    code = RC.code_digest()["combined"]
    REG.register_stage(RC.STAGE if hasattr(RC, "STAGE") else "confirmation", below_footer_sha256="a" * 64, code_digest=code,
                       git_head=None, addenda_count=8, note="rehearsal", path=reg_path)
    corpus, attrs = RS.build_corpus(corpus_dir)
    monkeypatch.setattr(paths, "MANIFESTS_DIR", tmp_path / "manifests_out")
    monkeypatch.setattr(RC, "confirmation_corpus", lambda: corpus)
    monkeypatch.setattr(RC, "pair_attributes", lambda: attrs)
    monkeypatch.setattr(RC, "load_fold_corpus", lambda: RS.fold_corpus_stub(corpus.frame))
    monkeypatch.setattr(RC, "build_v5_batched", RS.build_v5_batched)
    monkeypatch.setattr(RC, "build_v5pair_batched", RS.build_v5pair_batched)
    monkeypatch.setattr(RC, "build_v6_folds", RS.build_v6_folds)
    monkeypatch.setattr(RC, "_init_conf_worker", RS.init)
    monkeypatch.setattr(RC, "INCOMPLETE_RECORD_SETS", set())
    import dataclasses

    from gen19ct.folds import cell_holdout as CH
    monkeypatch.setitem(CH.VARIANTS, "strict", dataclasses.replace(CH.VARIANTS["strict"], thresholds=CH.PRIMARY))
    monkeypatch.setenv("G19_REHEARSAL_REGISTRY", str(reg_path))
    monkeypatch.setenv("G19_REHEARSAL_CORPUS", str(corpus_dir))
    SEAL = RS.seal()
    argv = ["--out-root", str(root), "--seed-store", str(store_path), "--workers", "2", "--expect-addenda", "8"]
    assert RC.main(argv, check=lambda: 0, digests=RS.gate_digests, seal=SEAL, registry_path=reg_path) == 0
    body = json.loads((root / "evaluation/confirmation/decisions/confirmation.json").read_text(encoding="utf-8"))
    # the exporter on that root
    assert ex.main(["--out-root", str(root), "--no-manifest"]) == 0
    d = root / "evaluation" / "confirmation"
    systems = pd.read_csv(d / "v6_systems.csv")
    rows = pd.read_csv(d / "v6_rows.csv")
    pairs = pd.read_csv(d / "v6_pairs.csv")
    assert len(systems) == len(body["s2"]["s2a"]["sign"]["per_system"])
    assert {"system", "metal_state", "log_D", "mean_logD", "lower_80", "upper_80"} <= set(rows.columns) and len(rows) > 0
    assert {"system", "observed_logsf", "predicted_logsf"} <= set(pairs.columns) and len(pairs) > 0
    assert (rows["n_seeds"] == len(body["seed_indices"])).all() and (pairs["n_seeds"] == len(body["seed_indices"])).all()
    # the pooled pair count per system is the run's own n_pairs per seed
    assert len(pairs) == body["s2"]["per_seed"][str(body["seed_indices"][0])]["n_pairs"]
    shutil.rmtree(root, ignore_errors=True)
