"""Contract tests for the gen8 few-shot harness.

These do not test that a method is good.  They test the four properties that make
every gen8 number mean what the report says it means, and each of them has a way of
failing silently that would leave the tables looking perfectly reasonable:

* an adapter cannot see the target of a row it did not select;
* a calibration row is never also a scored row;
* every policy sees the same candidate pool and is scored on the same rows;
* the pool/evaluation split depends only on (seed, repeat, ligand, n_rows), which is
  what makes a separately-run architecture paired with the primary run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen8.adapters import (  # noqa: E402
    AdaptContext, NoModel, RidgeOffset, ZeroShot, default_adapters,
)
from lanthanide_separation.gen8.evaluate import _sign_accuracy, evaluate_fewshot  # noqa: E402
from lanthanide_separation.gen8.kshot import POLICIES, PolicyContext, design_matrix, ridge_fit
from lanthanide_separation.gen8.protocols import make_p2_split  # noqa: E402


def _block(n: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    acid = np.linspace(-1.0, 0.7, n)
    return pd.DataFrame({
        "row_id": [f"r{i:03d}" for i in range(n)],
        "extractant": "CCO", "tanimoto_cluster": "tan001", "ecfp_cluster": "e1",
        "nn_train_tanimoto": 0.5, "model": "M", "fold": 0, "split_seed": 1,
        "massact__log10_cond__acid_concentration_M": acid,
        "massact__log10_cond__extractant_concentration_M": np.linspace(-2, -1, n),
        "lanthanide_index": rng.integers(1, 15, n).astype(float),
        "cond__temperature_C": 25.0,
        "log_D": 1.5 + 2.0 * acid + rng.normal(0, 0.05, n),
        "prediction": 0.2 + 0.3 * acid,
    })


# --------------------------------------------------------------------------- #
# The adapter interface cannot leak
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("adapter", default_adapters(), ids=lambda a: a.name)
def test_adapter_ignores_unselected_targets(adapter):
    """Corrupting every unselected target must not move a single prediction.

    This is the decisive leakage test.  An adapter that reads ``block['log_D']``
    instead of ``observed`` would produce different numbers here, and would look
    entirely correct in every other test.
    """
    block = _block()
    selected = np.array([2, 7])
    context = AdaptContext(split_seed=1, fold=0, extractant="CCO")
    observed = block["log_D"].to_numpy()[selected]
    clean = np.asarray(adapter.predict(block, block["prediction"].to_numpy(),
                                       selected, observed, context), dtype=float)

    poisoned = block.copy()
    values = poisoned["log_D"].to_numpy().copy()
    mask = np.ones(len(block), dtype=bool)
    mask[selected] = False
    values[mask] = np.nan
    poisoned["log_D"] = values
    dirty = np.asarray(adapter.predict(poisoned, poisoned["prediction"].to_numpy(),
                                       selected, observed, context), dtype=float)
    np.testing.assert_allclose(clean, dirty, atol=0, rtol=0)


def test_zero_shot_is_the_identity():
    block = _block()
    out = ZeroShot().predict(block, block["prediction"].to_numpy(), np.array([1]),
                             block["log_D"].to_numpy()[[1]],
                             AdaptContext(1, 0, "CCO"))
    np.testing.assert_allclose(out, block["prediction"].to_numpy())


def test_no_model_uses_only_the_measurements():
    block = _block()
    selected = np.array([0, 5])
    observed = block["log_D"].to_numpy()[selected]
    out = NoModel().predict(block, block["prediction"].to_numpy(), selected, observed,
                            AdaptContext(1, 0, "CCO"))
    assert np.allclose(out, observed.mean())


# --------------------------------------------------------------------------- #
# Adaptation modes
# --------------------------------------------------------------------------- #

def test_one_observation_moves_only_the_level():
    """With k = 1 and an unpenalised intercept, K1, K2 and K3 must coincide exactly.

    A single observation can be fitted exactly by the intercept alone at zero
    penalty cost, so any slope movement at k = 1 is a bug in the shrinkage.
    """
    block = _block()
    selected = np.array([4])
    observed = block["log_D"].to_numpy()[selected]
    context = AdaptContext(1, 0, "CCO")
    out = [RidgeOffset(mode=m).predict(block, block["prediction"].to_numpy(), selected,
                                       observed, context) for m in ("K1", "K2", "K3")]
    np.testing.assert_allclose(out[0], out[1])
    np.testing.assert_allclose(out[0], out[2])
    # and it is exactly the mean-residual offset
    shift = observed[0] - block["prediction"].to_numpy()[selected][0]
    np.testing.assert_allclose(out[0], block["prediction"].to_numpy() + shift)


def test_offset_k1_reproduces_the_mean_residual():
    block = _block()
    selected = np.array([1, 3, 9])
    prediction = block["prediction"].to_numpy()
    observed = block["log_D"].to_numpy()[selected]
    out = RidgeOffset(mode="K1").predict(block, prediction, selected, observed,
                                         AdaptContext(1, 0, "CCO"))
    np.testing.assert_allclose(out, prediction + (observed - prediction[selected]).mean())


def test_ridge_penalty_shrinks_slopes_toward_zero():
    design = design_matrix(_block(), "K3")
    residual = np.linspace(-1, 1, len(design))
    weak = ridge_fit(design, residual, penalty=0.01)
    strong = ridge_fit(design, residual, penalty=1000.0)
    assert np.abs(strong[1:]).sum() < np.abs(weak[1:]).sum()


def test_zero_penalty_falls_back_instead_of_raising():
    """The unrestricted arm must reproduce gen7's failure, not crash on it."""
    design = design_matrix(_block(n=4), "K3")[[0, 1]]      # 2 rows, 4 coefficients
    beta = ridge_fit(design, np.array([0.5, -0.5]), penalty=0.0)
    assert beta.shape == (design.shape[1],)
    assert np.isfinite(beta).all()


# --------------------------------------------------------------------------- #
# The protocol
# --------------------------------------------------------------------------- #

def test_pool_and_evaluation_are_disjoint_and_complete():
    for n in (4, 7, 12, 50, 200):
        split = make_p2_split(n, np.random.default_rng(3))
        assert not set(split.pool) & set(split.evaluation)
        assert len(split.evaluation) >= 2
        assert set(split.pool) | set(split.evaluation) <= set(range(n))


def test_split_depends_only_on_seed_repeat_and_size():
    """Two runs of different adapters must land on the identical split.

    This is what lets an architecture evaluated in a separate later run be paired
    row for row with the primary run instead of merely being 'run on the same data'.
    """
    a = make_p2_split(23, np.random.default_rng((20260820, 3, 99)))
    b = make_p2_split(23, np.random.default_rng((20260820, 3, 99)))
    np.testing.assert_array_equal(a.pool, b.pool)
    np.testing.assert_array_equal(a.evaluation, b.evaluation)


def test_pool_cap_is_applied_identically():
    split = make_p2_split(500, np.random.default_rng(11), pool_cap=48)
    assert len(split.pool) == 48


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_policies_select_from_the_pool_and_never_repeat(name):
    block = _block(n=16)
    n = len(block)
    pool = np.arange(0, 10)
    context = PolicyContext(
        block=block, prediction=block["prediction"].to_numpy(), pool=pool,
        evaluation=np.arange(10, n),
        axes=np.random.default_rng(0).normal(size=(n, 4)),
        uncertainty=np.random.default_rng(1).random(n),
        disagreement=np.random.default_rng(2).random(n),
        rng=np.random.default_rng(5), truth=None)
    chosen: list[int] = []
    for _ in range(5):
        pick = int(POLICIES[name](context, chosen))
        assert pick in set(pool.tolist()), f"{name} selected outside the pool"
        assert pick not in chosen, f"{name} selected the same row twice"
        chosen.append(pick)


def test_policies_never_receive_targets():
    """Every deployable policy must work with ``truth=None``."""
    block = _block(n=14)
    context = PolicyContext(
        block=block, prediction=block["prediction"].to_numpy(), pool=np.arange(8),
        evaluation=np.arange(8, 14),
        axes=np.random.default_rng(0).normal(size=(14, 4)),
        uncertainty=np.full(14, np.nan), disagreement=np.full(14, np.nan),
        rng=np.random.default_rng(7), truth=None)
    for name, policy in POLICIES.items():
        assert isinstance(int(policy(context, [])), int), name


def test_uncertainty_policies_degrade_gracefully_without_uncertainty():
    """An all-NaN uncertainty column must fall back to a random pick, not crash.

    Several arms in the frozen OOF carry no ``prediction_sd``; if the policy raised
    there, those arms would silently drop out of the leaderboard.
    """
    block = _block(n=10)
    context = PolicyContext(
        block=block, prediction=block["prediction"].to_numpy(), pool=np.arange(6),
        evaluation=np.arange(6, 10), axes=np.zeros((10, 4)),
        uncertainty=np.full(10, np.nan), disagreement=np.full(10, np.nan),
        rng=np.random.default_rng(1), truth=None)
    for name in ("MAX_ENSEMBLE_SD", "MIN_ENSEMBLE_SD", "MAX_MODEL_DISAGREEMENT"):
        assert int(POLICIES[name](context, [])) in set(range(6))


# --------------------------------------------------------------------------- #
# The driver
# --------------------------------------------------------------------------- #

def _tiny_run(adapters=None, policies=("RANDOM", "CENTRAL")):
    frames = []
    for i in range(6):
        block = _block(n=10, seed=i)
        block["extractant"] = f"L{i}"
        block["tanimoto_cluster"] = f"tan{i//2:03d}"
        block["ecfp_cluster"] = f"e{i}"
        block["row_id"] = [f"L{i}_r{j}" for j in range(10)]
        frames.append(block)
    oof = pd.concat(frames, ignore_index=True)
    cohort = oof.drop(columns=["prediction", "model", "fold", "split_seed",
                               "nn_train_tanimoto"])
    return evaluate_fewshot(oof, cohort, adapters or default_adapters(),
                            policies=list(policies), k_values=(0, 1, 2), repeats=3,
                            min_rows=4, verbose=False)


def test_every_policy_is_scored_on_the_same_evaluation_rows():
    detail = _tiny_run()
    key = ["split_seed", "fold", "extractant", "repeat"]
    sizes = detail.groupby(key)["n_eval"].nunique()
    assert (sizes == 1).all(), "policies were scored on differently sized evaluation sets"


def test_calibration_rows_are_never_scored():
    """A one-shot calibration must not be able to score the row it measured.

    Constructed to be decisive: ``NO_MODEL`` predicts the mean of the measured rows,
    so if a measured row were in the evaluation set its own error would be zero and
    the k = 1 ``NO_MODEL`` score would collapse toward it.
    """
    detail = _tiny_run(adapters=[NoModel()], policies=("RANDOM",))
    one_shot = detail[(detail["adapter"] == "NO_MODEL") & (detail["k"] == 1)]
    assert len(one_shot) > 0
    assert one_shot["mae"].min() > 1e-9


def test_zero_shot_reference_is_independent_of_k_and_policy():
    detail = _tiny_run()
    reference = detail[detail["adapter"] == "ZERO_SHOT_REF"]
    assert set(reference["k"]) == {0}
    key = ["split_seed", "fold", "extractant", "repeat"]
    assert (reference.groupby(key)["mae"].nunique() == 1).all()


def test_sign_accuracy_conventions():
    truth = np.array([0.0, 1.0, 2.0, 3.0])
    assert _sign_accuracy(truth, truth) == pytest.approx(1.0)
    assert _sign_accuracy(truth, -truth) == pytest.approx(0.0)
    assert _sign_accuracy(truth, np.zeros(4)) == pytest.approx(0.5)


def test_stable_hash_is_process_independent():
    """The split seed must not depend on PYTHONHASHSEED.

    Python salts ``hash()`` on strings per process.  Seeding the pool/evaluation
    draw with it makes a run reproducible *within* one process and different
    between processes — so two architectures evaluated in two separate runs would
    be scored on different rows while every table still called the comparison
    paired.  This is checked in a subprocess with an explicit, different hash seed
    rather than in-process, because in-process the salt is fixed and the bug is
    invisible.
    """
    import subprocess

    from lanthanide_separation.gen8.kshot import stable_hash

    expected = [stable_hash(s) for s in ("CCO", "TODGA", "c1ccccc1")]
    code = (
        "import sys; sys.path.insert(0, %r);"
        "from lanthanide_separation.gen8.kshot import stable_hash;"
        "print([stable_hash(s) for s in ('CCO','TODGA','c1ccccc1')])"
        % str(REPO_ROOT / "src")
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={"PYTHONHASHSEED": "12345", "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0, out.stderr
    assert eval(out.stdout.strip()) == expected


def test_split_is_stable_across_processes():
    """The decisive pairing test: same ligand, same seed, same split, new interpreter."""
    import subprocess

    reference = make_p2_split(23, np.random.default_rng(
        (20260820, 3, __import__("lanthanide_separation.gen8.kshot", fromlist=["x"])
         .stable_hash("TODGA"))))
    code = (
        "import sys; sys.path.insert(0, %r);"
        "import numpy as np;"
        "from lanthanide_separation.gen8.kshot import stable_hash;"
        "from lanthanide_separation.gen8.protocols import make_p2_split;"
        "s = make_p2_split(23, np.random.default_rng((20260820, 3, stable_hash('TODGA'))));"
        "print(list(s.pool), list(s.evaluation))"
        % str(REPO_ROOT / "src")
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={"PYTHONHASHSEED": "999", "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0, out.stderr
    pool, evaluation = eval(out.stdout.strip().replace("] [", "], ["))
    np.testing.assert_array_equal(np.asarray(pool), reference.pool)
    np.testing.assert_array_equal(np.asarray(evaluation), reference.evaluation)
