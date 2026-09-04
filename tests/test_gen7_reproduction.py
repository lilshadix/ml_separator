"""The reproduction guarantee: gen7 must return gen6's numbers, not approximate them.

This is the test the whole generation rests on.  If it fails, no gen7 number can be
put in a table beside a gen6 number, and the failure modes are not hypothetical —
during the gen7 session it caught two:

* deriving the learner's ``random_state`` from the *split* seed instead of the fixed
  model seed, which moved macro MAE by 0.006;
* ``pip install tabpfn`` silently downgrading scikit-learn 1.9.0 -> 1.6.1 and pandas
  3.0.5 -> 2.3.3, which moved it by 0.0014 — small absolutely, fatal against gen7
  effects of 0.004-0.02.

It needs the real bundle, so it skips cleanly when the data is not present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lanthanide_separation.gen7.harness import DATASET_PATH, DEFAULT_SEEDS

#: gen6 Experiment A, EXPANDED arm, ``MC_lig2d_ext_massaction``
#: (``runs/gen6_expA_5seed/arm_metrics.csv``): macro, offset, shape.
GEN6_REFERENCE = {
    104729: (1.067993, 0.936996, 0.502044),
    130363: (1.047784, 0.845103, 0.516803),
}
#: The gen6 numbers are printed to six decimals; the harness must land inside that.
TOLERANCE = 5e-6

pytestmark = pytest.mark.skipif(
    not Path(DATASET_PATH).exists(), reason="the 3D-structure bundle is not present")


@pytest.fixture(scope="module")
def cohort():
    from lanthanide_separation.gen7.harness import load_cohort
    return load_cohort()


def test_cohort_matches_the_gen6_composition(cohort):
    frame = cohort.frame
    assert len(frame) == 5248
    assert frame["extractant"].nunique() == 152
    assert frame["ecfp_cluster"].nunique() == 131
    assert frame["tanimoto_cluster"].nunique() == 79
    assert frame["metal_symbol"].nunique() == 14


def test_fingerprint_ignores_feature_columns(cohort):
    """Adding a feature block must not change the evaluation contract."""
    from lanthanide_separation.gen7.harness import load_cohort
    bare = load_cohort(with_gen7_blocks=False)
    assert bare.fingerprint == cohort.fingerprint
    assert len(bare.frame.columns) < len(cohort.frame.columns)


@pytest.mark.parametrize("seed", sorted(GEN6_REFERENCE))
def test_reference_arm_reproduces_gen6_exactly(cohort, seed):
    from lanthanide_separation.gen7.contenders import LevelTree
    from lanthanide_separation.gen7.harness import evaluate_contender, score_oof

    oof = evaluate_contender(LevelTree(arm="MC_lig2d_ext_massaction"), cohort,
                             seeds=(seed,), verbose=False)
    row = score_oof(oof).iloc[0]
    macro, offset, shape = GEN6_REFERENCE[seed]
    assert row["macro_mae"] == pytest.approx(macro, abs=TOLERANCE)
    assert row["offset_mae"] == pytest.approx(offset, abs=TOLERANCE)
    assert row["shape_mae"] == pytest.approx(shape, abs=TOLERANCE)


def test_extra_blocks_do_not_move_an_unrelated_arm(cohort):
    """The gen7 blocks are additive; an arm that does not ask for them must be
    bit-identical with and without."""
    import numpy as np
    from lanthanide_separation.gen7.contenders import LevelTree
    from lanthanide_separation.gen7.harness import evaluate_contender, load_cohort

    bare = load_cohort(with_gen7_blocks=False)
    seed = DEFAULT_SEEDS[0]
    a = evaluate_contender(LevelTree(arm="MC_donors"), bare, seeds=(seed,), verbose=False)
    b = evaluate_contender(LevelTree(arm="MC_donors"), cohort, seeds=(seed,), verbose=False)
    merged = a[["row_id", "prediction"]].merge(
        b[["row_id", "prediction"]], on="row_id", suffixes=("_a", "_b"))
    assert len(merged) == len(a)
    assert np.abs(merged["prediction_a"] - merged["prediction_b"]).max() < 1e-9
