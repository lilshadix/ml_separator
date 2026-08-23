"""gen7 — can architecture break the gen6 zero-shot ceiling?

gen6 settled that *chemical coverage* helps.  gen7 asks the complementary
question: with the rows we already own, how much of the remaining
held-out-chemotype error is removable by a better representation, objective,
decomposition or learner?

Everything in this package evaluates on **byte-identical test rows**: one shared
cohort built once at ``min_cells = 3``, folds that hold out whole Tanimoto-0.7
super-clusters, and the gen6 metric module unchanged.  A contender that changes
the cohort, the folds or the metric is not comparable and the runner refuses it.
"""

GEN7_LAYER = "gen7"

__all__ = ["GEN7_LAYER"]
