"""gen9 — teach the global model the shape of a titration, and learn which
experiment to run first.

gen8 localised the remaining error twice over.  It showed that the deployment unit
is a prediction *plus one requested measurement*, and it showed that the model's
last structural defect is not a missing molecular representation but a flattened
response surface: the median unseen extractant titration is measured with slope
2.57 and predicted with slope 0.12.  gen9 attacks exactly those two things and
nothing else.

* :mod:`.curves` — within-curve training pairs, fold-local by construction.
* :mod:`.objective` — ``CurveBoost``: boosted trees under
  ``L_row + lambda_delta L_delta + lambda_span L_span``.
* :mod:`.train` — the gen7 contender wrapper, same cohort/folds/design matrix.
* :mod:`.metrics` — the shape endpoints macro MAE cannot express.
* :mod:`.acquisition` — learning ``argmin |r_i - median(r)|`` from observables.
* :mod:`.series_adapter` — hierarchical, series-local few-shot calibration.
* :mod:`.audit` — the pairing, row-accounting and leakage artefacts.
"""

from __future__ import annotations

__all__ = ["curves", "objective", "train", "metrics", "acquisition", "series_adapter", "audit"]
