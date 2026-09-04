"""gen11 — does information about *other metals* transfer to lanthanides?

gen10 closed the architecture question: on the frozen cohort no representation
change met the inclusion bar, and the remaining error is chemical coverage
(0.52 of 0.97 macro MAE is a per-ligand constant that structure does not
predict), thin shape evidence (25 ligands, 8 chemotypes) and data quality.

gen11 asks one question and does not wander from it:

    can the audited multi-metal archive — 40 metals, mostly actinides — improve
    zero-shot and few-shot prediction for **unseen lanthanide** chemistry,
    without touching the frozen gen10 benchmark?

The answer has to survive the way it is measured, so the module is built
control-first:

``overlap``
    every relationship between a gen10 cohort row and an archive record —
    exact record, folded duplicate, series, publication, ligand, chemotype —
    and the per-fold auxiliary masks those relationships imply.  Nothing may be
    trained on until this says it is safe.
``auxfeatures``
    the archive rendered into the frozen 2,146-column feature contract
    (``METAL`` + ``COND`` + ``ECFP`` + ``MASSACTION`` + ``RECOVERED``).  Its
    correctness test is not a review: it must reproduce the frozen cohort's own
    columns bit-for-bit from the archive, or auxiliary rows are not comparable
    to training rows and no arm means anything.
``metalrep``
    the metal block was a 14-entry lanthanide dict.  This audits what it does
    with 40 metals and supplies the honest alternatives, with explicit
    missingness rather than silent row loss.
``arms``
    A_CONTROL … G_RANDOM_AUX_MATCHED, the matched samplers and the weighting
    schemes, all fixed before any result is seen.
``transfer``
    the three mechanisms — joint training, auxiliary pretraining then
    lanthanide fitting, shared representation with a lanthanide-only head —
    kept apart because they can disagree.
``runner``
    gen10's evaluation contract, unchanged: same cohort, same folds, same
    seeds, same metrics.  Auxiliary rows enter training only; they are never
    scored and never redefine a test row.

The frozen benchmark is `GEN10`.  It stays the control until an arm beats it
under the stopping rule in the brief.
"""

from __future__ import annotations

#: Every gen11 artefact lands under this run directory.
RUN_NAME = "gen11_transfer"

__all__ = ["RUN_NAME"]
