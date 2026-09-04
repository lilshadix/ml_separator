"""gen10 — the finalization sprint.

gen9's finding was that the model had no coordinate for *where a requested point
sits inside the design the user proposed*.  Five handcrafted columns supplied one,
and every shape endpoint moved.  gen10 takes that seriously as a **representation**
rather than as a feature, and asks the question the gen9 report could not:

    if a prediction depends on the candidate design, how much does it move when
    the candidate design changes?

That is not a rhetorical worry.  ``rel__position`` is ``(v - min) / (max - min)``
over the query set, so **one extra candidate point outside the measured window
rescales every other point's coordinate**.  A deployed model whose answer for
0.03 M depends on whether the user also asked about 0.30 M is not a response
surface, it is a lookup keyed on the question.  :mod:`.consistency` measures it,
:mod:`.features` supplies representations designed not to have the problem, and
:mod:`.architectures` makes every arm answer ``predict(query)`` so the measurement
is possible at all.

The modules, in the order the sprint uses them:

``querycurves``
    curve geometry rebuilt from an arbitrary frame of conditions — the deployment
    object, as opposed to gen9's cohort-wide table.
``features``
    every axis representation gen10 tests, absolute and design-relative, with the
    robustness of each to a changed query set stated up front.
``architectures``
    the ``fit -> predict(query)`` contract, the gen9 arms re-expressed in it (and
    asserted identical), and the level+shape family.
``consistency``
    the query-set consistency benchmark: nested windows, grid density, window
    extension, permutation, sparse grids, shifted windows, decoys.
``runner``
    gen7's evaluation contract, unchanged, driving a ``QueryPredictor``.
"""

from __future__ import annotations

#: Every gen10 artefact lands under this run directory.
RUN_NAME = "gen10_final"

__all__ = ["RUN_NAME"]
