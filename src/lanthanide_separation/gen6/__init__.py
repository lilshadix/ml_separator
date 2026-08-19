"""gen6 — the diversity generation.

Gen2–Gen5 asked *which model / which descriptors*.  Every answer landed in the
same place: on a genuinely new ligand the response **shape** is partly
predictable and the absolute **level** is not, and no amount of extra descriptor
or model capacity moved that (five 3D studies, one extended-descriptor study and
one selectivity-scale study all returned negative or split-sensitive).

Gen6 asks a different question — *what information is missing* — and its primary
hypothesis is that the binding constraint is the **chemical diversity of the
training cohort**, not model capacity.  The evidence that motivates it: the
``min_rows_per_extractant = 10`` eligibility rule keeps 91 of 190 extractants and
throws away disproportionately the unusual chemotypes; relaxing it to 3 costs
+7.5 % rows and buys +77 % ECFP clusters and +98 % Tanimoto super-clusters.

Modules
-------
``chemistry``   frozen all-190 chemistry map: identity, fingerprints, clusters,
                super-clusters, nearest-neighbour matrix, donor census, families.
``provenance``  what experimental provenance can and cannot be reconstructed,
                audited honestly; never guessed.
``cohorts``     BASE/EXPANDED membership as *row masks on one shared cohort*, so
                the two arms are byte-identical everywhere except the training
                rows, and acquisition policies for the learning curve.
``metrics``     level/shape/offset decomposition, hard-chemotype endpoints,
                OOD statistics and the chemistry-unit bootstrap.
``manifest``    run manifest: dataset/code/artifact hashes, split definition,
                validation report and ``_SUCCESS.json``.

Nothing here edits or re-runs a Gen2–Gen5 artifact; the older harness
(:mod:`lanthanide_separation.levels`) is imported and reused verbatim so that a
gen6 run of a gen5 arm reproduces the gen5 number exactly.
"""

from __future__ import annotations

GEN6_LAYER = "gen6_diversity"
