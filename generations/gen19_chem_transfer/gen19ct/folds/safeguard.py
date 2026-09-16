"""``folds/safeguard.py`` -- the nested-certificate safeguard sample (pre-registration section 2, "Leakage guard",
resolution of 2026-09-15).

``nested_certificate`` counts as "passes ``fold_isolation_check``" for inner splits of every design.  Safeguard: at
the start of discovery ``every_split`` is also run on 20 outer folds per design, drawn with
``numpy.random.default_rng(19)`` and stratified by metal category (Ln / An / other) where the design has cells; any
difference in residuals, quantiles or guard verdicts makes ``every_split`` mandatory for that design.  This module
draws the folds only; the comparison runs at discovery start.

Which designs (task X, findings VR-03 / VR-04 of 2026-09-15)
----------------------------------------------------------
The safeguard compares ``nested_certificate`` with ``every_split``, so it tests something only where an inner split
carries a certificate.  Under ``gen19ct.models.interface`` only the V5 inner design (``InnerCellCalibration``) builds
one; ``GroupKFoldCalibration`` (V1, V0) and ``InnerMetalCalibration`` (V2) never do, and ``ConformalWrapper._verify``
then checks the inner split itself, so both modes are identical by construction there.  A sample is therefore drawn for
every outer fold file that discovery fits with an arm tuned or calibrated with the V5 inner design:

* V5-primary, both populations: the exact leave-one-cell-out folds with a scored row (B6 / B6r0 inner tuning and the
  deterministic arms' multi-seed conformal calibration, section 15 resolution) and the heavy-arm batches of seed 104729
  (the seed on which the full section 7 inner design runs; the other discovery seeds use its first inner fold only,
  section 7 compute plan item 1);
* the V5 refit sensitivities whose certificate construction differs from the primary's -- cell-only
  (``component_aware=False``) and parent-structure (``component_map``) -- on their batches of seed 104729 (loose,
  strict and HNO3-only build the certificate exactly as the primary does and are not sampled separately);
* V5-P and V5-PAIR, both populations: the unbatched folds (closed-form arms, B6 / B6r0 included) and the heavy-arm
  batched folds of seed 104729 (section 3.1 resolution: tuned with the V5 inner design);

all in the selection half.  The V1 grouped folds and the V2 selection-half states keep their draw, labelled
``certificate_in_inner_splits: false`` / ``vacuous_under_current_code: true``: running the safeguard there compares a
check with itself unless a splitter of those designs gains a certificate.

Rule (:func:`draw_sample`):

* the population is the design's outer folds that discovery fits (:data:`REGISTERED_POPULATIONS`: seed filter, half
  filter and, for exact V5 folds, only folds with a scored row), sorted by fold id;
* a fresh ``numpy.random.default_rng(seed)`` per design;
* population <= n: every fold is taken and no draw is made;
* unstratified: ``rng.choice(N, size=n, replace=False)`` on the sorted population;
* stratified (folds carrying ``meta.cells``): the stratum of a fold is the metal category (:func:`metal_category`)
  held by the most of its cells, ties to the category with the fewest cells in the whole population, then the order
  Ln, An, other; the n draws are allocated to the strata by largest remainder of ``n * N_k / N`` (remainder ties in
  the order Ln, An, other), then every non-empty stratum that received none takes one from the stratum with the
  largest allocation (when n >= the number of non-empty strata); inside each stratum, in the order Ln, An, other,
  ``rng.choice(N_k, size=n_k, replace=False)`` on its sorted fold ids, all from the same generator.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from gen19ct.chemistry import support_graph as SG
from gen19ct.folds import io as FI

SAFEGUARD_SEED = 19
SAFEGUARD_N = 20
CATEGORY_ORDER: tuple[str, ...] = ("Ln", "An", "other")
KEY = "nested_certificate_safeguard_sample"
_V5_INNER = "InnerCellCalibration (V5 inner design; certificate = the inner cell's registered hiding on every MODEL row)"
#: the outer folds discovery fits, per fold file (section 7 compute plan; module docstring): seed filter, half filter,
#: stratify, only folds with a scored row, the inner splitter, and whether its inner splits carry a certificate
REGISTERED_POPULATIONS: dict[str, dict[str, Any]] = {
    "V5__primary__exact": {"seed": None, "half": "S", "stratify": True, "require_scored": True, "certificate": True,
                           "inner_splitter": _V5_INNER,
                           "population": "V5-primary exact leave-one-cell-out folds of the selection half with a scored "
                                         "row (B6 / B6r0 inner tuning; the deterministic arms' multi-seed conformal "
                                         "calibration)"},
    "V5__primary__batched": {"seed": 104729, "half": "S", "stratify": True, "require_scored": False,
                             "certificate": True, "inner_splitter": _V5_INNER,
                             "population": "V5-primary heavy-arm batches of seed 104729 in the selection half"},
    "V5__cell_only__batched": {"seed": 104729, "half": "S", "stratify": True, "require_scored": False,
                               "certificate": True, "inner_splitter": _V5_INNER + ", component_aware=False",
                               "population": "V5 cell-only refit-sensitivity batches of seed 104729 in the selection "
                                             "half (certificate without component-aware hiding)"},
    "V5__parent_structure__batched": {"seed": 104729, "half": "S", "stratify": True, "require_scored": False,
                                      "certificate": True,
                                      "inner_splitter": _V5_INNER + ", component_map=registered.parent_component_map()",
                                      "population": "V5 parent-structure refit-sensitivity batches of seed 104729 in the "
                                                    "selection half (certificate through the parent-structure component "
                                                    "map)"},
    "V5P__base__cell_x_group": {"seed": None, "half": "S", "stratify": True, "require_scored": False,
                                "certificate": True, "inner_splitter": _V5_INNER,
                                "population": "V5-P unbatched cell x publication-group folds of the selection half "
                                              "(closed-form arms, B6 / B6r0 included)"},
    "V5P__base__batched": {"seed": 104729, "half": "S", "stratify": True, "require_scored": False, "certificate": True,
                           "inner_splitter": _V5_INNER,
                           "population": "V5-P heavy-arm batches of seed 104729 in the selection half (section 3.1 "
                                         "resolution)"},
    "V5PAIR__primary__cell_pair": {"seed": None, "half": "S", "stratify": True, "require_scored": False,
                                   "certificate": True, "inner_splitter": _V5_INNER,
                                   "population": "V5-PAIR unbatched cell-pair folds of the selection half (closed-form "
                                                 "arms, B6 / B6r0 included)"},
    "V5PAIR__primary__batched": {"seed": 104729, "half": "S", "stratify": True, "require_scored": False,
                                 "certificate": True, "inner_splitter": _V5_INNER,
                                 "population": "V5-PAIR heavy-arm batches of seed 104729 in the selection half (section "
                                               "3.1 resolution)"},
    "V1__copy__grouped10": {"seed": 104729, "half": None, "stratify": False, "require_scored": False,
                            "certificate": False, "inner_splitter": "GroupKFoldCalibration (V1 inner design)",
                            "population": "the ten grouped V1 outer folds of seed 104729"},
    "V2__element__exact": {"seed": None, "half": "S", "stratify": False, "require_scored": False, "certificate": False,
                           "inner_splitter": "InnerMetalCalibration (V2 inner design)",
                           "population": "the V2 selection-half metal states (registered element-level hiding)"},
}
VACUOUS_NOTE = ("vacuous under current code: this design's inner splitter sets no certificate, so ConformalWrapper "
                "checks every inner split directly in nested_certificate mode and the two modes are identical by "
                "construction; the draw is kept in case a splitter of this design gains a certificate")


def metal_category(state: str | None) -> str:
    """``Ln`` / ``An`` / ``other`` of a metal state label."""
    if state is None:
        return "other"
    cat = SG.metal_properties(str(state))["category"]
    return {"lanthanide": "Ln", "actinide": "An"}.get(cat, "other")


def fold_strata(folds: Sequence[FI.Fold]) -> dict[str, str]:
    """Stratum of every fold with ``meta.cells``: the plurality metal category of its cells (ties: the category
    with the fewest cells in the population, then :data:`CATEGORY_ORDER`)."""
    per = {f.fold_id: Counter(metal_category(c[0]) for c in f.meta["cells"]) for f in folds}
    pop = Counter()
    for c in per.values():
        pop.update(c)
    out = {}
    for fid, c in per.items():
        if not c:
            raise ValueError(f"{fid}: a fold without cells cannot be stratified")
        top = max(c.values())
        tied = [k for k in CATEGORY_ORDER if c.get(k, 0) == top]
        out[fid] = min(tied, key=lambda k: (pop.get(k, 0), CATEGORY_ORDER.index(k)))
    return out


def allocate(sizes: Mapping[str, int], n: int) -> dict[str, int]:
    """Largest-remainder allocation of ``n`` draws to strata of ``sizes`` (see the module docstring)."""
    total = int(sum(sizes.values()))
    if n >= total:
        return {k: int(sizes.get(k, 0)) for k in CATEGORY_ORDER}
    quota = {k: n * sizes.get(k, 0) / total for k in CATEGORY_ORDER}
    alloc = {k: int(np.floor(q)) for k, q in quota.items()}
    rest = n - sum(alloc.values())
    for k in sorted(CATEGORY_ORDER, key=lambda k: (-(quota[k] - alloc[k]), CATEGORY_ORDER.index(k)))[:rest]:
        alloc[k] += 1
    nonempty = [k for k in CATEGORY_ORDER if sizes.get(k, 0) > 0]
    if n >= len(nonempty):
        for k in nonempty:
            if alloc[k] == 0:
                donor = max(CATEGORY_ORDER, key=lambda d: (alloc[d], -CATEGORY_ORDER.index(d)))
                alloc[donor] -= 1
                alloc[k] += 1
    if sum(alloc.values()) != n or any(alloc[k] > sizes.get(k, 0) for k in CATEGORY_ORDER):
        raise AssertionError(f"allocation {alloc} of {n} over {dict(sizes)}")
    return alloc


def draw_sample(fold_ids: Iterable[str], strata: Mapping[str, str] | None = None, n: int = SAFEGUARD_N,
                seed: int = SAFEGUARD_SEED) -> dict[str, Any]:
    """Draw ``n`` of ``fold_ids`` (see the module docstring).  Returns the sorted drawn ids and the draw record."""
    pop = sorted(set(str(x) for x in fold_ids))
    N = len(pop)
    rng = np.random.default_rng(seed)
    rec: dict[str, Any] = {"rng": f"numpy.random.default_rng({seed})", "n_requested": int(n), "population_size": N,
                           "stratified": strata is not None}
    if strata is not None:
        missing = [f for f in pop if f not in strata]
        if missing:
            raise KeyError(f"no stratum for {missing[:3]}")
        by = {k: [f for f in pop if strata[f] == k] for k in CATEGORY_ORDER}
        sizes = {k: len(v) for k, v in by.items()}
        rec["strata_population"] = sizes
    if N <= n:
        chosen = pop
        rec["all_taken"] = True
        if strata is not None:
            rec["allocation"] = dict(rec["strata_population"])
    elif strata is None:
        chosen = [pop[i] for i in rng.choice(N, size=int(n), replace=False)]
        rec["all_taken"] = False
    else:
        alloc = allocate(sizes, int(n))
        chosen = []
        for k in CATEGORY_ORDER:
            if alloc[k]:
                chosen += [by[k][i] for i in rng.choice(sizes[k], size=alloc[k], replace=False)]
        rec["all_taken"] = False
        rec["allocation"] = alloc
    rec["fold_ids"] = sorted(chosen)
    rec["n_drawn"] = len(chosen)
    return rec


def safeguard_sample(folds: Sequence[FI.Fold], *, seed_filter: int | None = None, half: str | None = None,
                     stratify: bool = True, require_scored: bool = False, n: int = SAFEGUARD_N,
                     seed: int = SAFEGUARD_SEED) -> dict[str, Any]:
    """The safeguard sample of one design's outer folds (optionally restricted to one discovery seed, one half and the
    folds with a scored row).  Stratified by metal category only when ``stratify`` and every fold carries
    ``meta.cells``."""
    pop = [f for f in folds if (seed_filter is None or f.seed == seed_filter) and (half is None or f.half == half)
           and (not require_scored or f.scored_row_ids)]
    if not pop:
        raise ValueError("safeguard_sample: empty population")
    has_cells = all("cells" in f.meta and f.meta["cells"] for f in pop)
    strata = fold_strata(pop) if (stratify and has_cells) else None
    rec = draw_sample([f.fold_id for f in pop], strata, n=n, seed=seed)
    by_id = {f.fold_id: f for f in pop}
    rec["fold_hashes"] = {fid: by_id[fid].fold_hash for fid in rec["fold_ids"]}
    if strata is not None:
        rec["stratum_of_drawn_fold"] = {fid: strata[fid] for fid in rec["fold_ids"]}
    rec["filters"] = {"seed": seed_filter, "half": half, "require_scored": bool(require_scored)}
    return rec


def registered_safeguard_samples(folds_dir: Path | None = None, n: int = SAFEGUARD_N,
                                 seed: int = SAFEGUARD_SEED) -> dict[str, Any]:
    """The registered safeguard sample of every design in :data:`REGISTERED_POPULATIONS`, drawn from the written
    (hash-verified) fold files -- the ``folds/INDEX.json`` entry ``nested_certificate_safeguard_sample``."""
    out: dict[str, Any] = {
        "rule": ("pre-registration section 2 (resolution 2026-09-15): every_split is also run at discovery start on "
                 f"{n} outer folds per design drawn with numpy.random.default_rng({seed}), stratified by metal category "
                 "(Ln / An / other) where the design has cells; a difference in residuals, quantiles or guard verdicts "
                 "makes every_split mandatory for that design. Draw rule: gen19ct.folds.safeguard (module docstring); "
                 "the comparison itself runs at discovery start"),
        "which_designs": ("every outer fold file discovery fits with an arm whose inner splits carry a certificate (the V5 "
                          "inner design, InnerCellCalibration): V5-primary exact and seed-104729 batches, the cell-only and "
                          "parent-structure batches (their certificate construction differs), V5-P and V5-PAIR unbatched "
                          "and seed-104729 batched folds; selection half. The V1 and V2 draws are kept and labelled "
                          "vacuous (their splitters set no certificate). Task X, findings VR-03 / VR-04"),
        "seed": seed, "n_per_design": n, "designs": {}}
    for stem, spec in REGISTERED_POPULATIONS.items():
        folds = FI.read_design(stem, folds_dir)
        rec = safeguard_sample(folds, seed_filter=spec["seed"], half=spec["half"], stratify=spec["stratify"],
                               require_scored=spec["require_scored"], n=n, seed=seed)
        rec["population"] = spec["population"]
        rec["inner_splitter"] = spec["inner_splitter"]
        rec["certificate_in_inner_splits"] = bool(spec["certificate"])
        rec["vacuous_under_current_code"] = not spec["certificate"]
        if not spec["certificate"]:
            rec["note"] = VACUOUS_NOTE
        rec["design_hash"] = FI.design_hash(folds)
        out["designs"][stem] = rec
    return out
