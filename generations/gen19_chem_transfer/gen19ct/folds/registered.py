"""``folds/registered.py`` -- helpers that pin the registered hold-out rules of ``preregistration_draft.md``.

Nothing here is fitted and nothing reads ``log_D``.  The functions are shared by ``scripts/g19_feasibility.py``
(which counts what the rules imply) and by the Phase C fold builders (which must apply them):

* :func:`v6_system_set` -- the V6 candidate systems (>= r Pr(III) and Nd(III) rows inside publications that
  measured both, plus >= L other Ln(III));
* :func:`v6_target_mask` -- ``V6_TARGET_ROWS``: Pr and Nd rows (known state and X(?)) of the V6 systems and of
  every system sharing a component with one of them.  They stay in training everywhere and are scored only in
  the single confirmation run;
* :func:`assert_not_scored` -- the guard every discovery or inner scoring index must pass;
* :func:`assign_halves` -- the registered selection / confirmation split of V5 systems, V1 groups and V2
  states (ladder decisions and claim freezing use the selection half only);
* :func:`parent_component_map` -- component SMILES -> stereo-free, salt-free parent key, the sensitivity
  reading of "shares a component".
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from gen19ct import paths
from gen19ct.chemistry import metals as MET
from gen19ct.chemistry import support_graph as SG

#: salt of the registered selection / confirmation split; changing it changes the split
HALF_SALT = "gen19-selection-confirmation-v1"
#: the registered V6 setting (min Pr and Nd rows in common publications, min other Ln(III))
V6_PRIMARY: tuple[int, int] = (5, 2)
PRND = ("Pr", "Nd")


def v6_system_set(frame: pd.DataFrame, min_rows: int = V6_PRIMARY[0], min_other_ln: int = V6_PRIMARY[1]) -> set[str]:
    """Systems with >= ``min_rows`` Pr(III) and Nd(III) rows inside publications that measured both, plus
    >= ``min_other_ln`` other Ln(III) states under the system (publication = ``g19_publication_id``)."""
    known = frame[frame[SG.METAL_COL].notna()]
    ln3 = {f"{e}(III)" for e in MET.LANTHANIDES}
    out = set()
    for key, g in known.groupby(SG.SYSTEM_COL):
        pr, nd = g[g[SG.METAL_COL] == "Pr(III)"], g[g[SG.METAL_COL] == "Nd(III)"]
        common = set(pr[SG.PUB_COL]) & set(nd[SG.PUB_COL])
        others = set(g[SG.METAL_COL]) & ln3 - {"Pr(III)", "Nd(III)"}
        if (pr[SG.PUB_COL].isin(common).sum() >= min_rows and nd[SG.PUB_COL].isin(common).sum() >= min_rows
                and len(others) >= min_other_ln):
            out.add(key)
    return out


def v6_target_mask(frame: pd.DataFrame, v6_systems: Iterable[str],
                   component_map: Mapping[str, str] | None = None) -> pd.Series:
    """``V6_TARGET_ROWS``: Pr and Nd rows (known state and X(?)) in the V6 systems and in every system sharing a
    component with one of them (boolean Series on ``frame.index``)."""
    keys = frame[SG.SYSTEM_COL].dropna().unique()
    touched = set(v6_systems)
    for k in list(touched):
        touched |= SG.systems_sharing_component(keys, k, component_map)
    return frame[SG.ELEMENT_COL].isin(PRND) & frame[SG.SYSTEM_COL].isin(touched)


def assert_not_scored(scoring_index: Iterable[Any], v6_mask: pd.Series, what: str = "scoring set") -> None:
    """Raise ``AssertionError`` when a discovery / inner scoring index contains a ``V6_TARGET_ROWS`` row."""
    idx = pd.Index(list(scoring_index))
    bad = idx.intersection(v6_mask.index[v6_mask.to_numpy(dtype=bool)])
    if len(bad):
        raise AssertionError(f"{what} scores {len(bad)} V6_TARGET_ROWS row(s); first: {list(bad[:5])}")


def assign_halves(weights: Mapping[Any, float], salt: str = HALF_SALT,
                  strata: Mapping[Any, str] | None = None) -> dict[Any, str]:
    """Registered selection ('S') / confirmation ('C') split.  Units are taken heaviest first (ties by
    ``sha256(salt|unit)``) and each goes to the lighter half of its stratum; a tie goes to the half holding
    fewer units overall, then to 'S'.  Deterministic and independent of any target value."""
    strata = strata or {u: "all" for u in weights}
    order = sorted(weights, key=lambda u: (-float(weights[u]), hashlib.sha256(f"{salt}|{u}".encode()).hexdigest()))
    load: dict[str, dict[str, float]] = defaultdict(lambda: {"S": 0.0, "C": 0.0})
    units = {"S": 0, "C": 0}
    out: dict[Any, str] = {}
    for u in order:
        st = strata[u]
        if load[st]["S"] != load[st]["C"]:
            half = "S" if load[st]["S"] < load[st]["C"] else "C"
        else:
            half = "S" if units["S"] <= units["C"] else "C"
        out[u] = half
        load[st][half] += float(weights[u])
        units[half] += 1
    return out


def parent_component_map(components_csv: Path | None = None) -> dict[str, str]:
    """Component SMILES -> ``parent:<stereo-free, salt-free InChIKey block>`` from ``extractant_components.csv``."""
    comps = pd.read_csv(components_csv or paths.DESCRIPTORS_DIR / "extractant_components.csv")
    st = comps[(comps["record_type"] == "STRUCTURE") & comps["parent_connectivity_key"].notna()]
    return dict(zip(st["smiles_canonical"], "parent:" + st["parent_connectivity_key"]))
