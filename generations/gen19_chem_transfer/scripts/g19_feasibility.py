"""``scripts/g19_feasibility.py`` -- brief section 28 Phase B (feasibility), sections 11 (V0-V7), 15/30
(actinide overlap), 16 (support graph) and 19 (candidate extractants).

Descriptive corpus counts only: nothing is trained or fitted.  The one place ``log_D`` is read is the
Q13 variance decomposition, which is labelled DESCRIPTIVE.  Every threshold is an argument and every
answer is reported over a sensitivity grid, never at one hand-picked number.

Population: MODEL rows (``g19_tier == "MODEL"``, 12,411).  "Metal state" = ``g19_metal_state``
(``Nd(III)``); rows with no recorded oxidation state (``Nd(?)``) never form a V5 / V2 cell of their own
but are counted as alias rows that a fold must hide with the cell (``leakage_metal_alias_risk.csv``).
"System" = ``extractant_system_key``; family / mechanism labels come from
``descriptors/extractant_systems.csv`` (system level, majority vote of the builder).  "Comparable
pair" = two rows of different known metal states with the same publication, the same system and an
identical ``normalize.condition_key`` (6 s.f.; the ``_nm`` variant drops metal concentration from the key).

Outputs (``data_audit/``): ``feasibility.json`` and ``feasibility_*.csv``; figure ``figures/F03_density_by_family.png``.

V5 cells are described under the REGISTERED hiding (``support_graph.hide_cell(component_aware=True)``): one
state-level rule applied in the cell's own system and in every other system sharing a component -- rows of the
hidden metal state plus the element's X(?) rows; other known states of the element stay in training.
Also written: state-, X(?)- and element-level and stereo-free parent-structure sharing counts, the acid media
pooled in each cell and an HNO3-only eligibility grid whose per-cell attributes are recomputed on HNO3 rows
alone, rows with an acidic co-extractant modifier, censoring-candidate rows
(exact-decade log D at a (study, system) floor or ceiling -- the only other place ``log_D`` is read, as a
value pattern, never as a target statistic), the V6 target rows, and the registered selection /
confirmation halves of the V5 systems, V1 groups and V2 states (``feasibility_halves.csv``).

    PYTHONIOENCODING=utf-8 PYTHONPATH=generations/gen19_chem_transfer \
        .venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_feasibility.py
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from gen19ct import paths  # noqa: E402
from gen19ct.chemistry import ligands as LIG  # noqa: E402
from gen19ct.chemistry import mechanisms as MECH  # noqa: E402
from gen19ct.chemistry import metals as MET  # noqa: E402
from gen19ct.chemistry import support_graph as SG  # noqa: E402
from gen19ct.data import load  # noqa: E402
from gen19ct.folds import registered as FR  # noqa: E402
from gen19ct.data import normalize as N  # noqa: E402
from gen19ct.manifest import Run, write_csv, write_json  # noqa: E402

NAME = "g19_feasibility"
OUT = paths.DATA_AUDIT_DIR
FIG_PATH = paths.FIGURES_DIR / "F03_density_by_family.png"
SYSTEMS_CSV = paths.DESCRIPTORS_DIR / "extractant_systems.csv"
COMPONENTS_CSV = paths.DESCRIPTORS_DIR / "extractant_components.csv"
METALS_CSV = paths.DESCRIPTORS_DIR / "metals.csv"
NAMED_CSV = paths.DATA_AUDIT_DIR / "named_extractant_presence.csv"
PUB_COMPONENTS_CSV = paths.DATA_AUDIT_DIR / "leakage_publication_components.csv"

SYS, PUB, STATE, ELEM = SG.SYSTEM_COL, SG.PUB_COL, SG.METAL_COL, SG.ELEMENT_COL
LN, AN = MET.LANTHANIDES, MET.ACTINIDES
LN_FOCUS = ("La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd")
CE_TO_SM = ("Ce", "Pr", "Nd", "Pm", "Sm")
CONTIGUOUS_RUN = ("Ce", "Pr", "Nd", "Sm")          # Pm skipped (radioactive, essentially unmeasured)
ACIDIC_P_FAMILIES = ("phosphoric_acid", "phosphonic_acid_monoester", "phosphonic_acid", "phosphinic_acid",
                     "thiophosphorus_acid", "dithiophosphinic_acid")
NAMED_TARGETS = ("PC88A", "Cyanex 272", "D2EHPA", "TODGA")
HALF_SALT = FR.HALF_SALT
#: primary V5 / V1 / V2 settings the halves and the V6 carve-out are computed for
PRIMARY_V5 = ("g19_publication_id", 10, 1, 3)
V6_PRIMARY = FR.V6_PRIMARY
#: censoring candidates (P8): exact-decade log D at the minimum (<= -2) / maximum (>= 2) of its
#: (study, system) group, shared by >= 2 rows
FLOOR_MAX_LOG_D, CEILING_MIN_LOG_D, CENSOR_MIN_ROWS = -2.0, 2.0, 2
BRIEF_EXPERT = {
    "ACIDIC_CATION_EXCHANGE": "E1_acidic_cation_exchange", "MIXED_ACIDIC": "E1_acidic_cation_exchange",
    "NEUTRAL_SOLVATING": "E2_neutral_solvating", "SOFT_N_DONOR": "E2_neutral_solvating",
    "MIXED_NEUTRAL": "E2_neutral_solvating", "ION_PAIR_BASIC": "E3_ion_pair_basic", "CHELATING": "E4_chelating",
    "SYNERGISTIC": "E5_synergistic", "UNKNOWN": "none_unknown",
}
CATEGORY_ORDER = ("lanthanide", "actinide", "alkaline_earth", "transition_metal", "rare_earth_non_lanthanide",
                  "post_transition_metal")
#: reference categorical palette slots 1-6 (dataviz skill references/palette.md, light mode, adjacent-validated)
CATEGORY_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")


def ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in str(text).split(",") if x.strip())


def floats(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in str(text).split(",") if x.strip())


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sig", type=int, default=N.DEFAULT_SIG, help="significant figures of the condition key")
    ap.add_argument("--v5-k", default="5,10,20", help="V5 min rows per cell")
    ap.add_argument("--v5-p", default="1,2", help="V5 min publications per cell")
    ap.add_argument("--v5-m", default="2,3,5", help="V5 min other systems for the metal / other metals for the system")
    ap.add_argument("--v5-pub-basis", default="g19_publication_id,group_primary_source_doi",
                    help="publication definitions for the V5 p threshold")
    ap.add_argument("--span", default="3,5,10", help="Q2/Q3 span thresholds")
    ap.add_argument("--span-cell-rows", default="1,5", help="Q2/Q3 min rows for a cell to count as spanned")
    ap.add_argument("--v3-rows", default="20,50,100")
    ap.add_argument("--v3-states", default="3,5")
    ap.add_argument("--v3-family-members", default="1,2,3")
    ap.add_argument("--v4-systems", default="2,3,5")
    ap.add_argument("--v4-rows", default="100,200,500")
    ap.add_argument("--expert-rows", default="100,200,500")
    ap.add_argument("--v1-rows", default="10,20,50")
    ap.add_argument("--v2-rows", default="50,100,200")
    ap.add_argument("--v2-systems", default="3,5,10")
    ap.add_argument("--v6-rows", default="3,5,10", help="V6 min Pr and Nd rows each in common publications")
    ap.add_argument("--v6-other-ln", default="2,5")
    ap.add_argument("--v7-rows", default="5,10,20")
    ap.add_argument("--v7-frac", default="0.1,0.2,0.3")
    ap.add_argument("--radius-tol", type=float, default=SG.RADIUS_NEIGHBOUR_TOL_A)
    ap.add_argument("--connected-share", type=float, default=0.95,
                    help="giant-component row share above which the graph counts as connected")
    return ap.parse_args(argv)


# --------------------------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------------------------- #

def pipe(values) -> str:
    return "|".join(str(v) for v in values)


def state_sort_key(label: str) -> tuple:
    p = SG.metal_properties(label)
    return (p["Z"] if p["Z"] is not None else 999, p["ox"] if p["ox"] is not None else 99, label)


def sort_states(labels) -> list[str]:
    return sorted(set(labels), key=state_sort_key)


def element_of(label: str) -> str:
    return str(label).split("(")[0]


def label_of(key: str, systems: pd.DataFrame) -> str:
    if key in systems.index:
        return str(systems.loc[key, "component_canonical_names"])
    return str(key)[:40]


def components_of(key: str) -> set[str]:
    return set(str(key).split("|"))


def cramers_v(a: pd.Series, b: pd.Series) -> dict:
    tab = pd.crosstab(a, b)
    n = int(tab.to_numpy().sum())
    chi2 = float(stats.chi2_contingency(tab.to_numpy(), correction=False)[0])
    r, c = tab.shape
    v = math.sqrt(chi2 / (n * (min(r, c) - 1))) if min(r, c) > 1 else float("nan")
    phi2c = max(0.0, chi2 / n - (r - 1) * (c - 1) / (n - 1))
    rc, cc = r - (r - 1) ** 2 / (n - 1), c - (c - 1) ** 2 / (n - 1)
    vc = math.sqrt(phi2c / min(rc - 1, cc - 1)) if min(rc, cc) > 1 else float("nan")
    return {"n": n, "rows": r, "cols": c, "chi2": chi2, "cramers_v": v, "cramers_v_bias_corrected": vc}


# --------------------------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------------------------- #

def load_frame(sig: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = load.load_model_rows()
    systems = pd.read_csv(SYSTEMS_CSV).set_index("extractant_system_key", drop=False)
    cv = N.condition_vector(model)
    fr = SG.prepare_support_frame(model, systems=systems, cv=cv)
    fr["state_or_alias"] = [SG.state_label_or_alias(s, e) for s, e in zip(fr[STATE], fr[ELEM])]
    fr["ck"] = N.condition_key(cv, sig).to_numpy()
    fr["ck_nm"] = N.condition_key(cv, sig, exclude=("metal_concentration_M",)).to_numpy()
    fr["acid_M_log10_grid"] = cv["acid_M_log10_grid"].to_numpy()
    fr["log_D"] = pd.to_numeric(model["log_D"], errors="coerce").astype(float)
    pc = pd.read_csv(PUB_COMPONENTS_CSV)
    for col in ("group_corrected_doi", "group_primary_source_doi", "group_cross_publication_copy",
                "group_near_duplicate_key", "group_compilation_doi"):
        fr[col] = fr[PUB].map(pc.set_index("g19_publication_id")[col])
    if fr["group_primary_source_doi"].isna().any():
        raise RuntimeError("leakage_publication_components.csv does not cover every MODEL publication")
    # acidic co-extractant recorded in the phase-modifier slot (HDEHP with TODGA / DOHyA): a different chemistry
    comps = pd.read_csv(COMPONENTS_CSV)
    mech_smiles = comps[comps["record_type"] == "STRUCTURE"].set_index("smiles_canonical")["mechanism"].to_dict()
    mech_name = comps[comps["record_type"] == "NAME_ONLY"].set_index("canonical_name")["mechanism"].to_dict()

    def coextractant(components) -> bool:
        for c in LIG.iter_component_dicts(components):
            if c.get("role") != "phase_modifier":
                continue
            smi, nm = c.get("smiles_canonical"), c.get("name")
            mech = mech_smiles.get(str(smi)) if isinstance(smi, str) and smi.strip() else \
                mech_name.get(str(nm)) if isinstance(nm, str) else None
            if mech in MECH.ACIDIC_SET:
                return True
        return False
    fr["acidic_coextractant_modifier"] = [coextractant(c) for c in model["components"]]
    got = fr.groupby(SYS)["acidic_coextractant_modifier"].sum()
    want = systems.set_index("extractant_system_key")["n_model_rows_acidic_coextractant_modifier"]
    bad = {k: (int(v), int(want.get(k, 0))) for k, v in got.items() if int(v) != int(want.get(k, 0))}
    if bad:
        raise RuntimeError(f"acidic co-extractant rows disagree with extractant_systems.csv: {bad}")
    # censoring candidates (P8)
    y = fr["log_D"].to_numpy(dtype=float)
    integer = np.abs(y - np.round(y)) < 1e-9
    grp = [fr["g19_study_id"], fr[SYS]]
    gmin = fr.groupby(grp)["log_D"].transform("min").to_numpy(dtype=float)
    gmax = fr.groupby(grp)["log_D"].transform("max").to_numpy(dtype=float)
    n_same = fr.groupby(grp + [fr["log_D"].round(9)])["log_D"].transform("size").to_numpy()
    fr["log_D_floor_candidate"] = integer & (y <= FLOOR_MAX_LOG_D) & (np.abs(y - gmin) < 1e-9) & (n_same >= CENSOR_MIN_ROWS)
    fr["log_D_ceiling_candidate"] = integer & (y >= CEILING_MIN_LOG_D) & (np.abs(y - gmax) < 1e-9) & (n_same >= CENSOR_MIN_ROWS)
    fr["log_D_censoring_candidate"] = fr["log_D_floor_candidate"] | fr["log_D_ceiling_candidate"]
    return fr, systems


def parent_component_map() -> dict[str, str]:
    return FR.parent_component_map(COMPONENTS_CSV)


assign_halves = FR.assign_halves


def v6_system_set(fr, min_rows: int, min_other_ln: int) -> set[str]:
    return FR.v6_system_set(fr, min_rows, min_other_ln)


def v6_target_mask(fr, v6_systems, component_map=None) -> pd.Series:
    return FR.v6_target_mask(fr, v6_systems, component_map)


# --------------------------------------------------------------------------------------------- #
# graph connectivity
# --------------------------------------------------------------------------------------------- #

def graph_block(fr: pd.DataFrame, share_thr: float) -> tuple[dict, pd.DataFrame]:
    res: dict = {"definition": "metal-state x extractant-system bipartite graph of MODEL rows; "
                               "'known_state' uses g19_metal_state only, 'incl_unknown_state' adds X(?) nodes",
                 "connected_share_threshold": share_thr}
    frames = []
    known = fr[fr[STATE].notna()]
    for variant, min_rows, unk in (("known_state_all_edges", 1, False), ("known_state_edges_ge5", 5, False),
                                   ("incl_unknown_state_all_edges", 1, True), ("incl_unknown_state_edges_ge5", 5, True)):
        pop = fr if unk else known
        pop_states = set(pop["state_or_alias"] if unk else pop[STATE])
        pop_systems = set(pop[SYS])
        G = SG.bipartite_graph(fr, min_rows=min_rows, include_unknown_state=unk)
        ct = SG.component_table(G)
        comps = SG.components_by_rows(G)
        giant = comps[0] if comps else set()
        giant_states = {SG.node_label(n) for n in giant if G.nodes[n]["node_type"] == "metal_state"}
        giant_systems = {SG.node_label(n) for n in giant if G.nodes[n]["node_type"] == "extractant_system"}
        in_graph_states = {SG.node_label(n) for n, d in G.nodes(data=True) if d["node_type"] == "metal_state"}
        in_graph_systems = {SG.node_label(n) for n, d in G.nodes(data=True) if d["node_type"] == "extractant_system"}
        edge_rows = int(sum(d["weight"] for _, _, d in G.edges(data=True)))
        giant_rows = int(ct.iloc[0]["n_rows"]) if len(ct) else 0
        share_edges = giant_rows / edge_rows if edge_rows else 0.0
        res[variant] = {
            "min_edge_rows": min_rows, "population_rows": int(len(pop)), "rows_on_graph_edges": edge_rows,
            "share_population_rows_on_graph_edges": edge_rows / len(pop), "n_components": int(len(ct)),
            "giant_rows": giant_rows, "giant_share_of_edge_rows": share_edges,
            "giant_share_of_population_rows": giant_rows / len(pop),
            "giant_metal_states": len(giant_states), "giant_systems": len(giant_systems),
            "population_metal_states": len(pop_states), "population_systems": len(pop_systems),
            "metal_states_outside_giant": sort_states(in_graph_states - giant_states),
            "metal_states_without_any_qualifying_edge": sort_states(pop_states - in_graph_states),
            "n_systems_outside_giant": len(in_graph_systems - giant_systems),
            "n_systems_without_any_qualifying_edge": len(pop_systems - in_graph_systems),
            "n_components_size_le_2_nodes": int(((ct["n_metal_states"] + ct["n_systems"]) <= 2).sum()),
            "verdict": "CONNECTED" if share_edges >= share_thr else "FRAGMENTED",
        }
        c2 = ct.copy()
        c2.insert(0, "variant", variant)
        c2["systems"] = c2["systems"].map(lambda s: s if len(s) < 400 else s[:400] + " ...(truncated)")
        frames.append(c2)
    res["support_multigraph"] = SG.graph_summary(SG.build_support_graph(fr))
    a, b = res["known_state_all_edges"], res["known_state_edges_ge5"]
    res["verdict_statement"] = (
        f"All-edge graph: {a['n_components']} component(s), giant holds {a['giant_share_of_edge_rows']:.3f} of "
        f"known-state MODEL rows -> {a['verdict']}.  The >=5-row backbone has {b['n_components']} component(s) "
        f"holding {b['giant_share_of_edge_rows']:.3f} of its edge rows ({b['share_population_rows_on_graph_edges']:.3f} "
        f"of known-state rows sit on >=5-row edges; {b['n_systems_without_any_qualifying_edge']} systems and "
        f"{len(b['metal_states_without_any_qualifying_edge'])} metal states have no >=5-row edge) -> {b['verdict']}.")
    return res, pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------------------------- #
# Q1  V5 cells
# --------------------------------------------------------------------------------------------- #

def _v5_cells_frame(known: pd.DataFrame, bases) -> pd.DataFrame:
    agg = {"n_rows": (PUB, "size")}
    for b in bases:
        agg[f"n_pubs__{b}"] = (b, "nunique")
    return known.groupby([STATE, SYS], sort=True).agg(**agg).reset_index()


#: per-cell attributes that depend on the rows of the medium; the HNO3-only grid reads their ``_hno3`` twins,
#: computed on HNO3 rows alone (rows, alias rows, component sharing, acids, flags, hidden support features)
MEDIUM_ATTRIBUTES: tuple[str, ...] = (
    "alias_unknown_state_rows_hidden_with_cell", "metal_rows_in_other_systems_sharing_a_component",
    "unknown_state_rows_in_other_systems_sharing_a_component", "hidden_rows_in_other_systems_sharing_a_component",
    "other_known_state_rows_in_other_systems_sharing_a_component", "element_rows_in_other_systems_sharing_a_component",
    "hidden_rows_in_other_systems_sharing_a_parent_structure", "element_rows_in_other_systems_sharing_a_parent_structure",
    "n_acids", "n_rows_non_hno3", "n_rows_acidic_coextractant_modifier", "n_rows_log_D_censoring_candidate",
    "connected_without_cell_edges_ge5", "hidden_n_series_neighbours_pm1", "hidden_series_bracketed",
    "hidden_nearest_radius_same_charge", "hidden_radius_bracketed_same_charge",
)


def _v5_grid_record(e: pd.DataFrame, medium: str, b, k, p, mm, base, ok, n_rows_col: str, sfx: str = "") -> dict:
    """One grid row; ``sfx`` selects the medium's twin of every :data:`MEDIUM_ATTRIBUTES` column."""
    a = {c: e[c + sfx] for c in MEDIUM_ATTRIBUTES}
    lan = e["metal_category"] == "lanthanide"
    return {
        "medium": medium, "pub_basis": b, "k_min_rows": k, "p_min_publications": p, "m_min_other": mm,
        "n_cells": int(ok.sum()), "n_rows_in_cells": int(e[n_rows_col].sum()),
        "n_systems": int(e["extractant_system_key"].nunique()), "n_metal_states": int(e["metal_state"].nunique()),
        "n_families": int(e["system_family"].nunique()),
        "n_lanthanide_cells": int(lan.sum()),
        "n_lanthanide_cells_series_pm1_present": int((lan & (a["hidden_n_series_neighbours_pm1"] >= 1)).sum()),
        "n_lanthanide_cells_bracketed_pm2": int((lan & a["hidden_series_bracketed"].astype(bool)).sum()),
        "n_cells_series_pm1_present_any_series": int((a["hidden_n_series_neighbours_pm1"] >= 1).sum()),
        "n_actinide_cells": int((e["metal_category"] == "actinide").sum()),
        "n_other_category_cells": int((~e["metal_category"].isin(["lanthanide", "actinide"])).sum()),
        "n_pr_nd_cells": int(e["is_pr_or_nd"].sum()),
        "n_cells_non_diglycolamide_family": int((e["system_family"] != "diglycolamide").sum()),
        "n_cells_with_alias_rows": int((a["alias_unknown_state_rows_hidden_with_cell"] > 0).sum()),
        # the registered (state-level) component-aware hiding removes rows beyond the cell's own system
        "n_cells_hidden_rows_in_system_sharing_component": int((a["hidden_rows_in_other_systems_sharing_a_component"] > 0).sum()),
        "n_cells_metal_in_system_sharing_component": int((a["metal_rows_in_other_systems_sharing_a_component"] > 0).sum()),
        "n_cells_unknown_state_in_system_sharing_component": int(
            (a["unknown_state_rows_in_other_systems_sharing_a_component"] > 0).sum()),
        "n_cells_other_known_state_kept_in_system_sharing_component": int(
            (a["other_known_state_rows_in_other_systems_sharing_a_component"] > 0).sum()),
        "n_cells_element_in_system_sharing_component": int((a["element_rows_in_other_systems_sharing_a_component"] > 0).sum()),
        "n_cells_hidden_rows_in_system_sharing_parent_structure": int(
            (a["hidden_rows_in_other_systems_sharing_a_parent_structure"] > 0).sum()),
        "n_cells_element_in_system_sharing_parent_structure": int(
            (a["element_rows_in_other_systems_sharing_a_parent_structure"] > 0).sum()),
        "n_cells_extra_under_parent_structure_rule": int(
            ((a["hidden_rows_in_other_systems_sharing_a_parent_structure"] > 0)
             & (a["hidden_rows_in_other_systems_sharing_a_component"] == 0)).sum()),
        "n_cells_multi_acid": int((a["n_acids"] > 1).sum()),
        "n_rows_non_hno3": int(a["n_rows_non_hno3"].sum()),
        "n_cells_with_acidic_coextractant_rows": int((a["n_rows_acidic_coextractant_modifier"] > 0).sum()),
        "n_rows_acidic_coextractant_modifier": int(a["n_rows_acidic_coextractant_modifier"].sum()),
        "n_cells_with_censoring_candidate_rows": int((a["n_rows_log_D_censoring_candidate"] > 0).sum()),
        "n_rows_log_D_censoring_candidate": int(a["n_rows_log_D_censoring_candidate"].sum()),
        "n_cells_nearest_radius_other_charge": int((~a["hidden_nearest_radius_same_charge"].astype(bool)).sum()),
        "n_cells_radius_bracketed_same_charge": int(a["hidden_radius_bracketed_same_charge"].astype(bool).sum()),
        "n_lanthanide_cells_radius_bracketed_same_charge": int((lan & a["hidden_radius_bracketed_same_charge"].astype(bool)).sum()),
        "n_cells_failing_only_connectivity": int((base & ~ok).sum()),
        "n_cells_also_connected_ge5_edges": int(a["connected_without_cell_edges_ge5"].astype(bool).sum()),   # e = eligible cells
    }


def v5_block(fr, systems, ns) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    k_grid, p_grid, m_grid = ints(ns.v5_k), ints(ns.v5_p), ints(ns.v5_m)
    bases = tuple(b for b in ns.v5_pub_basis.split(",") if b)
    known = fr[fr[STATE].notna()]
    cells = _v5_cells_frame(known, bases)
    state_nsys = known.groupby(STATE)[SYS].nunique()
    sys_nstate = known.groupby(SYS)[STATE].nunique()
    sys_states = known.groupby(SYS)[STATE].agg(lambda v: set(v))
    G = SG.bipartite_graph(fr)
    bridges = SG.bridge_set(G)
    # HNO3-only variant (acid media are not pooled; brief section 4.3)
    hno3 = fr[fr["acid_primary"] == "HNO3"]
    known_h = hno3[hno3[STATE].notna()]
    cells_h = _v5_cells_frame(known_h, bases).set_index([STATE, SYS])
    state_nsys_h = known_h.groupby(STATE)[SYS].nunique()
    sys_nstate_h = known_h.groupby(SYS)[STATE].nunique()
    bridges_h = SG.bridge_set(SG.bipartite_graph(hno3))
    # component / parent-structure sharing, element level
    all_keys = sorted(fr[SYS].dropna().unique())
    pmap = parent_component_map()
    share_cache: dict = {}

    def sharing(key, cmap):
        tag = (key, cmap is not None)
        if tag not in share_cache:
            share_cache[tag] = SG.systems_sharing_component(all_keys, key, cmap)
        return share_cache[tag]
    v6_sys = v6_system_set(fr, *V6_PRIMARY)
    v6_touched = set(v6_sys).union(*[sharing(k, None) for k in v6_sys]) if v6_sys else set()
    kmin = min(k_grid)
    cand = cells[cells["n_rows"] >= kmin].reset_index(drop=True)
    cond_cols = [SG.LOG_ACID_COL, SG.LOG_EXT_COL, SG.TEMP_COL]

    def medium_context(frm: pd.DataFrame) -> dict:
        kn = frm[frm[STATE].notna()]
        unknown = frm[frm[STATE].isna()]
        return {"frame": frm, "known": kn, "alias": unknown.groupby([ELEM, SYS]).size(),
                "st_sys_rows": kn.groupby([STATE, SYS]).size(), "el_sys_rows": frm.groupby([ELEM, SYS]).size(),
                "xq_sys_rows": unknown.groupby([ELEM, SYS]).size(),
                "acids": kn.groupby([STATE, SYS])["acid_primary"].agg(lambda v: v.value_counts().to_dict()),
                "coext": kn.groupby([STATE, SYS])["acidic_coextractant_modifier"].sum(),
                "censor": kn.groupby([STATE, SYS])["log_D_censoring_candidate"].sum(),
                "G5": SG.bipartite_graph(frm, min_rows=5),
                "groups": {kk: gg for kk, gg in kn.groupby([STATE, SYS], sort=False)}}

    def medium_attributes(ctx: dict, m: str, s: str, el: str, shared_sys: set, parent_sys: set) -> tuple[dict, dict | None]:
        """The :data:`MEDIUM_ATTRIBUTES` of one cell on one medium's rows, and the support features of the cell
        computed on those rows minus the registered hiding (None when the medium has no row of the cell)."""
        if (m, s) not in ctx["groups"]:
            return {c: None for c in MEDIUM_ATTRIBUTES}, None
        frm, g, G5 = ctx["frame"], ctx["groups"][(m, s)], ctx["G5"]
        mn, sn = SG.node_id("metal_state", m), SG.node_id("extractant_system", s)
        conn5 = (not SG.cell_is_bridge(G5, m, s)) if G5.has_edge(mn, sn) else \
            bool(mn in G5 and sn in G5 and nx.has_path(G5, mn, sn))
        cond = {col: float(np.nanmedian(g[col])) if g[col].notna().any() else float("nan") for col in cond_cols}
        cond[SG.ACID_ANION_COL] = g[SG.ACID_ANION_COL].mode().iloc[0]
        train = SG.hide_cell(frm, m, s, component_aware=True)          # the registered V5 hiding
        f = SG.SupportIndex(train, systems=systems, radius_tol=ns.radius_tol).features(m, s, cond)
        st_rows, el_rows, xq_rows = ctx["st_sys_rows"], ctx["el_sys_rows"], ctx["xq_sys_rows"]
        state_sh = int(sum(st_rows.get((m, k), 0) for k in shared_sys))
        xq_sh = int(sum(xq_rows.get((el, k), 0) for k in shared_sys))
        el_sh = int(sum(el_rows.get((el, k), 0) for k in shared_sys))
        alias_n = int(ctx["alias"].get((el, s), 0))
        # the hiding is exactly the state-level rule: cell + own-system X(?) + state and X(?) rows of sharing systems
        if f["exact_pair_exists"] or f["alias_unknown_state_rows"]:
            raise AssertionError(f"hidden cell {m} x {s} still visible in its training frame")
        scope = (train[SYS] == s) | train[SYS].isin(shared_sys)
        if (scope & ((train[STATE] == m) | (train[STATE].isna() & (train[ELEM] == el)))).any():
            raise AssertionError(f"hidden state {m} or {el}(?) still present in {s} or a system sharing a component")
        if len(frm) - len(train) != len(g) + alias_n + state_sh + xq_sh:
            raise AssertionError(f"hiding {m} x {s} removed {len(frm) - len(train)} rows, expected "
                                 f"{len(g) + alias_n + state_sh + xq_sh}")
        acids = ctx["acids"][(m, s)]
        attrs = {
            "alias_unknown_state_rows_hidden_with_cell": alias_n,
            "metal_rows_in_other_systems_sharing_a_component": state_sh,
            "unknown_state_rows_in_other_systems_sharing_a_component": xq_sh,
            "hidden_rows_in_other_systems_sharing_a_component": state_sh + xq_sh,
            "other_known_state_rows_in_other_systems_sharing_a_component": el_sh - state_sh - xq_sh,
            "element_rows_in_other_systems_sharing_a_component": el_sh,
            "hidden_rows_in_other_systems_sharing_a_parent_structure": int(
                sum(st_rows.get((m, k), 0) + xq_rows.get((el, k), 0) for k in parent_sys)),
            "element_rows_in_other_systems_sharing_a_parent_structure": int(sum(el_rows.get((el, k), 0) for k in parent_sys)),
            "n_acids": len(acids), "n_rows_non_hno3": int(sum(v for a, v in acids.items() if a != "HNO3")),
            "n_rows_acidic_coextractant_modifier": int(ctx["coext"][(m, s)]),
            "n_rows_log_D_censoring_candidate": int(ctx["censor"][(m, s)]),
            "connected_without_cell_edges_ge5": bool(conn5),
            "hidden_n_series_neighbours_pm1": f["n_series_neighbours_pm1"],
            "hidden_series_bracketed": f["series_bracketed"],
            "hidden_nearest_radius_same_charge": f["nearest_radius_same_charge"],
            "hidden_radius_bracketed_same_charge": f["radius_bracketed_same_charge"],
        }
        return attrs, f

    ctx_all, ctx_h = medium_context(fr), medium_context(hno3)
    st_sys_rows = ctx_all["st_sys_rows"]
    recs = []
    for _, c in cand.iterrows():
        m, s = c[STATE], c[SYS]
        props = SG.metal_properties(m)
        el = props["symbol"]
        mn, sn = SG.node_id("metal_state", m), SG.node_id("extractant_system", s)
        shared_sys = sharing(s, None)
        parent_sys = sharing(s, pmap)
        attrs, f = medium_attributes(ctx_all, m, s, el, shared_sys, parent_sys)
        attrs_h, _ = medium_attributes(ctx_h, m, s, el, shared_sys, parent_sys)
        acids = ctx_all["acids"][(m, s)]
        other_states_same_elem = sorted(x for x in sys_states[s] if x != m and element_of(x) == el)
        hc = cells_h.loc[(m, s)] if (m, s) in cells_h.index else None
        rec = {
            "metal_state": m, "element": el, "metal_category": props["category"],
            "series": props["series"], "extractant_system_key": s, "system_label": label_of(s, systems),
            "system_family": systems.loc[s, "system_family"], "mechanism": systems.loc[s, "mechanism"],
            "system_has_name_structure_conflict": bool(systems.loc[s, "has_name_structure_conflict"]),
            "n_rows": int(c["n_rows"]),
            **{f"n_publications__{b}": int(c[f"n_pubs__{b}"]) for b in bases},
            "other_systems_for_metal": int(state_nsys[m] - 1),
            "other_metal_states_for_system": int(sys_nstate[s] - 1),
            "same_element_other_state_under_system": pipe(other_states_same_elem),
            "connected_without_cell_all_edges": frozenset({mn, sn}) not in bridges,
            **attrs,
            "n_other_systems_sharing_a_component_with_metal": int(sum(1 for k in shared_sys if st_sys_rows.get((m, k), 0))),
            "n_other_systems_sharing_a_component_with_element": int(
                sum(1 for k in shared_sys if ctx_all["el_sys_rows"].get((el, k), 0))),
            "n_other_systems_sharing_a_parent_structure_with_element": int(
                sum(1 for k in parent_sys if ctx_all["el_sys_rows"].get((el, k), 0))),
            "other_systems_for_metal_after_registered_hiding": int(len(set(known.loc[known[STATE] == m, SYS]) - {s} - shared_sys)),
            "acids": ";".join(f"{a}:{n}" for a, n in sorted(acids.items())), "n_rows_hno3": int(acids.get("HNO3", 0)),
            "is_pr_or_nd": el in ("Pr", "Nd"),
            "is_v6_system": s in v6_sys,
            "in_v6_target_rows": bool(el in ("Pr", "Nd") and s in v6_touched),
            # HNO3-only eligibility inputs and the per-cell attributes of the HNO3-only grid (HNO3 rows alone)
            "n_rows_hno3_cell": int(hc["n_rows"]) if hc is not None else 0,
            **{f"n_publications_hno3__{b}": int(hc[f"n_pubs__{b}"]) if hc is not None else 0 for b in bases},
            "other_systems_for_metal_hno3": int(state_nsys_h.get(m, 0) - (1 if hc is not None else 0)),
            "other_metal_states_for_system_hno3": int(sys_nstate_h.get(s, 0) - (1 if hc is not None else 0)),
            "connected_without_cell_all_edges_hno3": bool(hc is not None and frozenset({mn, sn}) not in bridges_h),
            **{f"{k}_hno3": v for k, v in attrs_h.items()},
            # fold-wise support features, computed on MODEL rows minus the registered hiding of the cell
            "hidden_neighbour_metals": f["n_neighbour_metals"],
            "hidden_neighbour_metals_same_category": f["n_neighbour_metals_same_category"],
            "hidden_series_neighbours_present": f["series_neighbours_present"],
            "hidden_n_series_neighbours_pm2": f["n_series_neighbours_pm2"],
            "hidden_nearest_radius_metal": f["nearest_radius_metal"],
            "hidden_nearest_radius_distance_A": f["nearest_radius_distance_A"],
            "hidden_nearest_radius_same_species_charge": f["nearest_radius_same_species_charge"],
            "hidden_nearest_radius_basis": f["nearest_radius_basis"],
            "hidden_n_radius_neighbours_within_tol": f["n_radius_neighbours_within_tol"],
            "hidden_radius_bracket_lower_metal": f["radius_bracket_lower_metal"],
            "hidden_radius_bracket_upper_metal": f["radius_bracket_upper_metal"],
            "hidden_n_same_family_other_system_rows_for_metal": f["n_same_family_other_system_rows_for_metal"],
            "hidden_nearest_ligand_system_label": label_of(f["nearest_ligand_system"], systems)
            if f["nearest_ligand_system"] else None,
            "hidden_nearest_ligand_tanimoto": f["nearest_ligand_tanimoto"],
            "hidden_n_publications_system": f["n_publications_system"],
            "hidden_condition_distance_system_same_acid": f["condition_distance_system_same_acid"],
            "hidden_condition_distance_system": f["condition_distance_system"],
        }
        recs.append(rec)
    tab = pd.DataFrame(recs)
    grid = []
    for b, k, p, mm in itertools.product(bases, k_grid, p_grid, m_grid):
        base = (tab["n_rows"] >= k) & (tab[f"n_publications__{b}"] >= p) & \
               (tab["other_systems_for_metal"] >= mm) & (tab["other_metal_states_for_system"] >= mm)
        ok = base & tab["connected_without_cell_all_edges"]
        tab[f"eligible__{b}__k{k}_p{p}_m{mm}"] = ok
        grid.append(_v5_grid_record(tab[ok], "all", b, k, p, mm, base, ok, "n_rows"))
    for b, k, p, mm in itertools.product(bases, k_grid, p_grid, m_grid):
        base = (tab["n_rows_hno3_cell"] >= k) & (tab[f"n_publications_hno3__{b}"] >= p) & \
               (tab["other_systems_for_metal_hno3"] >= mm) & (tab["other_metal_states_for_system_hno3"] >= mm)
        ok = base & tab["connected_without_cell_all_edges_hno3"]
        tab[f"eligible_hno3__{b}__k{k}_p{p}_m{mm}"] = ok
        grid.append(_v5_grid_record(tab[ok], "HNO3_only", b, k, p, mm, base, ok, "n_rows_hno3_cell", sfx="_hno3"))
    grid_df = pd.DataFrame(grid)
    elig_cols = [c for c in tab.columns if c.startswith("eligible__")]
    tab["n_grid_settings_eligible"] = tab[elig_cols].sum(axis=1)
    hno3_cols = [c for c in tab.columns if c.startswith("eligible_hno3__")]
    tab = tab[(tab["n_grid_settings_eligible"] > 0) | tab[hno3_cols].any(axis=1)].sort_values(
        ["n_grid_settings_eligible", "n_rows", "metal_state", "extractant_system_key"],
        ascending=[False, False, True, True]).reset_index(drop=True)
    # registered primary: V6 carve-out (Pr/Nd cells in V6 systems or systems sharing a component with one)
    pcol = "eligible__{}__k{}_p{}_m{}".format(*PRIMARY_V5)
    tab["scored_primary_discovery"] = tab[pcol] & ~tab["in_v6_target_rows"]
    weights = tab[tab["scored_primary_discovery"]].groupby("extractant_system_key").size().to_dict()
    for k in tab["extractant_system_key"].unique():
        weights.setdefault(k, 0)
    fam_of = dict(zip(tab["extractant_system_key"], tab["system_family"]))
    halves = assign_halves(weights, strata={k: ("diglycolamide" if fam_of.get(k) == "diglycolamide" else "other")
                                            for k in weights})
    tab["selection_half"] = tab["extractant_system_key"].map(halves)
    g0 = grid_df[(grid_df["pub_basis"] == "g19_publication_id") & (grid_df["medium"] == "all")]
    prim = tab[tab["scored_primary_discovery"]]

    def setting(k, p, m, medium="all"):
        sel = grid_df[(grid_df.pub_basis == "g19_publication_id") & (grid_df.medium == medium) & (grid_df.k_min_rows == k)
                      & (grid_df.p_min_publications == p) & (grid_df.m_min_other == m)]
        return sel.iloc[0].to_dict() if len(sel) else None
    res = {
        "definition": ("cell = (g19_metal_state, extractant_system_key) over MODEL rows; eligible when rows >= k, "
                       "publications >= p, the metal keeps >= m other systems and the system >= m other metal "
                       "states, and the cell edge is not a bridge of the all-edge bipartite graph.  Registered "
                       "hiding (used for every hidden_* column; support_graph.hide_cell(component_aware=True)): the "
                       "same state-level rule in the cell's own system and in every other system sharing a component "
                       "structure -- rows of the hidden metal state plus the element's unknown-state X(?) rows; other "
                       "known states of the element stay in training everywhere.  element_* columns are "
                       "informational (every row of the element, any state).  medium HNO3_only recomputes the grid "
                       "and every per-cell attribute (MEDIUM_ATTRIBUTES, *_hno3 columns) on HNO3 rows alone."),
        "n_cells_ge1_row": int(len(cells)), "n_cells_ge_kmin_rows": int(len(cand)),
        "n_cells_listed_eligible_somewhere": int(len(tab)),
        "grid_g19_publication_id": g0.drop(columns=["pub_basis", "medium"]).to_dict(orient="records"),
        "headline_k5_p1_m2": setting(5, 1, 2), "primary_k10_p1_m3": setting(10, 1, 3),
        "strict_k20_p2_m5": setting(20, 2, 5),
        "hno3_only": {"headline_k5_p1_m2": setting(5, 1, 2, "HNO3_only"), "primary_k10_p1_m3": setting(10, 1, 3, "HNO3_only"),
                      "strict_k20_p2_m5": setting(20, 2, 5, "HNO3_only")},
        "v6_carve_out": {
            "rule": "Pr/Nd cells in a V6 system (rows5_otherln2) or in a system sharing a component with one",
            "n_v6_systems": len(v6_sys), "n_systems_touched_incl_component_sharing": len(v6_touched),
            "n_primary_cells_carved_out": int((tab[pcol] & tab["in_v6_target_rows"]).sum()),
            "n_primary_pr_nd_cells_in_v6_systems": int((tab[pcol] & tab["is_v6_system"] & tab["is_pr_or_nd"]).sum()),
            "n_primary_cells_scored_in_discovery": int(tab["scored_primary_discovery"].sum()),
            "n_primary_ln_cells_scored_in_discovery": int((prim["metal_category"] == "lanthanide").sum()),
            "n_primary_systems_scored_in_discovery": int(prim["extractant_system_key"].nunique())},
        "selection_halves_primary_scored": {
            h: {"n_cells": int((prim["selection_half"] == h).sum()),
                "n_systems": int(prim.loc[prim["selection_half"] == h, "extractant_system_key"].nunique()),
                "n_ln_cells": int(((prim["selection_half"] == h) & (prim["metal_category"] == "lanthanide")).sum()),
                "n_non_diglycolamide_cells": int(((prim["selection_half"] == h) & (prim["system_family"] != "diglycolamide")).sum()),
                "n_rows": int(prim.loc[prim["selection_half"] == h, "n_rows"].sum())} for h in ("S", "C")},
        "radius_interpolation_supply_primary_scored": {
            "n_cells": int(len(prim)),
            "n_cells_radius_bracketed_same_charge": int(prim["hidden_radius_bracketed_same_charge"].astype(bool).sum()),
            "n_cells_nearest_radius_other_charge": int((~prim["hidden_nearest_radius_same_charge"].astype(bool)).sum()),
            "n_lanthanide_cells": int((prim["metal_category"] == "lanthanide").sum()),
            "n_lanthanide_cells_radius_bracketed_same_charge": int(
                ((prim["metal_category"] == "lanthanide") & prim["hidden_radius_bracketed_same_charge"].astype(bool)).sum())},
    }
    return res, tab, grid_df, halves


# --------------------------------------------------------------------------------------------- #
# Q2 / Q3 spans, per-system and per-metal tables
# --------------------------------------------------------------------------------------------- #

def span_block(fr, ns) -> dict:
    known = fr[fr[STATE].notna()]
    cell = known.groupby([SYS, STATE]).size()
    out = {"systems_spanning_metal_states": [], "metal_states_spanning_systems": []}
    for min_rows, thr in itertools.product(ints(ns.span_cell_rows), ints(ns.span)):
        c = cell[cell >= min_rows]
        out["systems_spanning_metal_states"].append({
            "min_cell_rows": min_rows, "threshold": thr,
            "n_systems_known_state": int((c.groupby(level=0).size() >= thr).sum()),
            "n_systems_element_any_state": int((fr.groupby([SYS, ELEM]).size().pipe(lambda x: x[x >= min_rows])
                                               .groupby(level=0).size() >= thr).sum()),
        })
        out["metal_states_spanning_systems"].append({
            "min_cell_rows": min_rows, "threshold": thr,
            "n_metal_states_known_state": int((c.groupby(level=1).size() >= thr).sum()),
            "n_elements_any_state": int((fr.groupby([ELEM, SYS]).size().pipe(lambda x: x[x >= min_rows])
                                        .groupby(level=0).size() >= thr).sum()),
        })
    return out


def metal_state_table(fr, systems, ns) -> tuple[pd.DataFrame, dict]:
    known = fr[fr[STATE].notna()]
    mt = pd.read_csv(METALS_CSV)
    plaus = mt.set_index(mt["metal_state_label"].fillna(""))["state_plausible"]
    alias_by_elem = fr[fr[STATE].isna()].groupby(ELEM).size()
    rows = []
    for st, g in known.groupby(STATE):
        p = SG.metal_properties(st)
        cell = g.groupby(SYS).size()
        rows.append({
            "metal_state": st, "element": p["symbol"], "Z": p["Z"], "oxidation_state": p["ox"],
            "metal_category": g["metal_category"].iloc[0], "series": p["series"],
            "state_plausible": plaus.get(st, None),
            "n_model_rows": int(len(g)), "n_systems": int(g[SYS].nunique()),
            "n_systems_ge5_rows": int((cell >= 5).sum()), "n_publications": int(g[PUB].nunique()),
            "n_publication_groups_primary_source_doi": int(g["group_primary_source_doi"].nunique()),
            "n_families": int(g[SG.FAMILY_COL].nunique()), "families": pipe(sorted(g[SG.FAMILY_COL].unique())),
            "n_acids": int(g["acid_primary"].nunique()), "acids": pipe(sorted(g["acid_primary"].unique())),
            "unknown_state_rows_same_element": int(alias_by_elem.get(p["symbol"], 0)),
            "other_known_states_same_element": pipe(sort_states(set(known.loc[known[ELEM] == p["symbol"], STATE]) - {st})),
            "measured_in_one_publication_only": bool(g[PUB].nunique() == 1),
        })
    tab = pd.DataFrame(rows)
    tab = tab.iloc[sorted(range(len(tab)), key=lambda i: state_sort_key(tab.loc[i, "metal_state"]))].reset_index(drop=True)
    grid = []
    for r, s in itertools.product(ints(ns.v2_rows), ints(ns.v2_systems)):
        ok = (tab["n_model_rows"] >= r) & (tab["n_systems"] >= s)
        tab[f"v2_eligible__rows{r}_systems{s}"] = ok
        e = tab[ok]
        grid.append({"min_rows": r, "min_systems": s, "n_metal_states": int(ok.sum()),
                     "n_lanthanide": int((e["metal_category"] == "lanthanide").sum()),
                     "n_actinide": int((e["metal_category"] == "actinide").sum()),
                     "metal_states": pipe(e["metal_state"])})
    focus = []
    for el in LN_FOCUS:
        row = tab[tab["metal_state"] == f"{el}(III)"]
        r = row.iloc[0] if len(row) else None
        focus.append({"element": el, "model_rows_III": int(r["n_model_rows"]) if r is not None else 0,
                      "unknown_state_rows": int(alias_by_elem.get(el, 0)),
                      "systems_III": int(r["n_systems"]) if r is not None else 0,
                      "systems_ge5_rows_III": int(r["n_systems_ge5_rows"]) if r is not None else 0,
                      "publications_III": int(r["n_publications"]) if r is not None else 0,
                      "eligible_rows100_systems5": bool(r is not None and r["n_model_rows"] >= 100 and r["n_systems"] >= 5)})
    elig = tab[tab["v2_eligible__rows100_systems5"]] if "v2_eligible__rows100_systems5" in tab.columns else tab.iloc[0:0]
    strata = {st: ("ln_focus" if elem in LN_FOCUS and cat == "lanthanide" else cat)
              for st, elem, cat in zip(elig["metal_state"], elig["element"], elig["metal_category"])}
    v2_halves = assign_halves({st: 1.0 for st in elig["metal_state"]}, strata=strata)
    tab["v2_selection_half"] = tab["metal_state"].map(v2_halves)
    res = {"grid": grid, "ln_focus": focus,
           "v2_halves_rows100_systems5": {h: sorted((st for st, x in v2_halves.items() if x == h), key=state_sort_key)
                                          for h in ("S", "C")},
           "fraction_metal_states_in_one_publication": float(tab["measured_in_one_publication_only"].mean()),
           "implausible_states_present": pipe(tab.loc[tab["state_plausible"] == False, "metal_state"])}  # noqa: E712
    return tab, res


def system_table(fr, systems) -> pd.DataFrame:
    known = fr[fr[STATE].notna()]
    ln3 = {f"{e}(III)" for e in LN}
    rows = []
    for key, g in fr.groupby(SYS, sort=True):
        gk = known[known[SYS] == key]
        cell = gk.groupby(STATE).size()
        elems_any = set(g[ELEM])
        rows.append({
            "extractant_system_key": key, "system_label": label_of(key, systems),
            "system_family": systems.loc[key, "system_family"], "mechanism": systems.loc[key, "mechanism"],
            "brief_expert": BRIEF_EXPERT.get(systems.loc[key, "mechanism"], "none_unknown"),
            "n_organic_extractants": int(systems.loc[key, "n_organic_extractants"]),
            "n_model_rows": int(len(g)), "n_model_rows_known_state": int(len(gk)),
            "n_metal_states": int(gk[STATE].nunique()), "n_metal_states_ge5_rows": int((cell >= 5).sum()),
            "n_elements_any_state": len(elems_any),
            "n_ln_iii_states": len(set(gk[STATE]) & ln3),
            "n_actinide_rows": int((g["metal_category"] == "actinide").sum()),
            "n_lanthanide_rows": int((g["metal_category"] == "lanthanide").sum()),
            "n_publications": int(g[PUB].nunique()),
            "n_publication_groups_primary_source_doi": int(g["group_primary_source_doi"].nunique()),
            "n_acids": int(g["acid_primary"].nunique()), "acids": pipe(sorted(g["acid_primary"].unique())),
            "measured_in_one_publication_only": bool(g[PUB].nunique() == 1),
            "metal_states": pipe(sort_states(set(gk[STATE]))),
            "has_name_structure_conflict": bool(systems.loc[key, "has_name_structure_conflict"]),
            "n_model_rows_acidic_coextractant_modifier": int(g["acidic_coextractant_modifier"].sum()),
        })
    return pd.DataFrame(rows).sort_values(["n_model_rows", "extractant_system_key"], ascending=[False, True]) \
        .reset_index(drop=True)


# --------------------------------------------------------------------------------------------- #
# Q4  V3 / V4 / experts
# --------------------------------------------------------------------------------------------- #

def v3_v4_block(sys_tab, fr, systems, ns) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    st = sys_tab.copy()
    comp_rows: dict[str, set] = defaultdict(set)
    for key in st["extractant_system_key"]:
        for c in components_of(key):
            comp_rows[c].add(key)
    rows_by_key = st.set_index("extractant_system_key")["n_model_rows"]
    all_keys = list(st["extractant_system_key"])
    st["n_rows_other_systems_sharing_a_component"] = [
        int(sum(rows_by_key[k2] for k2 in SG.systems_sharing_component(all_keys, k))) for k in all_keys]
    pmap = parent_component_map()
    st["n_rows_other_systems_sharing_a_parent_structure"] = [
        int(sum(rows_by_key[k2] for k2 in SG.systems_sharing_component(all_keys, k, pmap))) for k in all_keys]
    grid3 = []
    for r, sm, fm in itertools.product(ints(ns.v3_rows), ints(ns.v3_states), ints(ns.v3_family_members)):
        big = st[st["n_model_rows"] >= r]
        fam_big = big.groupby("system_family").size()
        others = st["system_family"].map(fam_big).fillna(0) - (st["n_model_rows"] >= r).astype(int)
        ok = (st["n_model_rows"] >= r) & (st["n_metal_states"] >= sm) & (others >= fm) & ~st["has_name_structure_conflict"]
        st[f"v3_eligible__rows{r}_states{sm}_members{fm}"] = ok
        e = st[ok]
        grid3.append({"min_rows": r, "min_metal_states": sm, "min_other_family_members_with_min_rows": fm,
                      "n_extractant_systems": int(ok.sum()), "n_families": int(e["system_family"].nunique()),
                      "families": pipe(sorted(e["system_family"].unique())),
                      "n_non_diglycolamide": int((e["system_family"] != "diglycolamide").sum())})
    # V4 families (system_family); rows hidden by a strict family-out fold = any system with a component of it
    core = systems[["extractant_system_key", "component_core_families_resolved"]].rename(
        columns={"component_core_families_resolved": "component_core_families"})
    conflict = systems.set_index("extractant_system_key")["has_name_structure_conflict"].astype(bool)
    fr_nc = fr[~fr[SYS].map(conflict).fillna(False).astype(bool)]
    fam_rows = []
    known = fr[fr[STATE].notna()]
    for fam, g in fr.groupby(SG.FAMILY_COL):
        gk = known[known[SG.FAMILY_COL] == fam]
        parts = set(str(fam).split("+"))
        touching = core[core["component_core_families"].fillna("").map(
            lambda s: bool(parts & set(str(s).split("|"))))]["extractant_system_key"]
        fam_rows.append({
            "system_family": fam, "mechanisms": pipe(sorted(g[SG.MECH_COL].unique())),
            "n_systems": int(g[SYS].nunique()), "n_model_rows": int(len(g)),
            "n_metal_states": int(gk[STATE].nunique()),
            "n_ln_iii_states": len(set(gk[STATE]) & {f"{e}(III)" for e in LN}),
            "n_actinide_states": int(gk.loc[gk["metal_category"] == "actinide", STATE].nunique()),
            "n_publications": int(g[PUB].nunique()),
            "n_rows_lanthanide": int((g["metal_category"] == "lanthanide").sum()),
            "n_rows_actinide": int((g["metal_category"] == "actinide").sum()),
            "n_rows_any_system_containing_a_member_family": int(fr[SYS].isin(set(touching)).sum()),
            "n_systems_excl_name_structure_conflict": int(fr_nc.loc[fr_nc[SG.FAMILY_COL] == fam, SYS].nunique()),
            "n_model_rows_excl_name_structure_conflict": int((fr_nc[SG.FAMILY_COL] == fam).sum()),
        })
    fam_tab = pd.DataFrame(fam_rows).sort_values(["n_model_rows", "system_family"], ascending=[False, True]).reset_index(drop=True)
    grid4 = []
    for s_thr, r_thr in itertools.product(ints(ns.v4_systems), ints(ns.v4_rows)):
        ok = (fam_tab["n_systems"] >= s_thr) & (fam_tab["n_model_rows"] >= r_thr)
        ok_nc = (fam_tab["n_systems_excl_name_structure_conflict"] >= s_thr) & \
                (fam_tab["n_model_rows_excl_name_structure_conflict"] >= r_thr)
        fam_tab[f"v4_eligible__systems{s_thr}_rows{r_thr}"] = ok
        fam_tab[f"v4_eligible_excl_conflicts__systems{s_thr}_rows{r_thr}"] = ok_nc
        grid4.append({"min_systems": s_thr, "min_rows": r_thr, "n_families": int(ok.sum()),
                      "families": pipe(fam_tab.loc[ok, "system_family"]),
                      "n_families_excl_name_structure_conflicts": int(ok_nc.sum()),
                      "families_excl_name_structure_conflicts": pipe(fam_tab.loc[ok_nc, "system_family"])})
    # experts
    mech_rows = []
    for mech, g in fr.groupby(SG.MECH_COL):
        gk = known[known[SG.MECH_COL] == mech]
        mech_rows.append({"level": "mechanism_label", "label": mech, "brief_expert": BRIEF_EXPERT.get(mech, "none_unknown"),
                          "n_model_rows": int(len(g)), "n_systems": int(g[SYS].nunique()),
                          "n_metal_states": int(gk[STATE].nunique()), "n_publications": int(g[PUB].nunique()),
                          "n_families": int(g[SG.FAMILY_COL].nunique()),
                          "n_rows_lanthanide": int((g["metal_category"] == "lanthanide").sum()),
                          "n_rows_actinide": int((g["metal_category"] == "actinide").sum())})
    fr_exp = fr.assign(_expert=fr[SG.MECH_COL].map(BRIEF_EXPERT).fillna("none_unknown"))
    known_exp = fr_exp[fr_exp[STATE].notna()]
    for exp in sorted(set(BRIEF_EXPERT.values())):
        g = fr_exp[fr_exp["_expert"] == exp]
        gk = known_exp[known_exp["_expert"] == exp]
        mech_rows.append({"level": "brief_expert", "label": exp, "brief_expert": exp,
                          "n_model_rows": int(len(g)), "n_systems": int(g[SYS].nunique()),
                          "n_metal_states": int(gk[STATE].nunique()), "n_publications": int(g[PUB].nunique()),
                          "n_families": int(g[SG.FAMILY_COL].nunique()),
                          "n_rows_lanthanide": int((g["metal_category"] == "lanthanide").sum()),
                          "n_rows_actinide": int((g["metal_category"] == "actinide").sum())})
    mech_tab = pd.DataFrame(mech_rows)
    exp_grid = []
    et = mech_tab[mech_tab["level"] == "brief_expert"]
    for r_thr in ints(ns.expert_rows):
        ok = (et["n_model_rows"] >= r_thr) & (et["n_systems"] >= 3) & (et["n_metal_states"] >= 5)
        exp_grid.append({"min_rows": r_thr, "min_systems": 3, "min_metal_states": 5, "n_experts": int(ok.sum()),
                         "experts": pipe(et.loc[ok, "label"])})
    res = {"v3_definition": ("unit = extractant system; eligible when rows >= R, metal states >= S, its "
                             "system_family has >= F OTHER systems with >= R rows, and none of its components is a "
                             "name-structure conflict.  A strict fold must also hide "
                             "n_rows_other_systems_sharing_a_component (parent-structure sensitivity: "
                             "n_rows_other_systems_sharing_a_parent_structure)."),
           "v3_grid": grid3,
           "v4_definition": "unit = system_family; eligible when systems >= S and rows >= R",
           "v4_grid": grid4, "expert_grid": exp_grid,
           "expert_mapping_note": ("SOFT_N_DONOR and MIXED_NEUTRAL are grouped under brief expert 2 (neutral) and "
                                   "MIXED_ACIDIC under expert 1 -- a gen19 grouping (INFERRED), the builder's labels are kept "
                                   "in the mechanism_label rows")}
    return res, st, fam_tab, mech_tab


# --------------------------------------------------------------------------------------------- #
# Q5-Q8 named extractants and analogues; section 19 candidate table
# --------------------------------------------------------------------------------------------- #

def named_block(fr, systems) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    named = pd.read_csv(NAMED_CSV)
    comps = pd.read_csv(COMPONENTS_CSV)
    comps = comps[comps["record_type"] == "STRUCTURE"].copy()
    known = fr[fr[STATE].notna()]
    key_rows = fr.groupby(SYS).size()
    key_comp = {k: components_of(k) for k in key_rows.index}
    summary, analog_rows = {}, []
    for ref in NAMED_TARGETS:
        info = LIG.REFERENCE_EXTRACTANTS[ref]
        sub = named[named["reference_extractant"] == ref]
        name_rows = sub[sub["term_kind"] == "NAME_TEXT"]
        exact_struct_model = int(pd.to_numeric(name_rows["n_model_rows_with_reference_structure"], errors="coerce")
                                 .fillna(0).max()) if len(name_rows) else 0
        ref_can = LIG.canonical_smiles(info["smiles"])
        rows_containing = fr[fr[SYS].map(lambda k: ref_can in key_comp.get(k, set()))]
        name_only = name_rows[name_rows["verdict"] == "NAME_ONLY_NO_STRUCTURE"]
        trap = name_rows[name_rows["verdict"] == "NAME_TRAP_DIFFERENT_STRUCTURE_FAMILY"]
        if len(rows_containing):
            verdict = "PRESENT"
        elif int(name_only["n_model_rows"].sum()):
            verdict = "ABSENT_AS_STRUCTURE_NAME_ONLY_ROWS"
        else:
            verdict = "ABSENT"
        summary[ref] = {
            "verdict": verdict, "reference_smiles": info["smiles"], "expected_family": info["expected_family"],
            "exact_structure_model_rows_named_audit": exact_struct_model,
            "model_rows_in_systems_containing_structure": int(len(rows_containing)),
            "name_terms": {r["term"]: {"verdict": r["verdict"], "n_rows": int(r["n_rows"]), "n_model_rows": int(r["n_model_rows"])}
                           for _, r in name_rows.iterrows()},
            "name_only_model_rows": int(name_only["n_model_rows"].sum()),
            "name_trap_model_rows": int(trap["n_model_rows"].sum()),
            "family_substructure_verdicts": {r["term"]: r["verdict"] for _, r in sub[sub["term_kind"] != "NAME_TEXT"].iterrows()},
        }
        fq = SG.fingerprint(info["smiles"])
        sims = []
        for _, c in comps.iterrows():
            smi = c["smiles_canonical"]
            fc = SG.fingerprint(smi)
            sims.append(float("nan") if (fq is None or fc is None) else SG.tanimoto(info["smiles"], smi))
        comps[f"_sim_{ref}"] = sims
        top = comps.sort_values([f"_sim_{ref}", "n_model_rows"], ascending=[False, False]).head(5)
        fam_members = comps[comps["family"].isin(ACIDIC_P_FAMILIES) | comps["core_family"].isin(ACIDIC_P_FAMILIES)]
        chosen = pd.concat([top.assign(_why="top5_tanimoto_any_family"),
                            fam_members.assign(_why="acidic_organophosphorus_family_member")])
        for _, c in chosen.iterrows():
            smi = c["smiles_canonical"]
            keys = [k for k, cs in key_comp.items() if smi in cs]
            g = fr[fr[SYS].isin(keys)]
            gk = known[known[SYS].isin(keys)]
            analog_rows.append({
                "reference_extractant": ref, "reference_verdict": verdict, "why_listed": c["_why"],
                "component_id": c["component_id"],
                "canonical_name": c["canonical_name"] if pd.notna(c["canonical_name"])
                else "ALIAS:" + str(c["aliases"]).split("|")[0] if pd.notna(c["aliases"]) else None,
                "canonical_name_status": c["canonical_name_status"], "family": c["family"],
                "core_family": c["core_family"],
                "mechanism": c["mechanism"], "acidity_class": c["acidity_class"], "smiles_canonical": smi,
                "tanimoto_to_reference": c[f"_sim_{ref}"], "primary_role": c["primary_role"],
                "n_model_rows_component_any_role": int(c["n_model_rows"]),
                "n_model_rows_as_organic_extractant": int(len(g)), "n_systems_as_organic_extractant": len(keys),
                "metal_states_as_extractant": pipe(sort_states(set(gk[STATE]))),
                "n_publications_as_extractant": int(g[PUB].nunique()),
                "pr_rows": int((g[ELEM] == "Pr").sum()), "nd_rows": int((g[ELEM] == "Nd").sum()),
            })
    analog = pd.DataFrame(analog_rows).drop_duplicates(["reference_extractant", "component_id", "why_listed"])
    # TODGA by metal state
    t_can = LIG.canonical_smiles(LIG.REFERENCE_EXTRACTANTS["TODGA"]["smiles"])
    single = fr[fr[SYS] == t_can]
    contain = fr[fr[SYS].map(lambda k: t_can in key_comp.get(k, set()))]
    lab = sort_states(set(contain["state_or_alias"]))
    todga = pd.DataFrame([{
        "metal_state_or_alias": s, "metal_category": contain.loc[contain["state_or_alias"] == s, "metal_category"].iloc[0],
        "rows_todga_single_system": int((single["state_or_alias"] == s).sum()),
        "publications_todga_single_system": int(single.loc[single["state_or_alias"] == s, PUB].nunique()),
        "rows_any_system_containing_todga": int((contain["state_or_alias"] == s).sum()),
        "publications_any_system_containing_todga": int(contain.loc[contain["state_or_alias"] == s, PUB].nunique()),
    } for s in lab])
    summary["TODGA"]["single_system_rows"] = int(len(single))
    summary["TODGA"]["single_system_publications"] = int(single[PUB].nunique())
    summary["TODGA"]["containing_systems"] = int(contain[SYS].nunique())
    summary["TODGA"]["containing_publications"] = int(contain[PUB].nunique())
    summary["TODGA"]["containing_metal_states_known"] = int(contain[STATE].nunique())
    # section 19 candidate table
    cand = []
    ln3 = [f"{e}(III)" for e in LN]
    for ref in NAMED_TARGETS:
        info = LIG.REFERENCE_EXTRACTANTS[ref]
        can = LIG.canonical_smiles(info["smiles"])
        g = fr[fr[SYS].map(lambda k: can in key_comp.get(k, set()))]
        fam_smiles = set(comps.loc[(comps["family"] == info["expected_family"])
                                   | (comps["core_family"] == info["expected_family"]), "smiles_canonical"])
        fam_g = fr[fr["primary_extractant_smiles"].isin(fam_smiles)]
        states = set(g[STATE].dropna())
        cand.append({
            "extractant": ref, "presence_verdict": summary[ref]["verdict"], "expected_family": info["expected_family"],
            "direct_model_rows": int(len(g)), "direct_publications": int(g[PUB].nunique()),
            "same_family_model_rows_primary": int(len(fam_g)),
            "same_family_systems_primary": int(fam_g[SYS].nunique()),
            "pr_direct_rows": int((g[ELEM] == "Pr").sum()), "nd_direct_rows": int((g[ELEM] == "Nd").sum()),
            "neighbouring_ln_iii_states_direct": int(len(states & set(ln3) - {"Pr(III)", "Nd(III)"})),
            "actinide_rows_direct": int((g["metal_category"] == "actinide").sum()),
            "n_acids_direct": int(g["acid_primary"].nunique()),
            "acid_M_min_direct": float(g["acid_concentration_M"].min()) if len(g) else None,
            "acid_M_max_direct": float(g["acid_concentration_M"].max()) if len(g) else None,
            "feasibility_support_class": ("DIRECT_PR_AND_ND" if (g[ELEM] == "Pr").any() and (g[ELEM] == "Nd").any()
                                          else "SAME_FAMILY_ONLY" if len(fam_g) else "UNSUPPORTED_NO_DIRECT_NO_FAMILY"),
        })
    return summary, analog, todga, pd.DataFrame(cand)


# --------------------------------------------------------------------------------------------- #
# Q9-Q11 Pr / Nd and Ln neighbours
# --------------------------------------------------------------------------------------------- #

def prnd_block(fr) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    recs, summ = [], {}
    for el in ("Pr", "Nd"):
        g = fr[fr[ELEM] == el]
        summ[el] = {"model_rows": int(len(g)), "rows_known_III": int((g[STATE] == f"{el}(III)").sum()),
                    "rows_unknown_state": int(g[STATE].isna().sum()), "systems": int(g[SYS].nunique()),
                    "publications": int(g[PUB].nunique()), "families": int(g[SG.FAMILY_COL].nunique()),
                    "acids": {k: int(v) for k, v in g["acid_primary"].value_counts().sort_index().items()}}
        for dim, col in (("family", SG.FAMILY_COL), ("mechanism", SG.MECH_COL), ("acid", "acid_primary"),
                         ("diluent_family", SG.DILUENT_COL), ("state", "state_or_alias")):
            for val, h in g.groupby(col):
                recs.append({"element": el, "breakdown": dim, "value": val, "n_model_rows": int(len(h)),
                             "n_systems": int(h[SYS].nunique()), "n_publications": int(h[PUB].nunique())})
    tab = pd.DataFrame(recs).sort_values(["element", "breakdown", "n_model_rows", "value"],
                                         ascending=[True, True, False, True]).reset_index(drop=True)
    both = set(fr.loc[fr[ELEM] == "Pr", SYS]) & set(fr.loc[fr[ELEM] == "Nd", SYS])
    summ["systems_with_both_pr_and_nd"] = len(both)
    # Q11
    rows = []
    sys_with = sorted(set(fr.loc[fr[ELEM].isin(["Pr", "Nd"]), SYS]))
    for key in sys_with:
        g = fr[fr[SYS] == key]
        rec = {"extractant_system_key": key, "system_label": g["system_label"].iloc[0],
               "system_family": g[SG.FAMILY_COL].iloc[0], "mechanism": g[SG.MECH_COL].iloc[0]}
        for el in CE_TO_SM:
            rec[f"{el}_rows_any_state"] = int((g[ELEM] == el).sum())
            rec[f"{el}_rows_III"] = int((g[STATE] == f"{el}(III)").sum())
        present_any = [el for el in CE_TO_SM if rec[f"{el}_rows_any_state"] > 0]
        present_iii = [el for el in CE_TO_SM if rec[f"{el}_rows_III"] > 0]
        rec["neighbours_present_any_state"] = pipe(present_any)
        rec["neighbours_present_III"] = pipe(present_iii)
        rec["contiguous_ce_pr_nd_sm_any_state"] = all(el in present_any for el in CONTIGUOUS_RUN)
        rec["contiguous_ce_pr_nd_sm_III"] = all(el in present_iii for el in CONTIGUOUS_RUN)
        run_pubs = [set(g.loc[g[ELEM] == el, PUB]) for el in CONTIGUOUS_RUN]
        rec["contiguous_ce_pr_nd_sm_in_one_publication"] = bool(set.intersection(*run_pubs)) if all(run_pubs) else False
        rec["n_publications"] = int(g[PUB].nunique())
        rec["n_ln_elements_any_state"] = int(g.loc[g["metal_category"] == "lanthanide", ELEM].nunique())
        rows.append(rec)
    nb = pd.DataFrame(rows).sort_values(["contiguous_ce_pr_nd_sm_any_state", "n_ln_elements_any_state", "system_label"],
                                        ascending=[False, False, True]).reset_index(drop=True)
    summ["q11"] = {
        "n_systems_with_pr_or_nd": int(len(nb)),
        "n_contiguous_ce_pr_nd_sm_any_state": int(nb["contiguous_ce_pr_nd_sm_any_state"].sum()),
        "n_contiguous_ce_pr_nd_sm_III": int(nb["contiguous_ce_pr_nd_sm_III"].sum()),
        "n_contiguous_ce_pr_nd_sm_in_one_publication": int(nb["contiguous_ce_pr_nd_sm_in_one_publication"].sum()),
        "presence_counts_any_state": {el: int((nb[f"{el}_rows_any_state"] > 0).sum()) for el in CE_TO_SM},
        "presence_counts_III": {el: int((nb[f"{el}_rows_III"] > 0).sum()) for el in CE_TO_SM},
        "families_with_contiguous_run": {k: int(v) for k, v in nb.loc[nb["contiguous_ce_pr_nd_sm_any_state"], "system_family"]
                                         .value_counts().sort_index().items()},
    }
    return summ, tab, nb


# --------------------------------------------------------------------------------------------- #
# comparable pairs (pairwise supply, Q12, V6)
# --------------------------------------------------------------------------------------------- #

def comparable_pairs(fr, key_col: str) -> pd.DataFrame:
    """Long table: one row per (publication, system, condition key, state_a, state_b) with n_a*n_b row pairs."""
    known = fr[fr[STATE].notna()]
    cnt = known.groupby([PUB, SYS, key_col, STATE]).size()
    fam = known.groupby(SYS)[SG.FAMILY_COL].first()
    acid = known.groupby([PUB, SYS, key_col])["acid_primary"].agg(lambda v: pipe(sorted(set(v))))
    recs = []
    for (pub, key, ck), sub in cnt.groupby(level=[0, 1, 2], sort=True):
        if len(sub) < 2:
            continue
        states = sub.index.get_level_values(3)
        vals = dict(zip(states, sub.to_numpy()))
        for a, b in itertools.combinations(sort_states(states), 2):
            recs.append({PUB: pub, SYS: key, "ck": ck, "state_a": a, "state_b": b,
                         "n_row_pairs": int(vals[a] * vals[b]), "n_rows_a": int(vals[a]), "n_rows_b": int(vals[b]),
                         "system_family": fam[key], "acid_primary": acid[(pub, key, ck)]})
    return pd.DataFrame(recs)


def pair_supply(pairs: pd.DataFrame, pairs_nm: pd.DataFrame, fr) -> tuple[pd.DataFrame, dict]:
    cat = fr[fr[STATE].notna()].groupby(STATE)["metal_category"].first()

    def agg(p):
        return p.groupby(["state_a", "state_b"]).agg(
            n_condition_groups=("ck", "size"), n_row_pairs=("n_row_pairs", "sum"),
            n_systems=(SYS, "nunique"), n_publications=(PUB, "nunique"),
            families=("system_family", lambda v: pipe(sorted(set(v)))))
    a, b = agg(pairs), agg(pairs_nm).add_suffix("_nm")
    tab = a.join(b, how="outer").fillna({c: 0 for c in list(a.columns) + list(b.columns) if not c.startswith("families")})
    tab = tab.reset_index()
    tab["category_a"] = tab["state_a"].map(cat)
    tab["category_b"] = tab["state_b"].map(cat)
    tab["category_pair"] = [pipe(sorted([x, y])) for x, y in zip(tab["category_a"], tab["category_b"])]
    for c in tab.columns:
        if c.startswith("n_"):
            tab[c] = tab[c].astype(int)
    tab = tab.sort_values(["n_row_pairs", "state_a", "state_b"], ascending=[False, True, True]).reset_index(drop=True)
    res = {"total_row_pairs": int(tab["n_row_pairs"].sum()), "total_condition_group_pairs": int(tab["n_condition_groups"].sum()),
           "total_row_pairs_nm": int(tab["n_row_pairs_nm"].sum()),
           "n_metal_state_pairs": int((tab["n_row_pairs"] > 0).sum()),
           "n_systems_with_any_pair": int(pairs[SYS].nunique()) if len(pairs) else 0,
           "n_publications_with_any_pair": int(pairs[PUB].nunique()) if len(pairs) else 0,
           "row_pairs_by_category_pair": {k: int(v) for k, v in tab.groupby("category_pair")["n_row_pairs"].sum().items()},
           "row_pairs_by_acid": {k: int(v) for k, v in pairs.groupby("acid_primary")["n_row_pairs"].sum().items()},
           "top10": tab.head(10)[["state_a", "state_b", "n_row_pairs", "n_condition_groups", "n_systems", "n_publications"]]
           .to_dict(orient="records")}
    return tab, res


def an_ln_block(fr, pairs, pairs_nm) -> tuple[dict, pd.DataFrame]:
    cat = fr[fr[STATE].notna()].groupby(STATE)["metal_category"].first()
    by_sys = fr.groupby(SYS)["metal_category"].agg(lambda v: set(v))
    both = sorted(k for k, v in by_sys.items() if {"actinide", "lanthanide"} <= v)
    out_rows = []
    for variant, p in (("full_key", pairs), ("key_excl_metal_conc", pairs_nm)):
        if not len(p):
            continue
        ca, cb = p["state_a"].map(cat), p["state_b"].map(cat)
        m = ((ca == "lanthanide") & (cb == "actinide")) | ((ca == "actinide") & (cb == "lanthanide"))
        q = p[m].copy()
        q["actinide_state"] = np.where(ca[m] == "actinide", q["state_a"], q["state_b"])
        q["lanthanide_state"] = np.where(ca[m] == "actinide", q["state_b"], q["state_a"])
        g = q.groupby(["actinide_state", "lanthanide_state", "system_family"]).agg(
            n_condition_groups=("ck", "size"), n_row_pairs=("n_row_pairs", "sum"), n_systems=(SYS, "nunique"),
            n_publications=(PUB, "nunique")).reset_index()
        g.insert(0, "key_variant", variant)
        out_rows.append(g)
    tab = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()
    tab = tab.sort_values(["key_variant", "n_row_pairs", "actinide_state", "lanthanide_state", "system_family"],
                          ascending=[True, False, True, True, True]).reset_index(drop=True)
    full = tab[tab["key_variant"] == "full_key"]
    by_pair = full.groupby(["actinide_state", "lanthanide_state"]).agg(
        n_row_pairs=("n_row_pairs", "sum"), n_condition_groups=("n_condition_groups", "sum"),
        n_systems=("n_systems", "sum"), n_families=("system_family", "nunique")).reset_index() \
        .sort_values("n_row_pairs", ascending=False)
    ca, cb = pairs["state_a"].map(cat), pairs["state_b"].map(cat)
    anln = pairs[((ca == "lanthanide") & (cb == "actinide")) | ((ca == "actinide") & (cb == "lanthanide"))]
    res = {
        "n_systems_with_actinide_and_lanthanide_model_rows": len(both),
        "n_systems_with_comparable_an_ln_pairs_full_key": int(anln[SYS].nunique()),
        "n_publications_with_comparable_an_ln_pairs_full_key": int(anln[PUB].nunique()),
        "n_condition_groups_an_ln_full_key": int(anln[[PUB, SYS, "ck"]].drop_duplicates().shape[0]),
        "n_row_pairs_an_ln_full_key": int(anln["n_row_pairs"].sum()),
        "n_row_pairs_an_ln_key_excl_metal_conc": int(tab.loc[tab["key_variant"] == "key_excl_metal_conc", "n_row_pairs"].sum()),
        "families_with_an_ln_pairs_full_key": {k: int(v) for k, v in full.groupby("system_family")["n_row_pairs"].sum()
                                               .sort_values(ascending=False).items()},
        "top_metal_state_pairs_full_key": by_pair.head(15).to_dict(orient="records"),
        "am_eu_row_pairs_full_key": int(full.loc[(full.actinide_state == "Am(III)") & (full.lanthanide_state == "Eu(III)"),
                                                 "n_row_pairs"].sum()),
        "an_prnd_row_pairs_full_key": int(full.loc[full.lanthanide_state.isin(["Pr(III)", "Nd(III)"]), "n_row_pairs"].sum()),
    }
    return res, tab


def v6_block(fr, pairs, pairs_nm, ns) -> tuple[dict, pd.DataFrame]:
    known = fr[fr[STATE].notna()]
    ln3 = {f"{e}(III)" for e in LN}
    pp = pairs[(pairs.state_a == "Pr(III)") & (pairs.state_b == "Nd(III)")]
    pp_nm = pairs_nm[(pairs_nm.state_a == "Pr(III)") & (pairs_nm.state_b == "Nd(III)")]
    rows = []
    for key, g in known.groupby(SYS):
        pr, nd = g[g[STATE] == "Pr(III)"], g[g[STATE] == "Nd(III)"]
        if not len(pr) and not len(nd):
            continue
        common = set(pr[PUB]) & set(nd[PUB])
        others = sorted(set(g[STATE]) & ln3 - {"Pr(III)", "Nd(III)"}, key=state_sort_key)
        rows.append({
            "extractant_system_key": key, "system_label": g["system_label"].iloc[0],
            "system_family": g[SG.FAMILY_COL].iloc[0], "mechanism": g[SG.MECH_COL].iloc[0],
            "pr_iii_rows": int(len(pr)), "nd_iii_rows": int(len(nd)),
            "pr_unknown_state_rows": int(((fr[SYS] == key) & fr[STATE].isna() & (fr[ELEM] == "Pr")).sum()),
            "nd_unknown_state_rows": int(((fr[SYS] == key) & fr[STATE].isna() & (fr[ELEM] == "Nd")).sum()),
            "n_common_publications": len(common),
            "pr_rows_in_common_publications": int(pr[PUB].isin(common).sum()),
            "nd_rows_in_common_publications": int(nd[PUB].isin(common).sum()),
            "n_other_ln_iii": len(others), "other_ln_iii": pipe(others),
            "comparable_prnd_row_pairs": int(pp.loc[pp[SYS] == key, "n_row_pairs"].sum()),
            "comparable_prnd_condition_groups": int((pp[SYS] == key).sum()),
            "comparable_prnd_row_pairs_hno3": int(pp.loc[(pp[SYS] == key) & (pp["acid_primary"] == "HNO3"), "n_row_pairs"].sum()),
            "comparable_prnd_condition_groups_hno3": int(((pp[SYS] == key) & (pp["acid_primary"] == "HNO3")).sum()),
            "comparable_prnd_row_pairs_by_acid": ";".join(
                f"{a}:{int(v)}" for a, v in pp[pp[SYS] == key].groupby("acid_primary")["n_row_pairs"].sum().items()),
            "comparable_prnd_condition_groups_by_acid": ";".join(
                f"{a}:{int(v)}" for a, v in pp[pp[SYS] == key].groupby("acid_primary").size().items()),
            "comparable_prnd_row_pairs_nm": int(pp_nm.loc[pp_nm[SYS] == key, "n_row_pairs"].sum()),
            "n_publications": int(g[PUB].nunique()),
        })
    tab = pd.DataFrame(rows)
    grid = []
    for r, L in itertools.product(ints(ns.v6_rows), ints(ns.v6_other_ln)):
        ok = (tab["pr_rows_in_common_publications"] >= r) & (tab["nd_rows_in_common_publications"] >= r) & \
             (tab["n_other_ln_iii"] >= L)
        tab[f"v6_eligible__rows{r}_otherln{L}"] = ok
        e = tab[ok]
        grid.append({"min_pr_and_nd_rows_in_common_publications": r, "min_other_ln_iii": L,
                     "n_systems": int(ok.sum()), "n_families": int(e["system_family"].nunique()),
                     "families": pipe(sorted(e["system_family"].unique())),
                     "n_systems_non_diglycolamide": int((e["system_family"] != "diglycolamide").sum()),
                     "comparable_prnd_row_pairs": int(e["comparable_prnd_row_pairs"].sum()),
                     "comparable_prnd_row_pairs_hno3": int(e["comparable_prnd_row_pairs_hno3"].sum()),
                     "n_systems_with_ge1_comparable_pair": int((e["comparable_prnd_row_pairs"] > 0).sum()),
                     "n_systems_with_ge2_common_publications": int((e["n_common_publications"] >= 2).sum())})
    tab = tab.sort_values(["comparable_prnd_row_pairs", "pr_rows_in_common_publications", "system_label"],
                          ascending=[False, False, True]).reset_index(drop=True)
    res = {"definition": ("V6 candidate system: Pr(III) and Nd(III) each >= r rows inside publications that measured "
                          "both, plus >= L other Ln(III) under the system; comparable pairs = same publication + "
                          "identical condition key"),
           "grid": grid, "total_comparable_prnd_row_pairs": int(pp["n_row_pairs"].sum()),
           "total_comparable_prnd_row_pairs_by_acid": {a: int(v) for a, v in pp.groupby("acid_primary")["n_row_pairs"].sum().items()},
           "total_comparable_prnd_condition_groups": int(len(pp)),
           "total_comparable_prnd_row_pairs_nm": int(pp_nm["n_row_pairs"].sum()),
           "n_systems_with_comparable_prnd_pairs": int(pp[SYS].nunique()),
           "n_publications_with_comparable_prnd_pairs": int(pp[PUB].nunique())}
    for r, L_ in ((5, 2), (10, 5)):
        vs = set(tab.loc[tab[f"v6_eligible__rows{r}_otherln{L_}"], "extractant_system_key"])
        mask = v6_target_mask(fr, vs)
        own = fr[ELEM].isin(["Pr", "Nd"]) & fr[SYS].isin(vs)
        res[f"v6_target_rows__rows{r}_otherln{L_}"] = {
            "definition": "Pr and Nd rows (known state and X(?)) in the V6 systems and in every system sharing a component",
            "n_rows": int(mask.sum()), "n_rows_in_v6_systems": int(own.sum()),
            "n_rows_in_component_sharing_systems_only": int((mask & ~own).sum()),
            "n_systems_touched": int(fr.loc[mask, SYS].nunique())}
    return res, tab


# --------------------------------------------------------------------------------------------- #
# Q13 disentangling metal and conditions; V1; V7
# --------------------------------------------------------------------------------------------- #

def pooled_between(df: pd.DataFrame, group_cols: list[str], level_col: str) -> dict:
    """Pooled variance of level means (``level_col``) around their group mean, over groups with >= 2 levels."""
    means = df.groupby(group_cols + [level_col], dropna=False)["log_D"].mean().reset_index()
    ss, dof, ngroups, nlev = 0.0, 0, 0, 0
    for _, g in means.groupby(group_cols, dropna=False):
        if len(g) < 2:
            continue
        v = g["log_D"].to_numpy()
        ss += float(((v - v.mean()) ** 2).sum())
        dof += len(v) - 1
        ngroups += 1
        nlev += len(v)
    return {"n_groups": ngroups, "n_levels": nlev, "pooled_variance": ss / dof if dof else float("nan"),
            "pooled_sd": math.sqrt(ss / dof) if dof else float("nan"), "dof": dof}


def pooled_within(df: pd.DataFrame, group_cols: list[str]) -> dict:
    ss, dof, ng = 0.0, 0, 0
    for _, g in df.groupby(group_cols, dropna=False):
        if len(g) < 2:
            continue
        v = g["log_D"].to_numpy()
        ss += float(((v - v.mean()) ** 2).sum())
        dof += len(v) - 1
        ng += 1
    return {"n_groups": ng, "pooled_variance": ss / dof if dof else float("nan"),
            "pooled_sd": math.sqrt(ss / dof) if dof else float("nan"), "dof": dof}


def q13_block(fr, sys_tab) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    known = fr[fr[STATE].notna() & np.isfinite(fr["log_D"])]
    rows = []
    for key, g in known.groupby(SYS):
        if g[STATE].nunique() < 2:
            continue
        rec = {"extractant_system_key": key, "system_label": g["system_label"].iloc[0],
               "system_family": g[SG.FAMILY_COL].iloc[0], "n_rows_known_state": int(len(g)),
               "n_metal_states": int(g[STATE].nunique()), "n_publications": int(g[PUB].nunique())}
        for kc, suf in (("ck", ""), ("ck_nm", "_nm")):
            ns_ = g.groupby([PUB, kc])[STATE].nunique()
            shared = ns_[ns_ >= 2]
            in_shared = g.set_index([PUB, kc]).index.isin(shared.index)
            rec[f"n_condition_keys{suf}"] = int(len(ns_))
            rec[f"n_condition_keys_with_ge2_metal_states{suf}"] = int(len(shared))
            rec[f"fraction_rows_in_shared_condition_groups{suf}"] = float(in_shared.mean())
            # distinct condition keys measured for >= 2 publications, same metal state
            np_ = g.groupby([STATE, kc])[PUB].nunique()
            rec[f"n_state_condition_keys_with_ge2_publications{suf}"] = int((np_ >= 2).sum())
        rows.append(rec)
    per_sys = pd.DataFrame(rows).sort_values(["n_rows_known_state", "extractant_system_key"], ascending=[False, True]) \
        .reset_index(drop=True)
    tot_rows = per_sys["n_rows_known_state"].sum()
    assoc = {"metal_state_vs_publication": cramers_v(known[STATE], known[PUB]),
             "metal_state_vs_system": cramers_v(known[STATE], known[SYS]),
             "system_vs_publication": cramers_v(known[SYS], known[PUB]),
             "metal_state_vs_acid": cramers_v(known[STATE], known["acid_primary"])}
    comps = []
    for kc, variant in (("ck", "full_key"), ("ck_nm", "key_excl_metal_conc")):
        metal = pooled_between(known, [PUB, SYS, kc], STATE)
        pubs = pooled_between(known, [SYS, STATE, kc], PUB)
        rep = pooled_within(known, [PUB, SYS, kc, STATE])
        comps += [
            {"key_variant": variant, "component": "metal_effect_within_publication_system_condition",
             "group_definition": "(publication, system, condition key); levels = metal states", **metal},
            {"key_variant": variant, "component": "publication_effect_within_system_metal_condition",
             "group_definition": "(system, metal state, condition key); levels = publications", **pubs},
            {"key_variant": variant, "component": "replicate_within_publication_system_condition_metal",
             "group_definition": "(publication, system, condition key, metal state); rows", "n_levels": np.nan, **rep},
        ]
    # confounded contrasts, conditions NOT matched
    comps.append({"key_variant": "no_condition_match", "component": "publication_effect_within_system_metal_CONFOUNDED",
                  "group_definition": "(system, metal state); levels = publications; conditions differ",
                  **pooled_between(known, [SYS, STATE], PUB)})
    comps.append({"key_variant": "no_condition_match", "component": "metal_effect_within_system_CONFOUNDED",
                  "group_definition": "(system); levels = metal states; conditions and publications differ",
                  **pooled_between(known, [SYS], STATE)})
    var_tab = pd.DataFrame(comps)
    var_tab.insert(0, "label", "DESCRIPTIVE (no model; pooled one-way variance of level means)")
    res = {
        "n_systems_with_ge2_metal_states": int(len(per_sys)),
        "rows_in_those_systems": int(tot_rows),
        "fraction_rows_in_shared_condition_groups_weighted": float(
            (per_sys["fraction_rows_in_shared_condition_groups"] * per_sys["n_rows_known_state"]).sum() / tot_rows),
        "fraction_rows_in_shared_condition_groups_weighted_nm": float(
            (per_sys["fraction_rows_in_shared_condition_groups_nm"] * per_sys["n_rows_known_state"]).sum() / tot_rows),
        "n_systems_with_any_shared_condition_group": int((per_sys["n_condition_keys_with_ge2_metal_states"] > 0).sum()),
        "n_systems_with_any_cross_publication_condition_match": int((per_sys["n_state_condition_keys_with_ge2_publications"] > 0).sum()),
        "n_systems_with_any_cross_publication_condition_match_nm": int((per_sys["n_state_condition_keys_with_ge2_publications_nm"] > 0).sum()),
        "association_cramers_v": assoc,
        "fraction_metal_states_in_one_publication": float((known.groupby(STATE)[PUB].nunique() == 1).mean()),
        "fraction_systems_in_one_publication": float((fr.groupby(SYS)[PUB].nunique() == 1).mean()),
        "fraction_model_rows_in_single_publication_systems": float(
            fr[SYS].map(fr.groupby(SYS)[PUB].nunique() == 1).mean()),
        "variance_components": var_tab.drop(columns=["label"]).to_dict(orient="records"),
    }
    return res, per_sys, var_tab


def v1_halves(fr, min_rows: int = 20) -> dict:
    """Selection / confirmation halves of the registered V1 folds: groups of group_cross_publication_copy with
    >= min_rows MODEL rows, plus the remainder fold, balanced on MODEL rows."""
    sizes = fr.groupby("group_cross_publication_copy").size()
    units = {g: float(n) for g, n in sizes.items() if n >= min_rows}
    units["REMAINDER"] = float(sizes[sizes < min_rows].sum())
    return assign_halves(units)


def v1_block(fr, ns) -> tuple[dict, pd.DataFrame]:
    recs = []
    n = len(fr)
    for col in ("g19_publication_id", "group_corrected_doi", "group_primary_source_doi", "group_cross_publication_copy",
                "group_near_duplicate_key", "group_compilation_doi"):
        sizes = fr.groupby(col).size().sort_values(ascending=False)
        rec = {"grouping": col, "n_groups_with_model_rows": int(len(sizes)), "largest_group_rows": int(sizes.iloc[0]),
               "largest_group_share": float(sizes.iloc[0] / n), "top5_share": float(sizes.head(5).sum() / n)}
        for r in ints(ns.v1_rows):
            rec[f"n_groups_ge{r}_rows"] = int((sizes >= r).sum())
            rec[f"rows_in_groups_ge{r}"] = int(sizes[sizes >= r].sum())
        recs.append(rec)
    tab = pd.DataFrame(recs)
    return {"note": ("LOPO folds = one per group; groups below the row threshold can be pooled into a remainder "
                     "fold.  group_* columns merge publications that share DOIs / copies (leakage_publication_components.csv)"),
            "table": tab.to_dict(orient="records")}, tab


def v7_block(fr, ns) -> tuple[dict, pd.DataFrame]:
    variables = {"acid_M": SG.LOG_ACID_COL, "extractant_M": SG.LOG_EXT_COL, "metal_M": "log10_metal_M"}
    recs = []
    for unit, cols in (("system", [SYS]), ("system_x_acid", [SYS, "acid_primary"])):
        for keys, g in fr.groupby(cols):
            keys = keys if isinstance(keys, tuple) else (keys,)
            for vname, col in variables.items():
                v = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                if len(v) == 0:
                    continue
                lo, hi = float(v.min()), float(v.max())
                rng = hi - lo
                rec = {"unit": unit, "extractant_system_key": keys[0], "acid": keys[1] if len(keys) > 1 else "ALL",
                       "variable": vname, "n_rows_with_value": int(len(v)), "n_distinct_values": int(len(np.unique(v))),
                       "log10_min": lo, "log10_max": hi, "log10_range": rng}
                for frac in floats(ns.v7_frac):
                    top = int((v >= hi - frac * rng).sum()) if rng > 0 else 0
                    bot = int((v <= lo + frac * rng).sum()) if rng > 0 else 0
                    rec[f"n_top{int(frac * 100)}"] = top
                    rec[f"n_bottom{int(frac * 100)}"] = bot
                recs.append(rec)
    tab = pd.DataFrame(recs)
    grid = []
    for unit, frac, r in itertools.product(("system", "system_x_acid"), floats(ns.v7_frac), ints(ns.v7_rows)):
        t = tab[tab["unit"] == unit]
        pct = int(frac * 100)
        for vname in variables:
            tv = t[t["variable"] == vname]
            top_ok = (tv[f"n_top{pct}"] >= r) & ((tv["n_rows_with_value"] - tv[f"n_top{pct}"]) >= r)
            bot_ok = (tv[f"n_bottom{pct}"] >= r) & ((tv["n_rows_with_value"] - tv[f"n_bottom{pct}"]) >= r)
            grid.append({"unit": unit, "region_fraction_of_log_range": frac, "min_rows_inside_and_outside": r,
                         "variable": vname, "n_units_top": int(top_ok.sum()), "n_units_bottom": int(bot_ok.sum()),
                         "n_units_either": int((top_ok | bot_ok).sum()),
                         "n_distinct_systems_either": int(tv.loc[top_ok | bot_ok, "extractant_system_key"].nunique())})
    grid_df = pd.DataFrame(grid)
    res = {"definition": ("per unit (system, or system x acid) and variable: the log10 range of the unit's own values; "
                          "top region = values >= max - f*range, bottom = values <= min + f*range; a unit qualifies "
                          "when >= r rows lie inside the region AND >= r outside"),
           "acid_M_log10_grid_flag_rows": int(fr["acid_M_log10_grid"].sum()),
           "grid": grid_df.to_dict(orient="records")}
    return res, tab, grid_df


# --------------------------------------------------------------------------------------------- #
# figure F03
# --------------------------------------------------------------------------------------------- #

def figure_f03(fr, out: Path) -> dict:
    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": grid, "axes.labelcolor": ink2,
                         "xtick.color": ink2, "ytick.color": ink2, "text.color": ink})
    cats = [c for c in CATEGORY_ORDER if c in set(fr["metal_category"])]
    color = dict(zip(CATEGORY_ORDER, CATEGORY_COLORS))
    panels = []
    for dim in (SG.FAMILY_COL, SG.MECH_COL):
        rows = pd.crosstab(fr[dim], fr["metal_category"]).reindex(columns=cats, fill_value=0)
        sysc = fr.groupby([dim, "metal_category"])[SYS].nunique().unstack(fill_value=0).reindex(columns=cats, fill_value=0)
        uniq = fr.groupby(dim)[SYS].nunique()
        order = rows.sum(axis=1).sort_values(ascending=True).index
        panels.append((dim, rows.loc[order], sysc.loc[order], uniq.loc[order]))
    n_fam, n_mech = len(panels[0][1]), len(panels[1][1])
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 0.30 * (n_fam + n_mech) + 3.2),
                             gridspec_kw={"height_ratios": [n_fam, n_mech + 1.5], "width_ratios": [1.25, 1]},
                             facecolor=surface)
    for r, (dim, rows, sysc, uniq) in enumerate(panels):
        for cidx, (mat, what) in enumerate(((rows, "MODEL rows"), (sysc, "systems"))):
            ax = axes[r, cidx]
            ax.set_facecolor(surface)
            left = np.zeros(len(mat))
            y = np.arange(len(mat))
            for cat in cats:
                vals = mat[cat].to_numpy(dtype=float)
                ax.barh(y, vals, left=left, height=0.62, color=color[cat], edgecolor=surface, linewidth=1.1,
                        label=cat.replace("_", " "))
                left += vals
            tot = mat.sum(axis=1).to_numpy()
            xmax = tot.max() if len(tot) else 1
            for yi, t in zip(y, tot):
                txt = f"{int(t):,}"
                if what == "systems":
                    txt = f"{int(uniq.iloc[yi]):,} unique"
                ax.text(t + xmax * 0.01, yi, txt, va="center", ha="left", fontsize=7.5, color=ink2)
            ax.set_yticks(y)
            ax.set_yticklabels([str(i).replace("_", " ") for i in mat.index] if cidx == 0 else [""] * len(mat),
                               fontsize=8)
            ax.set_xlim(0, xmax * 1.18)
            ax.grid(axis="x", color=grid, linewidth=0.6)
            ax.set_axisbelow(True)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(axis="y", length=0)
            xl = what if what == "MODEL rows" else "systems (a system is counted once per metal category it covers)"
            ax.set_xlabel(xl, fontsize=8.5)
            ttl = "system family" if dim == SG.FAMILY_COL else "mechanism (system majority label)"
            ax.set_title(f"{what} by {ttl}", fontsize=10, loc="left", color=ink)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(cats), frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.955), title="metal category", title_fontsize=8.5)
    fig.suptitle("Which extractant families and mechanisms have enough data to support transfer? "
                 "MODEL rows and systems, stacked by metal category", fontsize=11.5, color=ink, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    paths.ensure_dir(out.parent)
    fig.savefig(out, dpi=130, facecolor=surface)
    plt.close(fig)
    return {"n_families_plotted": n_fam, "n_mechanisms_plotted": n_mech, "categories": cats}


def f03_table(fr) -> pd.DataFrame:
    recs = []
    for dim in (SG.FAMILY_COL, SG.MECH_COL):
        for (val, cat), g in fr.groupby([dim, "metal_category"]):
            recs.append({"dimension": "system_family" if dim == SG.FAMILY_COL else "mechanism", "label": val,
                         "metal_category": cat, "n_model_rows": int(len(g)), "n_systems": int(g[SYS].nunique())})
    return pd.DataFrame(recs).sort_values(["dimension", "label", "metal_category"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ns = parse_args(argv)
    outs: dict[str, Path] = {}

    def out(name: str) -> Path:
        p = OUT / name
        outs[name] = p
        return p

    with Run(NAME, args=vars(ns), seed=0) as run:
        run.inputs(paths.ARCHIVE_MASTER, SYSTEMS_CSV, COMPONENTS_CSV, METALS_CSV, NAMED_CSV, PUB_COMPONENTS_CSV)
        fr, systems = load_frame(ns.sig)
        result: dict = {"schema": "gen19.feasibility.v1", "population": {
            "model_rows": int(len(fr)), "known_state_rows": int(fr[STATE].notna().sum()),
            "unknown_state_rows": int(fr[STATE].isna().sum()), "systems": int(fr[SYS].nunique()),
            "known_metal_states": int(fr[STATE].nunique()), "publications": int(fr[PUB].nunique()),
            "condition_key_sig": ns.sig}}

        g_res, g_tab = graph_block(fr, ns.connected_share)
        result["graph_connectivity"] = g_res
        write_csv(g_tab, out("feasibility_graph_components.csv"))

        v5_res, v5_cells, v5_grid, v5_halves = v5_block(fr, systems, ns)
        result["Q1_V5_cells"] = v5_res
        write_csv(v5_cells, out("feasibility_v5_cells.csv"))
        write_csv(v5_grid, out("feasibility_v5_grid.csv"))

        result["Q2_Q3_spans"] = span_block(fr, ns)
        ms_tab, v2_res = metal_state_table(fr, systems, ns)
        result["V2_leave_metal_out"] = v2_res
        write_csv(ms_tab, out("feasibility_metal_states.csv"))

        sys_tab = system_table(fr, systems)
        v34_res, sys_tab, fam_tab, mech_tab = v3_v4_block(sys_tab, fr, systems, ns)
        result["Q4_V3_V4_experts"] = v34_res
        write_csv(fam_tab, out("feasibility_families.csv"))
        write_csv(mech_tab, out("feasibility_mechanisms.csv"))

        named_res, analog, todga, cand = named_block(fr, systems)
        result["Q5_Q8_named_extractants"] = named_res
        write_csv(analog, out("feasibility_named_analogues.csv"))
        write_csv(todga, out("feasibility_todga_by_metal_state.csv"))
        write_csv(cand, out("feasibility_candidate_extractants.csv"))
        result["section19_candidates"] = cand.to_dict(orient="records")

        prnd_res, prnd_tab, nb_tab = prnd_block(fr)
        result["Q9_Q10_prnd"] = {k: v for k, v in prnd_res.items() if k != "q11"}
        result["Q11_ln_neighbours"] = prnd_res["q11"]
        write_csv(prnd_tab, out("feasibility_prnd.csv"))
        write_csv(nb_tab, out("feasibility_ln_neighbours.csv"))

        pairs = comparable_pairs(fr, "ck")
        pairs_nm = comparable_pairs(fr, "ck_nm")
        supply, supply_res = pair_supply(pairs, pairs_nm, fr)
        result["pairwise_supply"] = supply_res
        write_csv(supply, out("feasibility_comparable_pairs.csv"))

        anln_res, anln_tab = an_ln_block(fr, pairs, pairs_nm)
        result["Q12_actinide_lanthanide_overlap"] = anln_res
        write_csv(anln_tab, out("feasibility_an_ln_overlap.csv"))

        v6_res, v6_tab = v6_block(fr, pairs, pairs_nm, ns)
        result["V6_prnd_double_cell"] = v6_res
        write_csv(v6_tab, out("feasibility_v6_systems.csv"))

        q13_res, q13_sys, var_tab = q13_block(fr, sys_tab)
        result["Q13_metal_vs_conditions"] = q13_res
        sys_tab = sys_tab.merge(q13_sys.drop(columns=["system_label", "system_family", "n_metal_states", "n_publications"]),
                                on="extractant_system_key", how="left")
        write_csv(sys_tab, out("feasibility_systems.csv"))
        write_csv(var_tab, out("feasibility_q13_variance.csv"))

        v1_res, v1_tab = v1_block(fr, ns)
        result["V1_leave_publication_out"] = v1_res
        write_csv(v1_tab, out("feasibility_v1_groups.csv"))

        # registered selection / confirmation halves (P4)
        h1 = v1_halves(fr)
        gsz = fr.groupby("group_cross_publication_copy").size()
        halves_rows = [{"design": "V1", "unit": g, "weight": float(gsz[g]) if g in gsz.index else
                        float(gsz[gsz < 20].sum()), "half": h} for g, h in sorted(h1.items())]
        v5w = v5_cells[v5_cells["scored_primary_discovery"]].groupby("extractant_system_key").size()
        halves_rows += [{"design": "V5_system", "unit": k, "weight": float(v5w.get(k, 0)), "half": h}
                        for k, h in sorted(v5_halves.items())]
        v2h = ms_tab.dropna(subset=["v2_selection_half"])
        halves_rows += [{"design": "V2_state", "unit": st, "weight": 1.0, "half": h}
                        for st, h in zip(v2h["metal_state"], v2h["v2_selection_half"])]
        halves = pd.DataFrame(halves_rows)
        write_csv(halves, out("feasibility_halves.csv"))
        result["selection_confirmation_halves"] = {
            "salt": HALF_SALT, "rule": "units heaviest first (ties by sha256(salt|unit)), each to the lighter half of its stratum",
            "V1": {h: {"n_folds": int((halves[(halves.design == "V1")].half == h).sum()),
                       "n_model_rows": int(halves[(halves.design == "V1") & (halves.half == h)].weight.sum())} for h in ("S", "C")},
            "V5_primary_scored": v5_res["selection_halves_primary_scored"],
            "V2": v2_res["v2_halves_rows100_systems5"]}

        cen = fr[fr["log_D_censoring_candidate"]]
        result["log_D_censoring_candidates"] = {
            "definition": ("exact-decade log D (|log D - round(log D)| < 1e-9) equal to the minimum (<= -2) or maximum "
                           "(>= 2) of its (g19_study_id, extractant_system_key) group and shared by >= 2 rows; a "
                           "detection-limit / saturation pattern (INFERRED), flagged, never altered"),
            "n_floor_rows": int(fr["log_D_floor_candidate"].sum()), "n_ceiling_rows": int(fr["log_D_ceiling_candidate"].sum()),
            "floor_values": {str(k): int(v) for k, v in fr.loc[fr["log_D_floor_candidate"], "log_D"].round().value_counts().sort_index().items()},
            "n_publications": int(cen[PUB].nunique()),
            "rows_by_publication": {k: int(v) for k, v in cen[PUB].value_counts().items()},
            "n_model_rows_log_D_eq_minus4": int((np.abs(fr["log_D"] + 4) < 1e-9).sum()),
            "n_model_rows_log_D_eq_minus3": int((np.abs(fr["log_D"] + 3) < 1e-9).sum())}
        result["acidic_coextractant_modifier_rows"] = {
            "n_model_rows": int(fr["acidic_coextractant_modifier"].sum()),
            "by_system": {label_of(k, systems): int(v) for k, v in fr[fr["acidic_coextractant_modifier"]].groupby(SYS).size().items()},
            "by_metal_state": {k: int(v) for k, v in fr[fr["acidic_coextractant_modifier"]]["state_or_alias"].value_counts().items()}}

        v7_res, v7_tab, v7_grid = v7_block(fr, ns)
        result["V7_condition_regions"] = v7_res
        write_csv(v7_tab, out("feasibility_v7_units.csv"))
        write_csv(v7_grid, out("feasibility_v7_grid.csv"))

        result["V0_random_split"] = {"note": "always computable (12,411 MODEL rows); diagnostic only, never a claim (brief V0)"}
        result["F03"] = figure_f03(fr, FIG_PATH)
        write_csv(f03_table(fr), out("feasibility_f03_data.csv"))

        write_json(out("feasibility.json"), result)
        run.outputs(*outs.values(), FIG_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
