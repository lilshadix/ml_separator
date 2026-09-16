"""``chemistry/support_graph.py`` -- the chemistry support graph (brief section 16).

Everything here is computed from a ``train_df`` that the caller passes in, so the same code runs
fold-wise later (brief section 12: anything fitted to data is fitted on the training rows only).
No ``support_score`` formula lives here -- that is registered later; this module returns the raw
support quantities only.  Nothing reads ``log_D``.

Input frame
-----------
:func:`prepare_support_frame` turns archive rows (``gen19ct.data.load.load_archive``) into the frame
the functions below read.  Required columns (:data:`REQUIRED_COLUMNS`):

``g19_metal_state``  ``"Nd(III)"``; ``None`` when the archive recorded no oxidation state
``g19_metal``        element symbol (used for the unknown-state alias rows)
``extractant_system_key``, ``g19_publication_id``
``system_family``, ``mechanism``, ``primary_extractant_smiles``  (from ``descriptors/extractant_systems.csv``)
``acid_anion``, ``diluent_family``, ``log10_acid_M``, ``log10_extractant_M``, ``temperature_C``

Graph (:func:`build_support_graph`, a ``networkx.MultiGraph``)
---------------------------------------------------------------
Node ids are ``"<prefix>|<label>"`` strings with a ``node_type`` attribute:

=====================  ======  ===============================================================
node_type              prefix  label
=====================  ======  ===============================================================
``metal_state``        ``M``   ``Nd(III)``; unknown-state rows become ``Nd(?)`` (``state_known`` False)
``extractant_system``  ``S``   ``extractant_system_key`` (canonical SMILES, ``|``-joined)
``family``             ``F``   ``system_family``
``publication``        ``P``   ``g19_publication_id``
``condition_regime``   ``R``   ``acid anion | log10(acid M) bin | diluent family``
=====================  ======  ===============================================================

Edges carry ``edge_type`` (also used as the multigraph key) and ``weight`` = number of training rows:

``measured``            metal_state -- extractant_system
``shared_family``       extractant_system -- family   (two systems share a family iff they share this hub)
``shared_publication``  extractant_system -- publication and metal_state -- publication
``shared_regime``       extractant_system -- condition_regime and metal_state -- condition_regime

The "shared X" relations of the brief are represented through hub nodes rather than as explicit
system-system cliques (the projection is exact and the graph stays linear in the data).

Metal-extractant bipartite graph (:func:`bipartite_graph`)
    Simple graph of the ``measured`` edges only; ``min_rows`` drops thin edges.  Components
    (:func:`component_table`) and bridges (:func:`cell_is_bridge`) answer "is the data graph
    connected" and "does hiding this cell cut its metal off from its system".

Support features (:class:`SupportIndex`, :func:`support_features`)
    The brief section 16 quantities for one requested ``(metal_state, system, conditions)``.  See
    :meth:`SupportIndex.features` for the exact definitions.
"""
from __future__ import annotations

import math
from collections import defaultdict
from functools import lru_cache
from typing import Any, Iterable, Mapping

import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from gen19ct import paths
from gen19ct.chemistry import metals as MET

RDLogger.DisableLog("rdApp.*")

NODE_TYPES: tuple[str, ...] = ("metal_state", "extractant_system", "family", "publication", "condition_regime")
EDGE_TYPES: tuple[str, ...] = ("measured", "shared_family", "shared_publication", "shared_regime")
NODE_PREFIX: dict[str, str] = {"metal_state": "M", "extractant_system": "S", "family": "F",
                               "publication": "P", "condition_regime": "R"}

METAL_COL = "g19_metal_state"
ELEMENT_COL = "g19_metal"
SYSTEM_COL = "extractant_system_key"
PUB_COL = "g19_publication_id"
FAMILY_COL = "system_family"
MECH_COL = "mechanism"
SMILES_COL = "primary_extractant_smiles"
ACID_ANION_COL = "acid_anion"
DILUENT_COL = "diluent_family"
LOG_ACID_COL = "log10_acid_M"
LOG_EXT_COL = "log10_extractant_M"
TEMP_COL = "temperature_C"
CONDITION_DIMS: tuple[str, ...] = (LOG_ACID_COL, LOG_EXT_COL, TEMP_COL)

REQUIRED_COLUMNS: tuple[str, ...] = (METAL_COL, ELEMENT_COL, SYSTEM_COL, PUB_COL, FAMILY_COL, MECH_COL, SMILES_COL,
                                     ACID_ANION_COL, DILUENT_COL) + CONDITION_DIMS

#: log10(acid M) bin width of the condition regime (0.5 decade).
ACID_BIN_WIDTH = 0.5
#: |delta radius| (angstrom) under which another metal state counts as an ionic-radius neighbour.
RADIUS_NEIGHBOUR_TOL_A = 0.05
#: Series-neighbour offsets in atomic number.
SERIES_OFFSETS: tuple[int, ...] = (-2, -1, 1, 2)

SYSTEMS_CSV = paths.DESCRIPTORS_DIR / "extractant_systems.csv"
METALS_CSV = paths.DESCRIPTORS_DIR / "metals.csv"

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# --------------------------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------------------------- #

def _missing(v: Any) -> bool:
    if v is None or v is pd.NA:
        return True
    if isinstance(v, str):
        return v.strip() == ""
    if isinstance(v, (float, np.floating)):
        return bool(np.isnan(v))
    return False


def node_id(node_type: str, label: Any) -> str:
    return f"{NODE_PREFIX[node_type]}|{label}"


def node_label(nid: str) -> str:
    return nid.split("|", 1)[1]


def state_label_or_alias(state: Any, element: Any) -> str | None:
    """``Nd(III)`` for a known state, ``Nd(?)`` for a row with an element but no state, else None."""
    if not _missing(state):
        return str(state)
    if not _missing(element):
        return f"{element}(?)"
    return None


def acid_bin(log10_acid: Any, width: float = ACID_BIN_WIDTH) -> str:
    """``"[-0.5,0)"``-style half-open log10(acid M) bin; ``"NA"`` when missing."""
    if _missing(log10_acid) or not np.isfinite(float(log10_acid)):
        return "NA"
    lo = math.floor(float(log10_acid) / width) * width
    return f"[{lo:g},{lo + width:g})"


def regime_label(acid_anion: Any, log10_acid: Any, diluent: Any, width: float = ACID_BIN_WIDTH) -> str:
    a = "NA" if _missing(acid_anion) else str(acid_anion)
    d = "NA" if _missing(diluent) else str(diluent)
    return f"{a}|{acid_bin(log10_acid, width)}|{d}"


def check_columns(df: pd.DataFrame, columns: Iterable[str] = REQUIRED_COLUMNS) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"support graph: columns missing {missing} (use prepare_support_frame)")


# --------------------------------------------------------------------------------------------- #
# static chemistry tables (descriptors, not fitted to targets)
# --------------------------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def load_system_table() -> pd.DataFrame:
    """``descriptors/extractant_systems.csv`` indexed by ``extractant_system_key``."""
    s = pd.read_csv(SYSTEMS_CSV)
    return s.set_index("extractant_system_key", drop=False)


@lru_cache(maxsize=1)
def _metals_table() -> pd.DataFrame | None:
    if not METALS_CSV.exists():
        return None
    m = pd.read_csv(METALS_CSV)
    return m[m["metal_state_label"].notna()].set_index("metal_state_label", drop=False)


@lru_cache(maxsize=512)
def metal_properties(label: str) -> dict[str, Any]:
    """Static descriptors of a metal-state label: symbol, oxidation state, Z, category, series
    (``Ln``/``An``/``none``), formal charge (= oxidation state), the charge of the aqueous species
    (``species_charge``: +1 for AnO2+, +2 for AnO2 2+, the bare-cation charge otherwise; None when unknown)
    and Shannon CN8 / CN6 radii.
    Values come from ``descriptors/metals.csv`` when the label is there, otherwise from
    :mod:`gen19ct.chemistry.metals` (same Shannon table)."""
    out: dict[str, Any] = {"label": label, "symbol": None, "ox": None, "Z": None, "category": None,
                           "series": "none", "charge": None, "species_charge": None, "r_cn8": np.nan, "r_cn6": np.nan}
    if _missing(label) or str(label).endswith("(?)"):
        sym = None if _missing(label) else str(label)[:-3]
        out.update(symbol=sym, Z=MET.ATOMIC_NUMBER.get(sym) if sym else None)
        if sym in MET.LANTHANIDES:
            out.update(series="Ln", category="lanthanide")
        elif sym in MET.ACTINIDES:
            out.update(series="An", category="actinide")
        return out
    sym, ox, _ = MET.normalize_metal(label)
    out.update(symbol=sym, ox=ox, charge=ox, Z=MET.ATOMIC_NUMBER[sym])
    out["series"] = "Ln" if sym in MET.LANTHANIDES else ("An" if sym in MET.ACTINIDES else "none")
    tab = _metals_table()
    if tab is not None and label in tab.index:
        row = tab.loc[label]
        out["category"] = row["category"]
        out["r_cn8"] = float(row["radius_cn8_A"]) if pd.notna(row["radius_cn8_A"]) else np.nan
        out["r_cn6"] = float(row["radius_cn6_A"]) if pd.notna(row["radius_cn6_A"]) else np.nan
        if "species_charge" in row.index and pd.notna(row["species_charge"]):
            out["species_charge"] = int(row["species_charge"])
    else:
        radii = MET.SHANNON_RADII.get((sym, ox), {}) if ox is not None else {}
        out["r_cn8"] = radii.get(8, np.nan)
        out["r_cn6"] = radii.get(6, np.nan)
        out["category"] = ("lanthanide" if out["series"] == "Ln" else "actinide" if out["series"] == "An" else None)
    return out


def series_neighbour_labels(label: str, offsets: Iterable[int] = SERIES_OFFSETS) -> dict[int, str]:
    """``{offset: label}`` of same-series, same-state neighbours (``Nd(III)`` -> ``{-2: Ce(III),
    -1: Pr(III), 1: Pm(III), 2: Sm(III)}``); empty outside the Ln / An series or for unknown states."""
    p = metal_properties(label)
    if p["series"] not in ("Ln", "An") or p["ox"] is None:
        return {}
    series = MET.LANTHANIDES if p["series"] == "Ln" else MET.ACTINIDES
    idx = series.index(p["symbol"])
    out: dict[int, str] = {}
    for off in offsets:
        j = idx + off
        if 0 <= j < len(series):
            out[off] = f"{series[j]}({MET.ROMAN[p['ox']]})"
    return out


@lru_cache(maxsize=4096)
def fingerprint(smiles: str):
    """Morgan r=2, 2048-bit fingerprint of the SMILES (None when RDKit cannot parse it)."""
    if _missing(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    return None if mol is None else _MORGAN.GetFingerprint(mol)


def tanimoto(smiles_a: str, smiles_b: str) -> float:
    fa, fb = fingerprint(smiles_a), fingerprint(smiles_b)
    if fa is None or fb is None:
        return float("nan")
    return float(DataStructs.TanimotoSimilarity(fa, fb))


# --------------------------------------------------------------------------------------------- #
# frame preparation
# --------------------------------------------------------------------------------------------- #

def prepare_support_frame(df: pd.DataFrame, systems: pd.DataFrame | None = None,
                          cv: pd.DataFrame | None = None) -> pd.DataFrame:
    """Archive rows -> the frame the support functions read (same index, archive columns kept).

    ``cv`` is ``gen19ct.data.normalize.condition_vector(df)`` (computed when omitted); ``systems``
    is :func:`load_system_table`.  System family / mechanism / primary SMILES are static chemistry
    labels joined on ``extractant_system_key``.
    """
    from gen19ct.data import normalize as N

    systems = load_system_table() if systems is None else systems
    cv = N.condition_vector(df) if cv is None else cv
    out = df.copy()
    keyed = systems.set_index("extractant_system_key") if "extractant_system_key" in systems.columns else systems
    for col, src in ((FAMILY_COL, "system_family"), (MECH_COL, "mechanism"), (SMILES_COL, "primary_extractant_smiles"),
                     ("system_label", "component_canonical_names")):
        out[col] = out[SYSTEM_COL].map(keyed[src])
    out[ACID_ANION_COL] = cv["acid_anion"]
    out[DILUENT_COL] = cv["diluent_family"]
    out[LOG_ACID_COL] = cv["log10_acid_M"]
    out[LOG_EXT_COL] = cv["log10_extractant_primary_M"]
    out["log10_metal_M"] = cv["log10_metal_M"]
    out[TEMP_COL] = cv["temperature_C"]
    return out


# --------------------------------------------------------------------------------------------- #
# graphs
# --------------------------------------------------------------------------------------------- #

def bipartite_graph(train_df: pd.DataFrame, min_rows: int = 1, include_unknown_state: bool = False) -> nx.Graph:
    """Metal-state -- extractant-system graph of ``measured`` edges with ``>= min_rows`` rows.

    Edge attributes: ``weight`` (rows), ``n_publications``.  Nodes: ``node_type``, ``label``.
    Unknown-state rows become ``X(?)`` nodes only when ``include_unknown_state``.
    """
    check_columns(train_df, (METAL_COL, ELEMENT_COL, SYSTEM_COL, PUB_COL))
    labels = [state_label_or_alias(s, e) if include_unknown_state else (None if _missing(s) else str(s))
              for s, e in zip(train_df[METAL_COL], train_df[ELEMENT_COL])]
    frame = pd.DataFrame({"m": labels, "s": train_df[SYSTEM_COL].to_numpy(), "p": train_df[PUB_COL].to_numpy()})
    frame = frame[frame["m"].notna() & frame["s"].notna()]
    G = nx.Graph()
    if frame.empty:
        return G
    agg = frame.groupby(["m", "s"], sort=True).agg(weight=("p", "size"), n_publications=("p", "nunique"))
    for (m, s), r in agg.iterrows():
        if int(r["weight"]) < min_rows:
            continue
        mn, sn = node_id("metal_state", m), node_id("extractant_system", s)
        G.add_node(mn, node_type="metal_state", label=m, state_known=not m.endswith("(?)"))
        G.add_node(sn, node_type="extractant_system", label=s)
        G.add_edge(mn, sn, weight=int(r["weight"]), n_publications=int(r["n_publications"]))
    return G


#: Separator of member lists in :func:`component_table` (system keys contain ``|``; SMILES never contain ``;``).
MEMBER_SEP = "; "


def components_by_rows(G: nx.Graph) -> list[set[str]]:
    """Connected components as node-id sets, ordered like :func:`component_table` (largest by rows first)."""
    comps = []
    for comp in nx.connected_components(G):
        rows = int(sum(d.get("weight", 1) for _, _, d in G.subgraph(comp).edges(data=True)))
        ms = sorted(node_label(n) for n in comp if G.nodes[n]["node_type"] == "metal_state")
        ss = sorted(node_label(n) for n in comp if G.nodes[n]["node_type"] == "extractant_system")
        comps.append((-rows, -len(ss), MEMBER_SEP.join(ms), MEMBER_SEP.join(ss), set(comp)))
    return [c[-1] for c in sorted(comps, key=lambda c: c[:4])]


def component_table(G: nx.Graph) -> pd.DataFrame:
    """One row per connected component, largest (by rows) first: ``component``, ``n_metal_states``,
    ``n_systems``, ``n_rows`` (sum of edge weights), ``share_rows``, ``is_giant``, members joined by
    :data:`MEMBER_SEP`."""
    recs = []
    total = sum(d.get("weight", 1) for _, _, d in G.edges(data=True)) or 0
    for comp in components_by_rows(G):
        sub = G.subgraph(comp)
        ms = sorted(node_label(n) for n in comp if G.nodes[n]["node_type"] == "metal_state")
        ss = sorted(node_label(n) for n in comp if G.nodes[n]["node_type"] == "extractant_system")
        rows = int(sum(d.get("weight", 1) for _, _, d in sub.edges(data=True)))
        recs.append({"n_metal_states": len(ms), "n_systems": len(ss), "n_rows": rows,
                     "metal_states": MEMBER_SEP.join(ms), "systems": MEMBER_SEP.join(ss)})
    if not recs:
        return pd.DataFrame(columns=["component", "n_metal_states", "n_systems", "n_rows", "share_rows", "is_giant",
                                     "metal_states", "systems"])
    out = pd.DataFrame(recs).reset_index(drop=True)
    out.insert(0, "component", np.arange(len(out)))
    out.insert(4, "share_rows", out["n_rows"] / total if total else np.nan)
    out.insert(5, "is_giant", out["component"] == 0)
    return out


def n_components(G: nx.Graph) -> int:
    return nx.number_connected_components(G) if G.number_of_nodes() else 0


def bridge_set(G: nx.Graph) -> set[frozenset[str]]:
    """All bridges of ``G`` as ``frozenset({u, v})`` (an edge whose removal disconnects its ends)."""
    return {frozenset(e) for e in nx.bridges(G)} if G.number_of_edges() else set()


def cell_is_bridge(G: nx.Graph, metal_state: str, system: str) -> bool:
    """True when the ``(metal_state, system)`` edge exists and removing it separates the two nodes."""
    u, v = node_id("metal_state", metal_state), node_id("extractant_system", system)
    if not G.has_edge(u, v):
        return False
    data = dict(G.edges[u, v])
    G.remove_edge(u, v)
    try:
        return not nx.has_path(G, u, v)
    finally:
        G.add_edge(u, v, **data)


def connected_without(G: nx.Graph, metal_state: str, system: str) -> bool:
    """True when metal and system are both in ``G`` and still joined by a path after removing
    their direct edge (False when either node is absent)."""
    u, v = node_id("metal_state", metal_state), node_id("extractant_system", system)
    if u not in G or v not in G:
        return False
    if not G.has_edge(u, v):
        return nx.has_path(G, u, v)
    return not cell_is_bridge(G, metal_state, system)


def build_support_graph(train_df: pd.DataFrame, acid_bin_width: float = ACID_BIN_WIDTH) -> nx.MultiGraph:
    """The full support multigraph of the module docstring, from training rows only."""
    check_columns(train_df)
    m = [state_label_or_alias(s, e) for s, e in zip(train_df[METAL_COL], train_df[ELEMENT_COL])]
    regimes = [regime_label(a, x, d, acid_bin_width) for a, x, d in
               zip(train_df[ACID_ANION_COL], train_df[LOG_ACID_COL], train_df[DILUENT_COL])]
    fam = ["NA" if _missing(f) else str(f) for f in train_df[FAMILY_COL]]
    frame = pd.DataFrame({"m": m, "s": train_df[SYSTEM_COL].to_numpy(), "p": train_df[PUB_COL].to_numpy(),
                          "f": fam, "r": regimes})
    frame = frame[frame["s"].notna()]
    G = nx.MultiGraph()

    def add_edges(a: str, a_type: str, b: str, b_type: str, etype: str, sub: pd.DataFrame) -> None:
        cnt = sub.groupby([a, b], sort=True).size()
        for (x, y), w in cnt.items():
            xn, yn = node_id(a_type, x), node_id(b_type, y)
            for n, t, lab in ((xn, a_type, x), (yn, b_type, y)):
                if n not in G:
                    attrs = {"node_type": t, "label": lab}
                    if t == "metal_state":
                        attrs["state_known"] = not str(lab).endswith("(?)")
                    G.add_node(n, **attrs)
            G.add_edge(xn, yn, key=etype, edge_type=etype, weight=int(w))

    with_metal = frame[frame["m"].notna()]
    add_edges("m", "metal_state", "s", "extractant_system", "measured", with_metal)
    add_edges("s", "extractant_system", "f", "family", "shared_family", frame)
    add_edges("s", "extractant_system", "p", "publication", "shared_publication", frame)
    add_edges("m", "metal_state", "p", "publication", "shared_publication", with_metal)
    add_edges("s", "extractant_system", "r", "condition_regime", "shared_regime", frame)
    add_edges("m", "metal_state", "r", "condition_regime", "shared_regime", with_metal)
    return G


def graph_summary(G: nx.MultiGraph) -> dict[str, Any]:
    """Node counts by type, edge counts / row weights by type, and connected components."""
    nodes = defaultdict(int)
    for _, d in G.nodes(data=True):
        nodes[d["node_type"]] += 1
    edges, weights = defaultdict(int), defaultdict(int)
    for _, _, d in G.edges(data=True):
        edges[d["edge_type"]] += 1
        weights[d["edge_type"]] += int(d.get("weight", 0))
    comps = sorted((len(c) for c in nx.connected_components(G)), reverse=True) if G.number_of_nodes() else []
    return {"nodes_by_type": {t: nodes.get(t, 0) for t in NODE_TYPES},
            "edges_by_type": {t: edges.get(t, 0) for t in EDGE_TYPES},
            "edge_row_weight_by_type": {t: weights.get(t, 0) for t in EDGE_TYPES},
            "n_components": len(comps), "component_sizes_nodes": comps[:10]}


# --------------------------------------------------------------------------------------------- #
# support features
# --------------------------------------------------------------------------------------------- #

class SupportIndex:
    """Precomputed training-set aggregates for :meth:`features` (fit on ``train_df`` only).

    ``systems`` (optional, static) supplies family / primary SMILES for a queried system that has no
    training rows (a hidden extractant); it never contributes counts.
    """

    def __init__(self, train_df: pd.DataFrame, systems: pd.DataFrame | None = None,
                 radius_tol: float = RADIUS_NEIGHBOUR_TOL_A):
        check_columns(train_df)
        self.radius_tol = float(radius_tol)
        self._systems = systems
        t = train_df
        state = t[METAL_COL].to_numpy(dtype=object)
        elem = t[ELEMENT_COL].to_numpy(dtype=object)
        sysk = t[SYSTEM_COL].to_numpy(dtype=object)
        pub = t[PUB_COL].to_numpy(dtype=object)
        fam = t[FAMILY_COL].to_numpy(dtype=object)
        known = np.array([not _missing(s) for s in state], dtype=bool)
        self.n_train_rows = int(len(t))

        self.pair_rows: dict[tuple[str, str], int] = defaultdict(int)
        self.pair_pubs: dict[tuple[str, str], set] = defaultdict(set)
        self.system_metals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.metal_systems: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.metal_family_system_rows: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.alias_rows: dict[tuple[str, str], int] = defaultdict(int)   # (element, system) unknown-state rows
        self.system_pubs: dict[str, set] = defaultdict(set)
        self.metal_pubs: dict[str, set] = defaultdict(set)
        self.system_family: dict[str, str] = {}
        self.system_smiles: dict[str, str] = {}
        for i in range(len(t)):
            s = sysk[i]
            if _missing(s):
                continue
            self.system_pubs[s].add(pub[i])
            if s not in self.system_family and not _missing(fam[i]):
                self.system_family[s] = str(fam[i])
            if known[i]:
                m = str(state[i])
                self.pair_rows[(m, s)] += 1
                self.pair_pubs[(m, s)].add(pub[i])
                self.system_metals[s][m] += 1
                self.metal_systems[m][s] += 1
                self.metal_pubs[m].add(pub[i])
                if not _missing(fam[i]):
                    self.metal_family_system_rows[(m, str(fam[i]))][s] += 1
            elif not _missing(elem[i]):
                self.alias_rows[(str(elem[i]), s)] += 1
        for s, smi in zip(sysk, t[SMILES_COL].to_numpy(dtype=object)):
            if not _missing(s) and s not in self.system_smiles and not _missing(smi):
                self.system_smiles[s] = str(smi)

        # condition standardisation fitted on the training rows
        X = np.column_stack([pd.to_numeric(t[c], errors="coerce").to_numpy(dtype=float) for c in CONDITION_DIMS]) \
            if len(t) else np.zeros((0, len(CONDITION_DIMS)))
        X[~np.isfinite(X)] = np.nan
        with np.errstate(all="ignore"):
            mu = np.nanmean(X, axis=0) if len(X) else np.full(len(CONDITION_DIMS), np.nan)
            sd = np.nanstd(X, axis=0) if len(X) else np.full(len(CONDITION_DIMS), np.nan)
        sd = np.where(~np.isfinite(sd) | (sd <= 0), 1.0, sd)
        mu = np.where(~np.isfinite(mu), 0.0, mu)
        self.cond_mean, self.cond_std = mu, sd
        Z = (X - mu) / sd
        Z[np.isnan(Z)] = 0.0          # missing training value -> train mean (imputation fitted on train)
        self._Z = Z
        self._anion = t[ACID_ANION_COL].to_numpy(dtype=object)
        self._metal_of_row = state
        idx: dict[str, list[int]] = defaultdict(list)
        for i, s in enumerate(sysk):
            if not _missing(s):
                idx[s].append(i)
        self._system_rows = {s: np.asarray(v, dtype=int) for s, v in idx.items()}

    # ----------------------------------------------------------------------------------------- #
    def _static_system(self, system: str, col: str) -> str | None:
        if self._systems is None:
            return None
        tab = self._systems.set_index("extractant_system_key") if "extractant_system_key" in self._systems.columns \
            else self._systems
        if system in tab.index and col in tab.columns and not _missing(tab.loc[system, col]):
            return str(tab.loc[system, col])
        return None

    def standardise(self, conditions: Mapping[str, Any]) -> np.ndarray:
        """Condition dict -> z vector (NaN where the query does not give the dimension).  Accepts
        ``log10_acid_M`` or ``acid_M``, ``log10_extractant_M`` or ``extractant_M``, ``temperature_C``."""
        def get_log(log_key: str, lin_key: str) -> float:
            if log_key in conditions and not _missing(conditions[log_key]):
                return float(conditions[log_key])
            if lin_key in conditions and not _missing(conditions[lin_key]) and float(conditions[lin_key]) > 0:
                return math.log10(float(conditions[lin_key]))
            return float("nan")
        q = np.array([get_log(LOG_ACID_COL, "acid_M"), get_log(LOG_EXT_COL, "extractant_M"),
                      float(conditions[TEMP_COL]) if TEMP_COL in conditions and not _missing(conditions[TEMP_COL])
                      else float("nan")])
        return (q - self.cond_mean) / self.cond_std

    def condition_distance(self, system: str, conditions: Mapping[str, Any] | None,
                           metal_state: str | None = None, same_acid_anion: bool = False) -> tuple[float, int]:
        """(Euclidean distance in train-standardised (log10 acid M, log10 extractant M, T) space to the
        nearest training row of ``system`` [of ``metal_state`` when given], number of dims used)."""
        if conditions is None or system not in self._system_rows:
            return float("nan"), 0
        z = self.standardise(conditions)
        use = np.isfinite(z)
        if not use.any():
            return float("nan"), 0
        rows = self._system_rows[system]
        if metal_state is not None:
            rows = rows[np.array([self._metal_of_row[i] == metal_state for i in rows], dtype=bool)]
        if same_acid_anion and not _missing(conditions.get(ACID_ANION_COL)):
            rows = rows[np.array([self._anion[i] == conditions[ACID_ANION_COL] for i in rows], dtype=bool)]
        if len(rows) == 0:
            return float("nan"), int(use.sum())
        d = np.sqrt(((self._Z[np.ix_(rows, np.where(use)[0])] - z[use]) ** 2).sum(axis=1))
        return float(d.min()), int(use.sum())

    # ----------------------------------------------------------------------------------------- #
    def features(self, metal_state: str, system: str, conditions: Mapping[str, Any] | None = None, *,
                 family: str | None = None, smiles: str | None = None) -> dict[str, Any]:
        """The brief section 16 raw support quantities for one request.

        exact pair
            ``exact_pair_exists`` / ``exact_pair_rows`` / ``exact_pair_publications``: training rows
            with this known metal state under this system.  ``alias_unknown_state_rows``: training rows
            of the same element with NO recorded state under this system (a V2/V5 fold must hide them
            too -- they are reported, never counted as the exact pair).
        neighbouring metals
            ``n_neighbour_metals``: other known metal states measured with this system
            (``_same_category``, ``_same_charge`` variants); ``neighbour_metals`` lists them.
        same family
            ``system_family``; ``n_same_family_rows_for_metal`` (rows of this metal in any system of the
            family, this system included), ``n_same_family_other_system_rows_for_metal`` and
            ``n_same_family_other_systems_for_metal`` (this system excluded).
        ionic radius
            ``nearest_radius_metal`` / ``nearest_radius_distance_A``: the other metal state measured with
            this system whose Shannon radius is closest.  The CANDIDATE POOL is chosen first: same formal
            charge and same aqueous-species charge (an actinyl stays with actinyls), else same formal
            charge, else every candidate; ``nearest_radius_same_charge`` /
            ``nearest_radius_same_species_charge`` say which pool won.  Inside the pool the basis is CN8
            when the query and a pool candidate have CN8 radii, else CN6 (``nearest_radius_basis``); a
            pool with no usable radius falls through to the next pool.  ``n_radius_neighbours_within_tol``:
            same-charge candidates within ``radius_tol`` on that basis.  ``radius_bracket_lower_metal`` /
            ``radius_bracket_upper_metal`` / ``radius_bracketed_same_charge``: the nearest same-charge
            candidates with a smaller and a larger radius on that basis (the two ends a radius
            interpolation, baseline B3i, would use).
        ligand
            ``nearest_ligand_system`` / ``nearest_ligand_tanimoto`` / ``nearest_ligand_distance``
            (1 - Tanimoto, Morgan r=2 2048 bits on the primary-extractant SMILES) over the OTHER systems
            measured with this metal.  ``n_systems_for_metal``.
        publications
            ``n_publications_pair`` / ``n_publications_system`` / ``n_publications_metal``.
        conditions
            ``condition_distance_system`` (nearest training row of this system, any metal),
            ``condition_distance_system_same_acid`` (same acid anion, when the query names one),
            ``condition_distance_pair`` (nearest row of the exact pair), ``condition_dims_used``.
        series neighbours
            ``series_neighbours_present``: same-series, same-state elements at Z-2..Z+2 measured with this
            system; ``n_series_neighbours_pm1`` / ``_pm2``; ``series_bracketed`` (a neighbour below AND
            above within +-2).
        """
        m, s = str(metal_state), str(system)
        props = metal_properties(m)
        out: dict[str, Any] = {"metal_state": m, "extractant_system_key": s}

        # exact pair
        rows = int(self.pair_rows.get((m, s), 0))
        out["exact_pair_exists"] = rows > 0
        out["exact_pair_rows"] = rows
        out["exact_pair_publications"] = len(self.pair_pubs.get((m, s), ()))
        out["alias_unknown_state_rows"] = int(self.alias_rows.get((props["symbol"], s), 0)) if props["symbol"] else 0

        # neighbouring metals under the system
        others = {k: v for k, v in self.system_metals.get(s, {}).items() if k != m}
        oprops = {k: metal_properties(k) for k in others}
        out["n_neighbour_metals"] = len(others)
        out["n_neighbour_metals_same_category"] = sum(1 for k in others if oprops[k]["category"] == props["category"]
                                                      and props["category"] is not None)
        out["n_neighbour_metals_same_charge"] = sum(1 for k in others if oprops[k]["charge"] == props["charge"]
                                                    and props["charge"] is not None)
        out["neighbour_metals"] = "|".join(sorted(others))

        # same family
        fam = self.system_family.get(s) or family or self._static_system(s, FAMILY_COL)
        out["system_family"] = fam
        fam_rows = self.metal_family_system_rows.get((m, fam), {}) if fam else {}
        out["n_same_family_rows_for_metal"] = int(sum(fam_rows.values()))
        out["n_same_family_other_system_rows_for_metal"] = int(sum(v for k, v in fam_rows.items() if k != s))
        out["n_same_family_other_systems_for_metal"] = sum(1 for k in fam_rows if k != s)

        # nearest ionic radius
        out.update(self._nearest_radius(props, oprops))

        # nearest ligand
        smi = self.system_smiles.get(s) or smiles or self._static_system(s, SMILES_COL)
        msys = [k for k in self.metal_systems.get(m, {}) if k != s]
        out["n_systems_for_metal"] = len(self.metal_systems.get(m, {}))
        best_sys, best_sim = None, float("nan")
        fq = fingerprint(smi) if smi else None
        if fq is not None and msys:
            cands = [(k, fingerprint(self.system_smiles.get(k))) for k in sorted(msys)]
            cands = [(k, f) for k, f in cands if f is not None]
            if cands:
                sims = DataStructs.BulkTanimotoSimilarity(fq, [f for _, f in cands])
                j = int(np.argmax(sims))
                best_sys, best_sim = cands[j][0], float(sims[j])
        out["nearest_ligand_system"] = best_sys
        out["nearest_ligand_tanimoto"] = best_sim
        out["nearest_ligand_distance"] = 1.0 - best_sim if np.isfinite(best_sim) else float("nan")

        # publications
        out["n_publications_pair"] = out["exact_pair_publications"]
        out["n_publications_system"] = len(self.system_pubs.get(s, ()))
        out["n_publications_metal"] = len(self.metal_pubs.get(m, ()))

        # conditions
        d_sys, k = self.condition_distance(s, conditions)
        d_acid, _ = self.condition_distance(s, conditions, same_acid_anion=True) \
            if conditions is not None and not _missing(conditions.get(ACID_ANION_COL)) else (float("nan"), 0)
        d_pair, _ = self.condition_distance(s, conditions, metal_state=m)
        out["condition_distance_system"] = d_sys
        out["condition_distance_system_same_acid"] = d_acid
        out["condition_distance_pair"] = d_pair
        out["condition_dims_used"] = k

        # series neighbours
        neigh = series_neighbour_labels(m)
        present = {off: lab for off, lab in neigh.items() if lab in self.system_metals.get(s, {})}
        out["series_neighbours_present"] = "|".join(present[o] for o in sorted(present))
        out["n_series_neighbours_pm1"] = sum(1 for o in present if abs(o) == 1)
        out["n_series_neighbours_pm2"] = len(present)
        out["series_bracketed"] = any(o < 0 for o in present) and any(o > 0 for o in present)
        return out

    def _nearest_radius(self, props: Mapping[str, Any], oprops: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        res = {"nearest_radius_metal": None, "nearest_radius_distance_A": float("nan"),
               "nearest_radius_same_charge": False, "nearest_radius_same_species_charge": False,
               "nearest_radius_basis": "NA", "n_radius_neighbours_within_tol": 0,
               "radius_bracket_lower_metal": None, "radius_bracket_upper_metal": None,
               "radius_bracketed_same_charge": False}
        q_charge, q_species = props.get("charge"), props.get("species_charge")
        same = {k for k, p in oprops.items() if q_charge is not None and p.get("charge") == q_charge}
        same_species = {k for k in same if q_species is not None and oprops[k].get("species_charge") == q_species}
        pools = ((same_species, True, True), (same, True, False), (set(oprops), False, False))
        for pool, is_same, is_same_species in pools:
            if not pool:
                continue
            for basis, key in (("CN8", "r_cn8"), ("CN6", "r_cn6")):
                rq = props.get(key, np.nan)
                if not np.isfinite(rq):
                    continue
                cands = {k: float(oprops[k][key]) for k in pool if np.isfinite(oprops[k].get(key, np.nan))}
                if not cands:
                    continue
                best = min(sorted(cands), key=lambda k: abs(cands[k] - rq))
                same_r = {k: float(oprops[k][key]) for k in same if np.isfinite(oprops[k].get(key, np.nan))}
                lower = sorted((k for k in same_r if same_r[k] < rq), key=lambda k: (rq - same_r[k], k))
                upper = sorted((k for k in same_r if same_r[k] > rq), key=lambda k: (same_r[k] - rq, k))
                res.update(nearest_radius_metal=best, nearest_radius_distance_A=float(abs(cands[best] - rq)),
                           nearest_radius_same_charge=is_same, nearest_radius_same_species_charge=is_same_species,
                           nearest_radius_basis=basis,
                           n_radius_neighbours_within_tol=sum(1 for r in same_r.values()
                                                              if abs(r - rq) <= self.radius_tol + 1e-12))
                if is_same:
                    res.update(radius_bracket_lower_metal=lower[0] if lower else None,
                               radius_bracket_upper_metal=upper[0] if upper else None,
                               radius_bracketed_same_charge=bool(lower and upper))
                return res
        return res


def support_features(metal_state: str, system: str, conditions: Mapping[str, Any] | None,
                     train_df: pd.DataFrame, *, systems: pd.DataFrame | None = None,
                     radius_tol: float = RADIUS_NEIGHBOUR_TOL_A, family: str | None = None,
                     smiles: str | None = None) -> dict[str, Any]:
    """Convenience wrapper: fit a :class:`SupportIndex` on ``train_df`` and return
    :meth:`SupportIndex.features` for one request.  For many requests on one fold, build the index
    once and call ``features`` repeatedly."""
    return SupportIndex(train_df, systems=systems, radius_tol=radius_tol).features(
        metal_state, system, conditions, family=family, smiles=smiles)


def system_components(system: Any, component_map: Mapping[str, str] | None = None) -> set[str]:
    """Component identities of an ``extractant_system_key`` (``|``-split SMILES, mapped through
    ``component_map`` when given, e.g. to a stereo-free parent key)."""
    parts = str(system).split("|")
    return {component_map.get(c, c) for c in parts} if component_map else set(parts)


def systems_sharing_component(system_keys: Iterable[Any], system: str,
                              component_map: Mapping[str, str] | None = None) -> set[str]:
    """The OTHER systems among ``system_keys`` that share at least one component with ``system``."""
    comps = system_components(system, component_map)
    return {k for k in set(system_keys) if not _missing(k) and k != system and system_components(k, component_map) & comps}


def hide_cell(df: pd.DataFrame, metal_state: str, system: str, include_unknown_state_alias: bool = True,
              component_aware: bool = False, component_map: Mapping[str, str] | None = None) -> pd.DataFrame:
    """``df`` without the ``(metal_state, system)`` rows -- and, by default, without the same element's
    unknown-state rows under that system (the alias rows that would otherwise leak the cell).

    ``component_aware=True`` is the registered V5 / V6 hiding.  The same STATE-LEVEL rule is applied in the
    cell's own system and in every other system that shares a component with ``system`` (TODGA|DHOA when
    Nd(III) x TODGA is hidden): rows of the hidden metal state, plus (by default) the element's X(?) rows.
    Other known states of the element stay in training everywhere -- Pu(IV) rows under TODGA|DHOA stay
    when Pu(VI) x TODGA is hidden; they are a different species, and element-level transfer is what V2
    tests.  ``component_map`` changes what "shares a component" means (default: identical canonical
    SMILES)."""
    check_columns(df, (METAL_COL, ELEMENT_COL, SYSTEM_COL))
    scope = df[SYSTEM_COL] == system
    if component_aware:
        sharing = systems_sharing_component(df[SYSTEM_COL].dropna().unique(), system, component_map)
        scope = scope | df[SYSTEM_COL].isin(sharing)
    drop = scope & (df[METAL_COL] == metal_state)
    sym = metal_properties(metal_state)["symbol"]
    if include_unknown_state_alias:
        drop = drop | (scope & df[METAL_COL].isna() & (df[ELEMENT_COL] == sym))
    return df[~drop.fillna(False).astype(bool)]


def hide_cells(df: pd.DataFrame, cells: Iterable[tuple[str, str]], **kwargs: Any) -> pd.DataFrame:
    """:func:`hide_cell` applied for every ``(metal_state, system)`` in ``cells`` (the V6 double cell, a
    V5 batch); each cell is hidden under the same state-level rule."""
    out = df
    for m, sy in cells:
        out = hide_cell(out, m, sy, **kwargs)
    return out
