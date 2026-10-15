"""Support/query design for one held-out cell: the fixed query set, two measurement models, the
closure rules and the label-free support policies.

Inside a cell every pairwise ``log SF`` is a difference of two per-metal ``log D`` values, so the
pairs form a complete graph whose values are exactly additive around every cycle.

Two measurement models are run (PRE_REGISTRATION.md §3):

* ``metal`` (registered, laboratory-faithful): a measurement is one metal's ``log D`` reading in
  the cell.  After ``k + 1`` metals are read, every separation factor among them is known, so the
  support set is the set ``S`` of read metals (revealed to adapters as the star spanning tree
  ``(s0, si)``; with the shared-reading noise model the choice of tree is immaterial) and the
  budget ``k = |S| - 1`` is the number of independent separation factors.  The query is fixed:
  every strict-adjacent pair of the cell, which forces ``S`` to contain no adjacent pair
  (``S`` is an independent set of the adjacency path); a metal is legal only if adding it keeps the
  maximum independent set reachable (path DP), so every policy reaches ``k_max``.
* ``pair`` (the brief's definition, exploratory): a measurement is one separation factor; the
  support set is a forest of non-adjacent pairs; no component may contain both metals of a query
  (per-pair closure), with the bipartite-lookahead rule so every policy reaches ``n_metals - 2``.
  Supports still fix linear combinations of the adjacent queries (any support is a sum of adjacent
  contrasts), so every pair-model row carries ``lab_known`` (both endpoints already read) and the
  arithmetic-only ``PROJ`` arm.

Nothing in this module reads a label: policies see metal identities, the query set, and (for the
covariance policies) a covariance estimated elsewhere from training data.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .data import N_LN, Z_VEC, CellView, is_gen13_adjacent, is_strict_adjacent, stable_seed

K_GRID: tuple[int, ...] = (0, 1, 2, 3, 5, 6, 8, 10)   # 6 = the metal model's maximum on a 14-metal cell
PAIR_POLICIES: tuple[str, ...] = ("RANDOM", "WIDEST", "UNIFORM", "DOPT_CURVE", "DOPT_Q")
METAL_POLICIES: tuple[str, ...] = ("RANDOM", "UNIFORM", "DOPT_CURVE", "DOPT_Q")
MODELS: tuple[str, ...] = ("metal", "pair")
N_RANDOM_DRAWS = 5


@dataclass(frozen=True)
class CellDesign:
    cell: CellView
    query: tuple[tuple[int, int], ...]          # fixed query pairs (lighter-first)
    candidates: tuple[tuple[int, int], ...]     # pair model: pairs a policy may measure
    query_name: str = "adjacent"
    model: str = "metal"

    @property
    def k_max(self) -> int:
        if self.model == "pair":
            return len(self.cell.metals) - (2 if self.query else 1)
        return max(max_independent_size(self.cell.metals, self.query, frozenset()) - 1, 0)


def make_design(cell: CellView, query_name: str = "adjacent", model: str = "metal") -> CellDesign:
    pairs = cell.pairs
    if query_name == "adjacent":
        q = tuple(p for p in pairs if is_strict_adjacent(*p))
    elif query_name == "adjacent_gen13":
        q = tuple(p for p in pairs if is_gen13_adjacent(*p))
    else:
        raise ValueError(query_name)
    if model not in MODELS:
        raise ValueError(model)
    qs = set(q)
    return CellDesign(cell=cell, query=q, candidates=tuple(p for p in pairs if p not in qs), query_name=query_name, model=model)


# --------------------------------------------------------------------------------------
# pair model: the forest and the per-pair closure rule
# --------------------------------------------------------------------------------------
def components(metals, edges) -> list[frozenset]:
    parent = {i: i for i in metals}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    comp: dict[int, set] = {}
    for i in metals:
        comp.setdefault(find(i), set()).add(i)
    return [frozenset(s) for s in comp.values()]


def _independent(s: frozenset, query) -> bool:
    return not any(a in s and b in s for a, b in query)


def _conflict_bipartite(comps, query) -> bool:
    m = len(comps)
    adj = {i: set() for i in range(m)}
    for i, j in itertools.combinations(range(m), 2):
        if not _independent(comps[i] | comps[j], query):
            adj[i].add(j)
            adj[j].add(i)
    colour: dict[int, int] = {}
    for s in range(m):
        if s in colour:
            continue
        colour[s] = 0
        stack = [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in colour:
                    colour[v] = 1 - colour[u]
                    stack.append(v)
                elif colour[v] == colour[u]:
                    return False
    return True


def closed_pairs(metals, edges, pairs) -> set[tuple[int, int]]:
    """Pairs whose two metals lie in one component of the support forest (arithmetically determined)."""
    comps = components(metals, edges)
    where = {i: k for k, c in enumerate(comps) for i in c}
    return {p for p in pairs if where[p[0]] == where[p[1]]}


def read_metals(supports) -> frozenset:
    return frozenset(m for e in supports for m in e)


def legal_candidates(design: CellDesign, edges) -> list[tuple[int, int]]:
    comps = components(design.cell.metals, edges)
    where = {i: k for k, c in enumerate(comps) for i in c}
    out = []
    for a, b in design.candidates:
        if (a, b) in edges or where[a] == where[b]:
            continue
        merged = comps[where[a]] | comps[where[b]]
        if not _independent(merged, design.query):
            continue
        new = [c for k, c in enumerate(comps) if k not in (where[a], where[b])] + [merged]
        if _conflict_bipartite(new, design.query):
            out.append((a, b))
    return out


# --------------------------------------------------------------------------------------
# metal model: independent sets of the adjacency path, with a maximality lookahead
# --------------------------------------------------------------------------------------
def max_independent_size(metals, query, chosen: frozenset) -> int:
    """Largest set of metals containing ``chosen`` with no query pair inside it, or -1 if ``chosen``
    itself contains a query pair.  Query pairs are strict-adjacent pairs, i.e. edges of a path in
    Z order, so a left-to-right DP over the cell's metals is exact."""
    ms = sorted(metals, key=lambda m: int(Z_VEC[m]))
    qs = set(query)
    if any(a in chosen and b in chosen for a, b in qs):
        return -1
    NEG = -10**6
    # dp over positions: best size with metal i taken / not taken
    take, skip = NEG, 0
    prev = None
    for m in ms:
        forced = m in chosen
        edge = prev is not None and ((prev, m) in qs or (m, prev) in qs)
        new_take = (skip if edge else max(take, skip)) + 1
        new_skip = NEG if forced else max(take, skip)
        take, skip = new_take, new_skip
        prev = m
    return max(take, skip)


def legal_metals(design: CellDesign, read: frozenset) -> list[int]:
    target = max_independent_size(design.cell.metals, design.query, read)
    out = []
    for m in design.cell.metals:
        if m in read:
            continue
        if max_independent_size(design.cell.metals, design.query, read | {m}) == target:
            out.append(m)
    return out


def star_pairs(order) -> list[tuple[int, int]]:
    """Supports revealed for an ordered metal set: each later metal against the first."""
    if len(order) < 2:
        return []
    s0 = order[0]
    return [(min(s0, m), max(s0, m)) for m in order[1:]]


def assert_metal_set(design: CellDesign, order) -> None:
    if len(set(order)) != len(order):
        raise AssertionError("a metal is read twice")
    if any(m not in design.cell.metals for m in order):
        raise AssertionError("a read metal is not measured in this cell")
    read = frozenset(order)
    bad = [q for q in design.query if q[0] in read and q[1] in read]
    if bad:
        raise AssertionError(f"query pairs with both metals read: {bad}")


# --------------------------------------------------------------------------------------
# shared closure validator (both models)
# --------------------------------------------------------------------------------------
def assert_no_closure(design: CellDesign, supports) -> None:
    """Pair model: no support is a query, none repeats, none closes a cycle, no query is closed.
    Metal model: supports must be the star tree of an admissible metal set."""
    qs = set(design.query)
    if any(p in qs for p in supports):
        raise AssertionError("a support pair is a query pair")
    if len(set(supports)) != len(supports):
        raise AssertionError("a support pair repeats")
    for n in range(1, len(supports) + 1):
        if closed_pairs(design.cell.metals, supports[:n - 1], [supports[n - 1]]):
            raise AssertionError("a support pair closes a cycle (it is redundant with earlier supports)")
    bad = closed_pairs(design.cell.metals, supports, design.query)
    if bad:
        raise AssertionError(f"query pairs arithmetically determined by the supports: {sorted(bad)}")
    if design.model == "metal" and supports:
        read = read_metals(supports)
        bad = [q for q in design.query if q[0] in read and q[1] in read]
        if bad:
            raise AssertionError(f"query pairs with both metals read: {bad}")


# --------------------------------------------------------------------------------------
# policies (label-free; each returns one nested support sequence of pairs)
# --------------------------------------------------------------------------------------
def _lex(p):
    return (int(Z_VEC[p[0]]), int(Z_VEC[p[1]]))


def _pair_vector(p) -> np.ndarray:
    d = np.zeros(N_LN)
    d[p[0]], d[p[1]] = 1.0, -1.0
    return d


def _posterior_update(post, d, noise_var):
    v = post @ d
    return post - np.outer(v, v) / (float(d @ v) + noise_var)


def _gain_curve(post, d, noise_var) -> float:
    v = post @ d
    den = float(d @ v) + noise_var
    return float(v @ v) / den if den > 1e-12 else -np.inf


def _gain_query(post, d, Q, noise_var) -> float:
    v = post @ d
    den = float(d @ v) + noise_var
    return float(((Q @ v) ** 2).sum()) / den if den > 1e-12 else -np.inf


# ---- pair model -------------------------------------------------------------------------
def _greedy_pairs(design: CellDesign, k: int, score) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for _ in range(min(k, design.k_max)):
        legal = legal_candidates(design, edges)
        if not legal:
            break
        best = max(legal, key=lambda p: (score(edges, p), tuple(-z for z in _lex(p))))
        edges.append(best)
    return edges


def widest(design: CellDesign, k: int) -> list[tuple[int, int]]:
    return _greedy_pairs(design, k, lambda edges, p: int(Z_VEC[p[1]] - Z_VEC[p[0]]))


def uniform(design: CellDesign, k: int) -> list[tuple[int, int]]:
    """Farthest-point coverage of the Z axis: widest first, then the pair whose endpoints are
    farthest (summed minimum |Z - Z_used|) from the metals already measured."""
    def score(edges, p):
        if not edges:
            return int(Z_VEC[p[1]] - Z_VEC[p[0]])
        used = read_metals(edges)
        return sum(min(abs(int(Z_VEC[m]) - int(Z_VEC[u])) for u in used) for m in p)
    return _greedy_pairs(design, k, score)


def random_policy(design: CellDesign, k: int, *, split_seed: int, draw: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(stable_seed("gen20", design.cell.cell_id, split_seed, "random", draw))
    edges: list[tuple[int, int]] = []
    for _ in range(min(k, design.k_max)):
        legal = sorted(legal_candidates(design, edges), key=_lex)
        if not legal:
            break
        edges.append(legal[int(rng.integers(len(legal)))])
    return edges


def dopt_curve(design: CellDesign, k: int, cov: np.ndarray, noise_var: float) -> list[tuple[int, int]]:
    post = cov.copy()
    edges: list[tuple[int, int]] = []
    for _ in range(min(k, design.k_max)):
        legal = legal_candidates(design, edges)
        if not legal:
            break
        best = max(legal, key=lambda p: (_gain_curve(post, _pair_vector(p), noise_var), tuple(-z for z in _lex(p))))
        post = _posterior_update(post, _pair_vector(best), noise_var)
        edges.append(best)
    return edges


def dopt_query(design: CellDesign, k: int, cov: np.ndarray, noise_var: float) -> list[tuple[int, int]]:
    Q = np.array([_pair_vector(q) for q in design.query]) if design.query else np.zeros((0, N_LN))
    post = cov.copy()
    edges: list[tuple[int, int]] = []
    for _ in range(min(k, design.k_max)):
        legal = legal_candidates(design, edges)
        if not legal:
            break
        best = max(legal, key=lambda p: (_gain_query(post, _pair_vector(p), Q, noise_var), tuple(-z for z in _lex(p))))
        post = _posterior_update(post, _pair_vector(best), noise_var)
        edges.append(best)
    return edges


# ---- metal model ------------------------------------------------------------------------
def _metal_sequence(design: CellDesign, k: int, choose) -> list[int]:
    """``choose(read_order, legal) -> metal``; returns the ordered metal set of size min(k, k_max) + 1."""
    order: list[int] = []
    target = min(k, design.k_max) + 1
    while len(order) < target:
        legal = legal_metals(design, frozenset(order))
        if not legal:
            break
        order.append(choose(order, legal))
    return order


def metal_random(design: CellDesign, k: int, *, split_seed: int, draw: int) -> list[int]:
    rng = np.random.default_rng(stable_seed("gen20", design.cell.cell_id, split_seed, "random_metal", draw))
    return _metal_sequence(design, k, lambda order, legal: sorted(legal, key=lambda m: int(Z_VEC[m]))[int(rng.integers(len(legal)))])


def metal_uniform(design: CellDesign, k: int) -> list[int]:
    """Lightest legal metal first, then the heaviest, then farthest-point coverage of the Z axis."""
    def choose(order, legal):
        if not order:
            return min(legal, key=lambda m: int(Z_VEC[m]))
        if len(order) == 1:
            return max(legal, key=lambda m: int(Z_VEC[m]))
        return max(legal, key=lambda m: (min(abs(int(Z_VEC[m]) - int(Z_VEC[u])) for u in order), -int(Z_VEC[m])))
    return _metal_sequence(design, k, choose)


def metal_dopt(design: CellDesign, k: int, cov: np.ndarray, noise_var: float, *, target: str) -> list[int]:
    """First metal: the lightest legal one (a single reading carries no contrast).  Then the metal
    whose contrast against the first metal gives the largest information gain (curve or query)."""
    Q = np.array([_pair_vector(q) for q in design.query]) if design.query else np.zeros((0, N_LN))
    state = {"post": cov.copy()}

    def choose(order, legal):
        if not order:
            return min(legal, key=lambda m: int(Z_VEC[m]))
        s0 = order[0]

        def gain(m):
            d = _pair_vector((min(s0, m), max(s0, m)))
            return _gain_curve(state["post"], d, noise_var) if target == "curve" else _gain_query(state["post"], d, Q, noise_var)
        best = max(legal, key=lambda m: (gain(m), -int(Z_VEC[m])))
        state["post"] = _posterior_update(state["post"], _pair_vector((min(s0, best), max(s0, best))), noise_var)
        return best
    return _metal_sequence(design, k, choose)


def support_sequence(design: CellDesign, how: str, k: int, *, split_seed: int = 0, draw: int = 0,
                     cov: np.ndarray | None = None, noise_var: float = 0.09) -> list[tuple[int, int]]:
    """One nested support sequence (pairs) for the design's model; budget ``k`` uses the first k."""
    if design.model == "pair":
        if how == "RANDOM":
            seq = random_policy(design, k, split_seed=split_seed, draw=draw)
        elif how == "WIDEST":
            seq = widest(design, k)
        elif how == "UNIFORM":
            seq = uniform(design, k)
        elif how == "DOPT_CURVE":
            seq = dopt_curve(design, k, cov, noise_var)
        elif how == "DOPT_Q":
            seq = dopt_query(design, k, cov, noise_var)
        else:
            raise ValueError(how)
    else:
        if how == "RANDOM":
            order = metal_random(design, k, split_seed=split_seed, draw=draw)
        elif how == "UNIFORM":
            order = metal_uniform(design, k)
        elif how == "DOPT_CURVE":
            order = metal_dopt(design, k, cov, noise_var, target="curve")
        elif how == "DOPT_Q":
            order = metal_dopt(design, k, cov, noise_var, target="query")
        else:
            raise ValueError(how)
        assert_metal_set(design, order)
        seq = star_pairs(order)
    assert_no_closure(design, seq)
    return seq


@dataclass(frozen=True)
class Support:
    """The only object an adapter may see from a held-out cell: measured pairs and their values."""
    pairs: tuple[tuple[int, int], ...]
    values: tuple[float, ...]

    def as_list(self) -> list[tuple[int, int, float]]:
        return [(a, b, float(v)) for (a, b), v in zip(self.pairs, self.values)]

    @property
    def k(self) -> int:
        return len(self.pairs)

    @property
    def read(self) -> frozenset:
        return read_metals(self.pairs)


def reveal(y_row: np.ndarray, pairs) -> Support:
    """Reveal the true values of the support pairs (and nothing else) from the cell's log D row."""
    vals = tuple(float(y_row[a] - y_row[b]) for a, b in pairs)
    if any(np.isnan(v) for v in vals):
        raise ValueError("a support pair is not measured in this cell")
    return Support(pairs=tuple(pairs), values=vals)
