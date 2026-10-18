"""Label-free support policies added in gen21.

``SERIES_STAGGER`` (attempt 3): the held-out cells of one condition series (same extractant, same
publication) are ordered by a hash of their cell id; cell j reads, greedily, the legal metal read
by the fewest earlier cells of its series (their full sequences), then the one farthest in Z from
its own reads so far, then the lighter one.  For the first cell of a series (and for any singleton)
this is exactly gen20's UNIFORM order (lightest legal, heaviest legal, farthest-point).  Legality is
gen20's: the read set stays an independent set of the cell's adjacency path and k_max stays
reachable, so no query pair of a cell is ever read at both ends *in that cell*.  It reads metal
identities and the series composition only, never a label.
"""
from __future__ import annotations

import hashlib

from gen20k import design as DS  # noqa: E402
from gen20k.data import Z_VEC  # noqa: E402

from .cache import KMAX

_MEMO: dict = {}


def _order_key(cell) -> str:
    return hashlib.sha256(str(cell["cell_id"]).encode()).hexdigest()


def _sequence(cell, cover: dict) -> list[int]:
    from .evaluate import _view
    dz = DS.make_design(_view(cell), "adjacent", "metal")
    target = min(dz.k_max, KMAX) + 1
    order: list[int] = []
    while len(order) < target:
        legal = DS.legal_metals(dz, frozenset(order))
        if not legal:
            break

        def key(m):
            spread = min(abs(int(Z_VEC[m]) - int(Z_VEC[u])) for u in order) if order else 0
            return (-cover.get(m, 0), spread, -int(Z_VEC[m]))
        order.append(max(legal, key=key))
    DS.assert_metal_set(dz, order)
    return order


def series_stagger(block: dict, cell: dict):
    cells = block["cells"]
    memo_key = (id(cells), len(cells), cells[0]["cell_id"] if cells else "")
    seqs = _MEMO.get(memo_key)
    if seqs is None:
        seqs = {}
        groups: dict = {}
        for c in cells:
            groups.setdefault((c["extractant"], c["publication"]), []).append(c)
        for members in groups.values():
            cover: dict = {}
            for c in sorted(members, key=_order_key):
                order = _sequence(c, cover)
                for m in order:
                    cover[m] = cover.get(m, 0) + 1
                seqs[c["ci"]] = DS.star_pairs(order)
        _MEMO[memo_key] = seqs
    return seqs[cell["ci"]]
