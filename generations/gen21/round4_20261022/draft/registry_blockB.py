"""DRAFT additions to r4/registry.py for block B (amendment A2).  Merged into the registry when A2 is sealed."""
from __future__ import annotations

# --- to insert in r4/registry.py ---------------------------------------------------------------
# from . import slots_b as SB
#
# def _b_mode(part: str, bench) -> dict:
#     """part = 'b:S05' | 'b:S06' | 'b:S22' | 'b:S09:<draw>'."""
#     bits = part.split(":")
#     sid = bits[1]
#     if sid == "S05":
#         return {"geometry": None}
#     if sid == "S06":
#         return {"geometry": "own"}
#     if sid == "S22":
#         return {"geometry": "radius_only"}
#     if sid == "S09":
#         return {"geometry": "shuffled", "map": SB.geometry_shuffle_map(bench, SB.GEO_SHUFFLE_SEEDS[int(bits[2])])}
#     raise KeyError(part)
#
# def _b_method(part: str):
#     def fn(block, cell, obs, k):
#         return SB.predict_levels(block["fitted"][part], obs, cell)
#     return fn
#
# in get():
#     if cid in ("S05", "S06"):
#         cols = {cid: f"b:{cid}"}
#     elif cid == "S09":
#         cols = {"S06": "b:S06", **{f"S09_d{i}": f"b:S09:{i}" for i in range(SB.N_DRAWS)}}
#     elif cid == "S22":
#         cols = {"S06": "b:S06", "S22": "b:S22"}
#     -> a slot config with parts ("hbk", "ref:B1", "ref:B2") + parts, methods B0 / B1 / B2 / cols, contrasts vs B1 / B2 / B0
#        (+ S06 vs each control column)
#
# in fit():
#     for p in own: if p.startswith("b:"): out[p] = SB.fit_slot(bench, rec, Y, _b_mode(p, bench), out["ref:B1"])
#
# slot_identity_check: a block-B column with n_trees = 0 must equal B1 exactly (fitted_info 'lambda' -> 'n_trees').
