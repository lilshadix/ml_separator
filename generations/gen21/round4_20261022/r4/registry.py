"""ID -> configuration for round 4 (EXPERIMENT_REGISTRY.csv is the authoritative list).

``R3:<ID>``  a round-3 configuration rebuilt by round 3's own registry (exact reproduction; not a new slot).
``REF``      the strict references B0 / B1 / B2 (PROTOCOL.md section 4) with the HIST_D7 diagnostic and
             round 3's R14 column for an identity check, scored with the fixed baselines on the same rows.
``S01`` ``S02`` ``S03``   stage-1 block A slots (r4.slots_a), each with B0 / B1 / B2 on the same rows.
``S04``      S02 with the electronic metal map permuted (5 seeded draws) + S02 itself on the same rows.
``S21:<route>``   the chemistry block-shuffle control (5 seeded draws) of route S01 or S02 + the route itself.
"""
from __future__ import annotations

from . import paths
from . import refs as RF
from . import slots_a as SA
from . import slots_b as SB
from . import slots_c as SCc
from r3 import configs as CF  # noqa: E402
from r3 import registry as R3G  # noqa: E402

CF.INNER_CACHE_DIR = paths.PRED / "_inner"      # never share round 3's inner cache
SHOW = ("B0", "B1", "B2", "HIST_D7", "R14_REPRO", "PWL_Z", "HBK@ML", "RIDGE2", "LINE", "G14@ZERO", "FLAT")
REF_PAIRS = (("B1", "B0"), ("B2", "B0"), ("B2", "B1"), ("HIST_D7", "B1"), ("B1", "R14_REPRO"), ("B0", "HBK@ML"),
             ("B0", "PWL_Z"))


def _ref_method(part: str):
    def fn(block, cell, obs, k):
        return RF.predict_levels(block["fitted"][part], obs, cell["metals"])
    return fn


def _hist_d7(block, cell, obs, k):
    return RF.predict_levels(RF.HIST_D7_FIT, obs, cell["metals"])


def _cls(f: dict, cell) -> str:
    """Donor class of a cell under the fitted slot's class rule (plain data, so fits pickle)."""
    m = f["class_map"]
    return SA.donor_class(str(cell["extractant"])) if m is None else m[str(cell["extractant"])]


def _slot_method(part: str):
    def fn(block, cell, obs, k):
        f = block["fitted"][part]
        return SA.predict_levels(f, obs, cell["metals"], _cls(f, cell))
    return fn


def _slot_spec(part: str, bench):
    """part = 'slot:S01' | 'slot:S04:<draw>' | 'slot:S21:<route>:<draw>' -> (spec, class map or None = donor rule)."""
    bits = part.split(":")
    sid = bits[1]
    if sid in SA.SPECS:
        return dict(SA.SPECS[sid]), None
    if sid == "S04":
        return dict(SA.SPECS["S02"], perm=SA.perm(SA.PERM_SEEDS[int(bits[2])])), None
    if sid == "S21":
        return dict(SA.SPECS[bits[2]]), SA.shuffled_class_map(bench, SA.SHUFFLE_SEEDS[int(bits[3])])
    raise KeyError(part)


def _b_mode(part: str, bench) -> dict:
    """Block B: part = 'b:S05' | 'b:S06' | 'b:S22' | 'b:S09:<draw>' -> CatBoost correction mode."""
    bits = part.split(":")
    sid = bits[1]
    if sid == "S05":
        return {"geometry": None}
    if sid == "S06":
        return {"geometry": "own"}
    if sid == "S22":
        return {"geometry": "radius_only"}
    if sid == "S09":
        return {"geometry": "shuffled", "map": SB.geometry_shuffle_map(bench, SB.GEO_SHUFFLE_SEEDS[int(bits[2])])}
    raise KeyError(part)


def _b_method(part: str):
    def fn(block, cell, obs, k):
        return SB.predict_levels(block["fitted"][part], obs, cell)
    return fn


def _c_method(part: str):
    def fn(block, cell, obs, k):
        return SCc.predict_levels(block["fitted"][part], obs, cell)
    return fn


def _slot_config(cid, cols: dict, extra=()) -> CF.Config:
    """cols: column name -> slot part.  Every slot config carries B0 / B1 / B2 on the same rows."""
    m = {"B1": _ref_method("ref:B1"), "B2": _ref_method("ref:B2")}
    m.update({c: (_b_method(p) if p.startswith("b:") else _c_method(p) if p.startswith("c:") else _slot_method(p))
              for c, p in cols.items()})
    pairs = [(c, r) for c in cols for r in ("B1", "B2", "B0")] + list(extra)
    cfg = CF.Config(cid, "A", f"stage-1 block A {cid}", "-", ("hbk", "ref:B1", "ref:B2") + tuple(cols.values()),
                    methods=m, ks=(3,), primary=tuple(cols), contrasts=tuple(pairs))
    cfg.r3 = False
    cfg.slot_cols = dict(cols)
    return cfg


def get(cid: str, bench) -> CF.Config:
    if cid.startswith("R3:"):
        cfg = R3G.get(cid[3:], bench)
        cfg.cid = "R3-" + cfg.cid
        cfg.r3 = True
        return cfg
    if cid == "REF":
        base = CF.baseline_methods(bench)
        m = {"B0": base["PWL_R"], "B1": _ref_method("ref:B1"), "B2": _ref_method("ref:B2"), "HIST_D7": _hist_d7,
             "R14_REPRO": CF.corrector_method("cor:Dnc")}
        cfg = CF.Config("REF", "REF", "strict references B0 / B1 / B2", "-", ("hbk", "cor:Dnc", "ref:B1", "ref:B2"),
                        methods=m, ks=(3,), primary=("B1", "B2", "HIST_D7"), contrasts=REF_PAIRS)
        cfg.r3 = False
        return cfg
    if cid in ("S01", "S02", "S03"):
        cfg = _slot_config(cid, {cid: f"slot:{cid}"})
    elif cid == "S04":
        cols = {"S02": "slot:S02", **{f"S04_d{i}": f"slot:S04:{i}" for i in range(SA.N_DRAWS)}}
        cfg = _slot_config(cid, cols, extra=[("S02", f"S04_d{i}") for i in range(SA.N_DRAWS)])
    elif cid.startswith("S21:") and cid.split(":")[1] in ("S01", "S02"):
        route = cid.split(":")[1]
        cols = {route: f"slot:{route}", **{f"S21_d{i}": f"slot:S21:{route}:{i}" for i in range(SA.N_DRAWS)}}
        cfg = _slot_config(cid.replace(":", "-"), cols, extra=[(route, f"S21_d{i}") for i in range(SA.N_DRAWS)])
    elif cid in ("S05", "S06"):
        cfg = _slot_config(cid, {cid: f"b:{cid}"})
    elif cid == "S09":
        cols = {"S06": "b:S06", **{f"S09_d{i}": f"b:S09:{i}" for i in range(SB.N_DRAWS)}}
        cfg = _slot_config(cid, cols, extra=[("S06", f"S09_d{i}") for i in range(SB.N_DRAWS)])
    elif cid == "S22":
        cfg = _slot_config(cid, {"S06": "b:S06", "S22": "b:S22"}, extra=[("S06", "S22")])
    elif cid == "S11":
        cfg = _slot_config(cid, {"S11": "c:S11"})
    elif cid == "S13":
        cfg = _slot_config(cid, {"S11": "c:S11", "S13": "c:S13"}, extra=[("S13", "S11")])
    elif cid in ("S14", "S15", "S24"):
        cfg = _slot_config(cid, {"S11": "c:S11", "S13": "c:S13", cid: f"c:{cid}"}, extra=[("S13", cid), ("S13", "S11")])
    elif cid == "S16":
        cols = {"S11": "c:S11", "S13": "c:S13", **{f"S16_d{i}": f"c:S16:{i}" for i in range(len(SCc.PERM_SEEDS))}}
        cfg = _slot_config(cid, cols, extra=[("S13", f"S16_d{i}") for i in range(len(SCc.PERM_SEEDS))] + [("S13", "S11")])
    else:
        raise KeyError(f"{cid} is not registered (EXPERIMENT_REGISTRY.csv)")
    base = CF.baseline_methods(bench)
    cfg.methods = {"B0": base["PWL_R"], **cfg.methods}
    return cfg


def fit(cfg, bench, rec, Y):
    if getattr(cfg, "r3", False):
        return R3G.fit(cfg, bench, rec, Y)
    own = [p for p in cfg.parts if p.startswith(("ref:", "slot:", "b:", "c:"))]
    out = CF.fit_parts(bench, rec, tuple(p for p in cfg.parts if p not in own), Y, None, version=cfg.version)
    for p in own:
        if p.startswith("ref:"):
            out[p] = RF.fit_reference(bench, rec, Y, p.split(":")[1])
    for p in own:
        if p.startswith("slot:"):
            spec, cmap = _slot_spec(p, bench)
            f = SA.fit_slot(bench, rec, Y, spec, out["ref:B1"], lambda cell, m=cmap: _cls({"class_map": m}, cell))
            f["class_map"] = cmap
            out[p] = f
    for p in own:
        if p.startswith("b:"):
            out[p] = SB.fit_slot(bench, rec, Y, _b_mode(p, bench), out["ref:B1"])
    for p in own:
        if p.startswith("c:"):
            bits = p.split(":")
            out[p] = SCc.fit_slot(bench, rec, Y, bits[1], out["ref:B1"], draw=int(bits[2]) if len(bits) > 2 else 0)
    return out


def base_methods(bench, cfg):
    return CF.baseline_methods(bench)


cells_of = R3G.cells_of
data_matrix = R3G.data_matrix
