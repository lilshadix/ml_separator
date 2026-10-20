"""ID -> Config (EXPERIMENT_REGISTRY.csv).  Every configuration also fits 'hbk' (attempt 4's fits), so the
fixed baselines PWL_R, PWL_Z, LINE, RIDGE2, HBK@ML, G14@ZERO, FLAT are scored on the same rows/supports."""
from __future__ import annotations

from . import configs as CF
from . import paths  # noqa: F401

SHOW_BASELINES = ("PWL_R", "PWL_Z", "HBK@ML", "RIDGE2", "LINE", "G14@ZERO", "FLAT")
_CHEM = None


def chem_block(bench):
    global _CHEM
    if _CHEM is None:
        from . import chem
        _CHEM = chem.ChemBlock(bench)
    return _CHEM


def get(cid: str, bench) -> CF.Config:
    pol_err = None
    if cid == "B0":
        return CF.Config("B0", "baseline", "fixed baselines", "-", ("hbk",), ks=(0, 1, 2, 3, 4, 5, 6), K=3)
    if cid == "R01":
        return CF.Config("R01", "A", "PCHIP_R", "-", ("hbk",), methods={"PCHIP_R": CF.pchip_method}, ks=(1, 2, 3),
                         primary=("PCHIP_R",))
    if cid in ("R02", "R03"):
        fset = {"R02": "G", "R03": "GP"}[cid]
        name = {"R02": "COR_G", "R03": "COR_GP"}[cid]
        return CF.Config(cid, "A", name, "-", ("hbk", f"cor:{fset}"), methods={name: CF.corrector_method(f"cor:{fset}")},
                         ks=(3,), primary=(name,))
    if cid == "R12":
        # best predictor (section-7 rule) on POL_ERR reads; configs/best_predictor_<ID>.txt names it
        from . import policy as PO
        spec = (paths.R3 / "configs" / f"best_predictor_{cid}.txt").read_text().strip()
        kind, val = spec.split("=")
        if kind == "cor":
            ch = chem_block(bench) if ("C" in val or "D" in val) else None
            part = "cor:TASKFS" if val == "TASKFS" else f"cor:{val}"
            m = {f"COR[{val}]": CF.corrector_method(part, ch)}
            parts = ("hbk", "pol_err", part)
        elif kind == "stack":
            m = {"STACK": CF.stack_method()}
            parts = ("hbk", "pol_err", "inner")
        else:
            raise KeyError(spec)
        return CF.Config(cid, "K", f"{spec} x POL_ERR", "-", parts, methods=m, policy=PO.PolErr(3), ks=(3,), primary=tuple(m))
    if cid == "R20":
        fset = _best_cor_fset()
        ch = chem_block(bench) if ("C" in fset or "D" in fset) else None
        name = f"COR[{fset}]@k"
        return CF.Config("R20", "F", name, "-", ("hbk",) + tuple(f"cor:{fset}:k{k}" for k in (3, 5, 6)),
                         methods={name: CF.corrector_by_budget(fset, (3, 5, 6), ch)}, ks=(3, 5, 6), K=6, primary=(name,))
    if cid in ("CTRL14W", "CTRL23W", "CTRL02W"):
        # PROTOCOL section 8 item 6 AS REGISTERED (within-pair-type permutation of the training residual targets,
        # the rule of R21) for the best learned candidates (2026-10-19, after the review found it had not been run
        # on them); leakage controls, not hypotheses, not selectable
        from . import policy as PO
        part, name, pol = {"CTRL14W": ("cor:Dnc:shuffle", "COR[Dnc]_SHUFFLED_W", "DOPT_Q"),
                           "CTRL23W": ("cor:Dnc:shuffle", "COR[Dnc]_SHUFFLED_W", PO.PolErr(3)),
                           "CTRL02W": ("cor:G:shuffle", "COR_G_SHUFFLED_W", "DOPT_Q")}[cid]
        parts = ("hbk", "pol_err", part) if not isinstance(pol, str) else ("hbk", part)
        m = {name: CF.corrector_method(part)}
        return CF.Config(cid, "CTRL", f"section 8.6 control ({cid})", "-", parts, methods=m, policy=pol, ks=(3,),
                         primary=tuple(m))
    if cid == "CTRL23":
        # post-hoc control for R23 (NOT PROTOCOL section 8.6, which permutes within pair type; see CTRL23W), not selectable:
        # the same pipeline with the training residual targets permuted across ALL rows (pair types and cells)
        from . import policy as PO
        m = {"COR[Dnc]_SHUFFLEDALL": CF.corrector_method("cor:Dnc:shuffleall")}
        return CF.Config("CTRL23", "CTRL", "R23 with all-row shuffled targets", "-", ("hbk", "pol_err", "cor:Dnc:shuffleall"),
                         methods=m, policy=PO.PolErr(3), ks=(3,), primary=tuple(m))
    if cid == "R14":
        # slot R14 (released) re-used for the REQUIRED crossing control of R23 (brief section 9, item 1):
        # R23's method (PWL_R + one Gd parameter) on gen20's DOPT_Q reads; not a new hypothesis
        m = {"COR[Dnc]": CF.corrector_method("cor:Dnc")}
        return CF.Config("R14", "CTRL", "PWL + one Gd parameter x DOPT_Q", "-", ("hbk", "cor:Dnc"), methods=m, ks=(3,),
                         primary=tuple(m))
    if cid == "R18":
        # R23 (separate POL_ERR regime) on data version V2 (POL_ERR and the Gd parameter refitted on V2 training rows),
        # against the V2 baselines on the same reads; defined when R23 was wrongly taken as the section-7 candidate
        # (AMENDMENTS D-9: the rule-defined R18 would be R02 on V2, which was not run)
        from . import policy as PO
        m = {"COR[Dnc]": CF.corrector_method("cor:Dnc")}
        return CF.Config("R18", "E", "R23 on V2", "-", ("hbk", "pol_err", "cor:Dnc"), methods=m, policy=PO.PolErr(3), ks=(3,),
                         primary=tuple(m), version="V2")
    if cid == "R23":
        # AMENDMENT 1: PWL_R + ONE shrunk shared Gd-dip parameter (no geometry, no chemistry) on POL_ERR reads
        from . import policy as PO
        m = {"COR[Dnc]": CF.corrector_method("cor:Dnc")}
        return CF.Config("R23", "AMEND", "PWL + one Gd parameter x POL_ERR", "-", ("hbk", "pol_err", "cor:Dnc"), methods=m,
                         policy=PO.PolErr(3), ks=(3,), primary=tuple(m))
    if cid == "R24":
        # AMENDMENT 2: budget F with placement: POL_ERR fitted at each k in (3, 5, 6), on C6; PWL baselines, HBK@ML and the
        # one-parameter Gd correction (R23's) trained at that k
        from . import policy as PO
        m = {"COR[Dnc]@k": CF.corrector_by_budget("Dnc", (3, 5, 6))}
        parts = ("hbk",) + tuple(f"pol_err_k{k}" for k in (3, 5, 6)) + tuple(f"cor:Dnc:k{k}" for k in (3, 5, 6))
        return CF.Config("R24", "AMEND", "budget x POL_ERR(k) on C6", "-", parts, methods=m, policy=PO.PolErrK(),
                         ks=(3, 5, 6), K=6, primary=tuple(m))
    if cid in ("R15", "R16"):
        # ablations of the best combination R12 (GD corrector x POL_ERR): R15 without the Gd block (G only);
        # R16 Gd dip without its chemistry interactions (GDnc)
        from . import policy as PO
        fset = {"R15": "G", "R16": "GDnc"}[cid]
        ch = chem_block(bench) if fset == "GD" else None
        part = f"cor:{fset}"
        m = {f"COR[{fset}]": CF.corrector_method(part, ch)}
        return CF.Config(cid, "K", f"ablation {fset} x POL_ERR", "-", ("hbk", "pol_err", part), methods=m,
                         policy=PO.PolErr(3), ks=(3,), primary=tuple(m))
    if cid == "R22":
        fset = _best_cor_fset()
        ch = chem_block(bench) if ("C" in fset or "D" in fset) else None
        part = f"cor:{fset}"
        return CF.Config("R22", "K", "STACK + corrector", "-", ("hbk", "inner", part, f"inner_cor:{fset}"),
                         methods={"STACK4": CF.stack_with_corrector_method(part, fset, ch), f"COR[{fset}]": CF.corrector_method(part, ch)},
                         ks=(3,), primary=("STACK4",), contrasts=(("STACK4", f"COR[{fset}]"),))
    if cid == "R17":
        return CF.Config("R17", "E", "V2 baselines", "-", ("hbk",), ks=(0, 1, 2, 3, 4, 5, 6), K=3, version="V2")
    if cid in ("R07", "R08"):
        fset, name = {"R07": ("GC", "COR_GC"), "R08": ("GD", "COR_GDC")}[cid]
        ch = chem_block(bench)
        return CF.Config(cid, "C", name, "-", ("hbk", f"cor:{fset}"), methods={name: CF.corrector_method(f"cor:{fset}", ch)},
                         ks=(3,), primary=(name,))
    if cid in ("R06", "R19", "R21"):
        bf = _best_cor_fset()
        part, name = {"R06": ("cor:TASKFS", "COR_TASKFS"), "R19": ("cor:G:robust", "ROBUST_COR"),
                      "R21": (f"cor:{bf}:shuffle", "COR_SHUFFLED")}[cid]
        fam = {"R06": "K", "R19": "E", "R21": "CTRL"}[cid]
        ch = chem_block(bench) if (cid == "R21" and ("C" in bf or "D" in bf)) else None
        return CF.Config(cid, fam, name, "-", ("hbk", part), methods={name: CF.corrector_method(part, ch)}, ks=(3,),
                         primary=(name,))
    if cid in ("R04", "R05"):
        mode = {"R04": "K", "R05": "KC"}[cid]
        name = {"R04": "HBK_TASK_K", "R05": "HBK_TASK_KC"}[cid]
        return CF.Config(cid, "B", name, "-", ("hbk", "inner"), methods={name: CF.hbk_task_method(mode)}, ks=(3,),
                         primary=(name,))
    if cid == "R13":
        return CF.Config("R13", "K", "STACK", "-", ("hbk", "inner"), methods={"STACK": CF.stack_method()}, ks=(3,),
                         primary=("STACK",))
    if cid in ("R09", "R10", "R11"):
        from . import policy as PO
        pol_err = PO.PolErr(3)
        return CF.Config(cid, "D", "POL_ERR", "-", ("hbk", "pol_err"), policy=pol_err, ks=(3,), primary=())
    raise KeyError(f"{cid} is not implemented yet (see EXPERIMENT_REGISTRY.csv)")


def _best_cor_fset() -> str:
    """R21 uses the best learned corrector's feature set (configs/best_corrector.txt once known; else G)."""
    f = paths.R3 / "configs" / "best_corrector.txt"
    return f.read_text().strip() if f.exists() else "G"


def base_methods(bench, cfg):
    return CF.baseline_methods(bench)


def fit(cfg, bench, rec, Y):
    chem = chem_block(bench) if any((p.startswith("cor:") or p.startswith("inner_cor:")) and
                                    ("C" in p.split(":")[1] or ("D" in p.split(":")[1] and "nc" not in p.split(":")[1]))
                                    for p in cfg.parts) else None
    return CF.fit_parts(bench, rec, cfg.parts, Y, chem, version=cfg.version)


def cells_of(cfg, blk):
    return [c for c in blk["cells"] if c["k_max"] >= min(cfg.ks)] if min(cfg.ks) > 0 else list(blk["cells"])


def data_matrix(version: str, bench):
    if version == "V1":
        return bench.Y
    if version == "V2":
        from . import datav2
        return datav2.matrix(bench)
    raise KeyError(version)
