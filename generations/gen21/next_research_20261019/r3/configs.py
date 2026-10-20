"""Registry configurations in code (the authoritative list with hypotheses is EXPERIMENT_REGISTRY.csv).

A config = (fit parts, predictor methods, policy, data version, budgets).  Fixed baselines are always
scored on the same rows and supports: PWL_R, PWL_Z, LINE, RIDGE2, HBK@ML, G14@ZERO, FLAT.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import corrector as CO
from . import interp
from . import paths  # noqa: F401
from g21 import evaluate as E  # noqa: E402
from g21 import methods as M  # noqa: E402
from gen20k import adapters as AD  # noqa: E402

BASELINES = ("PWL_R", "PWL_Z", "LINE", "RIDGE2", "HBK@ML", "G14@ZERO", "FLAT")


def baseline_methods(bench):
    g20 = E.gen20_baselines(bench)

    def pwl_r(block, cell, obs, k):
        return interp.pwl_levels(obs.y_masked, cell["metals"], "R")

    def pwl_z(block, cell, obs, k):
        return interp.pwl_levels(obs.y_masked, cell["metals"], "Z")

    def hbk_ml(block, cell, obs, k):
        f = block["fitted"]["hbk"]
        return f[f"HBK[{f['ml_kernel']}]"].predict(obs.support)
    return {"PWL_R": pwl_r, "PWL_Z": pwl_z, "LINE": g20["LINE"], "RIDGE2": g20["RIDGE2"], "HBK@ML": hbk_ml,
            "G14@ZERO": g20["G14@ZERO"], "FLAT": g20["FLAT"]}


@dataclass
class Config:
    cid: str
    family: str
    hypothesis: str
    contract: str
    parts: tuple                         # fit parts: 'hbk', 'cor:<fset>', 'inner', 'pol_err', ...
    methods: dict = field(default_factory=dict)
    policy: object = "DOPT_Q"
    ks: tuple = (0, 1, 2, 3)
    K: int = 3
    version: str = "V1"
    primary: tuple = ()                  # the config's own prediction columns
    contrasts: tuple = ()


INNER_CACHE_DIR = paths.PRED / "_inner"            # the final runner redirects this next to its own fold cache


def fit_parts(bench, rec, parts, Y, chem=None, version: str | None = None):
    out = {}
    for p in parts:
        if p == "hbk":
            out["hbk"] = M.attempt4_fit(bench, rec)
        elif p.startswith("cor:"):
            # 'cor:<fset>[:robust|:shuffle]'; 'cor:TASKFS' = feature set {G, GP} chosen by inner task loss
            bits = p.split(":")
            fset, flag = bits[1], (bits[2] if len(bits) > 2 else "")
            kk = 3
            if flag.startswith("k"):                 # 'cor:<fset>:k5' = trained at budget 5 (R20)
                kk, flag = int(flag[1:]), ""
            if fset == "TASKFS":
                cand = {fs: CO.fit_corrector(bench, rec, Y, fset=fs) for fs in ("G", "GP")}
                best = min(cand, key=lambda fs: cand[fs]["inner_best"])
                out[p] = dict(cand[best], taskfs_scores={fs: cand[fs]["inner_best"] for fs in cand})
                continue
            chem_of = (lambda ci: chem.vector(ci, rec)) if (chem is not None and ("C" in fset or "D" in fset)) else None
            out[p] = CO.fit_corrector(bench, rec, Y, fset=fset, chem_of=chem_of, n_chem=(chem.n if chem_of else 0),
                                      robust=(flag == "robust"), shuffle=(flag == "shuffle"), k=kk,
                                      shuffle_all=(flag == "shuffleall"))
        elif p == "inner":
            from . import inner_preds as IP
            import pickle
            tag = version or version_tag(Y, bench)
            cp = INNER_CACHE_DIR / f"{rec['design']}_{rec['split_seed']}_{rec['fold']}_{tag}.pkl"
            cp.parent.mkdir(parents=True, exist_ok=True)
            if cp.exists():                      # shared by every config that needs it (identical computation)
                with open(cp, "rb") as fh:
                    out["inner"], out["inner_lik"] = pickle.load(fh)
            else:
                out["inner"], out["inner_lik"] = IP.inner_table(bench, rec)
                with open(cp.with_suffix(".tmp"), "wb") as fh:
                    pickle.dump((out["inner"], out["inner_lik"]), fh)
                cp.with_suffix(".tmp").replace(cp)
        elif p.startswith("inner_cor:"):
            fs = p.split(":")[1]
            out["inner_cor"] = inner_corrector_column(bench, rec, Y, fs, chem if ("C" in fs or "D" in fs) else None)
        elif p == "pol_err":
            from . import policy as PO
            out["pol_err"] = PO.fit_error_table(bench, rec, Y)
        elif p.startswith("pol_err_k"):
            from . import policy as PO
            out[p] = PO.fit_error_table(bench, rec, Y, k=int(p[len("pol_err_k"):]))
        else:
            raise KeyError(p)
    return out


def version_tag(Y, bench) -> str:
    return "V1" if Y is bench.Y else "V2"


def corrector_method(part: str, chem=None):
    def fn(block, cell, obs, k):
        f = block["fitted"][part]
        c = chem.vector(cell["ci"], block["_rec"]) if (chem is not None and ("C" in f["fset"] or "D" in f["fset"])) else None
        return CO.predict_levels(f, obs, cell["metals"], c)
    return fn


def corrector_by_budget(fset: str, ks, chem=None):
    """R20: the corrector trained at the evaluated budget k (parts 'cor:<fset>:k<k>')."""
    fns = {k: corrector_method(f"cor:{fset}:k{k}", chem) for k in ks}

    def fn(block, cell, obs, k):
        return fns[k](block, cell, obs, k)
    return fn


def pchip_method(block, cell, obs, k):
    return interp.pchip_levels(obs.y_masked, cell["metals"], "R")


def hbk_task_method(mode: str):
    """B: HBK chosen by inner task loss.  mode 'K': kernel only (noise x1); 'KC': kernel x noise multiplier."""
    from . import inner_preds as IP

    def choice(block):
        f = block["fitted"]
        key = ("_task_" + mode)
        if key not in f:
            cands = [f"HBK[{kk}]x1.0" for kk in IP.KERNELS] if mode == "K" else \
                    [f"HBK[{kk}]x{c}" for kk in IP.KERNELS for c in IP.NOISE_MULT]
            best, sc = IP.task_choice(f["inner"], cands)
            f[key] = (best, sc)
        return f[key][0]

    def fn(block, cell, obs, k):
        best = choice(block)
        kern = best.split("[")[1].split("]")[0]
        c = float(best.rsplit("x", 1)[1])
        return IP.hbk_predict(block["fitted"]["hbk"][f"HBK[{kern}]"], obs.support, c)
    return fn


def stack_method(members=("PWL_R", "PWL_Z", "HBK@ML_inner"), step=0.05):
    """K4: convex blend fitted on the outer training fold's inner cross-fitted predictions (weighted squared
    error of hidden adjacent log SF; simplex grid); applied to the outer-fitted members."""
    import itertools

    def weights(block):
        f = block["fitted"]
        if "_stack_w" not in f:
            t = f["inner"]
            t = t[t.k == 3].copy()
            nq = t.groupby(["inner_fold", "ci"]).y.transform("size")
            wq = (t["w"] / nq).to_numpy()
            Pm = t[list(members)].to_numpy(float)
            y = t.y.to_numpy(float)
            grid = np.arange(0, 1 + 1e-9, step)
            best = None
            for w in itertools.product(grid, repeat=len(members) - 1):
                if sum(w) > 1 + 1e-9:
                    continue
                ww = np.array(list(w) + [1 - sum(w)])
                e = Pm @ ww - y
                s = float((wq * e * e).sum())
                if best is None or s < best[0]:
                    best = (s, ww)
            f["_stack_w"] = best[1]
        return f["_stack_w"]

    def fn(block, cell, obs, k):
        ww = weights(block)
        comps = {"PWL_R": interp.pwl_levels(obs.y_masked, cell["metals"], "R"),
                 "PWL_Z": interp.pwl_levels(obs.y_masked, cell["metals"], "Z")}
        f = block["fitted"]["hbk"]
        comps["HBK@ML_inner"] = f[f"HBK[{f['ml_kernel']}]"].predict(obs.support)
        # align the blend: HBK levels are centred (relative), PWL levels absolute; differences are what count,
        # so blend the level vectors after anchoring HBK to the mean of the read metals
        rr = list(obs.read)
        if rr:
            comps["HBK@ML_inner"] = comps["HBK@ML_inner"] + float(np.nanmean(obs.y_masked[rr] - comps["HBK@ML_inner"][rr]))
        out = np.zeros(14)
        for wgt, mname in zip(ww, members):
            out = out + wgt * np.nan_to_num(comps[mname], nan=0.0)
        return out
    return fn



STACK_COR_LAMBDA = 3.0          # fixed for the corrector member of R22 (declared before the run)


def inner_corrector_column(bench, rec, Y, fset, chem=None, lam=STACK_COR_LAMBDA):
    """For each inner block j: fit the corrector on block j's inner-training cells (masks rebuilt with block j's
    own nested covariance), predict block j's inner-validation DOPT_Q queries.  Returns {(inner_fold, ci, A, B): pred}."""
    out = {}
    pen = CO.penalty_vector(fset, chem.n if chem is not None else 0)
    for blk in rec["inner"]:
        tr_cells = set(map(int, blk["train_index"]))
        chem_of = (lambda ci: chem.vector(ci, rec)) if chem is not None else None
        fit_s = CO.training_situations(bench, rec, Y, fset=fset, chem_of=chem_of, cells_subset=tr_cells,
                                       dopt=(blk["cov"], 2 * blk["s2"]))
        rows = CO._stack(fit_s)
        coef = CO.ridge_fit(*rows, lam, pen)
        f = {"coef": coef, "fset": fset, "kind": "R"}
        for cell in blk["cells"]:
            if cell["k_max"] < 3:
                continue
            o = E.observe(Y[cell["ci"]], cell, cell["seqs"]["DOPT_Q"], 3)
            v = CO.predict_levels(f, o, cell["metals"], chem.vector(cell["ci"], rec) if chem is not None else None)
            for a, b in cell["query"]:
                out[(blk["inner_fold"], cell["ci"], a, b)] = float(v[a] - v[b])
    return out


def stack_with_corrector_method(part: str, fset: str, chem=None, step=0.05):
    """R22: convex blend of PWL_R, PWL_Z, HBK@ML and the corrector, weights fitted on the inner cross-fitted rows."""
    import itertools
    members = ("PWL_R", "PWL_Z", "HBK@ML_inner", "COR")

    def weights(block):
        f = block["fitted"]
        if "_stack4_w" not in f:
            t = f["inner"]
            t = t[t.k == 3].copy()
            t["COR"] = [f["inner_cor"].get((int(r.inner_fold), int(r.ci), int(r.A), int(r.B)), np.nan) for r in t.itertuples()]
            t = t[np.isfinite(t["COR"])]
            nq = t.groupby(["inner_fold", "ci"]).y.transform("size")
            wq = (t["w"] / nq).to_numpy()
            Pm, y = t[list(members)].to_numpy(float), t.y.to_numpy(float)
            grid = np.arange(0, 1 + 1e-9, step)
            best = None
            for w in itertools.product(grid, repeat=3):
                if sum(w) > 1 + 1e-9:
                    continue
                ww = np.array(list(w) + [1 - sum(w)])
                e = Pm @ ww - y
                sc = float((wq * e * e).sum())
                if best is None or sc < best[0]:
                    best = (sc, ww)
            f["_stack4_w"] = best[1]
        return f["_stack4_w"]

    cor = corrector_method(part, chem)

    def fn(block, cell, obs, k):
        ww = weights(block)
        comps = {"PWL_R": interp.pwl_levels(obs.y_masked, cell["metals"], "R"),
                 "PWL_Z": interp.pwl_levels(obs.y_masked, cell["metals"], "Z")}
        f = block["fitted"]["hbk"]
        h = f[f"HBK[{f['ml_kernel']}]"].predict(obs.support)
        rr = list(obs.read)
        comps["HBK@ML_inner"] = h + (float(np.nanmean(obs.y_masked[rr] - h[rr])) if rr else 0.0)
        comps["COR"] = cor(block, cell, obs, k)
        out = np.zeros(14)
        for wgt, mname in zip(ww, members):
            out = out + wgt * np.nan_to_num(comps[mname], nan=0.0)
        return out
    return fn
