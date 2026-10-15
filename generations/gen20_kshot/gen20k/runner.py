"""The evaluation grid: designs x seeds x folds x measurement models x held-out cells x policies x
budgets x arms.

For every fold: fit the zero-shot arms on the (masked) training cells; build each curve arm's
residual covariance from inner cross-fits of the same training fold; choose every free quantity
(noise factor, ridge penalty, neural configuration, k-shot mechanism and its penalty) on the fold's
inner split with a *nested* covariance, recording the inner scores of the baselines beside the
candidates; then for every held-out cell with at least one adjacent pair, fix the query set, run
every policy to the cell's maximum budget under each measurement model (supports chosen without
any query label, using the reference covariance so that every arm sees identical supports), and
at every budget adapt every arm and write one row per (model, policy, draw, k, pair) with one
prediction column per arm@adapter.

The long table is the only output; every metric and every interval is computed from it later.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import bootstrap
from . import adapters as AD
from . import arms as AR
from . import cov as CV
from . import design as DS
from . import nnc as NN
from .data import LANTHANIDES, Z_VEC, cell_view, pair_value
from .splits import folds_for

K_GRID = DS.K_GRID
FULL_POLICIES = {"metal": ("RANDOM", "UNIFORM", "DOPT_CURVE", "DOPT_Q", "UNC_SEQ"),
                 "pair": ("RANDOM", "WIDEST", "UNIFORM", "DOPT_CURVE", "DOPT_Q", "UNC_SEQ")}
LIGHT_POLICIES = {"metal": ("RANDOM", "UNIFORM", "DOPT_Q"), "pair": ("RANDOM", "WIDEST", "DOPT_Q")}
GEN15_NOISE_VAR = 0.09


@dataclass
class RunConfig:
    label: str
    designs: tuple[str, ...] = ("BP", "B")
    seeds: tuple[int, ...] = (104729, 130363, 155921, 196613, 262147)
    models: tuple[str, ...] = ("metal", "pair")
    query_name: str = "adjacent"
    k_grid: tuple[int, ...] = K_GRID
    n_random_draws: int = 5
    full_policy_designs: tuple[str, ...] = ("BP", "B")
    remaining_regime_designs: tuple[str, ...] = ("BP", "B")
    remaining_regime_sequences: tuple[tuple[str, int], ...] = (("DOPT_Q", 0), ("UNIFORM", 0), ("RANDOM", 0))
    arms: tuple[str, ...] = ("FLAT", "G14", "MEAN_CURVE", "NN_TANIMOTO", "NNC")
    nnc_configs: tuple[NN.NNCConfig, ...] = NN.CONFIG_GRID
    nnc_inner_seeds: int = 2
    nnc_outer_seeds: int = 8
    lam_head_grid: tuple[float, ...] = (0.1, 1.0, 10.0)
    lam_full_grid: tuple[float, ...] = (1e-3, 1e-2, 1e-1)
    lam_ridge_grid: tuple[float, ...] = (0.3, 1.0, 3.0, 10.0)
    noise_factor_grid: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    selection_ks: tuple[int, ...] = (0, 3)
    full_ft_policy: str = "DOPT_Q"
    full_ft_ks: tuple[int, ...] = (1, 3, 5, 10)
    full_ft_members: int = 4
    full_ft_steps: int = 30
    max_cells: int | None = None            # smoke runs only
    smoke: bool = False


# --------------------------------------------------------------------------------------
# inner selection (training fold only; nested covariance)
# --------------------------------------------------------------------------------------
def _ladder_mae(bench, cells, curve_fn, cov, noise_var, ks, *, model="metal", policy="DOPT_Q", query_name="adjacent"):
    """Adjacent MAE of ``curve_fn(ci, support) -> curve`` over ``cells`` at the budgets ``ks``."""
    errs = {k: [] for k in ks}
    for ci in cells:
        dz = DS.make_design(cell_view(bench, ci), query_name, model)
        if not dz.query or dz.k_max < max(ks):
            continue
        seq = DS.support_sequence(dz, policy, max(ks), cov=cov, noise_var=noise_var)
        yq = np.array([pair_value(bench.Y[ci], a, b) for a, b in dz.query])
        for k in ks:
            sup = DS.reveal(bench.Y[ci], seq[:k])
            p = AD.pairs_from_curve(curve_fn(ci, sup), dz.query)
            errs[k].extend(np.abs(p - yq).tolist())
    return {int(k): (float(np.mean(v)) if v else float("nan")) for k, v in errs.items()}


def _score(rec_scores: dict, ks) -> float:
    vals = [rec_scores.get(k, rec_scores.get(str(k), np.nan)) for k in ks]
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else np.inf


def select_on_inner(bench, fold, cfg: RunConfig) -> dict:
    """Every free choice is made here, on the fold's inner split: fit on the (masked) inner-training
    cells, score the adjacent ladder (metal model, DOPT_Q supports) on the inner-validation cells.
    The covariance used inside is estimated from inner cross-fits of the inner-training cells only
    (nested), so no scored cell's residual enters it.  Baseline arms are scored on the same ladder so
    the registered arm rule (PRE_REGISTRATION §4) can be decided from this record alone."""
    itr = CV.inner_mask(bench, fold.design, np.asarray(fold.inner_train_index), np.asarray(fold.inner_validation_index))
    iva = np.asarray(fold.inner_validation_index)
    if set(int(i) for i in itr) & set(int(i) for i in iva):
        raise AssertionError("inner-training and inner-validation cells overlap")
    ictx = CV.make_ctx(bench, fold, itr, iva)
    ifold = replace(fold, train_index=itr)                  # nested: covariance from itr's own inner cross-fits
    facts = AR.standard_arms()
    g14 = facts["G14"]().fit(ictx)
    flat = facts["FLAT"]().fit(ictx)
    cov_g14 = CV.fold_covariance(bench, ifold, AR.as_fit_curves(facts["G14"]))
    CV.assert_cov_provenance(cov_g14, itr)
    s2_robust = cov_g14.noise_s2
    ks = cfg.selection_ks
    rec = {"design": fold.design, "split_seed": fold.seed, "fold": fold.fold, "n_inner_train": int(len(itr)),
           "n_inner_val": int(len(iva)), "s2_robust": s2_robust, "scores": {}, "query_name": cfg.query_name}
    zero_g14 = {int(ci): g14.curves([ci])[0] for ci in iva}
    zero_flat = {int(ci): flat.curves([ci])[0] for ci in iva}
    # noise factor for every BLUP20 arm and for the covariance policies (G14@BLUP20 ladder decides)
    for f in cfg.noise_factor_grid:
        s2 = s2_robust * f
        rec["scores"][f"G14@BLUP20@noise{f}"] = _ladder_mae(
            bench, iva, lambda ci, sup, s2=s2: AD.blup(zero_g14[ci], cov_g14.cov, sup, CV.noise_matrix(sup.pairs, s2)),
            cov_g14.cov, 2 * s2, ks, query_name=cfg.query_name)
    best_f = min(cfg.noise_factor_grid, key=lambda f: _score(rec["scores"][f"G14@BLUP20@noise{f}"], ks))
    rec["noise_factor"] = float(best_f)
    s2 = s2_robust * best_f
    rec["s2"] = float(s2)
    nv = 2 * s2
    rec["scores"]["G14@BLUP20"] = rec["scores"][f"G14@BLUP20@noise{best_f}"]
    rec["scores"]["FLAT@BLUP20"] = _ladder_mae(bench, iva, lambda ci, sup: AD.blup(zero_flat[ci], cov_g14.cov, sup, CV.noise_matrix(sup.pairs, s2)),
                                               cov_g14.cov, nv, ks, query_name=cfg.query_name)
    rec["scores"]["LINE"] = _ladder_mae(bench, iva, lambda ci, sup: AD.naive_line(bench.basis, sup), cov_g14.cov, nv, ks, query_name=cfg.query_name)
    rec["scores"]["PROJ"] = _ladder_mae(bench, iva, lambda ci, sup: AD.project(sup), cov_g14.cov, nv, ks, query_name=cfg.query_name)
    prior = g14.prior_coef()
    for lam in cfg.lam_ridge_grid:
        rec["scores"][f"RIDGE2@lam{lam}"] = _ladder_mae(
            bench, iva, lambda ci, sup, lam=lam: AD.ridge_line(bench.basis, sup, prior, lam, CV.noise_matrix(sup.pairs, s2)),
            cov_g14.cov, nv, ks, query_name=cfg.query_name)
    rec["ridge_lambda"] = float(min(cfg.lam_ridge_grid, key=lambda l: _score(rec["scores"][f"RIDGE2@lam{l}"], ks)))
    if "NNC" not in cfg.arms:
        return rec
    for c in cfg.nnc_configs:
        c2 = NN.NNCConfig(**{**c.__dict__, "n_seeds": cfg.nnc_inner_seeds})
        arm = NN.NNCArm(c2).fit(ictx)
        cov_nn = CV.fold_covariance(bench, ifold, AR.as_fit_curves(lambda c2=c2: NN.NNCArm(c2)))
        CV.assert_cov_provenance(cov_nn, itr)
        zero = {int(ci): arm.curves([ci])[0] for ci in iva}
        rec["scores"][f"NNC[{c.name}]@BLUP20"] = _ladder_mae(
            bench, iva, lambda ci, sup: AD.blup(zero[ci], cov_nn.cov, sup, CV.noise_matrix(sup.pairs, s2)), cov_g14.cov, nv, ks, query_name=cfg.query_name)
        for lam in cfg.lam_head_grid:
            rec["scores"][f"NNC[{c.name}]@HEAD_FT@lam{lam}"] = _ladder_mae(
                bench, iva, lambda ci, sup, lam=lam: arm.head_ft_curves(ci, sup, lam).mean(0), cov_g14.cov, nv, ks, query_name=cfg.query_name)
        for lam in cfg.lam_full_grid:
            rec["scores"][f"NNC[{c.name}]@FULL_FT@lam{lam}"] = _ladder_mae(
                bench, iva, lambda ci, sup, lam=lam: arm.full_ft_curves(ci, sup, lam, steps=cfg.full_ft_steps, members=cfg.full_ft_members).mean(0),
                cov_g14.cov, nv, ks, query_name=cfg.query_name)
    cands = {k: v for k, v in rec["scores"].items() if k.startswith("NNC[")}
    key = min(cands, key=lambda k: _score(cands[k], ks))
    cfg_name = key.split("[")[1].split("]")[0]
    mech = key.split("@")[1]
    rec.update({"nnc_config": cfg_name, "nnc_mechanism": mech, "selected_key": key,
                "nnc_lambda": float(key.split("lam")[1]) if "lam" in key else None})
    # per-mechanism penalties for the selected configuration (used by the NNC@HEAD_FT / NNC@FULL_FT columns)
    rec["lambda_head"] = float(min(cfg.lam_head_grid, key=lambda l: _score(rec["scores"][f"NNC[{cfg_name}]@HEAD_FT@lam{l}"], ks)))
    rec["lambda_full"] = float(min(cfg.lam_full_grid, key=lambda l: _score(rec["scores"][f"NNC[{cfg_name}]@FULL_FT@lam{l}"], ks)))
    # the registered arm rule, decided here from inner scores at k = 3 (PRE_REGISTRATION §4)
    k3 = max(ks)
    nnc_k3 = cands[key].get(k3, np.nan)
    g14_k3 = rec["scores"]["G14@BLUP20"].get(k3, np.nan)
    rec["mstar_inner"] = {"nnc_k3": float(nnc_k3), "g14_blup20_k3": float(g14_k3),
                          "nnc_wins": bool(np.isfinite(nnc_k3) and np.isfinite(g14_k3) and nnc_k3 < g14_k3)}
    return rec


# --------------------------------------------------------------------------------------
# the fold loop
# --------------------------------------------------------------------------------------
def run_fold(bench, fold, cfg: RunConfig, log) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    ctx = CV.make_ctx(bench, fold, fold.train_index, fold.test_index)
    facts = AR.standard_arms()
    fitted = {n: facts[n]().fit(ctx) for n in cfg.arms if n in facts}
    covs = {n: CV.fold_covariance(bench, fold, AR.as_fit_curves(facts[n])) for n in fitted if n != "FLAT"}
    covs["FLAT"] = covs["G14"]                     # FLAT+BLUP uses the reference covariance (no chemistry)
    for c in covs.values():
        CV.assert_cov_provenance(c, fold.train_index)
    ref = covs["G14"]
    sel = select_on_inner(bench, fold, cfg)
    s2 = ref.noise_s2 * sel["noise_factor"]
    nv = 2 * s2
    nn_arm = None
    if "NNC" in cfg.arms:
        c = next(c for c in cfg.nnc_configs if c.name == sel["nnc_config"])
        nn_arm = NN.NNCArm(NN.NNCConfig(**{**c.__dict__, "n_seeds": cfg.nnc_outer_seeds})).fit(ctx)
        ci2 = NN.NNCConfig(**{**c.__dict__, "n_seeds": cfg.nnc_inner_seeds})
        covs["NNC"] = CV.fold_covariance(bench, fold, AR.as_fit_curves(lambda ci2=ci2: NN.NNCArm(ci2)))
        CV.assert_cov_provenance(covs["NNC"], fold.train_index)
    prior = fitted["G14"].prior_coef() if "G14" in fitted else np.zeros(2)
    full = fold.design in cfg.full_policy_designs
    n_draws = cfg.n_random_draws if full else 1
    remaining = fold.design in cfg.remaining_regime_designs
    kmax_grid = max(cfg.k_grid)
    test_cells = [int(c) for c in fold.test_index]
    if cfg.max_cells:
        test_cells = test_cells[: cfg.max_cells]
    rows = []
    zero_curves = {n: a.curves(test_cells) for n, a in fitted.items()}
    if nn_arm is not None:
        zero_curves["NNC"] = nn_arm.curves(test_cells)
    mech = sel.get("nnc_mechanism")
    for j, ci in enumerate(test_cells):
        cv = cell_view(bench, ci)
        yrow = bench.Y[ci]
        for model in cfg.models:
            dz = DS.make_design(cv, cfg.query_name, model)
            if not dz.query or dz.k_max < 1:
                continue
            kmax = min(dz.k_max, kmax_grid)
            policies = (FULL_POLICIES if full else LIGHT_POLICIES)[model]
            seqs: dict[tuple[str, int], list] = {}
            for pol in policies:
                if pol == "RANDOM":
                    for d in range(n_draws):
                        seqs[(pol, d)] = DS.support_sequence(dz, pol, kmax, split_seed=fold.seed, draw=d)
                elif pol == "UNC_SEQ":
                    if nn_arm is None:
                        continue
                    seqs[(pol, 0)] = unc_seq(bench, dz, nn_arm, ci, kmax, sel.get("lambda_head", 1.0))
                else:
                    seqs[(pol, 0)] = DS.support_sequence(dz, pol, kmax, cov=ref.cov, noise_var=nv)
            for (pol, d), seq in seqs.items():
                for k in cfg.k_grid:
                    if k > len(seq):
                        continue
                    sup = DS.reveal(yrow, seq[:k])
                    DS.assert_no_closure(dz, list(sup.pairs))
                    R = CV.noise_matrix(sup.pairs, s2, "shared")
                    R15 = GEN15_NOISE_VAR * np.eye(sup.k)
                    preds: dict[str, np.ndarray] = {}
                    for n, c0 in zero_curves.items():
                        c = c0[j]
                        preds[f"{n}@ZERO"] = c
                        preds[f"{n}@BLUP20"] = AD.blup(c, covs[n].cov, sup, R)
                        if n == "G14":
                            preds[f"{n}@BLUP_INDEP"] = AD.blup(c, covs[n].cov, sup, R15)
                    preds["LINE"] = AD.naive_line(bench.basis, sup)
                    preds["PROJ"] = AD.project(sup)
                    preds["RIDGE2"] = AD.ridge_line(bench.basis, sup, prior, sel["ridge_lambda"], R)
                    if "NN_TANIMOTO" in zero_curves:
                        preds["NN_TANIMOTO@RESCALE"] = AD.rescale(zero_curves["NN_TANIMOTO"][j], sup)
                    if nn_arm is not None:
                        preds["NNC@HEAD_FT"] = nn_arm.head_ft_curves(ci, sup, sel["lambda_head"]).mean(0)
                        want_full = (mech == "FULL_FT") or (pol == cfg.full_ft_policy and k in cfg.full_ft_ks)
                        if want_full:
                            preds["NNC@FULL_FT"] = nn_arm.full_ft_curves(ci, sup, sel["lambda_full"], steps=cfg.full_ft_steps,
                                                                         members=cfg.full_ft_members).mean(0)
                        preds["NNC@SELECTED"] = {"BLUP20": preds["NNC@BLUP20"], "HEAD_FT": preds["NNC@HEAD_FT"],
                                                 "FULL_FT": preds.get("NNC@FULL_FT")}[mech]
                    read = sup.read
                    closed = DS.closed_pairs(cv.metals, list(sup.pairs), cv.pairs)
                    targets = [(a, b, "Q") for a, b in dz.query]
                    if remaining and (pol, d) in cfg.remaining_regime_sequences:
                        sset = set(sup.pairs); qset = set(dz.query)
                        targets += [(a, b, "R") for a, b in cv.pairs if (a, b) not in sset and (a, b) not in qset and (a, b) not in closed
                                    and not (a in read and b in read)]
                    for a, b, regime in targets:
                        rec = {"design": fold.design, "split_seed": fold.seed, "fold": fold.fold, "cell_id": cv.cell_id,
                               "extractant": cv.extractant, "chemotype": cv.chemotype, "publication": cv.publication,
                               "n_metals": len(cv.metals), "model": model, "k_max": int(dz.k_max), "policy": pol, "draw": d,
                               "k": k, "n_read": len(read), "A": LANTHANIDES[a], "B": LANTHANIDES[b], "ia": a, "ib": b,
                               "dZ": int(Z_VEC[b] - Z_VEC[a]), "regime": regime, "lab_known": bool(a in read and b in read),
                               "y": pair_value(yrow, a, b)}
                        for n, c in preds.items():
                            rec[n] = float(c[a] - c[b]) if c is not None else np.nan
                        rows.append(rec)
        if log and (j + 1) % 50 == 0:
            log(f"    {fold.design} seed {fold.seed} fold {fold.fold}: {j + 1}/{len(test_cells)} cells, {len(rows)} rows, {time.time() - t0:.0f}s")
    table = pd.DataFrame(rows)
    meta = {"design": fold.design, "split_seed": fold.seed, "fold": fold.fold, "n_train": int(len(fold.train_index)),
            "n_test": int(len(fold.test_index)), "n_rows": int(len(table)), "seconds": time.time() - t0,
            "selection": sel, "s2_used": float(s2),
            "covariance": {n: {"n_rows": c.n_rows, "min_eig_before": c.min_eig_before, "noise_s2_robust": c.noise_s2,
                               "source": c.source, "n_cells": len(c.cells)} for n, c in covs.items()},
            "nnc_fit_log": nn_arm.fit_log if nn_arm is not None else None}
    return table, meta


def unc_seq(bench, dz: DS.CellDesign, nn_arm, ci: int, k: int, lam: float) -> list:
    """Sequential acquisition with the neural ensemble: after each revealed support the heads are
    refitted and the next measurement is the legal candidate with the largest ensemble sd of its
    predicted contrast.  Reads support labels only.  Metal model: candidates are legal metals,
    scored by the sd of their contrast against the first read metal."""
    if dz.model == "pair":
        edges: list = []
        for _ in range(min(k, dz.k_max)):
            legal = DS.legal_candidates(dz, edges)
            if not legal:
                break
            members = nn_arm.head_ft_curves(ci, DS.reveal(bench.Y[ci], edges), lam)
            best = max(legal, key=lambda p: (float((members[:, p[0]] - members[:, p[1]]).std()), -int(Z_VEC[p[0]]), -int(Z_VEC[p[1]])))
            edges.append(best)
        DS.assert_no_closure(dz, edges)
        return edges
    order: list = []
    target = min(k, dz.k_max) + 1
    while len(order) < target:
        legal = DS.legal_metals(dz, frozenset(order))
        if not legal:
            break
        if not order:
            order.append(min(legal, key=lambda m: int(Z_VEC[m])))
            continue
        s0 = order[0]
        members = nn_arm.head_ft_curves(ci, DS.reveal(bench.Y[ci], DS.star_pairs(order)), lam)
        best = max(legal, key=lambda m: (float((members[:, s0] - members[:, m]).std()), -int(Z_VEC[m])))
        order.append(best)
    DS.assert_metal_set(dz, order)
    seq = DS.star_pairs(order)
    DS.assert_no_closure(dz, seq)
    return seq


def run(bench, cfg: RunConfig, out_dir, log=print) -> dict:
    from pathlib import Path
    out_dir = Path(out_dir) if str(out_dir).startswith("/") else bootstrap.PREDICTIONS / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"label": cfg.label, "designs": {}, "config": {k: (v if not isinstance(v, tuple) or not v or not hasattr(v[0], "__dict__") else [x.__dict__ for x in v]) for k, v in cfg.__dict__.items()}}
    for design in cfg.designs:
        folds = folds_for(bench.frame, design, seeds=list(cfg.seeds))
        parts, metas = [], []
        for f in folds:
            t, m = run_fold(bench, f, cfg, log)
            parts.append(t); metas.append(m)
            log(f"  {design} seed {f.seed} fold {f.fold}: {m['n_rows']} rows in {m['seconds']:.0f}s")
        table = pd.concat(parts, ignore_index=True)
        for c in table.columns:
            if table[c].dtype == np.float64 and c != "y":
                table[c] = table[c].astype(np.float32)
        path = out_dir / f"pairs_{design}.parquet"
        table.to_parquet(path, index=False)
        with open(out_dir / f"folds_{design}.json", "w") as fh:
            json.dump(metas, fh, indent=1, default=str)
        summary["designs"][design] = {"rows": int(len(table)), "path": str(path), "seconds": float(sum(m["seconds"] for m in metas))}
        log(f"{design}: {len(table)} rows -> {path}")
    with open(out_dir / "run_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    return summary
