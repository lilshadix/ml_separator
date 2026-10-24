"""Stage-1 block C: selective An(III) transfer through a shared/private-head multitask scaffold (brief section 6
'Actinide transfer'; PROTOCOL section 12).  Every arm has the SAME architecture and the same Ln rows, targets, weights
and optimisation budget; arms differ only in the auxiliary data and how it enters.

Architecture (torch, CPU, float64, full batch, deterministic, one thread):
  encoder   E(z) -> h (8):  z = 8 RDKit descriptors (standardised on the fit's training Ln extractants; a scale below
            1e-6 is replaced by 1; standardised values clipped to +-5) + donor class (3);
            Linear(11, 16) - tanh - Linear(16, 8) - tanh
  main head (Ln selectivity residual): [h, r] -> correction at an unread metal; r = block B's metal / read-set /
            support-summary features WITHOUT chemistry (16, standardised on the fit's training rows; missing -> 0);
            Linear(24, 16) - tanh - Linear(16, 1).  Target: level residual t = y(m) - B1_S(m) of B1's training
            situations (B1's masks and weights); prediction at a held-out cell: level(m) = B1_S(m) + correction(m) at
            unread metals, read metals exact.
  Ln aux head (raw log D; every arm): [h, x(m), q(m)] -> log D of a training Ln cell at a measured metal;
            Linear(10, 16) - tanh - Linear(16, 1); row weight = chemotype-balanced cell weight / measured metals.
  private aux head (S13, S15, S16, S24): [h, 4-way metal one-hot] -> raw log D of the auxiliary domain;
            Linear(12, 16) - tanh - Linear(16, 1); row weight = 1 / (rows of its publication), so each publication
            weighs the same.
  loss = main weighted MSE / var(main targets) + w_Ln * Ln-aux weighted MSE / var(Ln-aux targets)
         + w_X * auxiliary weighted MSE / var(auxiliary targets)    (every term in units of its own target variance)
Optimiser: Adam, lr 0.01, weight decay WD (L2, every parameter), 400 full-batch epochs; 3 seeds averaged.
Selection (inner chemotype blocks, pair loss on inner-validation DOPT_Q masks, B1 refitted per block as in block B):
  S11  candidates {B1 (zero correction)} + {network with w_Ln in (0, 0.3, 1) x WD in (0.03, 0.3), w_X = 0};
       ties: B1, then the smaller w_Ln, then the larger WD
  S13  candidates {B1} + {network with (w_Ln, WD) = S11's best network, w_X in (0, 0.3, 1)}; ties: B1, then smaller w_X
       (w_X = 0 IS S11's best network, so S13 tunes exactly one more knob on the same Ln rows)
  controls use S13's grid with their own auxiliary data (inner tuning rerun per arm and draw):
  S14  naive shared head: the An rows go through the Ln aux head as pseudo-lanthanides with the same f count
       (Pu -> Sm, Am -> Eu, Cm -> Gd, Cf -> Dy), no private head; whenever w_X > 0 the Ln aux term enters with weight
       max(w_Ln, 0.3), so the head is always shared with Ln raw log D
  S15  source / diversity / weight-matched auxiliary control: the private head gets training-Ln raw log D rows matched
       to the An set's publication count, per-publication row counts, weights and (topped up across Ln publications)
       per-publication extractant count (metal one-hot = four Ln metal bins)
  S16  block-permuted An targets: each An extractant block takes the targets of another block (derangement; targets
       sampled with replacement inside the source block at the same metal); 5 seeded draws
  S24  metal-state-mapping shuffle: inside each An measurement series the metal labels are cycled over the series' own
       metal set (runs only if >= 10% of An rows change in every fold; label-free count, PROTOCOL section 12)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import an_data as AD
from . import refs as RF
from . import slots_a as SA
from . import slots_b as SB
from r3 import interp  # noqa: E402
from gen20k.data import N_LN  # noqa: E402
from gen13sep.metals import LANTHANIDES  # noqa: E402

W_LN_GRID = (0.0, 0.3, 1.0)
WD_GRID = (0.03, 0.3)
W_AN_GRID = (0.0, 0.3, 1.0)
S14_MIN_W_LN = 0.3
SEEDS = (0, 1, 2)
EPOCHS, LR = 400, 0.01
Z_CLIP, SD_FLOOR = 5.0, 1e-6
N_CHEM = 3 + len(SB.RD_DESC)              # donor one-hot + RDKit descriptors (last columns of block B's features)
ANALOGUE = {"Pu": "Sm", "Am": "Eu", "Cm": "Gd", "Cf": "Dy"}           # S14: same f count
PERM_SEEDS = tuple(20261323 + i for i in range(5))
MATCH_SEED = 20261423
NET_GRID = tuple((wl, wd) for wl in W_LN_GRID for wd in WD_GRID)


def net_key(wl: float, wd: float) -> str:
    return f"wl={wl}|wd={wd}"


def _z_raw(smiles: str) -> np.ndarray:
    cls = SA.donor_class(smiles)
    return np.array(list(SB.rdkit_desc(smiles)) + [float(cls == c) for c in SA.CLASSES], float)


def _metal_bin(m: int) -> int:
    return min(m // 4, 3) if m < 12 else 3


# ------------------------------------------------------------------------------------------
# data assembly for one fit (training cells only)
# ------------------------------------------------------------------------------------------
def _main_rows(sits, bench):
    X = np.vstack([s["X"][:, :-N_CHEM] for s in sits])
    t = np.concatenate([s["t"] for s in sits])
    w = np.concatenate([np.full(len(s["t"]), s["w"]) for s in sits])
    ext = [bench.frame.extractant.iat[s["ci"]] for s in sits for _ in range(len(s["t"]))]
    return X, t, w / w.sum(), ext


def _ln_aux_rows(bench, cells_ci):
    from gen13sep.amplitude_bench import cell_weights
    ci = np.array(sorted(cells_ci))
    cw = dict(zip(ci.tolist(), cell_weights(bench.groups[ci], bench.n_obs[ci]).tolist()))
    x = interp.X_R
    xs, ys, ws, ext = [], [], [], []
    for c in ci:
        ms = [m for m in range(N_LN) if np.isfinite(bench.Y[c, m])]
        for m in ms:
            xs.append([x[m], SB.Q[m]])
            ys.append(bench.Y[c, m])
            ws.append(cw[int(c)] / len(ms))
            ext.append(bench.frame.extractant.iat[c])
    w = np.array(ws)
    return np.array(xs), np.array(ys), w / w.sum(), ext


def _private_rows(an: pd.DataFrame, metal_key: str = "metal_symbol", order=AD.AN3):
    oh = np.array([[float(m == o) for o in order] for m in an[metal_key]])
    n_pub = an.groupby("publication").publication.transform("size").to_numpy(float)
    w = 1.0 / n_pub
    return oh, an.log_D.to_numpy(float), w / w.sum(), list(an.extractant)


def _matched_rows(bench, cells_ci, an: pd.DataFrame, seed: int):
    """S15: training-Ln raw log D rows matched to the An set's source structure: the same number of weight units
    ("publications"), the same per-unit row counts and weights (each unit weighs the same), and per unit as many
    distinct extractants as the An publication has.  Largest An publications are paired with the most diverse Ln
    publications (ties in a seeded order); when an Ln publication offers fewer extractants than needed, the unit is
    topped up with extractants of the next-ranked Ln publications (diversity is matched before source purity)."""
    rng = np.random.default_rng(seed)
    f = bench.frame
    by_pub = {}
    for c in sorted(cells_ci):
        ms = [m for m in range(N_LN) if np.isfinite(bench.Y[c, m])]
        if ms:
            by_pub.setdefault(str(f.publication_id.iat[c]), {})[c] = ms
    pubs = sorted(by_pub)
    order = rng.permutation(len(pubs))
    n_ext = {p: len({f.extractant.iat[c] for c in by_pub[p]}) for p in pubs}
    ranked = [pubs[order[i]] for i in sorted(range(len(pubs)), key=lambda i: (-n_ext[pubs[order[i]]], i))]
    target = sorted(((len(g), g.extractant.nunique()) for _, g in an.groupby("publication")), reverse=True)
    rows = []
    for j, (n, k) in enumerate(target):
        pick, cells = set(), {}
        for r in range(len(ranked)):
            src = by_pub[ranked[(j + r) % len(ranked)]]
            exts = sorted({f.extractant.iat[c] for c in src} - pick)
            take = {exts[i] for i in rng.permutation(len(exts))[:k - len(pick)]}
            pick |= take
            cells.update({c: ms for c, ms in src.items() if f.extractant.iat[c] in take})
            if len(pick) >= k:
                break
        pool = [(c, m) for c in sorted(cells) for m in cells[c]]
        rows += [(j,) + pool[i] for i in rng.integers(len(pool), size=n)]
    oh = np.array([[float(_metal_bin(m) == k) for k in range(4)] for _, _, m in rows])
    y = np.array([bench.Y[c, m] for _, c, m in rows])
    pid = pd.Series([j for j, _, _ in rows])
    w = 1.0 / pid.map(pid.value_counts()).to_numpy(float)
    return oh, y, w / w.sum(), [f.extractant.iat[c] for _, c, _ in rows]


def _permuted_an(an: pd.DataFrame, seed: int) -> pd.DataFrame:
    """S16: derangement of extractant blocks; each record takes a target sampled from its source block AT THE SAME
    METAL when the source block has that metal (otherwise from the whole source block), so the metal composition and
    the per-metal target distribution are kept while the extractant-target association is broken."""
    rng = np.random.default_rng(seed)
    blocks = sorted(an.extractant.unique())
    while True:
        perm = [blocks[i] for i in rng.permutation(len(blocks))]
        if len(blocks) < 2 or all(a != b for a, b in zip(blocks, perm)):
            break
    src = dict(zip(blocks, perm))
    pool_m = {k: g.log_D.to_numpy(float) for k, g in an.groupby(["extractant", "metal_symbol"])}
    pool = {e: g.log_D.to_numpy(float) for e, g in an.groupby("extractant")}
    new = []
    for e, m in zip(an.extractant, an.metal_symbol):
        p_ = pool_m.get((src[e], m), pool[src[e]])
        new.append(p_[rng.integers(len(p_))])
    return an.assign(log_D=new)


SERIES = ("publication", "extractant", "solvent_key", "acid_concentration_M", "extractant_primary_concentration_M")


def metal_cycled(an: pd.DataFrame) -> pd.Series:
    """S24: inside each measurement series (same publication, extractant, diluent, acid and extractant concentration)
    the series' own metal set, in AN3 order, is cycled (m_1 -> m_2 -> ... -> m_1), so every record of a multi-metal
    series carries another metal of the same series; single-metal series are unchanged.  (A global relabelling of the
    metal one-hot would be a pure reparametrisation of the private head and could not act as a control.)"""
    new = an.metal_symbol.copy()
    for _, idx in an.groupby(list(SERIES), dropna=False).groups.items():
        ms = sorted(set(an.metal_symbol.loc[idx]), key=AD.AN3.index)
        if len(ms) > 1:
            new.loc[idx] = an.metal_symbol.loc[idx].map(dict(zip(ms, ms[1:] + ms[:1])))
    return new


def s24_changed_fraction(an: pd.DataFrame) -> float:
    return float((metal_cycled(an) != an.metal_symbol).mean()) if len(an) else 0.0


def aux_data(arm: str, bench, cells_ci, an: pd.DataFrame, draw: int = 0) -> dict:
    """Auxiliary inputs of one fit.  Ln aux rows are identical in every arm."""
    out = {"ln": _ln_aux_rows(bench, cells_ci), "priv": None, "an_in_ln": None}
    if arm == "S13":
        out["priv"] = _private_rows(an)
    elif arm == "S14":
        x = interp.X_R
        an2 = an.assign(ln=an.metal_symbol.map(ANALOGUE))
        xs = np.array([[x[LANTHANIDES.index(l)], SB.Q[LANTHANIDES.index(l)]] for l in an2.ln]).reshape(len(an2), 2)
        n_pub = an2.groupby("publication").publication.transform("size").to_numpy(float)
        w = 1.0 / n_pub
        out["an_in_ln"] = (xs, an2.log_D.to_numpy(float), w / w.sum(), list(an2.extractant))
    elif arm == "S15":
        out["priv"] = _matched_rows(bench, cells_ci, an, MATCH_SEED)
    elif arm == "S16":
        out["priv"] = _private_rows(_permuted_an(an, PERM_SEEDS[draw]))
    elif arm == "S24":
        out["priv"] = _private_rows(an.assign(metal_shuffled=metal_cycled(an)), metal_key="metal_shuffled")
    return out


# ------------------------------------------------------------------------------------------
# network
# ------------------------------------------------------------------------------------------
class _Net:
    def __init__(self, seed: int, n_r: int):
        import torch
        torch.manual_seed(seed)
        nn = torch.nn
        self.enc = nn.Sequential(nn.Linear(N_CHEM, 16), nn.Tanh(), nn.Linear(16, 8), nn.Tanh())
        self.main = nn.Sequential(nn.Linear(8 + n_r, 16), nn.Tanh(), nn.Linear(16, 1))
        self.ln = nn.Sequential(nn.Linear(10, 16), nn.Tanh(), nn.Linear(16, 1))
        self.priv = nn.Sequential(nn.Linear(12, 16), nn.Tanh(), nn.Linear(16, 1))

    def params(self):
        return [p for m in (self.enc, self.main, self.ln, self.priv) for p in m.parameters()]


def _zmat(exts, zstat):
    mu, sd = zstat
    Z = np.vstack([_z_raw(e) for e in exts])
    Z[:, :len(SB.RD_DESC)] = np.clip((Z[:, :len(SB.RD_DESC)] - mu) / sd, -Z_CLIP, Z_CLIP)
    return Z


def _rstat(X):
    mu = np.nanmean(X, 0)
    sd = np.nanstd(X, 0)
    return mu, np.where(sd < SD_FLOOR, 1.0, sd)


def _rmat(X, rstat):
    mu, sd = rstat
    return np.nan_to_num((X - mu) / sd, nan=0.0)


def _wvar(y, w) -> float:
    return float(np.average((y - np.average(y, weights=w)) ** 2, weights=w)) + 1e-9


def train(main, aux, w_ln: float, w_x: float, wd: float, zstat, rstat, seeds=None, ln_weight_with_x=None) -> list:
    """main = (X, t, w, ext); aux = aux_data(...).  Returns trained nets (one per seed).  ln_weight_with_x: the Ln aux
    weight used when w_x > 0 (S14 keeps its Ln head shared); default w_ln."""
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seeds = SEEDS if seeds is None else seeds
    T = lambda a: torch.tensor(np.asarray(a, float), dtype=torch.float64)  # noqa: E731
    Xm, tm, wm, em = main
    R, Zm, tM, wM = T(_rmat(Xm, rstat)), T(_zmat(em, zstat)), T(tm), T(wm)
    lx, ly, lw, le = aux["ln"]
    LX, LZ, LY, LW = T(lx), T(_zmat(le, zstat)), T(ly), T(lw)
    mvar, lvar = _wvar(tm, wm), _wvar(ly, lw)
    w_ln_eff = w_ln if (w_x <= 0 or ln_weight_with_x is None) else ln_weight_with_x
    terms = []
    if aux["priv"] is not None and w_x > 0:
        px, py, pw, pe = aux["priv"]
        terms.append(("priv", T(px), T(_zmat(pe, zstat)), T(py), T(pw), _wvar(py, pw)))
    if aux["an_in_ln"] is not None and w_x > 0:
        ax, ay, aw, ae = aux["an_in_ln"]
        terms.append(("ln", T(ax), T(_zmat(ae, zstat)), T(ay), T(aw), _wvar(ay, aw)))
    nets = []
    for seed in seeds:
        net = _Net(seed, R.shape[1])
        for mod in (net.enc, net.main, net.ln, net.priv):
            mod.double()
        opt = torch.optim.Adam(net.params(), lr=LR, weight_decay=wd)
        for _ in range(EPOCHS):
            opt.zero_grad()
            pm = net.main(torch.cat([net.enc(Zm), R], 1)).squeeze(1)
            loss = (wM * (pm - tM) ** 2).sum() / mvar
            if w_ln_eff > 0:
                pl = net.ln(torch.cat([net.enc(LZ), LX], 1)).squeeze(1)
                loss = loss + w_ln_eff * (LW * (pl - LY) ** 2).sum() / lvar
            for kind, X_, Z_, Y_, W_, var in terms:
                head = net.priv if kind == "priv" else net.ln
                p_ = head(torch.cat([net.enc(Z_), X_], 1)).squeeze(1)
                loss = loss + w_x * (W_ * (p_ - Y_) ** 2).sum() / var
            loss.backward()
            opt.step()
        nets.append(net)
    return nets


def predict_corr(nets, R_raw, exts, zstat, rstat) -> np.ndarray:
    """R_raw: main-head inputs (block B's features without the chemistry columns)."""
    import torch
    with torch.no_grad():
        R = torch.tensor(_rmat(R_raw, rstat), dtype=torch.float64)
        Z = torch.tensor(_zmat(exts, zstat), dtype=torch.float64)
        return np.mean([net.main(torch.cat([net.enc(Z), R], 1)).squeeze(1).numpy() for net in nets], axis=0)


def net_state(nets) -> list:
    """Every parameter of every trained net as numpy (fit poisoning compares these bit for bit)."""
    return [[p.detach().numpy().copy() for p in n.params()] for n in nets]


# ------------------------------------------------------------------------------------------
# fit with inner selection
# ------------------------------------------------------------------------------------------
def _stats(sits, bench, cells_ci):
    exts = sorted({bench.frame.extractant.iat[c] for c in cells_ci})
    Zr = np.vstack([_z_raw(e) for e in exts])[:, :len(SB.RD_DESC)]
    sd = Zr.std(0)
    zstat = (Zr.mean(0), np.where(sd < SD_FLOOR, 1.0, sd))
    rstat = _rstat(np.vstack([s["X"][:, :-N_CHEM] for s in sits]))
    return zstat, rstat


def _val_loss(nets, val_s, bench, zstat, rstat) -> tuple[float, float]:
    if nets is None:
        corr = [np.zeros(len(s["t"])) for s in val_s]
    else:
        X = np.vstack([s["X"][:, :-N_CHEM] for s in val_s])
        ext = [bench.frame.extractant.iat[s["ci"]] for s in val_s for _ in range(len(s["t"]))]
        pv = predict_corr(nets, X, ext, zstat, rstat)
        corr = np.split(pv, np.cumsum([len(s["t"]) for s in val_s])[:-1])
    return SB._pair_loss(val_s, corr)


def _blocks(bench, rec, Y, b1, an_fold, mode):
    blocks = []
    for blk in rec["inner"]:
        tr = set(map(int, blk["train_index"]))
        va = set(map(int, blk["val_index"]))
        b1j = SB._inner_b1(bench, rec, Y, b1, tr, blk)
        fit_s = SB._rows(bench, rec, Y, mode, b1j, cells_subset=tr, dopt=(blk["cov"], 2 * blk["s2"]))
        val_s = [s for s in SB._rows(bench, rec, Y, mode, b1j, cells_subset=va) if s["mask"] == "DOPT_Q"]
        if not fit_s or not val_s:
            continue
        tr_cells = {s["ci"] for s in fit_s} | {int(c) for c in tr}
        zstat, rstat = _stats(fit_s, bench, tr_cells)
        blocks.append((fit_s, val_s, tr_cells, zstat, rstat, AD.inner_an(bench, rec, blk, an_fold)))
    return blocks


def fit_slot(bench, rec, Y, arm: str, b1: dict, draw: int = 0, gate_only: bool = False) -> dict:
    """Inner selection of the candidate set (see module docstring), then a refit on the outer training fold.
    gate_only: stop after the inner selection (scripts/r4_gate_c.py; no outer refit, no prediction)."""
    mode = {"geometry": None}
    an_fold = AD.fold_an(bench, rec)
    blocks = _blocks(bench, rec, Y, b1, an_fold, mode)
    # stage 1: S11 (Ln-only scaffold); memoised per bench object, Y object and fold (identical in every arm).
    # The cache value keeps bench and Y alive, so their ids cannot be reused by another object while the entry exists.
    key = (id(bench), id(Y), rec["design"], rec["split_seed"], rec["fold"], tuple(np.round(b1["coef"], 15)))
    hit = _S11_CACHE.get(key)
    if hit is not None and hit[0] is bench and hit[1] is Y:
        sc11, best = hit[2], hit[3]
    else:
        sc11, best = _s11_stage(blocks, bench)
        _S11_CACHE[key] = (bench, Y, sc11, best)
    if arm == "S11":
        order = ["zero"] + [net_key(*g) for g in _tie_order()]
        choice = min(order, key=lambda k: (sc11[k], order.index(k)))
        cfg = None if choice == "zero" else (*_tie_order()[order.index(choice) - 1], 0.0)
        scores = sc11
    else:
        scores, cfg = _x_stage(blocks, bench, arm, draw, sc11, best)
    if gate_only:
        return {"inner_scores": scores, "choice": cfg, "best_net": best}
    return _refit(bench, rec, Y, arm, b1, draw, cfg, scores, an_fold, mode, best)


_S11_CACHE: dict = {}


def _tie_order():
    """S11's network order for ties: smaller w_Ln first, then larger WD."""
    return sorted(NET_GRID, key=lambda g: (g[0], -g[1]))


def _s11_stage(blocks, bench):
    sse = {"zero": 0.0, **{g: 0.0 for g in NET_GRID}}
    wsum = {k: 0.0 for k in sse}
    for fit_s, val_s, tr_cells, zstat, rstat, an_j in blocks:
        main = _main_rows(fit_s, bench)
        aux = aux_data("S11", bench, tr_cells, an_j)
        s, n = _val_loss(None, val_s, bench, zstat, rstat)
        sse["zero"] += s
        wsum["zero"] += n
        for wl, wd in NET_GRID:
            s, n = _val_loss(train(main, aux, wl, 0.0, wd, zstat, rstat), val_s, bench, zstat, rstat)
            sse[(wl, wd)] += s
            wsum[(wl, wd)] += n
    sc11 = {"zero": sse["zero"] / max(wsum["zero"], 1e-12)}
    sc11.update({net_key(*g): sse[g] / max(wsum[g], 1e-12) for g in NET_GRID})
    order = _tie_order()
    best = min(order, key=lambda g: (sc11[net_key(*g)], order.index(g)))
    return sc11, best


def _ln_with_x(arm: str, wl: float):
    return max(wl, S14_MIN_W_LN) if arm == "S14" else None


def _x_stage(blocks, bench, arm, draw, sc11, best):
    """Selection over the auxiliary weight with (w_Ln, WD) fixed at S11's best network (w_X = 0 = that network)."""
    wl, wd = best
    sx = {"zero": sc11["zero"]}
    for wx in W_AN_GRID:
        if wx == 0:                       # w_X = 0 trains exactly S11's best network (no auxiliary term enters the loss)
            sx[str(wx)] = sc11[net_key(wl, wd)]
            continue
        sse_t, n_t = 0.0, 0.0
        for fit_s, val_s, tr_cells, zstat, rstat, an_j in blocks:
            aux = aux_data(arm, bench, tr_cells, an_j, draw)
            nets = train(_main_rows(fit_s, bench), aux, wl, wx, wd, zstat, rstat, ln_weight_with_x=_ln_with_x(arm, wl))
            s, n = _val_loss(nets, val_s, bench, zstat, rstat)
            sse_t += s
            n_t += n
        sx[str(wx)] = sse_t / max(n_t, 1e-12)
    cands = ["zero"] + [str(w) for w in W_AN_GRID]
    choice = min(cands, key=lambda k: (sx[k], cands.index(k)))
    cfg = None if choice == "zero" else (wl, wd, float(choice))
    return {"S11": sc11, arm: sx}, cfg


def _refit(bench, rec, Y, arm, b1, draw, cfg, scores, an_fold, mode, best):
    # final refit on all outer-training situations
    sits = SB._rows(bench, rec, Y, mode, b1)
    tr_all = {s["ci"] for s in sits} | {int(c) for c in rec["outer"]["train_index"]}
    zstat, rstat = _stats(sits, bench, tr_all)
    nets = None
    if cfg is not None:
        wl, wd, wx = cfg
        nets = train(_main_rows(sits, bench), aux_data(arm, bench, tr_all, an_fold, draw), wl, wx, wd, zstat, rstat,
                     ln_weight_with_x=_ln_with_x(arm, wl))
    return {"arm": arm, "draw": draw,
            "choice": "zero" if cfg is None else {"w_Ln": cfg[0], "WD": cfg[1], "w_X": cfg[2]},
            "best_net": {"w_Ln": best[0], "WD": best[1]}, "inner_scores": scores, "b1": b1, "nets": nets,
            "net_state": None if nets is None else net_state(nets), "zstat": zstat, "rstat": rstat,
            "an_counts": AD.counts(an_fold), "n_rows": int(sum(len(s["t"]) for s in sits))}


def predict_levels(fit: dict, obs, cell) -> np.ndarray:
    base = RF.predict_levels(fit["b1"], obs, cell["metals"])
    if fit["nets"] is None:
        return base
    feats = SB.row_features(cell, obs.y_masked, {"geometry": None}, float(fit["b1"]["coef"][0]))
    out = base.copy()
    if feats:
        ms = sorted(feats)
        X = np.vstack([feats[m] for m in ms])[:, :-N_CHEM]
        pv = predict_corr(fit["nets"], X, [str(cell["extractant"])] * len(ms), fit["zstat"], fit["rstat"])
        for m, v in zip(ms, pv):
            out[m] = base[m] + float(v)
    return out
