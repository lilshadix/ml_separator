"""Anchor regression with publication identity as the anchor (Rothenhaeusler et al., JRSS-B 2021).

The anchor A is the cell's publication.  Because A is a set of disjoint dummies the L2 projection
Pi_A is exactly the (weighted) per-publication mean operator, so the finite-sample estimator

    b_gamma = argmin_b ||(I - Pi_A)(y - Xb)||^2_w + gamma ||Pi_A (y - Xb)||^2_w

is plain weighted ridge on the row-transformed data W_gamma = (I - Pi_A) + sqrt(gamma) Pi_A,
because the two residual components are orthogonal in the w-inner product.  gamma = 1 is OLS/ridge,
gamma = 0 is the partialling-out (publication fixed-effect) estimator, gamma -> inf is the IV /
between-publication estimator.  NOTE the direction: gamma < 1 is the "remove the lab effect" side,
gamma > 1 is the shift-robust side.  Only the training fold is transformed; prediction is X_te @ b.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gen15_curve"))
from gen15 import valuebench as V           # noqa: E402
from gen15 import arms as A                 # noqa: E402
from gen15.valuebench import Ctx            # noqa: E402

DESIGNS = ("BP", "A", "B")
GAMMAS = (0.0, 0.05, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0, 256.0)
ALPHA = 10.0                                 # ridge penalty, held fixed so gamma is the only mover


def _pub(ctx: Ctx) -> np.ndarray:
    return ctx.bench.frame["publication_id"].astype(str).to_numpy()


def _wmean_by(group: np.ndarray, M: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted per-group mean, broadcast back to rows.  This is Pi_A applied to M."""
    out = np.empty_like(M, dtype=float)
    for g in np.unique(group):
        m = group == g
        ww = w[m]
        out[m] = (ww[:, None] * M[m]).sum(0) / max(ww.sum(), 1e-12)
    return out


def anchor(gamma: float, alpha: float = ALPHA, cols: str | None = None):
    """Anchor-regularised ridge on the 209-column lean block, both curve coefficients."""
    def _arm(ctx: Ctx) -> np.ndarray:
        tr, te = ctx.train, ctx.test
        X = ctx.X if cols is None else ctx.feat(cols)
        Xtr, Xte, Y, w = X[tr], X[te], ctx.bench.coef[tr], ctx.w
        w = w / w.sum()
        mu = (w[:, None] * Xtr).sum(0)
        sd = np.sqrt((w[:, None] * (Xtr - mu) ** 2).sum(0)); sd[sd < 1e-9] = 1.0
        Z, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
        ym = (w[:, None] * Y).sum(0); Yc = Y - ym
        g = _pub(ctx)[tr]
        PZ, PY = _wmean_by(g, Z, w), _wmean_by(g, Yc, w)
        s = np.sqrt(gamma)
        Zt, Yt = (Z - PZ) + s * PZ, (Yc - PY) + s * PY
        sw = np.sqrt(w)[:, None]
        Zw, Yw = Zt * sw, Yt * sw
        G = Zw.T @ Zw + alpha * np.eye(Z.shape[1])
        B = np.linalg.solve(G, Zw.T @ Yw)
        return Zte @ B + ym
    return _arm


ARMS: dict = {"G14": A.g14, "MEAN_CURVE": A.mean_curve}
for gm in GAMMAS:
    ARMS[f"ANC{gm:g}"] = anchor(gm)

COMPS = {f"ANC{gm:g}_vs_G14": ("G14", f"ANC{gm:g}") for gm in GAMMAS}
COMPS.update({f"ANC{gm:g}_vs_ANC1": ("ANC1", f"ANC{gm:g}") for gm in GAMMAS if gm != 1.0})

if __name__ == "__main__":
    t0 = time.time()
    bench = V.load()
    print(f"[anchor] bench loaded {time.time()-t0:.0f}s; {len(ARMS)} arms x {len(DESIGNS)} designs", flush=True)
    B, C, tables = V.score(bench, ARMS, DESIGNS, comps=COMPS)
    out = ROOT / "gen15_curve" / "results"; out.mkdir(parents=True, exist_ok=True)
    B.to_csv(out / "g15_anchor_board.csv", index=False)
    C.to_csv(out / "g15_anchor_contrasts.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n=== extractant-macro MAE of log SF (lower is better) ===")
    print(V.wide(B).round(4).to_string())
    if len(C):
        print("\n=== paired contrasts (positive = candidate better) ===")
        print(C[["design","comparison","point","ci95_low","ci95_high","p_two_sided","passes_P1"]]
              .round(4).to_string(index=False))
    print(f"\n[anchor] total {time.time()-t0:.0f}s")
