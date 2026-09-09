"""Step 4d: the honest ceiling for a *two-term* model, which is what every arm in this programme is.

``O_BOTH`` = 0.181 is ``E|residual difference|`` where the residual is a cell's deviation from its
own fitted quadratic.  The coefficient was fitted to the same values it is scored against, so the
noise in the scored pair is partly inside the prediction.  The honest object is

    error = (what the two-term basis cannot express) + (an independent unit of measurement noise)

which is exactly what you get by scoring the fitted quadratic against the observed data plus one
fresh draw of noise.  Run at the sigma the whiteness test leaves for measurement error.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from noise import replicate_arrays  # noqa: E402
from gen15 import valuebench as V  # noqa: E402
from gen13sep.amplitude_bench import _pair_frame  # noqa: E402
from gen13sep.metrics import per_extractant, summarise  # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
SIGMAS = {"white part, m>=14 (0.117)": 0.117, "white part, m>=10 (0.137)": 0.137,
          "residual upper bound, m>=13 (0.167)": 0.167,
          "residual upper bound, pooled m>=5 (0.201)": 0.201}
N_DRAWS = 5


def main() -> None:
    bench = V.load()
    obs = ~np.isnan(bench.Y)
    nr, _ = replicate_arrays(bench)
    rep = np.where(np.isfinite(nr), np.maximum(nr, 1.0), 1.0)
    quad = np.where(obs, bench.coef @ bench.basis, np.nan)
    rng = np.random.default_rng(97531)
    rows = []
    for label, s in SIGMAS.items():
        se = np.where(obs, s / np.sqrt(rep), np.nan)
        parts = []
        for d in range(N_DRAWS):
            e = rng.normal(size=bench.Y.shape) * np.nan_to_num(se)
            Yp = np.where(obs, bench.Y + e, np.nan)
            t = _pair_frame(bench.frame, Yp, np.arange(len(Yp)), d, 0)
            ia, ib, loc = t["ia"].to_numpy(), t["ib"].to_numpy(), t["cell_local"].to_numpy()
            tt = t.drop(columns=["ia", "ib", "cell_local"]).copy()
            q = np.nan_to_num(quad)
            y0 = np.nan_to_num(bench.Y)
            tt["CEIL_2TERM"] = q[loc, ia] - q[loc, ib]          # best two-term curve, fresh labels
            tt["FLOOR_NOISE"] = y0[loc, ia] - y0[loc, ib]       # a model that knows the exact means
            tt["FLAT"] = 0.0
            parts.append(tt)
        tab = pd.concat(parts, ignore_index=True)
        names = ["CEIL_2TERM", "FLOOR_NOISE", "FLAT"]
        sm = summarise(per_extractant(tab, names), tab, names).set_index("arm")
        rows.append({"sigma_label": label, "sigma": s,
                     "ceiling_two_term": float(sm.loc["CEIL_2TERM", "macro_mae_extractant"]),
                     "floor_noise_only": float(sm.loc["FLOOR_NOISE", "macro_mae_extractant"]),
                     "flat_on_same_labels": float(sm.loc["FLAT", "macro_mae_extractant"])})
        print(f"  {label:44s} ceiling(2-term) {rows[-1]['ceiling_two_term']:.4f}  "
              f"floor(noise only) {rows[-1]['floor_noise_only']:.4f}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "s4d_two_term_ceiling.csv", index=False)
    print("\n=== honest ceilings, extractant-macro MAE ===")
    print(d.round(4).to_string(index=False))
    print("\nfor reference on the same rows with no added noise: O_BOTH (in sample) 0.1811, "
          "OWNBOTH_LOPO 0.2736, G14 (BP) 0.500, FLAT 0.589")


if __name__ == "__main__":
    main()
