"""Steps 2 and 3: score de-noised labels, precision weights and the attenuation correction
under all five designs.

Nothing here changes the *evaluation* -- the endpoint is still the extractant-macro MAE of
predicted log SF against the raw observed log D of held-out cells, on the frozen pair rows.  Only
what the model is fitted to changes.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from arms_labelerr import labelerr_arm  # noqa: E402
from gen15 import arms as A, valuebench as V  # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

ARMS = {
    "FLAT": A.flat,
    "G14": A.g14,
    "G14R": labelerr_arm(),                                            # must reproduce G14
    # --- step 2: the one-stage / hierarchical label -------------------------------------
    "SH_DIR": labelerr_arm(dir_label="shrunk"),
    "SH_MAG": labelerr_arm(mag_source="shrunk"),
    "SH_BOTH": labelerr_arm(dir_label="shrunk", mag_source="shrunk"),
    "SH_BOTH_G": labelerr_arm(dir_label="shrunk", mag_source="shrunk", group="none"),
    "SH_ALL": labelerr_arm(dir_label="shrunk", mag_source="shrunk", curv="shrunk"),
    # --- step 3: errors in variables ----------------------------------------------------
    "PRECW": labelerr_arm(weights="precision"),
    "MAGDEATT": labelerr_arm(mag_source="deattenuated"),
    "PRECW_DEATT": labelerr_arm(weights="precision", mag_source="deattenuated"),
    "SH_PRECW": labelerr_arm(dir_label="shrunk", mag_source="shrunk", weights="precision"),
}
COMPS = {f"{k}_vs_G14": ("G14", k) for k in ARMS if k not in ("FLAT", "G14")}
COMPS.update({f"{k}_vs_FLAT": ("FLAT", k) for k in ARMS if k not in ("FLAT",)})


def main() -> None:
    t0 = time.time()
    bench = V.load()
    B, C, tables = V.score(bench, ARMS, ["B", "BR", "BQ", "A", "BP"], comps=COMPS)
    B.to_csv(OUT / "s2s3_board.csv", index=False)
    C.to_csv(OUT / "s2s3_contrasts.csv", index=False)
    for d, (tab, pe) in tables.items():
        pe.to_csv(OUT / f"s2s3_per_extractant_{d}.csv", index=False)

    print("\n=== extractant-macro MAE (lower better) ===")
    print(V.wide(B).round(4).to_string())
    print("\n=== macro sign accuracy on strong pairs ===")
    print(V.wide(B, "macro_sign_acc_strong").round(4).to_string())
    print("\n=== macro pair Spearman ===")
    print(V.wide(B, "macro_pair_spearman").round(4).to_string())
    print("\n=== chemotype-macro MAE ===")
    print(V.wide(B, "macro_mae_chemotype").round(4).to_string())
    print("\n=== chemotype-blocked paired bootstrap (positive = candidate better) ===")
    cols = [c for c in ["design", "comparison", "delta", "ci_low", "ci_high", "bca_low",
                        "bca_high", "p_value", "seeds_positive", "passes_P1", "mde_80",
                        "loco_min_delta"] if c in C.columns]
    print(C[cols].round(4).to_string(index=False))
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
