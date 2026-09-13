"""Score the external-data arms on the gen15 curve bench under all five designs."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "generations" / "gen15_curve"))
sys.path.insert(0, str(HERE))

from gen15 import valuebench as V, arms as A       # noqa: E402
import extarms as X                                # noqa: E402

ARMS = {
    "FLAT": A.flat,
    "ALWAYS_HEAVY": X.always_heavy,
    "G14": A.g14,
    "LOGK_DIR": X.logk_dir(balanced=False),
    "LOGK_DIR_CB": X.logk_dir(balanced=True),
    "LOGK_NN": X.logk_nn,
    "G14_AUG_0.25": X.g14_augmented(0.25),
    "G14_AUG_1.0": X.g14_augmented(1.0),
}
COMPS = {
    "LOGK_DIR_vs_FLAT": ("FLAT", "LOGK_DIR"),
    "LOGK_DIR_vs_ALWAYS_HEAVY": ("ALWAYS_HEAVY", "LOGK_DIR"),
    "LOGK_DIR_vs_G14": ("G14", "LOGK_DIR"),
    "LOGK_DIR_CB_vs_ALWAYS_HEAVY": ("ALWAYS_HEAVY", "LOGK_DIR_CB"),
    "LOGK_NN_vs_ALWAYS_HEAVY": ("ALWAYS_HEAVY", "LOGK_NN"),
    "G14_vs_ALWAYS_HEAVY": ("ALWAYS_HEAVY", "G14"),
    "G14_AUG_0.25_vs_G14": ("G14", "G14_AUG_0.25"),
    "G14_AUG_1.0_vs_G14": ("G14", "G14_AUG_1.0"),
}


def main() -> None:
    bench = V.load()
    B, C, tables = V.score(bench, ARMS, ["B", "BR", "BQ", "A", "BP"], comps=COMPS)
    B.to_csv(HERE / "bench_board.csv", index=False)
    C.to_csv(HERE / "bench_contrasts.csv", index=False)
    for value in ("macro_mae_extractant", "macro_sign_acc_strong", "macro_pair_spearman"):
        try:
            w = V.wide(B, value)
        except KeyError:
            continue
        print(f"\n=== {value} ===")
        print(w.round(4).to_string())
        w.round(6).to_csv(HERE / f"wide_{value}.csv")
    print("\n=== chemotype-blocked paired contrasts (mae_all, positive favours the candidate) ===")
    print(C.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
