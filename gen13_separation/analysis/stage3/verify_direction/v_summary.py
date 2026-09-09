import pandas as pd
from pathlib import Path
OUT = Path("D:/ml_separator_gh/gen13_separation/analysis/stage3/verify_direction")
a = pd.read_csv(OUT/"v_attack_target_units.csv"); b = pd.read_csv(OUT/"v_baselines.csv")
c = pd.read_csv(OUT/"v_contrasts.csv"); h = pd.read_csv(OUT/"v_headline.csv")
print("== independent re-derivation (my own pipeline, L2 logistic) ==")
print(h[["model","features","n_cols","macro_acc","baseline_heavy","gain","lo","hi","p_two_sided"]].round(4).to_string(index=False))
print("\n== target attacks (original ExtraTrees estimator, design BP) ==")
print(a[["tag","n_cells","n_ext","n_chem","macro","baseline_heavy","gain","gain_lo","gain_hi","p"]].round(4).to_string(index=False))
print("\n== baselines and estimators (design BP) ==")
print(b[["tag","n_ext","macro","acc_lo","acc_hi","gain_vs_always_heavy","gain_lo","gain_hi","p"]].round(4).to_string(index=False))
print("\n== paired contrasts ==")
print(c.round(4).to_string(index=False))
