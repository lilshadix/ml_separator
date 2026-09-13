"""Assemble the gen14 report tables from the result CSVs (markdown, ready to paste)."""
import glob
from pathlib import Path
import pandas as pd

R = Path("generations/gen14_direction/results")
D = ["B", "BR", "BQ", "A", "BP"]
out = []


def md(df, title):
    out.append(f"\n### {title}\n")
    out.append(df.round(4).to_markdown())


# 1. direction accuracy, five designs
b = pd.concat([pd.read_csv(R / "g14_baseline_board.csv"), pd.read_csv(R / "g14_sweep_board.csv"),
               pd.read_csv(R / "g14_sweep2_board.csv")]).drop_duplicates(["design", "model"])
p = b.pivot_table(index="model", columns="design", values="macro_accuracy")
cols = [c for c in D if c in p.columns]
md(p[cols].loc[[m for m in ["LOGIT_TOPO39", "G13_ET_TOPO39", "CHEM137_ET", "LEAN209_ET",
                            "DONORS13_ET", "ALWAYS_HEAVY"] if m in p.index]],
   "direction: macro accuracy over 82 extractants")

# 2. paired gains
g = pd.concat([pd.read_csv(R / "g14_baseline_gains.csv"), pd.read_csv(R / "g14_sweep_gains.csv"),
               pd.read_csv(R / "g14_sweep2_gains.csv")]).drop_duplicates(["design", "reference", "candidate"])
for ref, cand in [("ALWAYS_HEAVY", "LOGIT_TOPO39"), ("DONORS13_ET", "LOGIT_TOPO39"),
                  ("G13_ET_TOPO39", "LOGIT_TOPO39")]:
    s = g[(g.reference == ref) & (g.candidate == cand)].set_index("design")
    s = s.loc[[d for d in D if d in s.index]]
    md(s[["gain", "ci_low", "ci_high", "p_two_sided", "units_better", "units_worse",
          "loco_sign_stable"]], f"direction: {cand} minus {ref}")

# 3. value
v = pd.concat([pd.read_csv(f) for f in glob.glob(str(R / "g14_value_*.csv"))
               if "contrast" not in f])
p = v.pivot_table(index="arm", columns="design", values="macro_mae_extractant")
cols = [c for c in D if c in p.columns]
md(p[cols], "value: extractant-macro MAE of log SF")
c = pd.concat([pd.read_csv(f) for f in glob.glob(str(R / "g14_value_contrasts_*.csv"))])
for comp in ["G14hard_vs_MEANCURVE", "G14hard_vs_FULL", "G14hard_vs_G13dir", "ORACLE_vs_G14hard"]:
    s = c[c.comparison == comp].set_index("design")
    s = s.loc[[d for d in D if d in s.index]]
    md(s[["point", "ci95_low", "ci95_high", "p_two_sided", "units_improved", "n_units",
          "seeds_positive", "loco_sign_stable", "passes_P1"]], f"value: {comp}")

# 4. magnitude, ceilings, post-hoc
for f, title in [("g14_magnitude_BP.csv", "magnitude priors under BP"),
                 ("g14_ceiling_bands_BP.csv", "accuracy against label determination (BP)"),
                 ("g14_ceiling_coverage_BP.csv", "accuracy with abstention (BP)"),
                 ("g14_ceiling_summary_BP.csv", "ceilings (BP)")]:
    t = pd.read_csv(R / f)
    if "arm" in t.columns:
        t = t[["arm", "macro_mae_extractant", "macro_mae_chemotype", "macro_mae_adjacent",
               "macro_mae_far", "macro_sign_acc_strong"]]
    md(t, title)
s = pd.read_csv(R / "g14_steric_board.csv").pivot_table(index="model", columns="design",
                                                        values="macro_accuracy")
md(s[[c for c in D if c in s.columns]], "POST-HOC steric block (falsified)")
st = pd.read_csv(R / "g14_steric_strong.csv").pivot_table(index="model", columns="design",
                                                          values="macro_accuracy_strong")
md(st[[c for c in D if c in st.columns]], "POST-HOC steric block on strongly directed extractants")

Path(R / "TABLES.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
