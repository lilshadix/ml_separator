"""P4 skeptic re-check (label-blind): independent recomputation of the decision-relevant P4 numbers.

Reads ONLY bibliographic / extractant-name metadata columns of the gen19 archive, the P4 candidate table
(read-only) and P2's G80 table (read-only). No D / log D / SF column is loaded (asserted). Prints:
  A. full-string author search in the archive for every author surname of the new-lab candidates
     (substring match over publication_authors / reference_authors / reference_other, including non-DOI
     references, which P4_check_candidates.py's author index skips);
  B. archive extractant-name hits for the extractant families P4 calls "absent from the corpus";
  C. the expected-yield arithmetic, recomputed per paper (lab converts if any of its papers converts)
     and with the non-DGA restriction;
  D. every P2 G80 cell (pool B1-B0, one-sided 0.025) reached by each supply scenario;
  E. sensitivity of the verified / probable / expected new-lab counts to the skeptic's re-classifications
     (P4-063 not verified from the accessible text; title-only "no abstract" E2 rows moved to U as P4's own
     tier rule says; institution links applied as P4 applied them to JNC-Koma and ANL-Peppard).

Run from D:/ml_separator_gh:
    .venv/Scripts/python.exe generations/gen21/round5_design_20261023/probes/scripts/P4_skeptic_recheck.py
Writes nothing.
"""
from __future__ import annotations

import re
import unicodedata
from itertools import product
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
PROBES = Path(__file__).resolve().parents[1]
ARCH = ROOT / "dataset_all_metals" / "clean" / "master_clean.parquet"
CAND = PROBES / "P4_candidates.csv"
G80 = PROBES / "P2_out" / "P2_G80.csv"

COLS = ["doi_primary_corrected", "publication_authors", "reference_authors", "reference_other",
        "publication_title", "reference_title", "extractant_primary_name", "extractant_names",
        "extractant_name_raw", "is_lanthanide", "metal_symbol"]
assert not {"D_raw", "D_value", "log_D"} & set(COLS)


def norm(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)).lower()


def lab_cons(lab: str) -> str:  # same merge as P4_check_candidates.py
    if lab in {"CIAC-Li", "CIAC-Liao", "CIAC-Deng"}:
        return "CIAC"
    if lab.startswith("UFV-") or lab.startswith("Aveiro-"):
        return "UFV/Aveiro-Schaeffer"
    if lab.startswith("KULeuven"):
        return "KULeuven-Binnemans"
    if "Turanov" in lab:
        return "ISSP-Turanov"
    return lab


def main() -> None:
    arch = pd.read_parquet(ARCH, columns=COLS)
    arch["key"] = arch["doi_primary_corrected"].fillna("REF:" + arch["reference_other"].fillna("").astype(str).str[:60])
    txt = (arch[["publication_authors", "reference_authors", "reference_other"]].fillna("").astype(str)
           .agg(" | ".join, axis=1).map(norm))
    per_key = txt.groupby(arch["key"]).agg(lambda s: " || ".join(sorted(set(s))))
    print(f"archive keys {len(per_key)}; non-DOI keys {sum(not k.startswith('10.') for k in per_key.index)}")

    c = pd.read_csv(CAND, dtype=str).fillna("")
    c["lab_cons"] = c["lab_id"].map(lab_cons)

    # ---- A. author search for every candidate whose lab is flagged N or P
    print("\n=== A. archive author substring hits (surnames >= 4 letters; word-boundary match)")
    for _, r in c[c["lab_in_corpus"].isin(["N", "P"])].iterrows():
        sn = [norm(s).strip() for s in r["author_surnames"].split(";") if s.strip() and s.strip() != "..."]
        hits = {}
        for s in sn:
            if len(s) < 4:
                continue
            pat = re.compile(r"\b" + re.escape(s) + r"\b")
            ks = [k for k, t in per_key.items() if pat.search(t)]
            if ks:
                hits[s] = ks[:4] + (["..."] if len(ks) > 4 else [])
        if hits:
            print(f"{r['cand_id']} [{r['eligibility']}/{r['lab_in_corpus']}] {r['lab_id']}: {hits}")

    # distinctive names / institutions of the 16 E1+E2 new labs, incl. co-authors named in the report
    extra = ["binnemans", "hoogerstraete", "cardinaels", "katsuta", "turanov", "karandashev", "atanassova",
             "stepanov", "tsivadze", "kostikova", "boyarintsev", "shinohara", "nagame", "haba", "toyoshima",
             "deng", "liao", "chakravortty", "belova", "voshkin", "vasilenko", "schaeffer", "coutinho",
             "hespanhol", "imura", "okamura", "hasegawa", "shafer", "servis", "shkrob", "soldenhoff",
             "odinets", "baulin", "leuven", "chiba", "changchun", "ciac", "dubna", "frumkin"]
    print("\n--- distinctive-name search (word boundary)")
    for s in extra:
        pat = re.compile(r"\b" + re.escape(s) + r"\b")
        ks = [k for k, t in per_key.items() if pat.search(t)]
        print(f"  {s:14s} {len(ks):3d} {ks[:5]}")

    # ---- B. extractant-family presence in the archive
    print("\n=== B. archive extractant names matching families P4 calls absent")
    names = (arch[["extractant_primary_name", "extractant_names", "extractant_name_raw"]].fillna("").astype(str)
             .agg(" | ".join, axis=1).map(norm))
    fam = {
        "acidic organoP (HDEHP/D2EHPA/P507/PC88A/P204/Cyanex 272/phosphoric/phosphinic/phosphonic acid)":
            r"hdehp|d2ehpa|dehpa|p507|pc ?88a|p204|cyanex ?272|ehehpa|hehehp|phosphoric acid|phosphinic acid|phosphonic acid|bis\(2-ethylhexyl\) ?(hydrogen )?phosph",
        "neutral phosphine oxide (TOPO/Cyanex 923/925/TRPO)": r"topo|trioctylphosphine oxide|cyanex ?92[35]|trpo|phosphine oxide",
        "TBP / trialkyl phosphate": r"\btbp\b|tributyl ?phosphate|tri-n-butyl",
        "quaternary ammonium / phosphonium (Aliquat/A336/TOMA/P66614)": r"aliquat|a336|\btoma|methyltrioctylammonium|trioctylmethylammonium|p66614|phosphonium|ammonium",
        "beta-diketone / TTA / 8-HQ / pyrazolone / tropolone": r"\btta\b|thenoyltrifluoroacetone|diketone|acetylacetone|benzoylacetone|hydroxyquinolin|pyrazolon|tropolone",
        "CMPO": r"cmpo|carbamoylmethyl",
        "crown / calixarene": r"crown|calix",
    }
    for lab, pat in fam.items():
        m = names.str.contains(pat, regex=True)
        ln = arch["is_lanthanide"].astype(str).str.lower().eq("true")
        top = arch.loc[m, "extractant_primary_name"].astype(str).value_counts().head(6).to_dict()
        print(f"  {lab}: rows {int(m.sum())} (Ln rows {int((m & ln).sum())}), keys {arch.loc[m, 'key'].nunique()}; top {top}")

    # ---- C. expected yield, recomputed
    print("\n=== C. expected new-lab yield")
    new = c[(c["lab_in_corpus"] == "N") & c["eligibility"].isin(["E1", "E2", "U"])]
    labs = new.groupby("lab_cons").agg(
        e1=("eligibility", lambda s: int((s == "E1").sum())),
        e2=("eligibility", lambda s: int((s == "E2").sum())),
        u=("eligibility", lambda s: int((s == "U").sum())),
        classes=("ext_class", lambda s: ",".join(sorted(set(s)))),
    )
    labs["nonDGA"] = labs["classes"].map(lambda x: any(k != "DGA" for k in x.split(",")))
    print(labs.to_string())
    for p2, pu in [(0.5, 0.25), (0.12, 0.0), (0.88, 0.5), (0.5, 0.0)]:
        lab_tier = (labs["e1"] > 0).astype(float)
        p_lab = 1 - (1 - lab_tier) * (1 - p2) ** labs["e2"] * (1 - pu) ** labs["u"]
        rep = (labs["e1"] > 0) * 1.0 + ((labs["e1"] == 0) & (labs["e2"] > 0)) * p2 + ((labs["e1"] == 0) & (labs["e2"] == 0)) * pu
        print(f"  p(E2)={p2}, p(U)={pu}: per-lab-any-paper {p_lab.sum():.2f} (nonDGA {p_lab[labs['nonDGA']].sum():.2f}); "
              f"report's per-lab tier rule {rep.sum():.2f} (nonDGA {rep[labs['nonDGA']].sum():.2f})")

    # ---- E. sensitivity of the headline counts to the skeptic's re-classifications
    print("\n=== E. sensitivity: verified / probable / expected new labs under re-classifications")

    def summarise(cc: pd.DataFrame, p2: float, pu: float) -> str:
        nn = cc[(cc["lab_in_corpus"] == "N") & cc["eligibility"].isin(["E1", "E2", "U"])]
        lb = nn.groupby("lab_cons").agg(e1=("eligibility", lambda s: int((s == "E1").sum())),
                                        e2=("eligibility", lambda s: int((s == "E2").sum())),
                                        u=("eligibility", lambda s: int((s == "U").sum())))
        tier = (lb["e1"] > 0) * 1.0 + ((lb["e1"] == 0) & (lb["e2"] > 0)) * p2 + ((lb["e1"] == 0) & (lb["e2"] == 0)) * pu
        anyp = 1 - (1 - (lb["e1"] > 0).astype(float)) * (1 - p2) ** lb["e2"] * (1 - pu) ** lb["u"]
        return (f"E1 labs {int((lb['e1'] > 0).sum())}, E1+E2 labs {int(((lb['e1'] + lb['e2']) > 0).sum())}, "
                f"E1+E2+U labs {len(lb)}; expected (tier rule, p2={p2:.2f}, pU={pu}) {tier.sum():.2f}; "
                f"(any-paper rule) {anyp.sum():.2f}")

    base = c.copy()
    s1 = base.copy()
    s1.loc[s1["cand_id"] == "P4-063", "eligibility"] = "E2"          # adjacent pair not visible on the landing text
    s2 = s1.copy()
    s2.loc[(s2["eligibility"] == "E2") & (s2["screen_level"] == "TITLE"), "eligibility"] = "U"  # P4's own U rule: no abstract
    s3 = s2.copy()
    s3.loc[s3["cand_id"].isin(["P4-077", "P4-073", "P4-047"]), "lab_in_corpus"] = "P"   # CSM-Shafer: ANL co-authors (ANL-Peppard precedent)
    s4 = s3.copy()
    s4.loc[s4["cand_id"].isin(["P4-083"]), "lab_in_corpus"] = "P"   # Osaka: Nagame/Asai/Toyoshima (JAEA; affiliation not in metadata)
    s4.loc[s4["cand_id"].isin(["P4-041", "P4-064", "P4-065"]), "lab_in_corpus"] = "P"   # Kanazawa: JAEA co-author Okamura
    for name, cc, p2, pu in [
        ("S0 as reported", base, 0.5, 0.25),
        ("S0 without the unmeasured U conversion", base, 0.5, 0.0),
        ("S1 P4-063 -> E2, conversion 2/6", s1, 2 / 6, 0.25),
        ("S1 without U conversion", s1, 2 / 6, 0.0),
        ("S2 S1 + title-only E2 -> U", s2, 2 / 6, 0.25),
        ("S3 S2 + CSM-Shafer -> P (ANL co-authors)", s3, 2 / 6, 0.25),
        ("S4 S3 + Osaka, Kanazawa -> P (JAEA co-authors)", s4, 2 / 6, 0.25),
        ("S4 without U conversion", s4, 2 / 6, 0.0),
    ]:
        print(f"  {name:48s}: {summarise(cc, p2, pu)}")

    # ---- D. G80 cells reached
    print("\n=== D. P2 G80 (pool B1-B0, alpha 0.025) reached by each supply scenario")
    g = pd.read_csv(G80, dtype={"alpha": str})
    g = g[(g["alpha"] == "025") & g["pool"].isin(["B1-B0", "B1-B0|nonDGA"])].copy()
    pub = g[(g["unit_type"] == "publication") & (g["pool"] == "B1-B0") & (g["test"] == "SEALED")]
    print("  publication SEALED rows:", pub[["tau", "G80_interp", "power_G40", "power_G80"]].to_dict("records"))
    for G in [5, 6, 7, 9, 10, 11, 12, 13, 13.5, 16, 19, 24]:
        ok = g[g["G80_interp"] <= G]
        print(f"  G={G}: {len(ok)} cells -> " + "; ".join(
            f"{r.pool}/{r.unit_type}/tau{r.tau}/{r.test}={r.G80_interp:.1f}" for r in ok.sort_values("G80_interp").itertuples()))


if __name__ == "__main__":
    main()
