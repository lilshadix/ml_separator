"""P4 (literature supply probe): novelty check and unit counting for probes/P4_candidates.csv.

Label-blind: reads only bibliographic metadata of the archive (DOI and author columns, no D / log D column)
plus the hand-screened candidate table. Prints
  1. archive / round-4-cohort DOI novelty of every candidate,
  2. author-surname overlap with the archive (evidence for the hand-assigned lab_in_corpus flag),
  3. counts of new laboratory x chemotype units under the counting rules stated in P4_literature_supply.md.

Run from D:/ml_separator_gh:
    .venv/Scripts/python.exe generations/gen21/round5_design_20261023/probes/scripts/P4_check_candidates.py
"""
from __future__ import annotations

import ast
import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
PROBES = Path(__file__).resolve().parents[1]
ARCH = ROOT / "dataset_all_metals" / "clean" / "master_clean.parquet"
COH = ROOT / "generations" / "gen21" / "round4_20261022" / "results" / "provenance" / "sources_by_doi.csv"
CAND = PROBES / "P4_candidates.csv"

META = ["doi_primary", "doi_primary_corrected", "doi_all", "doi_source_all", "archive_citation_doi",
        "reference_authors", "publication_authors", "is_lanthanide", "metal_symbol"]
assert not {"D_raw", "D_value", "log_D"} & set(META)

COMMON = {  # surnames too common to identify a laboratory on their own
    "li", "wang", "zhang", "liu", "chen", "yang", "zhao", "wu", "zhou", "xu", "sun", "ma", "zhu", "hu", "guo", "he",
    "gao", "lin", "luo", "zheng", "liang", "xie", "tang", "han", "cao", "deng", "feng", "huang", "ye", "lu", "cui",
    "jiang", "pan", "wei", "shi", "xiao", "tian", "dong", "yu", "yuan", "peng", "fang", "su", "ding", "jin", "qi",
    "suzuki", "sasaki", "tanaka", "yamamoto", "kobayashi", "ito", "kim", "lee", "park", "nguyen", "kumar", "singh",
    "reddy", "ali", "petrov", "ivanov", "smith", "wilson", "mohamed", "sharma", "xue", "bi", "yao", "ban", "kaneko",
}
CIAC = {"CIAC-Li", "CIAC-Liao", "CIAC-Deng"}


def norm_doi(x) -> str | None:
    if not isinstance(x, str) or not x.strip():
        return None
    x = x.strip().lower()
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", x)


def strip_acc(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower()


def archive_dois(df: pd.DataFrame) -> set[str]:
    out: set[str] = set()
    for col in ["doi_primary", "doi_primary_corrected", "archive_citation_doi"]:
        out |= {d for d in df[col].map(norm_doi).dropna()}
    for col in ["doi_all", "doi_source_all"]:
        for v in df[col].dropna().astype(str):
            try:
                items = ast.literal_eval(v) if v.startswith("[") else [v]
            except (ValueError, SyntaxError):
                items = re.split(r"[;,\s]+", v)
            out |= {d for d in map(norm_doi, items) if d}
    return out


def archive_author_index(df: pd.DataFrame) -> dict[str, set[str]]:
    """surname -> set of archive DOIs whose author strings contain it."""
    idx: dict[str, set[str]] = {}
    g = df.assign(doi=df["doi_primary_corrected"].map(norm_doi)).dropna(subset=["doi"])
    for doi, sub in g.groupby("doi"):
        txt = " ; ".join(set(sub["reference_authors"].dropna().astype(str)) | set(sub["publication_authors"].dropna().astype(str)))
        for tok in re.split(r"[;,&]|\band\b", txt):
            parts = [p for p in re.split(r"[\s.]+", strip_acc(tok)) if p]
            if not parts:
                continue
            for p in {parts[-1], parts[0]}:  # 'Given Surname' and 'Surname I.' styles
                if len(p) > 1:
                    idx.setdefault(p, set()).add(doi)
    return idx


def main() -> int:
    arch = pd.read_parquet(ARCH, columns=META)
    adois = archive_dois(arch)
    coh = pd.read_csv(COH)
    cdois = set(coh["doi"].map(norm_doi))
    aidx = archive_author_index(arch)

    c = pd.read_csv(CAND, dtype=str).fillna("")
    c["doi_n"] = c["doi"].map(norm_doi)
    print(f"candidates screened: {len(c)} | unique DOIs: {c['doi_n'].nunique()} | archive DOIs: {len(adois)} | cohort DOIs: {len(cdois)}")
    hits = c[c["doi_n"].isin(adois) | c["doi_n"].isin(cdois)]
    print("candidates whose DOI is already in the archive or the round-4 cohort:", hits["cand_id"].tolist() or "none")

    # ---- author overlap evidence
    print("\n--- author-surname overlap with the archive (common surnames ignored) vs hand flag lab_in_corpus")
    rows = []
    for _, r in c.iterrows():
        sn = [strip_acc(s).strip() for s in r["author_surnames"].split(";") if s.strip() and s.strip() != "..."]
        sn_unc = [s for s in sn if s not in COMMON]
        ov = {s: sorted(aidx.get(s, set())) for s in sn_unc if s in aidx}
        last = strip_acc(r["last_author"]).strip()
        last_hit = last in aidx and last not in COMMON
        # DOIs shared by >= 2 distinct uncommon surnames = strong lab link
        cnt: dict[str, int] = {}
        for s, ds in ov.items():
            for d in ds:
                cnt[d] = cnt.get(d, 0) + 1
        strong = sorted(d for d, k in cnt.items() if k >= 2)
        auto = "Y" if (strong or last_hit) else ("P" if ov else "N")
        rows.append((r["cand_id"], r["lab_id"], r["lab_in_corpus"], auto, ",".join(sorted(ov)), ";".join(strong[:3])))
    ov_df = pd.DataFrame(rows, columns=["cand_id", "lab_id", "hand", "auto", "overlap_surnames", "strong_link_dois"])
    with pd.option_context("display.width", 250, "display.max_colwidth", 70, "display.max_rows", 200):
        print(ov_df.to_string(index=False))
    mism = ov_df[ov_df["hand"] != ov_df["auto"]]
    print(f"\nhand vs auto disagreements: {len(mism)} -> {mism[['cand_id', 'lab_id', 'hand', 'auto']].values.tolist()}")

    # ---- unit counts
    def lab_cons(lab: str) -> str:
        if lab in CIAC:
            return "CIAC"
        if lab.startswith("UFV-") or lab.startswith("Aveiro-"):
            return "UFV/Aveiro-Schaeffer"
        if lab.startswith("KULeuven"):
            return "KULeuven-Binnemans"
        if "Turanov" in lab:
            return "ISSP-Turanov"
        return lab

    c["lab_cons"] = c["lab_id"].map(lab_cons)
    out = {}
    for tier, keep in (("E1", {"E1"}), ("E1+E2", {"E1", "E2"}), ("E1+E2+U", {"E1", "E2", "U"})):
        s = c[c["eligibility"].isin(keep)]
        new = s[s["lab_in_corpus"] == "N"]
        res = {
            "papers": len(s),
            "papers_new_lab": len(new),
            "new_labs_liberal": new["lab_id"].nunique(),
            "new_labs_conservative": new["lab_cons"].nunique(),
            "new_lab_x_chemotype_units_liberal": new[["lab_id", "chemotype_label"]].drop_duplicates().shape[0],
            "new_lab_x_chemotype_units_conservative": new[["lab_cons", "chemotype_label"]].drop_duplicates().shape[0],
            "new_labs_on_new_chemotype_conservative": new[new["chemotype_in_corpus"] == "N"]["lab_cons"].nunique(),
            "units_new_lab_or_new_chemotype_for_known_lab_conservative": pd.concat([
                new[["lab_cons", "chemotype_label"]],
                s[(s["lab_in_corpus"] != "N") & (s["chemotype_in_corpus"] == "N")][["lab_cons", "chemotype_label"]],
            ]).drop_duplicates().shape[0],
            "cpg_new_chemotypes_any_lab": s[s["chemotype_in_corpus"] == "N"]["chemotype_label"].nunique(),
            "new_labs_m22_rule_conservative": new[new["m22_rule_8Ln_EuGdTb"] == "Y"]["lab_cons"].nunique(),
            "new_labs_with_sibling_conditions_conservative": new[new["sibling_conditions"] == "Y"]["lab_cons"].nunique(),
            "new_labs_open_access_conservative": new[~new["oa_status"].str.startswith("closed")]["lab_cons"].nunique(),
            "new_labs_by_class_conservative": new.groupby("ext_class")["lab_cons"].nunique().to_dict(),
            "new_lab_list_conservative": sorted(new["lab_cons"].unique()),
        }
        out[tier] = res
    out["eligibility_counts"] = c["eligibility"].value_counts().to_dict()
    out["screen_level_counts"] = c["screen_level"].value_counts().to_dict()
    out["oa_counts"] = c["oa_status"].str.split(" ").str[0].value_counts().to_dict()
    out["class_counts_all_screened"] = c["ext_class"].value_counts().to_dict()
    out["lab_in_corpus_counts_eligible_E1E2"] = c[c["eligibility"].isin({"E1", "E2"})]["lab_in_corpus"].value_counts().to_dict()
    print("\n--- unit counts")
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
