#!/usr/bin/env python
"""Provenance inventory (brief section 5; stage 1).  Label-free except where a column says otherwise.

1. ``cells.csv``: every one of the 521 V1 cells joined to its archive records (gen19 ``master_clean`` via
   safe_exp_id -> source_record_id): DOI(s), citation, data location(s) (table / figure), curator(s), ORNL-import
   status, extractant SMILES / name / InChIKey / explicit stereo, chemotype, donor-defined DGA, gen19 copy group,
   source cluster, exact condition key and parsed conditions, aqueous complexant presence, metals, k_max, and the
   round-3 data-quality flags (V2 rules, smoothness-only suspects, the suspected low-D floor -- the last two are
   value-derived flags, reported, never used to select rows).
2. ``sources_by_doi.csv``: per DOI the cells, C3 cells, C3 pairs, extractants, chemotypes, curators, locations.
3. ``fold_training_<design>.csv``: per dev outer fold, what survives in TRAINING after the design's exclusions:
   cells, C3 cells, C3 strict-adjacent pairs, publications, DOIs, extractants, chemotypes, copy groups, exact-
   condition series.
4. ``auxiliary_ln.csv`` / ``auxiliary_an3.csv`` + ``auxiliary_summary.json``: archive records outside the cohort that
   could only ever be TRAINING data (extra lanthanide series; An(III) records), with the per-fold survival after
   excluding every record that shares a publication, a copy group or an exact extractant with the fold's held-out
   cells.  Chemotype isolation of new extractants is NOT guaranteed by these counts (no chemotype assignment for
   extractants outside the cohort) -- a stage-1 transfer slot must add it before use.

    .venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/scripts/r4_provenance.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r4 import donor as DN  # noqa: E402
from g21 import cache as C  # noqa: E402
from g21.cache import cell_view  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k import design as DS  # noqa: E402
from gen20k.splits import copy_group_map, folds_for  # noqa: E402
from gen13sep.metals import LANTHANIDES  # noqa: E402

sys.path.insert(0, str(paths.ROOT / "generations" / "gen19_chem_transfer"))
sys.path.insert(0, str(paths.R3DIR))
from gen19ct.data.load import load_archive  # noqa: E402
from r3 import datav2 as V2  # noqa: E402

OUT = paths.RES / "provenance"
SMOOTHNESS_SUSPECTS = ("6ff6056da7469332", "8cb0deef770e3ca0", "543ce167788b1f3e", "c950dc2982e96b83",
                       "326289072f176f80", "9586db05ac5e234b")      # round 3 inventory_data_quality.md D8
CROSS_ACID_COPY_SUSPECTS = ("44012938119033a5", "5169c377777d282b")
AN3 = {"Am", "Cm", "Cf", "Pu", "Bk", "Es"}


def _canon(smi: str) -> str:
    from rdkit import Chem
    m = Chem.MolFromSmiles(str(smi))
    return Chem.MolToSmiles(m) if m is not None else str(smi)


def _inchikey(smi: str) -> str:
    from rdkit import Chem
    m = Chem.MolFromSmiles(str(smi))
    return Chem.MolToInchiKey(m) if m is not None else ""


def _join(v) -> str:
    flat = []
    for x in v:
        if isinstance(x, (list, tuple, np.ndarray)):
            flat += [str(y) for y in x]
        elif x is not None and not (isinstance(x, float) and np.isnan(x)) and x is not pd.NA:
            flat.append(str(x))
    return ";".join(sorted({s for s in flat if s not in ("", "nan", "None", "<NA>")}))


def cohort_cells(bench, arch) -> pd.DataFrame:
    f = bench.frame.reset_index(drop=True)
    arch = arch.assign(sid=arch.source_record_id.astype(int))
    rows = []
    cg = copy_group_map()
    for i, r in f.iterrows():
        ids = [int(x.split(":")[-1]) for x in str(r.safe_exp_ids).split(";") if x]
        a = arch[arch.sid.isin(ids)]
        dz = DS.make_design(cell_view(bench, i), "adjacent", "metal")
        comments = a.comments_raw.astype(str).str.lower()
        y = bench.Y[i]
        rows.append({
            "ci": i, "cell_id": r.cell_id, "c3": bool(dz.query) and dz.k_max >= 3, "k_max": int(dz.k_max),
            "n_queries": len(dz.query), "n_metals": int(np.isfinite(y).sum()),
            "metals": ",".join(LANTHANIDES[m] for m in range(14) if np.isfinite(y[m])),
            "publication_id": r.publication_id, "copy_group": cg.get(str(r.publication_id), ""),
            "doi": _join(a.doi_primary_corrected), "title": _join(a.reference_title)[:200],
            "year": _join(a.reference_year.astype("Int64").astype(str)), "journal": _join(a.reference_journal),
            "n_records": len(a), "records_found": len(a) == len(ids),
            "data_location": _join(a.data_location), "curator": _join(a.entry_author),
            "ornl_import_all": bool(len(a) and comments.str.contains("ornl").all()),
            "ornl_import_any": bool(comments.str.contains("ornl").any()),
            "extractant_smiles": r.extractant, "extractant_name": r.extractant_name,
            "inchikey": _inchikey(r.extractant), "explicit_stereo": any(ch in str(r.extractant) for ch in "@/\\"),
            "chemotype": r.chemotype, "dga_donor": DN.is_dga(str(r.extractant)), "condition_key": r.condition_key,
            "acid": _join(a.acid_primary), "acid_M": _join(a.acid_concentration_M.round(4)),
            "extractant_M": _join(a.extractant_primary_concentration_M.round(5)), "solvent": _join(a.solvent_key),
            "aqueous_complexant": _join(a.complexant_names), "temperature_C": _join(a.temperature_C),
            "n_replicated_metals": int(r.n_replicated_metals) if pd.notna(r.n_replicated_metals) else 0,
            "v2_1_ornl_reimport": r.cell_id in V2.V2_1_CELLS, "v2_2_duplicate_entry": r.cell_id in V2.V2_2_DROP,
            "v2_3_hidden_merge": r.cell_id in V2.V2_3_CELLS, "v2_5_resolution_floor": r.cell_id in V2.V2_5_FLOOR,
            "flag_smoothness_suspect_value_derived": r.cell_id in SMOOTHNESS_SUSPECTS,
            "flag_cross_acid_copy_suspect": r.cell_id in CROSS_ACID_COPY_SUSPECTS,
            "flag_logD_le_-3.5_value_derived": bool(np.nanmin(y) <= -3.5) if np.isfinite(y).any() else False,
        })
    t = pd.DataFrame(rows)
    # source cluster = connected components of chemotype, publication, copy group (as r4.metrics, over all cells)
    par = {}

    def find(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for c, p, g in t[["chemotype", "publication_id", "copy_group"]].itertuples(index=False):
        par[find("c:" + str(c))] = find("p:" + str(p))
        par[find("p:" + str(p))] = find("g:" + str(g))
    t["source_cluster"] = t.chemotype.map(lambda c: find("c:" + str(c)))
    return t


def by_doi(cells: pd.DataFrame) -> pd.DataFrame:
    c = cells.assign(doi=cells.doi.str.split(";")).explode("doi")
    g = c.groupby("doi")
    return pd.DataFrame({
        "publication_ids": g.publication_id.agg(_join), "cells": g.cell_id.size(), "c3_cells": g.c3.sum(),
        "c3_pairs": g.apply(lambda h: int(h.loc[h.c3, "n_queries"].sum()), include_groups=False),
        "extractants": g.extractant_smiles.nunique(), "chemotypes": g.chemotype.agg(_join),
        "curators": g.curator.agg(_join), "data_locations": g.data_location.agg(_join),
        "ornl_import_cells": g.ornl_import_all.sum(), "flagged_cells": g.apply(
            lambda h: int((h.v2_1_ornl_reimport | h.v2_2_duplicate_entry | h.v2_3_hidden_merge | h.v2_5_resolution_floor).sum()),
            include_groups=False),
        "title": g.title.first(), "year": g.year.first()}).reset_index().sort_values("c3_pairs", ascending=False)


def fold_training(bench, cells: pd.DataFrame, design: str) -> pd.DataFrame:
    rows = []
    for f in folds_for(bench.frame, design, seeds=list(paths.DEV_SEEDS)):
        rec = C.load(f.design, f.seed, f.fold)
        tr = cells.set_index("ci").loc[np.asarray(rec["outer"]["train_index"])]
        c3 = tr[tr.c3]
        rows.append({"design": design, "seed": f.seed, "fold": f.fold, "train_cells": len(tr), "train_c3_cells": len(c3),
                     "train_c3_pairs": int(c3.n_queries.sum()), "publications": tr.publication_id.nunique(),
                     "dois": len({d for s in tr.doi for d in s.split(";") if d}), "extractants": tr.extractant_smiles.nunique(),
                     "chemotypes": tr.chemotype.nunique(), "copy_groups": tr.copy_group.nunique(),
                     "source_clusters": tr.source_cluster.nunique(), "condition_series": tr.condition_key.nunique(),
                     "c3_publications": c3.publication_id.nunique(), "c3_chemotypes": c3.chemotype.nunique(),
                     "test_cells": len(rec["outer"]["test_index"])})
    return pd.DataFrame(rows)


def auxiliary(bench, cells: pd.DataFrame, arch: pd.DataFrame) -> dict:
    """Archive series outside the cohort (training-only candidates) and their per-fold survival."""
    used = {int(x.split(":")[-1]) for s in bench.frame.safe_exp_ids for x in str(s).split(";") if x}
    a = arch[~arch.source_record_id.astype(int).isin(used) & arch.log_D.notna()].copy()
    a["ext"] = a.extractant_primary_smiles.astype(str).map(_canon)
    cg = copy_group_map()
    a["copy_group"] = a.g19_publication_id.astype(str).map(lambda p: cg.get(p, "own:" + p))
    cohort_ext = {_canon(s) for s in cells.extractant_smiles}
    cohort_pub = set(cells.publication_id.astype(str))
    out, summary = {}, {}
    for name, sel in (("ln", a.metal_category == "lanthanide"),
                      ("an3", (a.metal_category == "actinide") & a.metal_symbol.isin(AN3) &
                       (a.metal_oxidation_state.astype(str).isin(["3", "3.0"])))):
        s = a[sel]
        ser = s.groupby("series_id").agg(publication=("g19_publication_id", "first"), copy_group=("copy_group", "first"),
                                         doi=("doi_primary_corrected", "first"), ext=("ext", "first"),
                                         metals=("metal_symbol", lambda v: ",".join(sorted(set(v)))),
                                         n_metals=("metal_symbol", "nunique"), n_rows=("log_D", "size"),
                                         complexant=("complexant_names", lambda v: _join(v)),
                                         model_ready=("is_model_ready", "mean")).reset_index()
        ser["doi_in_cohort"] = ser.publication.astype(str).isin(cohort_pub)
        ser["extractant_in_cohort"] = ser.ext.isin(cohort_ext)
        out[name] = ser
        keep = ser[ser.n_metals >= (4 if name == "ln" else 1)]
        folds = []
        for d in ("BP", "BPG"):
            for f in folds_for(bench.frame, d, seeds=list(paths.DEV_SEEDS)):
                rec = C.load(f.design, f.seed, f.fold)
                te = cells.set_index("ci").loc[np.asarray(rec["outer"]["test_index"])]
                bad = (keep.publication.astype(str).isin(set(te.publication_id.astype(str))) |
                       keep.copy_group.isin(set(te.copy_group)) | keep.ext.isin({_canon(x) for x in te.extractant_smiles}))
                k = keep[~bad]
                folds.append({"design": d, "seed": f.seed, "fold": f.fold, "series": len(k), "rows": int(k.n_rows.sum()),
                              "publications": k.publication.nunique(), "extractants": k.ext.nunique()})
        fd = pd.DataFrame(folds)
        summary[name] = {
            "series_total": len(ser), "series_kept_rule": "n_metals >= 4" if name == "ln" else "any",
            "series_kept": len(keep), "rows_kept": int(keep.n_rows.sum()), "publications_kept": int(keep.publication.nunique()),
            "extractants_kept": int(keep.ext.nunique()), "kept_with_cohort_doi": int(keep.doi_in_cohort.sum()),
            "kept_with_cohort_extractant": int(keep.extractant_in_cohort.sum()),
            "kept_with_aqueous_complexant": int((keep.complexant != "").sum()),
            "per_fold_after_exclusions_min": {c: int(fd[c].min()) for c in ("series", "rows", "publications", "extractants")},
            "per_fold_after_exclusions_median": {c: float(fd[c].median()) for c in ("series", "rows", "publications", "extractants")},
            "chemotype_isolation": "not checked (no chemotype assignment outside the cohort)"}
        fd.to_csv(OUT / f"auxiliary_{name}_folds.csv", index=False)
    out["ln"].to_csv(OUT / "auxiliary_ln.csv", index=False)
    out["an3"].to_csv(OUT / "auxiliary_an3.csv", index=False)
    return summary


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    bench = D.load_bench()
    arch = load_archive()
    cells = cohort_cells(bench, arch)
    cells.to_csv(OUT / "cells.csv", index=False)
    by_doi(cells).to_csv(OUT / "sources_by_doi.csv", index=False)
    folds = {d: fold_training(bench, cells, d) for d in ("BP", "BPG")}
    for d, t in folds.items():
        t.to_csv(OUT / f"fold_training_{d}.csv", index=False)
    c3 = cells[cells.c3]
    summary = {
        "cohort": {"cells": len(cells), "records_all_found": bool(cells.records_found.all()),
                   "dois": len({d for s in cells.doi for d in s.split(";") if d}), "publications": cells.publication_id.nunique()},
        "c3": {"cells": len(c3), "pairs": int(c3.n_queries.sum()), "dois": len({d for s in c3.doi for d in s.split(";") if d}),
               "publications": c3.publication_id.nunique(), "extractants": c3.extractant_smiles.nunique(),
               "chemotypes": c3.chemotype.nunique(), "copy_groups": c3.copy_group.nunique(),
               "source_clusters": c3.source_cluster.nunique(), "condition_series": c3.condition_key.nunique(),
               "cells_with_data_location": int((c3.data_location != "").sum()), "ornl_import_cells": int(c3.ornl_import_all.sum()),
               "explicit_stereo_extractants": int(c3.drop_duplicates("extractant_smiles").explicit_stereo.sum()),
               "cells_with_aqueous_complexant": int((c3.aqueous_complexant != "").sum()),
               "cells_flagged_v2": int((c3.v2_1_ornl_reimport | c3.v2_2_duplicate_entry | c3.v2_3_hidden_merge |
                                        c3.v2_5_resolution_floor).sum()),
               "cells_value_flags": int((c3["flag_smoothness_suspect_value_derived"] | c3["flag_logD_le_-3.5_value_derived"]).sum()),
               "curators": c3.curator.str.split(";").explode().value_counts().to_dict()},
        "training_after_exclusions": {d: {c: {"min": int(t[c].min()), "median": float(t[c].median()), "max": int(t[c].max())}
                                          for c in ("train_c3_cells", "train_c3_pairs", "c3_publications", "c3_chemotypes",
                                                    "publications", "copy_groups", "source_clusters")}
                                      for d, t in folds.items()},
        "auxiliary": auxiliary(bench, cells, arch)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
