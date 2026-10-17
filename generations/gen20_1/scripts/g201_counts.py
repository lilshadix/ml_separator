"""gen20.1 stage S3 (COUNTS, Stage 0a): cohort assertions, fold gates (a)-(e), row manifests, and the
recomputation of every value the three gen20.1 documents mark [CNT], [DMD] or [INNER].

Label-free (IMPLEMENTATION_PLAN.md §9 S3): no log D or log SF value is read beyond NaN masks and the
equality assertions; no SD/mean/variance/metric of a target; no model fit.  The one exception is the
frozen gen13 quarantine rule itself: the cohort-chain counts (5,992 -> 5,860 -> 2,395 cells ...) re-run
``gen13sep.cohort.apply_quarantine``, whose sentinel test reads ``log_D <= -6`` as a boolean row mask
(the missing-value code -12.467 of three Pr rows, as in gen5-gen13); ``log_D`` is dropped right after and
enters nothing else.  This read is declared in counts_g201.json ("label_reads").  Replicate counts
(``nrep__``) and provenance run identifiers are identity data.

Outputs (the S3 reference set; all deterministic, no timestamps):
  fold_plan_g201.json (gates (a)-(e), per-fold sizes, inner-fold sizes),
  fold_records_g201.parquet (gen20k.splits.fold_record of the six designs),
  row_manifests/<population>.parquet + row_manifests/DIGESTS.json: the scored populations of §2 per
  (design, split_seed, fold) test cells (COH-D-C3, COH-SF-C3, COH-D-C3M, COH-D-ALL, COH-SF-ANY, COH-SF-C1,
  COH-SF-NDSM, COH-SC009-D/-SF, COH-NOSC009-D/-SF, each also @NODMD), COH-TRAIN, COH-REP, COH-INNER (per
  unit), COH-FROZEN, COH-AUG and COH-REPPAIR (cohort level); COH-REPCAP is label-derived and written at S5,
  counts_g201.json (every comparison, recorded values, gates, label reads, status);
plus the run manifest g201_counts__<label>.json (git, environment, dataset digests, timings, the coverage
audit of the tagged document blocks, and on a re-run the byte comparison).

Every output is first written to a fresh temporary directory, never into --out directly:
  * first run (--out holds none of the reference files): on PASS the files are copied into --out; on STOP
    nothing is copied and the temporary directory is kept for diagnosis;
  * re-run (--out already holds a reference file, e.g. the S6 re-run of plan §9): the sha256 of every
    reference file is compared with the existing one; any difference (or a missing/extra file) is a STOP,
    the existing files are never touched, and the temporary copies are kept for diagnosis (deleted on a match).
--label names the run manifest (default S3; the S6 re-run passes --label S6); an existing
g201_counts__<label>.json is never overwritten.  --no-subprocess (tests only) skips the PYTHONHASHSEED
subprocess check of gates (a), (b), (e): it is refused for the default manifests directory and always ends
INCOMPLETE (exit 1).  The forbidden-writer guard is installed, and the repository outside generations/gen20_1
(plus the three gen20_1 documents) is snapshotted before the bench is loaded and compared before
counts_g201.json is written and again at the end.
Exit status 0 only if every assertion, gate, documented value and coverage check passes (else STOP
INCOMPARABLE_FOLDS / count mismatch, exit 1).

Usage (repo root):  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B generations/gen20_1/scripts/g201_counts.py
                    [--label S6] [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
G201 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(G201))

from gen201 import bootstrap  # noqa: E402
from gen201 import data as D  # noqa: E402
from gen201 import folds as FO  # noqa: E402
from gen201 import guard  # noqa: E402
from gen201.run import Run, sha256_file  # noqa: E402

guard.install()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DOCS = {"PLAN": bootstrap.PLAN, "PREREG": bootstrap.PREREG, "README": bootstrap.G201 / "README.md"}
TAG_RE = re.compile(r"\[(CNT|DMD|INNER)\]")


# --------------------------------------------------------------------------------------
# comparison table
# --------------------------------------------------------------------------------------
class Table:
    def __init__(self):
        self.rows: list[dict] = []
        self.infos: list[dict] = []

    def check(self, cid: str, source: str, tag: str, what: str, expected, got, *, tol: float | None = None,
              ndigits: int | None = None) -> bool:
        if ndigits is not None:
            ok = round(float(got), ndigits) == round(float(expected), ndigits)
        elif tol is not None:
            ok = abs(float(got) - float(expected)) <= tol
        elif isinstance(expected, (list, tuple)):
            ok = list(_py(expected)) == list(_py(got))
        else:
            ok = _py(expected) == _py(got)
        self.rows.append({"id": cid, "source": source, "tag": tag, "what": what, "expected": _py(expected),
                          "recomputed": _py(got), "match": bool(ok),
                          "rule": ("round %d" % ndigits) if ndigits is not None else (f"|d| <= {tol}" if tol else "exact")})
        return ok

    def info(self, cid: str, source: str, what: str, got) -> None:
        """A recomputed value with no documented expectation (recorded and printed, never a mismatch)."""
        self.infos.append({"id": cid, "source": source, "tag": "info", "what": what, "recomputed": _py(got)})

    @property
    def mismatches(self) -> list[dict]:
        return [r for r in self.rows if not r["match"]]

    def print(self) -> None:
        w = max(len(r["id"]) for r in self.rows + self.infos)
        print("\n" + "=" * 140)
        print(f"{'id':<{w}}  {'tag':<11} {'match':<5}  {'expected':<34} {'recomputed':<34} what (source)")
        print("-" * 140)
        for r in self.rows:
            e, g = _fmt(r["expected"]), _fmt(r["recomputed"])
            print(f"{r['id']:<{w}}  {r['tag']:<11} {'OK' if r['match'] else 'FAIL':<5}  {e:<34} {g:<34} {r['what']} ({r['source']})")
        print("-" * 140)
        print(f"{len(self.rows)} values compared, {len(self.mismatches)} mismatches")
        if self.infos:
            print("\nrecorded without a documented expectation (info):")
            for r in self.infos:
                print(f"  {r['id']:<{w}}  {json.dumps(r['recomputed'])}  {r['what']} ({r['source']})")


def _py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (list, tuple)):
        return [_py(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _py(x) for k, x in v.items()}
    return v


def _fmt(v) -> str:
    s = json.dumps(v) if not isinstance(v, str) else v
    return s if len(s) <= 34 else s[:31] + "..."


# --------------------------------------------------------------------------------------
# the cohort chain from the bundle (identity columns; log_D only inside gen13's quarantine mask)
# --------------------------------------------------------------------------------------
def cohort_chain(bench) -> dict:
    import pyarrow.parquet as pq
    from gen13sep import cohort as C, paths
    paths.assert_bundle_unchanged()
    cond = [c for c in pq.read_schema(paths.BUNDLE_PARQUET).names if c.startswith("cond__")]
    cols = ["safe_exp_id", "metal", "canonical_smiles", "extractant_name", "log_D", *cond]
    b0 = pd.read_parquet(paths.BUNDLE_PARQUET, columns=cols)
    bq, audit = C.apply_quarantine(b0)                     # frozen rule; the only log_D read (boolean mask)
    b0 = b0.drop(columns=["log_D"])
    bq = bq.drop(columns=["log_D"])
    quarantined = set(b0["safe_exp_id"]) - set(bq["safe_exp_id"])
    prov = pd.read_parquet(paths.GEN6_PROVENANCE_PARQUET, columns=["safe_exp_id", "publication_id"])

    def chain(b, with_pub: bool) -> dict:
        b = b.merge(prov, on="safe_exp_id", how="left", validate="one_to_one")
        ck = C._condition_key(b, cond)
        key = b["canonical_smiles"] + ("@@" + b["publication_id"].astype(str) if with_pub else "") + "@@" + ck
        b = b.assign(cell_key=key, condition_key=ck)
        nmet = b.groupby("cell_key")["metal"].nunique()
        multi = set(nmet.index[nmet >= 2])
        single = set(nmet.index[nmet < 2])
        inm = b["cell_key"].isin(multi)
        ent_size = b[inm].groupby(["cell_key", "metal"]).size()
        rep = ent_size[ent_size >= 2]
        return {"b": b, "multi": multi, "inm": inm, "single": single,
                "cells_all": int(len(nmet)), "entries_all": int(b.groupby(["cell_key", "metal"]).ngroups),
                "cells_multi": int(len(multi)), "rows_multi": int(inm.sum()), "entries_multi": int(len(ent_size)),
                "cells_single": int(len(single)), "rows_single": int(b["cell_key"].isin(single).sum()),
                "replicated_entries": int(len(rep)), "replicated_cells": int(rep.index.get_level_values(0).nunique()),
                "replicated_rows": int(rep.sum())}

    q_pub = chain(bq, True)
    q_nopub = chain(bq, False)
    raw_pub = chain(b0, True)
    # the multi-metal cell keys must be exactly the 521 bench cells (gen13 cell_id = blake2b(extractant@@pub@@key))
    bm = q_pub["b"][q_pub["inm"]].drop_duplicates("cell_key")
    ids = {hashlib.blake2b((s + "@@" + str(p) + "@@" + k).encode(), digest_size=8).hexdigest()
           for s, p, k in zip(bm["canonical_smiles"], bm["publication_id"], bm["condition_key"])}
    same_cells = ids == set(bench.frame["cell_id"].astype(str))
    rb = raw_pub["b"]
    q_rows_in_multi = int(rb[rb["safe_exp_id"].isin(quarantined)]["cell_key"].isin(raw_pub["multi"]).sum())
    # COH-AUG (not run, O23): the single-metal gen13-key cells, identity columns only
    sb = q_pub["b"][q_pub["b"]["cell_key"].isin(q_pub["single"])]
    aug = (sb.assign(cell_id=[hashlib.blake2b(k.encode(), digest_size=8).hexdigest() for k in sb["cell_key"]])
             .groupby("cell_id", sort=True)
             .agg(extractant=("canonical_smiles", "first"), publication_id=("publication_id", "first"),
                  metal=("metal", "first"), n_metal_values=("metal", "nunique"), n_rows=("safe_exp_id", "size"),
                  safe_exp_ids=("safe_exp_id", lambda s: ";".join(sorted(map(str, s)))))
             .reset_index())
    if (aug["n_metal_values"] != 1).any():
        raise AssertionError("a single-metal cell holds more than one metal")
    aug = aug.drop(columns=["n_metal_values"])
    strip = lambda r: {k: v for k, v in r.items() if k not in ("b", "multi", "inm", "single")}  # noqa: E731
    return {"quarantine_audit": audit, "n_quarantined_rows": len(quarantined),
            "quarantined_rows_in_multimetal_cells_raw": q_rows_in_multi,
            "quarantined_pub": strip(q_pub), "quarantined_nopub": strip(q_nopub), "raw_pub": strip(raw_pub),
            "multi_metal_cells_equal_bench": bool(same_cells), "_aug": aug}


# --------------------------------------------------------------------------------------
# row manifests
# --------------------------------------------------------------------------------------
def content_digest(t: pd.DataFrame) -> str:
    return hashlib.sha256(t.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def _write_manifest(t: pd.DataFrame, name: str, out_dir: Path, **meta) -> dict:
    p = out_dir / f"{name}.parquet"
    t = t.reset_index(drop=True)
    t.to_parquet(p, index=False)
    return {"file": f"row_manifests/{p.name}", **meta, "rows_total": int(len(t)), "columns": list(t.columns),
            "sha256_file": sha256_file(p), "sha256_content_csv": content_digest(t)}


def write_training_manifests(bench, F: dict, dm, out_dir: Path) -> tuple[dict, pd.DataFrame]:
    """COH-TRAIN (every (design, seed, fold) training entry, with nrep and the DMDPhPDA flag: COH-NODMD's
    training rows are those with is_dmd False), COH-REP (its rows with nrep >= 2) and COH-INNER (the four
    masked chemotype inner folds of every unit: role train / val / masked_out, skipped flags for the primary
    and the COH-NODMD variant).  Returns the digests and the per-unit training summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    f = bench.frame
    dmd = D.dmd_mask(bench)
    cell_id = f["cell_id"].astype(str).to_numpy()
    tr_parts, in_parts = [], []
    for d in FO.DESIGNS:
        for fo in F[d]:
            t = D.train_logd_rows(bench, fo.train_index)[["ci", "cell_id", "metal", "im", "nrep", "is_dmd"]]
            t.insert(0, "fold", int(fo.fold))
            t.insert(0, "split_seed", int(fo.seed))
            t.insert(0, "design", d)
            tr_parts.append(t)
            prim = FO.inner_folds(bench, d, fo)
            nod = FO.inner_folds(bench, d, fo, drop=dm)
            for p_, q_ in zip(prim, nod):
                masked_out = np.setdiff1d(p_.train_pos_unmasked, p_.train_pos)
                for role, idx in (("train", p_.train_pos), ("val", p_.val_pos), ("masked_out", masked_out)):
                    idx = np.sort(np.asarray(idx, dtype=int))
                    in_parts.append(pd.DataFrame({
                        "design": d, "split_seed": int(fo.seed), "fold": int(fo.fold), "inner": int(p_.j),
                        "role": role, "ci": idx, "cell_id": cell_id[idx], "is_dmd": dmd[idx],
                        "skipped": bool(p_.skipped), "skipped_nodmd": bool(q_.skipped)}))
    TR = pd.concat(tr_parts, ignore_index=True)
    IN = pd.concat(in_parts, ignore_index=True)
    if TR.duplicated(["design", "split_seed", "fold", "ci", "im"]).any():
        raise AssertionError("COH-TRAIN: duplicated keys")
    if IN.duplicated(["design", "split_seed", "fold", "inner", "ci"]).any():
        raise AssertionError("COH-INNER: a cell holds two roles in one inner fold")
    RP = TR[TR["nrep"] >= 2]
    digests = {
        "COH-TRAIN": _write_manifest(TR, "COH-TRAIN", out_dir, target="logd", side="train",
                                     definition="every measured entry of the masked outer-training cells per "
                                                "(design, split_seed, fold); nrep = replicate count; COH-NODMD "
                                                "training rows = rows with is_dmd False"),
        "COH-REP": _write_manifest(RP, "COH-REP", out_dir, target="logd", side="train",
                                   definition="COH-TRAIN rows with nrep >= 2 (replicated entries of the outer-training "
                                              "cells; counts only, never values)"),
        "COH-INNER": _write_manifest(IN, "COH-INNER", out_dir, target="cell", side="inner",
                                     definition="4-fold chemotype inner cross-fit of every outer training fold: "
                                                "seeded_group_kfold(chemotype[train], 4, inner_seed(seed, fold)), "
                                                "gen20k.cov.inner_mask applied (role masked_out), skipped if < 20 "
                                                "masked inner-train cells; skipped_nodmd after removing the 5 "
                                                "DMDPhPDA cells (rows with is_dmd True)"),
    }
    summ = (TR.groupby(["design", "split_seed", "fold"])
              .agg(n_train_rows=("im", "size"), n_train_rep_entries=("nrep", lambda s: int((s >= 2).sum())),
                   n_train_rows_nodmd=("is_dmd", lambda s: int((~s).sum())))
              .reset_index())
    return digests, summ


def write_cohort_manifests(bench, aug: pd.DataFrame, reppair: pd.DataFrame, out_dir: Path) -> dict:
    """COH-FROZEN (the 521 cells), COH-AUG (single-metal gen13-key cells; not run, O23) and COH-REPPAIR
    (strict-adjacent pairs with both endpoints replicated and their run pairing; identity columns only)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    f = bench.frame
    obs = D.observed(bench)
    frozen = pd.DataFrame({"ci": np.arange(len(f)), "cell_id": f["cell_id"].astype(str).to_numpy(),
                           "extractant": f["extractant"].astype(str).to_numpy(),
                           "chemotype": f["chemotype"].astype(str).to_numpy(),
                           "publication_id": f["publication_id"].astype(str).to_numpy(),
                           "n_metals": obs.sum(1).astype(int), "n_rows": f["n_rows"].to_numpy().astype(int),
                           "is_dmd": D.dmd_mask(bench)})
    return {
        "COH-FROZEN": _write_manifest(frozen, "COH-FROZEN", out_dir, target="cell", side="cohort",
                                      definition="the 521 cells of the frozen gen13 exact cohort (universe for folds "
                                                 "and training); its entries are COH-D-ALL"),
        "COH-AUG": _write_manifest(aug, "COH-AUG", out_dir, target="cell", side="cohort",
                                   definition="single-metal gen13-key cells (canonical SMILES @@ publication_id @@ "
                                              "64-column condition key) after the quarantine; NOT RUN (O23)"),
        "COH-REPPAIR": _write_manifest(reppair, "COH-REPPAIR", out_dir, target="logsf", side="cohort",
                                       definition="strict-adjacent pairs whose two endpoints are replicated (nrep >= 2), "
                                                  "with the number of run values (key column) in which a and b each "
                                                  "have exactly one reading; run_paired = >= 2 such runs; identity "
                                                  "columns only (O16 feasibility)"),
    }


def write_row_manifests(pops: dict, F: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name in sorted(pops):
        rs = pops[name]
        parts = []
        for d in FO.DESIGNS:
            for f in F[d]:
                te = set(np.asarray(f.test_index).tolist())
                sub = rs.rows[rs.rows["ci"].isin(te)].copy()
                sub.insert(0, "fold", int(f.fold))
                sub.insert(0, "split_seed", int(f.seed))
                sub.insert(0, "design", d)
                parts.append(sub)
        t = pd.concat(parts, ignore_index=True)
        keys = ["design", "split_seed", "fold", "ci"] + (["im"] if rs.target == "logd" else ["ia", "ib"])
        if t.duplicated(keys).any():
            raise AssertionError(f"{name}: duplicated manifest keys")
        for d in FO.DESIGNS:
            for s in FO.SEEDS:
                n = int(((t.design == d) & (t.split_seed == s)).sum())
                if n != rs.n_rows:
                    raise AssertionError(f"{name}: design {d} seed {s} has {n} rows, population has {rs.n_rows}")
        digests[name] = _write_manifest(t, name, out_dir, target=rs.target, side="test", definition=rs.definition,
                                        rows_per_design_seed=rs.n_rows)
    return digests


def write_digests(digests: dict, out_dir: Path) -> Path:
    p = out_dir / "DIGESTS.json"
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(_py(digests), fh, indent=1, sort_keys=True)
    return p


# --------------------------------------------------------------------------------------
# coverage audit: every number in a tagged block must appear among the expected values (else STOP)
# --------------------------------------------------------------------------------------
NUM_RE = re.compile(r"(?<![\w.:/#-])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\w%])")


def _num_variants(v) -> set[str]:
    """The spellings under which an expected value may appear in a document: the integer, the float at 1..16
    decimals (with and without trailing zeros, never rounded to an integer: 12.97 does not cover '13'), and a
    share in (0, 1) as a whole percent."""
    out = set()
    if isinstance(v, bool) or v is None:
        return out
    if isinstance(v, (list, tuple)):
        for x in v:
            out |= _num_variants(x)
        return out
    if isinstance(v, (int, float)):
        f = float(v)
        if f == int(f):
            out.add(str(int(f)))
        for nd in range(1, 17):
            s = f"{f:.{nd}f}"
            t = s.rstrip("0").rstrip(".")
            if "." not in t and f != int(f):
                continue                           # rounded to an integer: 12.97 never covers '13' or '13.0'
            out.add(s)
            out.add(t)
        if 0 < f < 1:
            out.add(str(round(100 * f)))           # shares quoted as percent
    return out


#: numbers that occur in a tagged block but are not label-free counts of gen20.1: each key is a regular expression
#: that matches the number IN ITS CONTEXT (so the same digits used as a count elsewhere are still audited); the
#: value is the reason.  Matches are blanked (on the joined block text, before citations are removed) before the
#: numbers of the block are extracted.  There is no blanket list of small integers.
NOT_A_COUNT: dict[str, str] = {
    # dates, versions, machine readings, timings
    r"\b\d{4}-\d{2}-\d{2}\b": "a date",
    r"\bdraft \d+\b": "draft number of the document",
    r"\b\d+\.\d+ s\b": "[TIM] mean single-fit seconds of gen13's timings.csv (cited), not a count",
    r"\b12 logical CPUs, 8,464,097,280 bytes RAM, Python \d+\.\d+\.\d+, numpy \d+\.\d+\.\d+":
        "[ENV] machine reading (recorded by gen201.env.environment at run time), not a count",
    # decision constants of the pre-registration (thresholds, interval levels, budgets)
    r"τ = 0\.4 / 0\.5": "decision thresholds tau",
    r"RULED_OUT at 0\.4, RULED_OUT at 0\.5": "decision thresholds tau",
    r"\b(?:95|98\.75) %": "interval levels (K = 4 Bonferroni gives 98.75 %)",
    r"\b(?:2\.5|97\.5)th percentiles?": "percentile levels of the 95 % interval",
    r"\b10,000 seed-mean draws": "bootstrap replicate count (design constant)",
    r"\bK = 4 Bonferroni": "number of deciding claims (design constant)",
    r"≥ 4 of 5 per-seed values": "RULE-T per-seed criterion (design constant)",
    r"Fewer than 10 clusters": "'preliminary' threshold on the number of clusters (recovery plan :304)",
    r"repsd > 0\.5": "COH-REPCAP replicate-sd cap (design constant)",
    # definitions (adjacency, cohort and fold constants)
    r"\|ΔZ\| = 1(?:, 12 types)?": "strict adjacency |dZ| = 1 and its 12 pair types (La/Ce .. Yb/Lu, Pm absent): definitional",
    r"\by < 0\b": "sign convention of log SF",
    r"\bk(?:_max)? *(?:=|≥|>|<) *\d+": "k-shot level / gen20's metal-model k_max cohort threshold (definitional)",
    r"≥ [12] (?:lanthanides|metals|such pair|scored)": "cohort / population membership thresholds (definitional)",
    r"\bnrep ≥ 2\b": "replicate definition (nrep >= 2)",
    r"\bddof 1\b": "sample-sd convention of repsd__",
    r"\b64-column condition key": "width of gen13's condition key (definitional)",
    r"\b\d+-fold\b": "number of outer or inner folds (design constant)",
    r"\b5 split seeds × 5 folds": "split seeds x folds (design constant)",
    r"\b5-seed mean": "number of split seeds (design constant)",
    r"< 20(?= masked inner-training|, skipped)": "MIN_INNER_TRAIN = 20, the skip threshold of an inner fold (gen20k/cov.py:84)",
    r"\brow \d+\b": "a table row reference",
    r"^\| \d+ \|": "the row number of the PRE_REGISTRATION §12 deviations table",
    r"(?<![\w(])\(\d\)(?= [A-Z0-9`\"])": "an enumerator '(1)'..'(7)' of the plan's corrections",
    r"\bpossibly 0\b": "a hedge on a count that S3 recomputes (COH-REPPAIR feasibility, recorded as info)",
    r"\babout 4 effective chemotypes": "a rounding of the Kish values 3.85 / 3.43 that are checked themselves",
    # values cited from other generations' files (not gen20.1 label-free counts; never recomputed at S3)
    r"\b111 rows of other extractants": "cited breakdown of the 129 quarantined rows (runs/gen7_architecture/ambiguity_audit.md:311)",
    r"\b18 legitimate \"TODGA,DHOA\" mixture rows": "cited breakdown of the 129 quarantined rows (ambiguity_audit.md:311)",
    r"89 %/11 %, variance 2\.776, adjacent sd 0\.231 with n = 2,115, replicate median 0\.237":
        "DATA_AUDIT §3 statistics, cited (pre-quarantine target statistics; S3 reads no target value)",
    r"\bfor 0\.237\b": "DATA_AUDIT replicate median, cited (a target statistic)",
    r"\+4/\+3/\+2/\+1 decades in the 1–4 M cells and by \+1 for La, Ce and Pr in the 5 M cell":
        "gen8's DMDPhPDA source audit (cited): decade shifts and nitrate molarities, not counts",
    r"\b59 of its\b": "gen8's DMDPhPDA source audit (cited; label-derived: entries that differ from the curated copy)",
    r"the 5 M Pr/Nd pair \(\+1\.0\)": "gen8's DMDPhPDA source audit (cited): molarity and decade shift",
    r"= 42 \+ fold·1009 \+ 9,999,991": "model-seed formula constants (gen13sep/splits.py:24-29)",
    r"seed·31 \+ fold·7919 \+ 17": "inner-seed formula constants (gen20k/splits.py:31-33)",
    r"\(275 cells, 1,980 strict-adjacent pairs": "gen20's pair-model C3 cohort, cited (levels_BP_pair_C3_DOPT_Q.csv:6,18); not a gen20.1 population",
    r"\b12 % of replicated readings": "cited (GEN20_REPORT.md:402); label-derived, never recomputed at S3",
    r"\b13/25 near tie": "gen20's selection near ties, cited (GEN20_REPORT.md)",
    r"0\.2583 \(gen5 BASE91": "cited pooled R2 (gen5), a result, not a count",
    r"0\.2721 \(gen7": "cited pooled R2 (gen7)",
    r"0\.174–0\.303 per seed": "cited per-seed R2 range (gen6 MONO_ET)",
    r"0\.42–0\.43 only with near analogues": "cited R2 range (gen5)",
    r"unseen_ligand 0\.4253": "cited R2 (gen5 unseen_ligand)",
    r"pooled R² 0\.421": "cited R2 (gen12.2 DEC_L4_S1)",
    r"pre-registered selection 0\.240": "cited R2 (gen12.2 selection)",
    r"< 0\.4 \(gen6 within-cell shape R² 0\.30": "threshold tau and a cited R2 (gen6 phase 2)",
}


def tagged_blocks(lines: list[str]) -> list[tuple[int, int]]:
    """1-based inclusive (first, last) line ranges of the blocks that carry a [CNT]/[DMD]/[INNER] tag.  A block
    is a paragraph or one list item with its continuation lines (ended by a blank line or the next list item,
    heading or table row); a table row or heading is a block of its own.  The whole block is audited, so values
    that sit after their tag ('counts [CNT] unless stated: ...') are seen as well as values before it."""
    spans, start = [], None
    for i, line in enumerate(lines, 1):
        if not line.strip():
            if start is not None:
                spans.append((start, i - 1))
                start = None
            continue
        if start is not None and _BLOCK_START.match(line):
            spans.append((start, i - 1))
            start = None
        if start is None:
            start = i
        if _OWN_BLOCK.match(line):
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(lines)))
    return [(a, b) for a, b in spans if any(TAG_RE.search(lines[k - 1]) for k in range(a, b + 1))]


_BLOCK_START = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s|#|\|)")
_OWN_BLOCK = re.compile(r"^\s*(?:#|\|)")
#: a file:line citation, its line list possibly continued after a comma ('splits.py:24-29, 58-60', 'csv:6,18,30')
_CITE = r"(?::\d+(?:[\-–]\d+)?(?:,(?:\d+(?:[\-–]\d+)?|\s:?\d+[\-–]\d+|\s:\d+(?:[\-–]\d+)?))*)?"


def _clean(text: str) -> str:
    """Remove what is not a number of the text: every NOT_A_COUNT match, code spans, file:line citations, tags,
    section references, identifiers (T1, O23, MET-04, C3, ...) and the five discovery split seeds."""
    clean = text
    for pat in NOT_A_COUNT:
        clean = re.sub(pat, " ", clean)
    clean = re.sub(r"`[^`]*`", " ", clean)                                                   # code spans
    clean = re.sub(r"\S+\.(?:md|csv|py|json|parquet|rtf|txt|pkl)" + _CITE, " ", clean)        # file:line citations
    clean = re.sub(r"\b(?:csv|RTF|brief|line|lines)\s*" + _CITE, " ", clean)                  # 'csv:6,18', 'RTF:42'
    clean = re.sub(r"(?:(?<=[\s,(;])|^)" + _CITE[3:-2], " ", clean)                           # ' :126-127', '(:304)'
    clean = re.sub(r"\[(?:LEV|BRIEF|CNT|DMD|INNER|TIM|ENV)\]" + _CITE, " ", clean)
    clean = re.sub(r"§\s*\d+(?:\.\d+)*[a-z]?", " ", clean)                                     # section refs
    clean = re.sub(r"\b(?:O|H|T|S|K|C|D|MET-|LT-|EX-|TAB-|RK|OUT|RULE-[A-Z]\d*)\d+[a-z]?\b", " ", clean)  # IDs
    clean = re.sub(r"\b(?:%s)\b" % "|".join(str(s) for s in FO.SEEDS), " ", clean)            # discovery split seeds
    return clean


def coverage_audit(table: Table, extra_ok: set[str] = frozenset()) -> list[dict]:
    """For every tagged block of the three documents (see ``tagged_blocks``), list the numbers that are not
    among the expected values of the comparison table, after ``_clean``.  Any unmapped number is a STOP."""
    covered = set(extra_ok)
    for r in table.rows:
        covered |= _num_variants(r["expected"])
    report = []
    for key, path in DOCS.items():
        lines = path.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")
        for a, b in tagged_blocks(lines):
            block = " ".join(x.strip() for x in lines[a - 1:b])
            toks = [t.replace(",", "") for t in NUM_RE.findall(_clean(block))]
            unmapped = sorted({t for t in toks if t not in covered})
            report.append({"doc": key, "lines": [a, b], "tags": sorted(set(TAG_RE.findall(block))),
                           "n_numbers": len(toks), "unmapped": unmapped})
    return report


# --------------------------------------------------------------------------------------
# the S3 reference set, the re-run comparison (plan §9 S6) and publication of a first run
# --------------------------------------------------------------------------------------
REFERENCE_FILES = ("fold_plan_g201.json", "counts_g201.json", "fold_records_g201.parquet", "row_manifests/DIGESTS.json")


def reference_files(d) -> list[str]:
    """The S3 reference files present in ``d`` (posix paths relative to it): the four fixed files and every
    row_manifests/*.parquet."""
    d = Path(d)
    names = [n for n in REFERENCE_FILES if (d / n).is_file()]
    if (d / "row_manifests").is_dir():
        # COH-REPCAP* manifests are label-derived S5 outputs (g201_describe.py, digests in describe_g201.json),
        # never written by this script, so they are not S3 reference files
        names += [f"row_manifests/{p.name}" for p in (d / "row_manifests").glob("*.parquet")
                  if not p.name.startswith("COH-REPCAP")]
    return sorted(names)


def compare_reference(ref_dir, new_dir) -> dict:
    """sha256 of every reference file on both sides.  ``identical`` only if both sides hold the same file
    names and every file is byte-identical (a missing or an extra file is a difference)."""
    ref, new = set(reference_files(ref_dir)), set(reference_files(new_dir))
    files = []
    for n in sorted(ref | new):
        a = sha256_file(Path(ref_dir) / n) if n in ref else None
        b = sha256_file(Path(new_dir) / n) if n in new else None
        files.append({"file": n, "reference_sha256": a, "rerun_sha256": b, "identical": a is not None and a == b})
    differing = [r["file"] for r in files if not r["identical"]]
    return {"n_files": len(files), "identical": bool(files) and not differing, "differing": differing, "files": files}


def publish(work, out) -> list[Path]:
    """Copy the reference files of a passing first run from the temporary directory into ``out``."""
    written = []
    for n in reference_files(work):
        dst = bootstrap.assert_under_g201(Path(out) / n)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(work) / n, dst)
        written.append(dst)
    return written


def _same_path(a, b) -> bool:
    return os.path.normcase(str(Path(a).resolve())) == os.path.normcase(str(Path(b).resolve()))


# --------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(bootstrap.MANIFESTS), help="manifest directory (default gen20_1/manifests)")
    ap.add_argument("--label", default="S3",
                    help="run label; names the run manifest g201_counts__<label>.json (default S3; the S6 re-run passes S6)")
    ap.add_argument("--no-subprocess", action="store_true",
                    help="skip the PYTHONHASHSEED subprocess gate (tests only): refused for the default manifests "
                         "directory, always ends INCOMPLETE (exit 1)")
    args = ap.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.label):
        raise SystemExit(f"--label {args.label!r}: letters, digits, '_', '.' and '-' only")
    out = bootstrap.assert_under_g201(args.out)
    if args.no_subprocess and _same_path(out, bootstrap.MANIFESTS):
        raise SystemExit("--no-subprocess skips gate (b) in subprocesses, so its outputs can never be the S3 reference: "
                         "refused for the default manifests directory (pass --out to a scratch directory)")
    manifest_path = out / f"g201_counts__{args.label}.json"
    if manifest_path.exists():
        raise SystemExit(f"{manifest_path} exists: a run manifest is never overwritten; pass a new --label "
                         "(the S6 re-run passes --label S6)")
    out.mkdir(parents=True, exist_ok=True)
    rerun = bool(reference_files(out))
    work = bootstrap.assert_under_g201(tempfile.mkdtemp(prefix=f"g201_counts_{args.label}_"))
    T = Table()
    gates: dict = {}
    with Run("g201_counts", label=args.label, args=vars(args), seeds=list(FO.SEEDS), manifest_dir=out) as run:
        mode = "rerun" if rerun else "first"
        run.record("mode", {"mode": mode, "reference_dir": str(out), "work_dir": str(work),
                            "reference_files_present": reference_files(out)})
        run.log(f"{mode} run: every output is written to {work}; "
                + (f"it is compared byte for byte with the reference files in {out}, which are never touched" if rerun
                   else f"on PASS the files are copied into {out}"))
        before = guard.snapshot()                           # the repository outside gen20_1 + the three documents
        run.log(f"snapshot of {len(before)} files outside generations/gen20_1 (+ the gen20.1 documents) taken")
        # ---------------- cohort assertions (§1.1) ----------------
        bench = D.load(check=True)
        rec = dict(D.LOAD_RECORD)
        run.log(f"cohort assertions passed: 521-cell manifest equality, bench.Y == frame logD__*, bundle "
                f"{rec['bundle_sha256'][:8]}, parquet {rec['cohort_parquet_sha256'][:8]}, bench.pkl {rec['bench_pkl_sha256'][:8]}, "
                f"fingerprint {rec['fingerprint_obtained']}")
        f = bench.frame
        obs = D.observed(bench)
        meta = D.cell_meta(bench)
        nrep = f[list(D.NREP_COLS)].to_numpy(dtype=float)

        chain = cohort_chain(bench)
        aug = chain.pop("_aug")
        qp, qn, rp = chain["quarantined_pub"], chain["quarantined_nopub"], chain["raw_pub"]
        qa = chain["quarantine_audit"]
        T.check("C11-bundle", "PLAN §1.1", "audit", "bundle raw rows", 5992, qa["rows_in"])
        T.check("C11-quar", "PLAN §1.1", "audit", "rows after quarantine", 5860, qa["rows_out"])
        T.check("C11-todga", "PLAN §1.1", "audit", "TODGA-SMILES/name-mismatch rows", 129, qa["todga_name_mismatch_rows"])
        T.check("C11-sentinel", "PLAN §1.1", "audit", "sentinel rows (log D <= -6)", 3, qa["sentinel_rows"])
        T.check("C11-allcells", "PLAN §1.1", "[CNT]", "all gen13 cells (any metal count)", 2395, qp["cells_all"])
        T.check("C11-allobs", "PLAN §1.1", "[CNT]", "all gen13 (cell, metal) obs", 5233, qp["entries_all"])
        T.check("C11-multi", "PLAN §1.1/§2", "[CNT]", "multi-metal cells (>= 2 lanthanides)", 521, qp["cells_multi"])
        T.check("C11-multi-bench", "PLAN §1.1", "assert", "chain multi-metal cell_ids == bench cell_ids", True,
                chain["multi_metal_cells_equal_bench"])
        T.check("C11-rows", "PLAN §1.1", "audit", "raw rows in the 521 cells", 3871, qp["rows_multi"])
        T.check("C11-rows-frame", "PLAN §1.1", "audit", "sum frame.n_rows == sum nrep__", [3871, 3871],
                [int(f["n_rows"].sum()), int(np.nansum(nrep))])
        T.check("C11-entries", "PLAN §1.1/§0", "[CNT]", "replicate-mean (cell, metal) entries (chain)", 3359, qp["entries_multi"])
        T.check("C11-entries-bench", "PLAN §1.1", "[CNT]", "non-null logD__ of the cohort (NaN mask)", 3359, int(obs.sum()))
        T.check("C11-ext", "PLAN §1.1/§2", "[CNT]", "extractants of the 521 cells", 90, f["extractant"].nunique())
        T.check("C11-chem", "PLAN §1.1/§2", "[CNT]", "chemotypes of the 521 cells", 45, f["chemotype"].nunique())
        T.check("C11-pub", "PLAN §1.1/§2", "[CNT]", "publications of the 521 cells", 58, f["publication_id"].nunique())
        rep_entries = int(((nrep >= 2) & obs).sum())
        T.check("C11-rep", "PLAN §1.1/§2", "audit", "replicated entries (nrep >= 2)", 216, rep_entries)
        T.check("C11-rep-chain", "PLAN §1.1", "audit", "replicated entries (bundle chain)", 216, qp["replicated_entries"])
        T.check("C11-repcells", "PLAN §1.1/§2", "audit", "cells with a replicated entry", 41, int(((nrep >= 2) & obs).any(1).sum()))
        T.check("C11-reprows", "PLAN §1.1", "[CNT]", "raw rows of the replicated entries", 728, qp["replicated_rows"])
        T.check("C11-reprows-nrep", "PLAN §1.1", "[CNT]", "sum nrep over replicated entries", 728,
                int(nrep[(nrep >= 2) & obs].sum()))
        T.check("C11-identity", "PLAN §1.1 corr.(1)", "[CNT]", "3,871 - 728 + 216", 3359, 3871 - qp["replicated_rows"] + rep_entries)
        vc = f["chemotype"].value_counts()
        T.check("C11-largest", "PLAN §1.1", "audit", "largest chemotype (cells, extractants, name)", [375, 23, "sc009"],
                [int(vc.iloc[0]), int(f[f.chemotype == vc.index[0]].extractant.nunique()), str(vc.index[0])])
        T.check("C11-excl", "PLAN §1.1/§2 COH-AUG", "[CNT]", "single-metal gen13-key cells / raw rows", [1874, 1989],
                [qp["cells_single"], qp["rows_single"]])
        T.check("C11-quar0", "PLAN §1.1 corr.(2)", "[CNT]", "quarantine rows in >= 2-metal cells / quarantined rows",
                [0, 132], [chain["quarantined_rows_in_multimetal_cells_raw"], chain["n_quarantined_rows"]])
        T.check("C11-quar-nochange", "PLAN §1.1 corr.(2)", "[CNT]", "multi-metal cells/rows/entries without quarantine",
                [521, 3871, 3359], [rp["cells_multi"], rp["rows_multi"], rp["entries_multi"]])
        T.check("C11-nopub", "PLAN §1.1 corr.(2)", "[CNT]", "key without publication_id: raw rows / entries", [3878, 3364],
                [qn["rows_multi"], qn["entries_multi"]])
        run.log("cohort chain recomputed")

        # ---------------- populations (§2) ----------------
        pops = D.populations(bench)
        cards = {k: D.card(v) for k, v in pops.items()}
        c3 = meta.index[meta.is_c3].to_numpy()
        dC3, sC3 = cards["COH-D-C3"], cards["COH-SF-C3"]
        T.check("P-C3-cells", "PLAN §0/§2; PREREG §2; README", "[CNT]", "C3 cells", 229, dC3["n_cells"])
        T.check("P-D-C3", "PLAN §0/§2; PREREG §2; README", "[CNT]", "COH-D-C3 entries per seed", 2425, dC3["n_rows"])
        T.check("P-SF-C3", "PLAN §0/§2; PREREG §2", "[CNT]", "COH-SF-C3 pairs per seed", 1802, sC3["n_rows"])
        T.check("P-C3-ext", "PLAN §0/§2; PREREG §2", "[CNT]", "C3 extractants", 79, dC3["n_extractants"])
        T.check("P-C3-chem", "PLAN §0/§2/§6; PREREG §2/§6", "[CNT]", "C3 chemotypes (both tables)",
                [38, 38], [dC3["n_chemotypes"], sC3["n_chemotypes"]])
        T.check("P-C3-pub", "PLAN §0/§2/§6; PREREG §2/§6", "[CNT]", "C3 publications (both tables)",
                [39, 39], [dC3["n_publications"], sC3["n_publications"]])
        same_lists = (sorted(pops["COH-D-C3"].rows.chemotype.unique()) == sorted(pops["COH-SF-C3"].rows.chemotype.unique())
                      and sorted(pops["COH-D-C3"].rows.publication_id.unique()) == sorted(pops["COH-SF-C3"].rows.publication_id.unique()))
        T.check("P-C3-lists", "PREREG §2 (LT-17)", "[CNT]", "D-C3 and SF-C3 share chemotype & publication lists", True, same_lists)
        T.check("P-D-C3-kish", "PLAN §2; PREREG §2", "[CNT]", "COH-D-C3 Kish (chemotypes, rows)", 3.430, dC3["kish_chemotypes"], ndigits=3)
        T.check("P-D-C3-share", "PLAN §2; PREREG §7", "[CNT]", "COH-D-C3 largest chemotype share (sc009)", 0.524,
                dC3["largest_chemotype_share"], ndigits=3)
        T.check("P-D-C3-rep", "PLAN §2; PREREG §10", "[CNT]", "replicated entries in COH-D-C3", 161,
                int(meta.loc[c3, "n_rep_entries"].sum()))
        nmc = meta.loc[c3, "n_obs"]
        T.check("P-C3-metals", "PLAN §2; PREREG §2; README", "[CNT]", "measured metals per C3 cell (min, max)",
                [5, 14], [int(nmc.min()), int(nmc.max())])
        T.check("P-SF-C3-kish", "PLAN §2; PREREG §2 ([LEV]:6)", "[CNT]", "COH-SF-C3 Kish (chemotypes, pairs)",
                3.8511214634053776, sC3["kish_chemotypes"], tol=1e-12)
        T.check("P-SF-C3-share", "PLAN §2; PREREG §2/§7", "[CNT]", "COH-SF-C3 largest share (sc009)", 0.489,
                sC3["largest_chemotype_share"], ndigits=3)
        T.check("P-SF-C3-sc009", "PLAN §2", "[CNT]", "largest chemotype of C3 is sc009", ["sc009", "sc009"],
                [dC3["largest_chemotype"], sC3["largest_chemotype"]])
        T.check("P-D-C3M", "PLAN §2; PREREG §2; README", "[CNT]", "COH-D-C3M entries", 2208, cards["COH-D-C3M"]["n_rows"])
        T.check("P-D-C3-noSF", "PREREG §2 (O27)", "[CNT]", "C3 entries without a scored log SF counterpart", 217,
                dC3["n_rows"] - cards["COH-D-C3M"]["n_rows"])
        T.check("P-D-ALL", "PLAN §2", "[CNT]", "COH-D-ALL entries", 3359, cards["COH-D-ALL"]["n_rows"])
        T.check("P-D-ALL-kish", "PLAN §2", "[CNT]", "COH-D-ALL Kish rows", 2.734, cards["COH-D-ALL"]["kish_chemotypes"], ndigits=3)
        a = cards["COH-SF-ANY"]
        any_cells = meta.index[meta.is_any].to_numpy()
        T.check("P-SF-ANY", "PLAN §2; PREREG §2", "[CNT]", "COH-SF-ANY pairs/cells/ext/chem/pub",
                [2114, 378, 82, 40, 45], [a["n_rows"], a["n_cells"], a["n_extractants"], a["n_chemotypes"], a["n_publications"]])
        T.check("P-SF-ANY-entries", "PLAN §2", "[CNT]", "COH-SF-ANY cells' entries", 2980, int(obs[any_cells].sum()))
        T.check("P-SF-ANY-kish", "PLAN §2", "[CNT]", "COH-SF-ANY Kish pairs", 3.296, a["kish_chemotypes"], ndigits=3)
        c = cards["COH-SF-C1"]
        c1_cells = meta.index[meta.is_c1].to_numpy()
        T.check("P-SF-C1", "PLAN §2; PREREG §2", "[CNT]", "COH-SF-C1 pairs/cells/ext/chem/pub",
                [2067, 331, 82, 40, 44], [c["n_rows"], c["n_cells"], c["n_extractants"], c["n_chemotypes"], c["n_publications"]])
        T.check("P-SF-C1-entries", "PLAN §2", "[CNT]", "COH-SF-C1 cells' entries", 2886, int(obs[c1_cells].sum()))
        n = cards["COH-SF-NDSM"]
        nd_cells = meta.index[meta.is_ndsm].to_numpy()
        T.check("P-SF-NDSM", "PLAN §2; PREREG §2", "[CNT]", "COH-SF-NDSM pairs/cells/ext/chem/pub",
                [1984, 236, 79, 39, 39], [n["n_rows"], n["n_cells"], n["n_extractants"], n["n_chemotypes"], n["n_publications"]])
        T.check("P-SF-NDSM-entries", "PLAN §2", "[CNT]", "COH-SF-NDSM cells' entries", 2466, int(obs[nd_cells].sum()))
        ndsm_overlap = {"ndsm_cells_in_c3": int(np.isin(nd_cells, c3).sum()), "c3_cells_in_ndsm": int(np.isin(c3, nd_cells).sum()),
                        "ndsm_only_cells": int((~np.isin(nd_cells, c3)).sum()),
                        "ndsm_pairs_strict": int(pops["COH-SF-NDSM"].rows.strict.sum()),
                        "ndsm_pairs_nd_sm": int((~pops["COH-SF-NDSM"].rows.strict).sum())}
        s9d, s9s = cards["COH-SC009-D"], cards["COH-SC009-SF"]
        T.check("P-SC009", "PLAN §2; PREREG §2/§6", "[CNT]", "COH-SC009 cells/ext/pub/entries/pairs",
                [129, 23, 15, 1270, 882], [s9d["n_cells"], s9d["n_extractants"], s9d["n_publications"], s9d["n_rows"], s9s["n_rows"]])
        n9d, n9s = cards["COH-NOSC009-D"], cards["COH-NOSC009-SF"]
        T.check("P-NOSC009", "PLAN §2; PREREG §2", "[CNT]", "COH-NOSC009 cells/ext/chem/pub/entries/pairs",
                [100, 56, 37, 29, 1155, 920],
                [n9d["n_cells"], n9d["n_extractants"], n9d["n_chemotypes"], n9d["n_publications"], n9d["n_rows"], n9s["n_rows"]])
        T.check("P-NOSC009-kish-rows", "PLAN §2", "[CNT]", "COH-NOSC009 Kish rows", 13.14, n9d["kish_chemotypes"], ndigits=2)
        T.check("P-NOSC009-kish-pairs", "PLAN §2", "[CNT]", "COH-NOSC009 Kish pairs", 12.97, n9s["kish_chemotypes"], ndigits=2)
        # DMDPhPDA rule
        dm = D.dmd_cells(bench)
        dmd_entries = int(obs[dm].sum())
        dmd_pairs = int(meta.loc[dm, "n_qadj"].sum())
        T.check("DMD-rule", "PLAN §1.1(6); PREREG §2", "[DMD]", "DMDPhPDA cells / entries / strict-adjacent pairs",
                [5, 70, 60], [len(dm), dmd_entries, dmd_pairs])
        T.check("DMD-in-C3", "PLAN §1.1(6); PREREG §2; README", "[DMD]", "DMDPhPDA cells in C3", 5, int(meta.loc[dm, "is_c3"].sum()))
        nd3, nds = cards["COH-D-C3@NODMD"], cards["COH-SF-C3@NODMD"]
        T.check("NODMD-C3", "PLAN §2; PREREG §2", "[CNT]+[DMD]", "COH-NODMD C3 cells/entries/pairs",
                [224, 2355, 1742], [nd3["n_cells"], nd3["n_rows"], nds["n_rows"]])
        ndall = cards["COH-D-ALL@NODMD"]
        T.check("NODMD-cohort", "PLAN §2", "[CNT]+[DMD]", "COH-NODMD cohort cells/entries", [516, 3289],
                [ndall["n_cells"], ndall["n_rows"]])
        T.check("REP-cohort", "PLAN §2 COH-REP", "audit", "COH-REP cohort entries / cells (nrep >= 2)", [216, 41],
                [rep_entries, int(((nrep >= 2) & obs).any(1).sum())])
        T.check("AUG", "PLAN §2 COH-AUG", "[CNT]", "COH-AUG manifest: single-metal cells / raw rows", [1874, 1989],
                [int(len(aug)), int(aug["n_rows"].sum())])
        run.log("populations recomputed")

        # ---------------- folds (§1.2, §4) ----------------
        F = FO.build(bench)
        once = FO.tested_once_per_seed(F, len(f))
        T.check("F-once", "PLAN §1.2; LT-01", "[CNT]", "each cell tested once per seed (6 designs)",
                [True] * 6, [once[d] for d in FO.DESIGNS])
        dig = FO.gen16_digest(F)
        anc = FO.anchors_reference()
        ga, gb = FO.gate_a(dig), FO.gate_b(dig)
        ga["anchors_json_agrees"] = (anc["sha256"] == FO.EXPECTED_TEST_PLAN_SHA256
                                     and anc["sha256_train_test"] == FO.EXPECTED_TRAIN_TEST_SHA256)
        gb["anchors_json_agrees"] = all(anc["per_design"][k] == v for k, v in FO.EXPECTED_PER_DESIGN_SHA256.items())
        gates["a"], gates["b"] = ga, gb
        T.check("G-a-test", "PLAN §1.2/§4 (a)", "gate", "test-index plan digest 7046c640", FO.EXPECTED_TEST_PLAN_SHA256, dig["sha256"])
        T.check("G-a-tt", "PLAN §1.2/§4 (a)", "gate", "train+test digest 9a24b6e9", FO.EXPECTED_TRAIN_TEST_SHA256, dig["sha256_train_test"])
        for d in FO.GEN16_DESIGN_ORDER:
            T.check(f"G-b-{d}", "PLAN §1.2/§4 (b)", "gate", f"per-design digest {d}", FO.EXPECTED_PER_DESIGN_SHA256[d], dig["per_design"][d])
        T.check("G-ab-anchors", "anchors.json:31-40", "gate", "constants equal anchors.json", [True, True],
                [ga["anchors_json_agrees"], gb["anchors_json_agrees"]])
        pdig = FO.plan_digest(bench, F)                      # gate (e), in process
        if not args.no_subprocess:
            gs = FO.gate_b_subprocess(("1", "2"), plan_sha256=pdig["sha256"])
            gates["b_subprocess"] = gs
            T.check("G-b-subproc", "PLAN §4 (b); LT-05", "gate",
                    "(a)+(b) and the (e) digest in subprocesses, PYTHONHASHSEED 1 and 2", True, gs["pass"])
        gc = FO.gate_c(F["BP"])
        gates["c"] = gc
        T.check("G-c", "PLAN §1.2/§4 (c); PREREG §2", "[CNT]", "BP (seed, fold, n_train, n_test) == folds_BP.json", 25, gc["n_match"])
        gd = FO.gate_d(F)
        gates["d"] = gd
        T.check("G-d", "PLAN §1.2/§4 (d); PREREG §2", "[CNT]", "test sets equal BP's: B, BR, BQ, BPG, A",
                [25, 25, 25, 25, 0], [gd["n_equal_to_BP"][d] for d in ("B", "BR", "BQ", "BPG", "A")])

        # per-fold sizes, COH-TRAIN, giant folds, COH-REP per fold
        c3set = set(c3.tolist())
        is_c3 = meta.is_c3.to_numpy()
        nq = meta.n_qadj.to_numpy()
        per_fold = []
        for d in FO.DESIGNS:
            for fo in F[d]:
                tr, te = np.asarray(fo.train_index), np.asarray(fo.test_index)
                per_fold.append({"design": d, "seed": int(fo.seed), "fold": int(fo.fold), "model_seed": int(fo.model_seed),
                                 "n_train_cells": int(len(tr)), "n_train_rows": int(obs[tr].sum()),
                                 "n_test_cells": int(len(te)), "n_test_c3_cells": int(is_c3[te].sum()),
                                 "n_test_c3_entries": int(obs[te][is_c3[te]].sum()),
                                 "n_test_c3_pairs": int(nq[te][is_c3[te]].sum()),
                                 "test_has_sc009": bool((f.chemotype.to_numpy()[te] == D.SC009).any()),
                                 "n_train_rep_entries": int(((nrep[tr] >= 2) & obs[tr]).sum()),
                                 "n_stored_inner_val": int(len(fo.inner_validation_index))})
        PF = pd.DataFrame(per_fold)
        exp_train = {"BP": (76, 492), "B": (103, 507), "BR": (76, 492), "BQ": (76, 492), "BPG": (73, 492), "A": (241, 494)}
        for d, (lo, hi) in exp_train.items():
            s = PF[PF.design == d]
            T.check(f"TRAIN-{d}", "PLAN §2 COH-TRAIN; PREREG §2", "[CNT]", f"design {d} training cells per fold (min, max)",
                    [lo, hi], [int(s.n_train_cells.min()), int(s.n_train_cells.max())])
        s = PF[PF.design == "BP"]
        T.check("TRAIN-BP-rows", "PLAN §2 COH-TRAIN; PREREG §2", "[CNT]", "design BP training log D rows per fold (min, max)",
                [725, 3084], [int(s.n_train_rows.min()), int(s.n_train_rows.max())])
        giant = s[s.n_train_cells < 100].sort_values("seed")
        exp_giant = [[104729, 0, 76, 764, 418, 148, 1070], [130363, 3, 98, 931, 394, 142, 989],
                     [155921, 3, 82, 725, 415, 158, 1150], [196613, 1, 85, 803, 400, 151, 1102],
                     [262147, 2, 85, 919, 400, 136, 935]]
        got_giant = giant[["seed", "fold", "n_train_cells", "n_train_rows", "n_test_cells", "n_test_c3_cells",
                           "n_test_c3_pairs"]].astype(int).values.tolist()
        T.check("GIANT-tuples", "PLAN §4", "[CNT]", "giant BP folds (seed, fold, tr cells, tr rows, te cells, te C3 cells, te C3 pairs)",
                exp_giant, got_giant)
        T.check("GIANT-one-per-seed", "PLAN §4", "[CNT]", "one giant fold per seed, each holds out sc009",
                [5, 5, 5], [int(len(giant)), int(giant.seed.nunique()), int(giant.test_has_sc009.sum())])
        sh = giant.n_test_c3_pairs / sC3["n_rows"]
        T.check("GIANT-share", "PLAN §4/§12; PREREG §9", "[CNT]", "giant folds' share of the seed's C3 pairs, % (min, max)",
                [52, 64], [int(round(100 * sh.min())), int(round(100 * sh.max()))])
        T.check("GIANT-train", "PREREG §9", "[CNT]", "giant folds' training cells (min, max)", [76, 98],
                [int(giant.n_train_cells.min()), int(giant.n_train_cells.max())])
        iv = sorted(s.n_stored_inner_val.tolist())
        small = [v for v in iv if v <= 11]
        T.check("GEN20-INNER-VAL", "PREREG §12 row 6 / O12; PLAN §4", "[CNT]",
                "gen20 single inner split: BP folds with 10-11 validation cells (n, min, max)",
                [4, 10, 11], [len(small), int(min(small)) if small else None, int(max(small)) if small else None])

        # inner chemotype cross-fit
        inner_rows, first_eq, nodmd_units, nodmd_skip_changes = [], {}, 0, []
        for d in FO.DESIGNS:
            first_eq[d] = 0
            for fo in F[d]:
                first_eq[d] += int(FO.first_inner_matches(bench, fo))
                prim = FO.inner_folds(bench, d, fo)
                nod = FO.inner_folds(bench, d, fo, drop=dm)
                fnod = FO.drop_cells(fo, dm)
                ok = (set(np.asarray(fnod.train_index).tolist()) == set(np.asarray(fo.train_index).tolist()) - set(dm.tolist())
                      and np.array_equal(fnod.test_index, fo.test_index))
                nodmd_units += int(ok)
                for p_, q_ in zip(prim, nod):
                    inner_rows.append({"design": d, "seed": int(fo.seed), "fold": int(fo.fold), "inner": p_.j,
                                       "n_inner_train": int(len(p_.train_pos)), "n_inner_train_unmasked": int(len(p_.train_pos_unmasked)),
                                       "n_inner_val": int(len(p_.val_pos)), "skipped": bool(p_.skipped),
                                       "n_inner_train_nodmd": int(len(q_.train_pos)), "skipped_nodmd": bool(q_.skipped)})
                    if p_.skipped != q_.skipped:
                        nodmd_skip_changes.append([d, int(fo.seed), int(fo.fold), p_.j])
        IN = pd.DataFrame(inner_rows)
        for d in FO.DESIGNS:
            T.check(f"INNER-first-{d}", "PLAN §4; PREREG §2", "[INNER]", f"first chemotype inner fold == stored Fold.inner_* ({d})",
                    0 if d == "A" else 25, first_eq[d])
        ib = IN[IN.design == "BP"]
        T.check("INNER-BP-n", "PLAN §2/§4; PREREG §2", "[INNER]", "BP inner folds (count, skipped)", [100, 1],
                [int(len(ib)), int(ib.skipped.sum())])
        T.check("INNER-BP-min", "PLAN §2/§4; PREREG §2", "[INNER]", "BP minimum masked inner-train cells", 18, int(ib.n_inner_train.min()))
        sk = ib[ib.skipped][["seed", "fold", "inner", "n_inner_train"]].astype(int).values.tolist()
        T.check("INNER-BP-skipped", "PLAN §4; PREREG §2", "[INNER]", "BP skipped inner fold (seed, fold, inner, cells)",
                [[155921, 2, 3, 18]], sk)
        ibb = IN[IN.design == "B"]
        T.check("INNER-B", "PLAN §2/§4; PREREG §2", "[INNER]", "design B: skipped inner folds, minimum inner-train",
                [0, 59], [int(ibb.skipped.sum()), int(ibb.n_inner_train.min())])
        T.check("LT25-nodmd-train", "PLAN §2 COH-NODMD; LT-25", "[DMD]", "NODMD train = primary train minus the 5 cells (units)",
                150, nodmd_units)
        run.log("folds, gates (a)-(d) and inner cross-fit recomputed")

        # gate (e) and fold records
        FR = FO.fold_record(bench, F)
        fr_path = work / "fold_records_g201.parquet"
        FR.to_parquet(fr_path, index=False)
        previous = FO.recorded_plan(out / "fold_plan_g201.json")     # the reference: a re-run (S6) must reproduce it
        ge = FO.gate_e(pdig, recorded=previous,
                       subprocess_runs=gates["b_subprocess"]["runs"] if "b_subprocess" in gates else None)
        ge["previous_run_present"] = previous is not None
        gates["e"] = ge
        T.check("G-e", "PLAN §4 (e)", "gate", "outer+inner plan digest: subprocess-identical (and = previous S3 run)",
                True, ge["pass"])
        fold_plan = {"schema": "gen20.1 fold plan (S3); deterministic, no timestamps",
                     "designs": list(FO.DESIGNS), "split_seeds": list(FO.SEEDS), "n_inner": FO.N_INNER,
                     "min_inner_train": FO.MIN_INNER_TRAIN,
                     "gate_a": ga, "gate_b": gb, "gate_b_subprocess": gates.get("b_subprocess"),
                     "gate_c": gc, "gate_d": gd,
                     "gate_e": {"sha256": pdig["sha256"], "per_design": pdig["per_design"], "definition": pdig["definition"],
                                "subprocess_identical": ge.get("subprocess_identical")},
                     "gen16_digest": dig,
                     "first_inner_fold_equal": first_eq,
                     "fold_records": {"file": fr_path.name, "sha256_file": sha256_file(fr_path),
                                      "sha256_content_csv": hashlib.sha256(FR.to_csv(index=False, lineterminator="\n").encode()).hexdigest(),
                                      "rows": int(len(FR))},
                     "per_fold": _py(PF.to_dict(orient="records")),
                     "inner": _py(IN.to_dict(orient="records")),
                     "nodmd": {"cells": [int(x) for x in dm], "cell_ids": [str(f.cell_id.iat[i]) for i in dm],
                               "units_train_equal_primary_minus_cells": nodmd_units,
                               "inner_skip_status_changes": nodmd_skip_changes}}
        with open(work / "fold_plan_g201.json", "w", encoding="utf-8", newline="\n") as fh:   # temporary dir only
            json.dump(_py(fold_plan), fh, indent=1)
        run.log(f"gate (e) digest {pdig['sha256'][:12]}... (pass {ge['pass']}) written to the temporary fold_plan_g201.json")

        # COH-REP per fold and COH-REPPAIR feasibility (identity/run columns only)
        rep_fold = PF.groupby("design").n_train_rep_entries.agg(["min", "max"]).astype(int).to_dict(orient="index")
        readings = D.reading_index(bench)
        rp_tables = []
        rp_out = {"n_readings": int(len(readings)),
                  "experiment_id_unique_per_reading": bool(readings.experiment_id.nunique() == len(readings)),
                  "experiment_id_equals_safe_exp_id": float((readings.experiment_id.astype(str) == readings.safe_exp_id.astype(str)).mean()),
                  "status": {c: readings[c].astype(str).value_counts().to_dict()
                             for c in ("experiment_id_status", "experiment_series_id_status", "replicate_id_status")},
                  "bundle_metal_equals_provenance_metal_symbol": float((readings.metal.astype(str) == readings.metal_symbol.astype(str)).mean()),
                  "keys": {}}
        bp_folds = F["BP"]
        for key in D.PROVENANCE_RUN_COLUMNS:
            cand = D.reppair_candidates(bench, readings, key)
            rp_tables.append(cand.assign(key=key))
            paired = cand[cand.run_paired]
            perfold = []
            for fo in bp_folds:
                trs = set(np.asarray(fo.train_index).tolist())
                pf = paired[paired.ci.isin(trs)]
                perfold.append({"seed": int(fo.seed), "fold": int(fo.fold), "pairs": int(len(pf)), "chemotypes": int(pf.chemotype.nunique())})
            pfd = pd.DataFrame(perfold)
            rp_out["keys"][key] = {
                "both_endpoints_replicated_strict_pairs": int(len(cand)),
                "both_endpoints_replicated_cells": int(cand.ci.nunique()) if len(cand) else 0,
                "run_paired_pairs": int(len(paired)), "run_paired_cells": int(paired.ci.nunique()) if len(paired) else 0,
                "run_paired_chemotypes": int(paired.chemotype.nunique()) if len(paired) else 0,
                "run_paired_pairs_in_c3": int(paired.ci.isin(c3set).sum()) if len(paired) else 0,
                "bp_training_folds_min_pairs": int(pfd.pairs.min()), "bp_training_folds_max_pairs": int(pfd.pairs.max()),
                "bp_training_folds_min_chemotypes": int(pfd.chemotypes.min()),
                "bp_folds_estimable_(>=20_pairs_>=5_chemotypes)": int(((pfd.pairs >= 20) & (pfd.chemotypes >= 5)).sum()),
                "run_paired_chemotype_counts": paired.chemotype.value_counts().sort_index().to_dict() if len(paired) else {},
                "bp_training_folds": perfold}
        best = max(rp_out["keys"].values(), key=lambda v: v["run_paired_pairs"])
        rp_out["feasible"] = bool(best["bp_folds_estimable_(>=20_pairs_>=5_chemotypes)"] == 25)
        rp_out["verdict"] = ("log SF noise estimable from run-paired replicated adjacent pairs in every BP training fold"
                             if rp_out["feasible"] else
                             "log SF noise NOT ESTIMABLE from run-paired replicated adjacent pairs (fewer than 20 pairs "
                             "from 5 chemotypes per BP training fold under every available run key); S5 reports "
                             "'not estimable' plus the labelled sqrt(2)*sigma bound (O16 default)")
        run.log(f"COH-REPPAIR: {rp_out['verdict']}")

        # row manifests: scored (test-side) populations, training-side populations, cohort-level sets
        rm_dir = work / "row_manifests"
        rm = write_row_manifests(pops, F, rm_dir)
        rm_train, train_summ = write_training_manifests(bench, F, dm, rm_dir)
        reppair = pd.concat(rp_tables, ignore_index=True)
        reppair["a"] = [D.LANTHANIDES[i] for i in reppair["ia"]]
        reppair["b"] = [D.LANTHANIDES[i] for i in reppair["ib"]]
        rm_cohort = write_cohort_manifests(bench, aug, reppair, rm_dir)
        rm = {**rm, **rm_train, **rm_cohort}
        dig_path = write_digests(rm, rm_dir)
        chk = train_summ.merge(PF.rename(columns={"seed": "split_seed"}), on=["design", "split_seed", "fold"])
        T.check("M-TRAIN-rows", "PLAN §2 COH-TRAIN", "assert", "COH-TRAIN manifest rows per unit == training entries (units)",
                150, int((chk.n_train_rows_x == chk.n_train_rows_y).sum()))
        T.check("M-REP-rows", "PLAN §2 COH-REP", "assert", "COH-REP manifest rows per unit == replicated training entries",
                150, int((chk.n_train_rep_entries_x == chk.n_train_rep_entries_y).sum()))
        dmd_train_rows = []
        for d in FO.DESIGNS:
            for fo in F[d]:
                dmd_train_rows.append(int(obs[np.intersect1d(np.asarray(fo.train_index), dm)].sum()))
        chk = chk.sort_values(["design", "split_seed", "fold"]).reset_index(drop=True)
        exp_nodmd = (pd.DataFrame([(d, int(fo.seed), int(fo.fold), r) for (d, fo), r in
                                   zip([(d, fo) for d in FO.DESIGNS for fo in F[d]], dmd_train_rows)],
                                  columns=["design", "split_seed", "fold", "dmd_rows"])
                     .merge(chk, on=["design", "split_seed", "fold"]))
        T.check("M-NODMD-rows", "PLAN §2 COH-NODMD", "assert",
                "every unit's NODMD training rows = training rows minus the DMDPhPDA entries (units)",
                150, int((exp_nodmd.n_train_rows_x - exp_nodmd.dmd_rows == exp_nodmd.n_train_rows_nodmd).sum()))
        run.log(f"{len(rm)} row manifests written ({dig_path.name})")

        # values recomputed without a documented expectation
        T.info("I-NDSM-overlap", "PREREG §2 (overlap with COH-SF-C3 counted at S3)", "COH-SF-NDSM cells in C3 / NDSM-only "
               "cells / NDSM strict pairs / NDSM Nd-Sm pairs",
               [ndsm_overlap["ndsm_cells_in_c3"], ndsm_overlap["ndsm_only_cells"], ndsm_overlap["ndsm_pairs_strict"],
                ndsm_overlap["ndsm_pairs_nd_sm"]])
        for d in FO.DESIGNS:
            T.info(f"I-REP-{d}", "PLAN §2 COH-REP (per fold at S3)", f"design {d}: replicated training entries per fold (min, max)",
                   [rep_fold[d]["min"], rep_fold[d]["max"]])
        bp_tr = chk[chk.design == "BP"].sort_values(["split_seed", "fold"])
        T.info("I-REP-BP-folds", "PLAN §2 COH-REP", "design BP: replicated training entries, 25 folds in (seed, fold) order",
               bp_tr.n_train_rep_entries_x.astype(int).tolist())
        ks = rp_out["keys"]["experiment_series_id"]
        T.info("I-REPPAIR", "PLAN §2/§7 COH-REPPAIR (O16)", "strict pairs with both endpoints replicated (pairs, cells); "
               "run-paired by experiment_series_id (pairs, cells, chemotypes, in C3)",
               [ks["both_endpoints_replicated_strict_pairs"], ks["both_endpoints_replicated_cells"], ks["run_paired_pairs"],
                ks["run_paired_cells"], ks["run_paired_chemotypes"], ks["run_paired_pairs_in_c3"]])
        T.info("I-REPPAIR-BP", "PLAN §7 (>= 20 pairs from >= 5 chemotypes per fold)", "BP training folds: run-paired pairs "
               "(min, max), chemotypes (min), folds estimable", [ks["bp_training_folds_min_pairs"], ks["bp_training_folds_max_pairs"],
                                                                 ks["bp_training_folds_min_chemotypes"],
                                                                 ks["bp_folds_estimable_(>=20_pairs_>=5_chemotypes)"]])
        T.info("I-REPPAIR-feasible", "O16", "COH-REPPAIR feasible", rp_out["feasible"])
        T.info("I-bench-pkl", "PLAN §1.1 (recorded)", "bench.pkl sha256 / bytes", [rec["bench_pkl_sha256"], rec["bench_pkl_bytes"]])
        T.info("I-fingerprint", "PLAN §1.1 (drift from 4c3c6628ea0be949 expected)", "Gen13Cohort fingerprint obtained",
               rec["fingerprint_obtained"])
        T.info("I-gate-e", "PLAN §4 (e)", "outer+inner plan digest (six designs)", pdig["sha256"])
        T.info("I-NODMD-skip", "PLAN §2 COH-NODMD", "inner folds whose skip status changes when the 5 DMD cells are removed",
               nodmd_skip_changes)

        # coverage audit of the tagged document blocks (a number not among the expected values is a STOP); it
        # depends on the documents' wording, so it goes to the run manifest, not into the compared counts file
        cov = coverage_audit(T)
        unm = [c for c in cov if c["unmapped"]]
        run.record("coverage_audit", {"not_a_count": NOT_A_COUNT, "blocks": cov})

        # frozen trees: nothing outside generations/gen20_1 (nor a gen20.1 document) changed since before the load
        changed = guard.diff(before, guard.snapshot())
        frozen_ok = not any(changed.values())
        run.record("frozen_trees_check_before_counts", {"files": len(before), "unchanged": frozen_ok,
                                                         "changed": {k: v[:20] for k, v in changed.items() if v}})
        gate_fail = [k for k, v in gates.items() if not v.get("pass")]
        stop_reasons = ([f"{len(T.mismatches)} count mismatches"] if T.mismatches else []) \
            + ([f"failed gates {gate_fail}"] if gate_fail else []) \
            + ([f"{len(unm)} tagged document blocks hold numbers not among the expected values"] if unm else []) \
            + ([f"files outside generations/gen20_1 changed: { {k: v[:5] for k, v in changed.items() if v} }"]
               if not frozen_ok else [])
        s3_status = "STOP" if stop_reasons else ("INCOMPLETE" if args.no_subprocess else "PASS")

        from gen13sep import cohort as C13, paths as P13
        label_reads = [{
            "what": "gen13sep.cohort.apply_quarantine sentinel mask log_D <= LOG_D_FLOOR on the raw bundle "
                    "(the frozen gen13 cohort-membership rule)",
            "column": "log_D", "threshold": float(C13.LOG_D_FLOOR),
            "source": Path(P13.BUNDLE_PARQUET).resolve().relative_to(bootstrap.ROOT.resolve()).as_posix(),
            "rows_read": int(qa["rows_in"]), "rows_flagged": int(qa["sentinel_rows"]),
            "rows_in_521_cells": int(chain["quarantined_rows_in_multimetal_cells_raw"]),
            "rows_in_521_cells_definition": "quarantined rows (either rule) inside the multi-metal cells of the "
                                            "unquarantined chain, which are the 521 bench cells; the sentinel rows "
                                            "are a subset",
            "used_for": "cohort-chain membership counts only (C11-*); the column is dropped right after the mask; "
                        "no statistic of it is computed"}]
        counts = {"schema": "gen20.1 S3 counts; label-free except the declared frozen quarantine mask (label_reads); "
                            "deterministic, no timestamps",
                  "s3_status": s3_status, "label_reads": label_reads, "frozen_trees_unchanged": frozen_ok,
                  "cohort_assertions": {k: v for k, v in rec.items()},
                  "cohort_chain": chain, "populations": cards, "ndsm_overlap_with_c3": ndsm_overlap,
                  "dmd": {"cells": [int(x) for x in dm], "entries": dmd_entries, "strict_adjacent_pairs": dmd_pairs},
                  "coh_rep": {"cohort_entries": rep_entries, "cohort_cells": int(((nrep >= 2) & obs).any(1).sum()),
                              "c3_entries": int(meta.loc[c3, "n_rep_entries"].sum()),
                              "per_fold_train_min_max_by_design": rep_fold},
                  "coh_reppair": rp_out, "row_manifests": rm, "gates": {k: v.get("pass") for k, v in gates.items()},
                  "comparison": T.rows, "recorded_info": T.infos,
                  "n_compared": len(T.rows), "n_mismatches": len(T.mismatches),
                  "n_coverage_unmapped_blocks": len(unm)}
        with open(work / "counts_g201.json", "w", encoding="utf-8", newline="\n") as fh:   # temporary dir only
            json.dump(_py(counts), fh, indent=1)
        run.record("s3_status", s3_status)
        run.record("label_reads", label_reads)
        run.record("frozen_trees_unchanged", frozen_ok)
        run.record("n_compared", len(T.rows))
        run.record("n_mismatches", len(T.mismatches))
        run.record("gates", {k: v.get("pass") for k, v in gates.items()})
        run.record("gate_e_sha256", pdig["sha256"])
        run.record("gate_e_check", ge)

        T.print()
        print("\nCOH-NDSM overlap with C3:", ndsm_overlap)
        print("COH-REP training replicated entries per fold (min, max):", rep_fold)
        print("COH-REPPAIR:", json.dumps(_py({k: {kk: vv for kk, vv in v.items() if kk != "bp_training_folds"}
                                              for k, v in rp_out["keys"].items()}), indent=1))
        print("COH-REPPAIR verdict:", rp_out["verdict"])
        print("gate (e) sha256:", pdig["sha256"])
        print(f"coverage audit: {len(cov)} tagged blocks, {sum(c['n_numbers'] for c in cov)} numbers, "
              f"{len(unm)} blocks with unmapped numbers")
        for c in unm:
            print(f"  UNMAPPED {c['doc']}:{c['lines'][0]}-{c['lines'][1]} {c['tags']} -> {c['unmapped']}")

        # re-run: byte comparison with the reference; first run: publish on PASS / INCOMPLETE
        if rerun:
            cmp = compare_reference(out, work)
            run.record("rerun_comparison", cmp)
            print(f"\nre-run byte comparison with {out}: {cmp['n_files']} reference files, "
                  f"{len(cmp['differing'])} differ")
            for n in cmp["differing"]:
                print("  DIFFERS", n)
            if not cmp["identical"]:
                stop_reasons.append(f"re-run not byte-identical to the reference: {cmp['differing']}")
        elif not stop_reasons:
            published = publish(work, out)
            run.record("outputs", run.outputs(published))
            run.log(f"{len(published)} reference files copied into {out}")
        changed_end = guard.diff(before, guard.snapshot())
        run.record("frozen_trees_unchanged_at_end", not any(changed_end.values()))
        if any(changed_end.values()) and frozen_ok:
            stop_reasons.append(f"files outside generations/gen20_1 changed at the end: "
                                f"{ {k: v[:5] for k, v in changed_end.items() if v} }")

        if stop_reasons:
            print(f"\nSTOP INCOMPARABLE_FOLDS / count mismatch ({args.label}): " + "; ".join(stop_reasons))
            for r in T.mismatches:
                print("  MISMATCH", r)
            print(f"the outputs of this run are kept for diagnosis in {work}"
                  + (f"; the reference files in {out} were not touched" if rerun else f"; nothing was copied into {out}"))
            run.record("status", "STOP")
            run.record("stop_reasons", stop_reasons)
            return 1
        shutil.rmtree(work, ignore_errors=True)
        if args.no_subprocess:
            print(f"\n{args.label} INCOMPLETE: --no-subprocess skipped the PYTHONHASHSEED subprocess check of gates "
                  "(a), (b) and (e); this run can never stand as the S3 reference")
            run.record("status", "INCOMPLETE")
            return 1
        run.record("status", "PASS")
        print(f"\n{args.label} PASS: every assertion, gate and documented [CNT]/[DMD]/[INNER] value reproduced"
              + (f"; every reference file in {out} reproduced byte for byte" if rerun else f"; reference files written to {out}"))
        return 0


if __name__ == "__main__":
    sys.exit(main())
