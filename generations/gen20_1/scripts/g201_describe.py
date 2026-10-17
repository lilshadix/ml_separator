"""gen20.1 stage S5 (DESCRIBE, Stage 0b): target statistics after the seal, before any model fit.

IMPLEMENTATION_PLAN.md §9 S5, §5 MET-13/MET-09, §7; PRE_REGISTRATION.md §10.  Sealed gate first
(``require_sealed(PRE_REGISTRATION.md)``), forbidden-writer guard installed, frozen trees snapshotted and compared.

Computes (descriptive only, never in a rule; no model fit, no prediction):
  * a MET-13 card for every scored population (every test-side S3 row manifest, primary and @NODMD, and the
    COH-REPCAP versions): n_rows, n_cells, n_extractants, n_chemotypes, n_publications, Kish (chemotype rows),
    Kish (extractant rows), largest chemotype share, y_mean, y_sd0, y_var0 (ddof 0), y_rms0 (log SF only), s_b and
    s_w = pi (log D only); rows are seed-invariant, so one card per population (asserted across the seeds);
  * per pair type (log SF populations): n, n_extractants, y_mean, y_sd0;
  * MET-19 log D replicate noise per outer (design, seed, fold) from the masked TRAINING cells only:
    sigma2_pool = sum (n_e - 1) s_e^2 / sum (n_e - 1) over entries with nrep >= 2, the same with s_e <= 0.5, the
    median s_e (s_e <= 0.5; gen20's training_noise_s2 is its square), entry counts; median and min-max over the 25
    BP folds; log SF noise "not estimable" (S3: COH-REPPAIR reaches 20 pairs from 5 chemotypes in 0 of 25 BP folds),
    only the labelled sqrt(2)*sigma upper bound is printed;
  * COH-REPCAP (label-derived, scoring only): the drop list of entries with nrep >= 2 and repsd > 0.5
    (``row_manifests/COH-REPCAP.parquet``) and every test-side S3 manifest without those entries / without pairs
    containing one (``row_manifests/COH-REPCAP@<population>.parquet``).

Outputs (deterministic, no timestamps): manifests/describe_g201.json and the COH-REPCAP row manifests, built in a
staging directory inside the output directory first (``<out>/.g201_describe_*``, gitignored: these are label-derived
target statistics and never leave generations/gen20_1; removed on success, kept for diagnosis on a STOP or an error).
First run: copied into place.  Re-run (S6): every existing file must be byte-identical
(sha256), otherwise STOP; nothing is ever overwritten.  The S3 row_manifests/DIGESTS.json is never touched; the
REPCAP digests live in describe_g201.json.  Run manifest: manifests/g201_describe__<label>.json (default S5; an
existing one is never overwritten).

Usage (repo root):  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B generations/gen20_1/scripts/g201_describe.py [--label S6]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
G201 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(G201))

from gen201 import bootstrap  # noqa: E402
from gen201 import guard  # noqa: E402
from gen201.seal import require_sealed  # noqa: E402

guard.install()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gen201 import data as D  # noqa: E402
from gen201 import folds as FO  # noqa: E402
from gen201.run import Run, sha256_file  # noqa: E402

REPCAP_SD = 0.5          # COH-REPCAP: drop entries with nrep >= 2 and repsd > 0.5 (and pairs containing one)
NOISE_CAP = 0.5          # gen20k.cov.REPLICATE_SD_CAP
REPSD_COLS = tuple(f"repsd__{m}" for m in D.LANTHANIDES)
OUT_JSON = "describe_g201.json"
STAGING_PREFIX = ".g201_describe_"        # staging directories inside the output directory (.gitignore)


def staging_dir(out_root: Path) -> Path:
    """A fresh staging directory inside ``out_root`` (itself inside generations/gen20_1 or a test's temporary
    directory), never in the system temporary directory."""
    base = bootstrap.assert_under_g201(out_root)
    base.mkdir(parents=True, exist_ok=True)
    return bootstrap.assert_under_g201(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=str(base)))


# --------------------------------------------------------------------------------------
# statistics (pure functions; unit-tested on synthetic arrays only)
# --------------------------------------------------------------------------------------
def target_values(Y: np.ndarray, rows: pd.DataFrame, target: str) -> np.ndarray:
    ci = rows["ci"].to_numpy(dtype=int)
    if target == "logd":
        return np.asarray(Y, dtype=float)[ci, rows["im"].to_numpy(dtype=int)]
    return np.asarray(Y, dtype=float)[ci, rows["ia"].to_numpy(dtype=int)] - np.asarray(Y, dtype=float)[ci, rows["ib"].to_numpy(dtype=int)]


def kish_of(labels) -> float:
    c = pd.Series(np.asarray(labels).astype(str)).value_counts().to_numpy(dtype=float)
    return float(c.sum() ** 2 / (c ** 2).sum()) if c.size else float("nan")


def card(rows: pd.DataFrame, y: np.ndarray, target: str) -> dict:
    """MET-13 card of one population (one row per scored entry or pair, y aligned)."""
    y = np.asarray(y, dtype=float)
    n = int(len(y))
    if not np.isfinite(y).all():
        raise AssertionError("a scored row has a non-finite target")
    vc = rows["chemotype"].astype(str).value_counts()
    out = {"target": target, "n_rows": n, "n_cells": int(rows["ci"].nunique()),
           "n_extractants": int(rows["extractant"].nunique()), "n_chemotypes": int(rows["chemotype"].nunique()),
           "n_publications": int(rows["publication_id"].nunique()),
           "kish_chemotypes_rows": kish_of(rows["chemotype"]), "kish_extractants_rows": kish_of(rows["extractant"]),
           "largest_chemotype": str(vc.index[0]) if n else None,
           "largest_chemotype_share": float(vc.iloc[0] / vc.sum()) if n else float("nan")}
    if n == 0:
        return out
    ybar = float(y.mean())
    var0 = float(np.mean((y - ybar) ** 2))
    out.update({"y_mean": ybar, "y_sd0": float(np.sqrt(var0)), "y_var0": var0})
    if target == "logsf":
        out["y_rms0"] = float(np.sqrt(np.mean(y * y)))
    else:
        ci = rows["ci"].to_numpy(dtype=int)
        _, code = np.unique(ci, return_inverse=True)
        nc = np.bincount(code).astype(float)
        yc = np.bincount(code, weights=y) / nc
        sst = float(np.sum((y - ybar) ** 2))
        ssb = float(np.sum(nc * (yc - ybar) ** 2))
        out["sst"] = sst
        if sst > 0:
            out["s_b"] = ssb / sst
            out["s_w"] = 1.0 - ssb / sst
            out["pi"] = out["s_w"]
        else:
            out["s_b"] = out["s_w"] = out["pi"] = None
            out["s_reason"] = "constant target"
    return out


def pair_types(rows: pd.DataFrame, y: np.ndarray) -> list[dict]:
    t = rows.assign(_y=np.asarray(y, dtype=float))
    out = []
    for (pair, ia), g in sorted(t.groupby(["pair", "ia"]), key=lambda kv: kv[0][1]):
        v = g["_y"].to_numpy(dtype=float)
        out.append({"pair": str(pair), "n": int(len(v)), "n_extractants": int(g["extractant"].nunique()),
                    "y_mean": float(v.mean()), "y_sd0": float(np.sqrt(np.mean((v - v.mean()) ** 2)))})
    return out


def noise_record(frame: pd.DataFrame, train_index, observed: np.ndarray) -> dict:
    """MET-19 log D replicate noise of one outer training fold, reading the training cells' ``nrep__``/``repsd__``
    only (LT-19)."""
    tr = np.sort(np.asarray(train_index, dtype=int))
    nrep = frame[list(D.NREP_COLS)].to_numpy(dtype=float)[tr]
    sd = frame[list(REPSD_COLS)].to_numpy(dtype=float)[tr]
    obs = np.asarray(observed, dtype=bool)[tr]
    sel = obs & (nrep >= 2) & np.isfinite(sd)
    n_e, s_e = nrep[sel], sd[sel]
    cap = s_e <= NOISE_CAP
    dof = n_e - 1.0
    rec = {"n_train_cells": int(len(tr)), "n_rep_entries": int(sel.sum()), "n_rep_entries_cap": int(cap.sum()),
           "n_rep_cells": int(sel.any(axis=1).sum())}
    rec["sigma2_pool"] = float(np.sum(dof * s_e ** 2) / np.sum(dof)) if dof.sum() > 0 else None
    rec["sigma2_pool_cap"] = float(np.sum(dof[cap] * s_e[cap] ** 2) / np.sum(dof[cap])) if dof[cap].sum() > 0 else None
    rec["median_s_e_cap"] = float(np.median(s_e[cap])) if cap.any() else None
    rec["gen20_training_noise_s2"] = None if rec["median_s_e_cap"] is None else rec["median_s_e_cap"] ** 2
    rec["sqrt2_sigma_bound_logsf"] = None if rec["sigma2_pool_cap"] is None else float(np.sqrt(2.0 * rec["sigma2_pool_cap"]))
    return rec


def summary(values) -> dict:
    v = np.array([x for x in values if x is not None], dtype=float)
    if not v.size:
        return {"n": 0, "median": None, "min": None, "max": None}
    return {"n": int(v.size), "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}


def repcap_drop(frame: pd.DataFrame, observed: np.ndarray) -> pd.DataFrame:
    """Label-derived drop list: observed entries with nrep >= 2 and repsd > 0.5."""
    nrep = frame[list(D.NREP_COLS)].to_numpy(dtype=float)
    sd = frame[list(REPSD_COLS)].to_numpy(dtype=float)
    hit = np.asarray(observed, dtype=bool) & (nrep >= 2) & np.isfinite(sd) & (sd > REPCAP_SD)
    ci, im = np.nonzero(hit)
    return pd.DataFrame({"ci": ci.astype(int), "cell_id": frame["cell_id"].astype(str).to_numpy()[ci],
                         "im": im.astype(int), "metal": [D.LANTHANIDES[i] for i in im],
                         "nrep": nrep[ci, im].astype(int)})


def apply_repcap(rows: pd.DataFrame, drop: pd.DataFrame, target: str) -> pd.DataFrame:
    bad = set(zip(drop["ci"].astype(int), drop["im"].astype(int)))
    ci = rows["ci"].to_numpy(dtype=int)
    if target == "logd":
        keep = np.array([(c, m) not in bad for c, m in zip(ci, rows["im"].to_numpy(dtype=int))], dtype=bool)
    else:
        keep = np.array([(c, a) not in bad and (c, b) not in bad
                         for c, a, b in zip(ci, rows["ia"].to_numpy(dtype=int), rows["ib"].to_numpy(dtype=int))], dtype=bool)
    return rows[keep].reset_index(drop=True)


def content_digest(t: pd.DataFrame) -> str:
    return hashlib.sha256(t.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def _py(v):
    if isinstance(v, dict):
        return {str(k): _py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_py(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if np.isfinite(f) else None
    if isinstance(v, np.bool_):
        return bool(v)
    return v


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def seed_invariant_rows(man: pd.DataFrame, target: str) -> pd.DataFrame:
    """The population's rows (one design and seed), asserting every (design, seed) holds the same keys."""
    keys = ["ci", "im"] if target == "logd" else ["ci", "ia", "ib"]
    ref = None
    for (d, s), g in man.groupby(["design", "split_seed"]):
        k = set(map(tuple, g[keys].astype(int).to_numpy().tolist()))
        if ref is None:
            ref = k
        elif k != ref:
            raise AssertionError(f"population rows differ between (design, seed) units ({d}, {s})")
    first = man[(man["design"] == man["design"].iloc[0]) & (man["split_seed"] == man["split_seed"].iloc[0])]
    return first.sort_values(keys, kind="mergesort").reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", default="S5")
    ap.add_argument("--out", default=None, help="output directory (default generations/gen20_1/manifests; tests: a temporary "
                                                "directory)")
    args = ap.parse_args(argv)
    seal_state = require_sealed(bootstrap.PREREG)
    out_root = bootstrap.assert_under_g201(args.out if args.out else bootstrap.MANIFESTS)
    run_manifest = out_root / f"g201_describe__{args.label}.json"
    if run_manifest.exists():
        print(f"STOP: {run_manifest} exists; a re-run passes a new --label", file=sys.stderr)
        return 1
    before = guard.snapshot()
    rm_dir = bootstrap.MANIFESTS / "row_manifests"
    digests = json.loads((rm_dir / "DIGESTS.json").read_text(encoding="utf-8"))
    status = 1
    with Run("g201_describe", label=args.label, args=vars(args), seeds=list(FO.SEEDS), manifest_dir=out_root) as run:
        run.record("preregistration", seal_state)
        bench = D.load(check=True)
        run.record("cohort", D.LOAD_RECORD)
        f = bench.frame
        Y = np.asarray(bench.Y, dtype=float)
        obs = D.observed(bench)
        F = FO.build(bench)
        run.record("fold_gate_e", FO.gate_e(FO.plan_digest(bench, F), recorded=FO.recorded_plan()))
        if not run.records["fold_gate_e"]["pass"]:
            raise FO.FoldGateError("STOP INCOMPARABLE_FOLDS: outer+inner fold digest differs from the S3 record")
        out = {"schema": "gen20.1 S5 describe (deterministic; label-derived target statistics, descriptive only)",
               "preregistration_digest": seal_state["digest"], "cards": {}, "pair_types": {},
               "noise_logd": {}, "repcap": {}, "repcap_manifests": {}}
        test_pops = sorted(k for k, v in digests.items() if v.get("side") == "test")
        drop = repcap_drop(f, obs)
        work = staging_dir(out_root)
        try:
            (work / "row_manifests").mkdir()
            new_files = {}
            p = work / "row_manifests" / "COH-REPCAP.parquet"
            drop.to_parquet(p, index=False)
            new_files["row_manifests/COH-REPCAP.parquet"] = p
            out["repcap"] = {"definition": "observed entries with nrep >= 2 and repsd > 0.5 (label-derived; scoring "
                                           "only); log SF pairs containing one are dropped",
                             "n_entries_dropped": int(len(drop)), "n_cells_touched": int(drop["ci"].nunique()),
                             "per_population": {}}
            out["repcap_manifests"]["COH-REPCAP"] = {"file": "row_manifests/COH-REPCAP.parquet",
                                                    "sha256_file": sha256_file(p), "sha256_content_csv": content_digest(drop),
                                                    "rows_total": int(len(drop))}
            for name in test_pops:
                target = digests[name]["target"]
                man = pd.read_parquet(rm_dir / f"{name}.parquet")
                rows = seed_invariant_rows(man, target)
                y = target_values(Y, rows, target)
                out["cards"][name] = card(rows, y, target)
                if target == "logsf":
                    out["pair_types"][name] = pair_types(rows, y)
                rc_name = f"COH-REPCAP@{name}"
                rc_all = apply_repcap(man, drop, target)
                rc_rows = seed_invariant_rows(rc_all, target)
                yr = target_values(Y, rc_rows, target)
                out["cards"][rc_name] = card(rc_rows, yr, target)
                if target == "logsf":
                    out["pair_types"][rc_name] = pair_types(rc_rows, yr)
                out["repcap"]["per_population"][name] = {"n_rows": int(len(rows)), "n_rows_repcap": int(len(rc_rows)),
                                                         "n_dropped": int(len(rows) - len(rc_rows))}
                q = work / "row_manifests" / f"{rc_name}.parquet"
                rc_all.to_parquet(q, index=False)
                new_files[f"row_manifests/{rc_name}.parquet"] = q
                out["repcap_manifests"][rc_name] = {"file": f"row_manifests/{rc_name}.parquet", "parent": name,
                                                   "target": target, "side": "test", "sha256_file": sha256_file(q),
                                                   "sha256_content_csv": content_digest(rc_all),
                                                   "rows_total": int(len(rc_all)), "rows_per_design_seed": int(len(rc_rows))}
            for d in FO.DESIGNS:
                recs = []
                for fo in F[d]:
                    recs.append({"split_seed": int(fo.seed), "fold": int(fo.fold), **noise_record(f, fo.train_index, obs)})
                out["noise_logd"][d] = recs
            bp = out["noise_logd"]["BP"]
            out["noise_logd_summary_BP"] = {k: summary([r[k] for r in bp]) for k in
                                            ("sigma2_pool", "sigma2_pool_cap", "median_s_e_cap", "n_rep_entries",
                                             "n_rep_entries_cap", "sqrt2_sigma_bound_logsf")}
            out["noise_logsf"] = {"status": "not estimable",
                                  "reason": "S3 (counts_g201.json): COH-REPPAIR run pairing via experiment_series_id gives "
                                            "115 candidate pairs in 20 cells from 2 chemotypes; 0 of 25 BP training folds "
                                            "reach 20 pairs from 5 chemotypes",
                                  "bound": "sqrt(2) * sigma_pool_cap per fold (labelled upper bound, never an estimate)"}
            jp = work / OUT_JSON
            with open(jp, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(_py(out), fh, indent=1, sort_keys=True)
            new_files[OUT_JSON] = jp
            # publish or compare (never overwrite)
            existing = {rel: out_root / rel for rel in new_files if (out_root / rel).exists()}
            cmp = {}
            if existing:
                for rel, src in new_files.items():
                    dst = out_root / rel
                    cmp[rel] = dst.exists() and sha256_file(dst) == sha256_file(src)
                run.record("rerun_byte_comparison", cmp)
                if not all(cmp.values()):
                    raise AssertionError(f"STOP: S5 outputs differ from the existing files: "
                                         f"{[k for k, v in cmp.items() if not v]}; temporary copies kept in {work}")
            else:
                for rel, src in new_files.items():
                    dst = bootstrap.assert_under_g201(out_root / rel)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
            run.record("outputs", {rel: sha256_file(out_root / rel) for rel in new_files})
            run.record("summary", {"n_populations": len(out["cards"]), "repcap_entries_dropped": int(len(drop))})
            shutil.rmtree(work, ignore_errors=True)
        except Exception:
            run.log(f"staging directory kept for diagnosis: {work}")
            raise
        guard.assert_unchanged(before)
        run.record("frozen_trees_unchanged", True)
        status = 0
    return status


if __name__ == "__main__":
    sys.exit(main())
