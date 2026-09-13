"""Script 4 of DESIGN.md section 11: fit the pre-registered D model M1 per system.

For every system of the E1 cohort (``--all``; frozen list ``results/audit/e1_cohort.csv``) or
the systems named by ``--system-id``, fit the pooled-slope mass-action model of
PRE_REGISTRATION.md section 3 (``gen18proc.evalproto.fit_mass_action``) on the fit-eligible
records of the ``--band`` temperature band, with the reliability of section 6 (jackknife by
publication, split-half by publication).  Writes ``results/dmodels/<system_id>.json``
(parameters, SE, reliability, domain, flags), ``<system_id>_split_half.csv`` (per repeat),
``fits_summary.csv``, ``manifest.json``, and the fitted block of DESIGN.md section 3.8 into
``systems/<system_id>.json`` (``params[ligand][band]``; ``fit_manifest_sha256`` = SHA-256 of the
results JSON; the ligand's ``ligands_per_metal`` set from the fitted ``n``; metals of the
entry's records without a training intercept get a ``value: null`` log K so validator V8
passes).

Refuses to run unless ``g18_seal_prereg.py --check`` passes (PRE_REGISTRATION.md section 0),
EXCEPT under ``--unsealed-dry-run``: then it fits the two-system subset ``DRYRUN_SYSTEMS`` and
writes everything under ``results/_dryrun/`` (the database files are copied there and the
frozen ``systems/`` directory is not touched).

Usage (from the repository root, one process):
    .venv/Scripts/python.exe generations/gen18_process/scripts/g18_fit_dmodels.py \\
        [--system-id SID ... | --all] [--band 20-30C] [--seed 18] [--unsealed-dry-run]

No wall-clock value is written.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen18proc import DEFAULT_SEED, paths  # noqa: E402
from gen18proc.evalproto import (  # noqa: E402
    DEFAULT_BAND,
    FitResult,
    as_solvating_params,
    fit_mass_action,
)
from gen18proc.report import write_manifest, write_table  # noqa: E402
from gen18proc.systems import load_registry, load_system, write_system  # noqa: E402
from gen18proc.types import Mechanism, Sourced  # noqa: E402

DRYRUN_SYSTEMS: tuple[str, ...] = ("sys_5cb78e5000d40860", "sys_a7195d8a9d8696e0")
"""The two-system subset of the unsealed dry run (TODGA and TEHDGA, nitrate / aliphatic; both
in E1 and both carrying E2 loading series)."""

DRYRUN_DIR = paths.RESULTS_DIR / "_dryrun"


def require_sealed(dry_run: bool) -> str:
    """Run ``g18_seal_prereg.py --check``; refuse (``SystemExit``) when it fails unless
    ``dry_run``.  Returns the check's message."""
    proc = subprocess.run([sys.executable, str(paths.SCRIPTS_DIR / "g18_seal_prereg.py"),
                           "--check"], capture_output=True, text=True, cwd=str(paths.REPO_ROOT),
                          check=False)
    msg = (proc.stdout or proc.stderr).strip()
    if proc.returncode != 0 and not dry_run:
        raise SystemExit("refused: PRE_REGISTRATION.md is not sealed or its digest differs "
                         f"({msg}); seal it with g18_seal_prereg.py or pass --unsealed-dry-run "
                         "to run on the two-system subset into results/_dryrun/")
    if proc.returncode != 0:
        print(f"UNSEALED DRY RUN ({msg}): two-system subset, outputs under {DRYRUN_DIR}")
    return msg


def e1_systems() -> pd.DataFrame:
    path = paths.RESULTS_AUDIT_DIR / "e1_cohort.csv"
    if not path.exists():
        raise SystemExit(f"{path} missing: run g18_audit.py first")
    return pd.read_csv(path)


def primary_ligand(registry: pd.DataFrame, system_id: str) -> str:
    row = registry.loc[registry["system_id"] == system_id]
    if row.empty:
        raise SystemExit(f"{system_id} is not in the registry")
    return str(row["ligands"].iloc[0]).split(";")[0]


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not math.isfinite(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (frozenset, set)):
        return sorted(_jsonable(v) for v in obj)
    if hasattr(obj, "value") and hasattr(obj, "name") and not isinstance(obj, Sourced):
        return obj.value                      # enums
    if dataclasses.is_dataclass(obj):
        return _jsonable(dataclasses.asdict(obj))
    return obj


def fit_json(fit: FitResult, band: str, n_records_system: int) -> dict[str, Any]:
    split = None
    if fit.split_half is not None:
        split = {k: v for k, v in fit.split_half.items() if k != "per_repeat"}
    return {
        "schema": "gen18.dmodel_fit.1",
        "system_id": fit.system_id, "ligand": fit.ligand, "band": band, "status": fit.status,
        "mechanism": fit.mechanism, "model_id": f"M1_pooled_{fit.system_id}_{band}",
        "n": fit.n, "p_eff": fit.p_eff, "slope_status": fit.slope_status,
        "priors": fit.priors, "intercepts": fit.intercepts,
        "publication_effects": fit.publication_effects, "se": fit.se, "se_hc1": fit.se_hc1,
        "jackknife_se": fit.jackknife_se, "split_half": split,
        "interpretable": fit.interpretable, "n_points": fit.n_points,
        "n_publications": fit.n_publications, "n_rows": fit.n_rows,
        "n_records_system": int(n_records_system), "distinct_levels": fit.distinct_levels,
        "residual_sd": fit.residual_sd, "param_names": list(fit.param_names),
        "covariance": fit.covariance, "covariance_hc1": fit.covariance_hc1,
        "flags": fit.flags, "domain": fit.domain,
        "record_ids_by_metal": {m: list(v) for m, v in fit.record_ids_by_metal.items()},
        "regime": {"cohort": f"fit-eligible records of {fit.system_id}, band {band}, "
                             "aggregated per replicate group",
                   "holdout": "none (in-sample fit); reliability by jackknife / split-half "
                              "over publications",
                   "averaging_unit": "aggregated point (weight n_rep)",
                   "status_of_parameters": "fitted_from_corpus"},
    }


def write_fitted_block(entry_path: Path, out_path: Path, fit: FitResult, band: str,
                       sha: str, prepared_metals: set[str]) -> None:
    """The fitted block of DESIGN.md 3.8 into ``params[ligand][band]`` of the entry."""
    entry = load_system(entry_path)
    fit_h = dataclasses.replace(fit, fit_manifest_sha256=sha)
    block = as_solvating_params(fit_h, medium_anion=entry.medium.anion)
    log_k = dict(block.log_k)
    record_metals = {r.metal for r in entry.records}
    for m in sorted(record_metals - set(log_k)):
        log_k[m] = Sourced.unknown("1", note="no training intercept in this band (validator V8 "
                                             "key; value unknown)")
    block = dataclasses.replace(block, log_k=log_k)
    params = {k: dict(v) for k, v in entry.params.items()}
    params.setdefault(fit.ligand, {})[band] = block
    ligands = []
    for lig in entry.organic_ligands:
        if lig.name == fit.ligand and lig.mechanism == Mechanism.SOLVATING:
            st = dataclasses.replace(lig.stoichiometry, ligands_per_metal=block.n_solvation)
            lig = dataclasses.replace(lig, stoichiometry=st)
        ligands.append(lig)
    # max loading fraction studied, from the loading-series records (corpus-measured)
    phase = entry.phase
    best = None
    for r in entry.records:
        if not r.loading_series_id or not r.fit_eligible:
            continue
        mm = r.metals_initial_mM.get(r.metal)
        lig_m = r.ligand_M.get(fit.ligand)
        if mm is None or lig_m is None or not (mm > 0 and lig_m > 0):
            continue
        lam = float(fit_h.n) * float(mm) * 1e-3 / float(lig_m)
        if best is None or lam > best[0]:
            best = (lam, r)
    if best is not None and phase.max_loading_fraction_studied.value is None:
        lam, r = best
        from gen18proc.types import Provenance, ProvStatus, Source
        phase = dataclasses.replace(phase, max_loading_fraction_studied=Sourced(
            lam, "1", Provenance(ProvStatus.MEASURED_CORPUS, Source(
                "corpus", None, "largest n_fitted * metal_initial_mM * 1e-3 / ligand_M over "
                "the loading-series records (O/A 1 assumed)", r.publication_id, (r.record_id,)),
                note=f"loading series {r.loading_series_id}; n = {fit_h.n:.4g}")))
    new_entry = dataclasses.replace(entry, params=params, organic_ligands=tuple(ligands),
                                    phase=phase)
    write_system(new_entry, out_path)
    load_system(out_path)          # raises SystemValidationError on any error-level rule


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--system-id", nargs="*", default=None)
    ap.add_argument("--all", action="store_true", help="every system of the E1 cohort")
    ap.add_argument("--band", default=DEFAULT_BAND)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--unsealed-dry-run", action="store_true",
                    help="run on DRYRUN_SYSTEMS into results/_dryrun/ without the seal")
    ap.add_argument("--systems-dir", default=str(paths.SYSTEMS_DIR))
    args = ap.parse_args()
    seal_msg = require_sealed(args.unsealed_dry_run)
    systems_dir = Path(args.systems_dir)
    if args.unsealed_dry_run:
        out_dir = DRYRUN_DIR / "dmodels"
        entries_out = DRYRUN_DIR / "systems"
        wanted = list(DRYRUN_SYSTEMS)
    else:
        out_dir = paths.RESULTS_DMODELS_DIR
        entries_out = systems_dir
        if args.all:
            wanted = e1_systems()["system_id"].astype(str).tolist()
        elif args.system_id:
            wanted = list(args.system_id)
        else:
            ap.error("pass --all or --system-id")
    out_dir.mkdir(parents=True, exist_ok=True)
    entries_out.mkdir(parents=True, exist_ok=True)
    registry = load_registry(systems_dir)
    e1 = e1_systems()
    records_path = systems_dir / "corpus_records.csv"
    records = pd.read_csv(records_path, low_memory=False)
    outputs: list[Path] = []
    summary_rows = []
    for sid in wanted:
        sub = records.loc[records["system_id"] == sid]
        if sub.empty:
            print(f"{sid}: no records; skipped")
            continue
        row = e1.loc[e1["system_id"] == sid]
        ligand = str(row["ligand"].iloc[0]) if not row.empty else primary_ligand(registry, sid)
        fit = fit_mass_action(sub, ligand=ligand, band=args.band, seed=args.seed,
                              reliability=True)
        if fit.status != "fitted":
            print(f"{sid}: {fit.status} ({fit.n_rows} rows in band {args.band}); skipped")
            continue
        payload = fit_json(fit, args.band, int(len(sub)))
        json_path = out_dir / f"{sid}.json"
        json_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True,
                                        ensure_ascii=True) + "\n", encoding="utf-8",
                             newline="\n")
        outputs.append(json_path)
        if fit.split_half is not None:
            sh_path = write_table(fit.split_half["per_repeat"], out_dir / f"{sid}_split_half.csv",
                                  regime={"cohort": f"E1 records of {sid}, band {args.band}",
                                          "holdout": "split-half by publication (20 seeded "
                                                     "repeats, seed 18)",
                                          "averaging_unit": "repeat"})
            outputs.append(sh_path)
        sha = paths.sha256_of(json_path)
        entry_path = systems_dir / f"{sid}.json"
        out_entry = entries_out / f"{sid}.json"
        write_fitted_block(entry_path, out_entry, fit, args.band, sha,
                           set(fit.intercepts))
        outputs.append(out_entry)
        jk = fit.jackknife_se or {}
        sh = fit.split_half or {}
        summary_rows.append({
            "system_id": sid, "ligand": ligand, "band": args.band, "n": fit.n,
            "p_eff": fit.p_eff, "se_n": fit.se.get("n", np.nan),
            "se_p_eff": fit.se.get("p_eff", np.nan),
            "jackknife_se_n": (jk.get("n") or (np.nan, np.nan))[1],
            "jackknife_se_p_eff": (jk.get("p_eff") or (np.nan, np.nan))[1],
            "split_half_n_agree": sh.get("n_sign_agreement", np.nan),
            "split_half_p_agree": sh.get("p_eff_sign_agreement", np.nan),
            "split_half_intercept_r_median": sh.get("intercept_r_median", np.nan),
            "slope_status_n": fit.slope_status["n"],
            "slope_status_p_eff": fit.slope_status["p_eff"],
            "interpretable_n": fit.interpretable["n"],
            "interpretable_p_eff": fit.interpretable["p_eff"],
            "n_points": fit.n_points, "n_publications": fit.n_publications,
            "n_rows": fit.n_rows, "n_metals": len(fit.intercepts),
            "residual_sd": fit.residual_sd, "fit_manifest_sha256": sha,
        })
        jk_n = summary_rows[-1]["jackknife_se_n"]
        print(f"{sid} ({ligand}): n = {fit.n:.3f} (jk se {jk_n:.3f}), p_eff = {fit.p_eff:.3f}, "
              f"{fit.n_points} points, {fit.n_publications} pubs, "
              f"interpretable {fit.interpretable}")
    if summary_rows:
        summ = pd.DataFrame(summary_rows)
        outputs.append(write_table(summ, out_dir / "fits_summary.csv", regime={
            "cohort": "E1 cohort systems" + (" (dry-run subset)" if args.unsealed_dry_run else ""),
            "holdout": "in_sample fit; reliability by jackknife and split-half over publications",
            "averaging_unit": "system", "status_of_parameters": "fitted_from_corpus"},
            regime_table=True))
    write_manifest(out_dir / "manifest.json", outputs,
                   [records_path, paths.RESULTS_AUDIT_DIR / "e1_cohort.csv",
                    paths.PRE_REGISTRATION_MD, paths.RESULTS_EVAL_DIR / "prereg_sha256.txt"],
                   args.seed, arguments=vars(args),
                   extra={"seal_check": seal_msg, "systems": wanted,
                          "dry_run": bool(args.unsealed_dry_run)})
    print(f"wrote {len(outputs)} files under {out_dir} (+ entries under {entries_out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
