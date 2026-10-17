"""gen20.1 stage S8 (SMOKE + TIMING): one split seed, the BP fold that holds out sc009 (the giant fold) plus the BP
fold whose training set is closest to 490 cells, at most 40 scored test cells each, both variants, every arm.

IMPLEMENTATION_PLAN.md §9 S8 and §9 "Compute estimate"; PRE_REGISTRATION.md §8.1 compute rule.  Smoke computes NO
metric against any outer-test y and tunes nothing; its output (label ``smoke*``) is refused by the report code.
Checks: coverage of the scored rows (``fitting.check_coverage``, row manifests restricted to the scored cells),
finiteness of every arm, C_PAIR antisymmetry (f(a,b) = -f(b,a) exactly), the centred learner C fits and predicts
finite values, frozen trees untouched (whole-repository snapshot outside generations/gen20_1).  Records seconds and
process peak RSS (Windows PeakWorkingSetSize, monotone over the process) after every fit, and projects the full
two-pass discovery time (150 units x {primary, nodmd}, C_PAIR on BP/B primary, A_PRIME on BP primary): per arm
family, seconds per unit are interpolated linearly in the unit's training-row count between the two measured units
and summed over the training-row counts of all 150 units (COH-TRAIN).  The S8 gate compares the projection with 2 x
the plan's two-pass estimate (about 12 h): above 24 h the pre-registered reduction rule applies (designs in separate
later invocations, procedure unchanged, nothing dropped).

The smoke pass flags (C, C_PAIR, coverage, finite, frozen) are written to manifests/g201_smoke__<label>.json; the
discovery script reads them (a learner that fails smoke is "not run").  A failure of the centred learner C (a
non-finite C / C_K* / C_ACFG / C_selSF column, or any exception inside its fits) or of C_PAIR (non-finite column or an
exception inside its fit / prediction) is recorded, not raised: ``fitting`` raises it as ``ArmFailure`` naming the
learner, the learner is switched off ("not run", PRE_REGISTRATION.md §4.2 / O11 for C, §4.5 for C_PAIR) and the
smoke run is repeated from scratch without it; the reasons go to ``records.arm_failures``.  Any other failure raises.
The exit status is 0 when coverage, finiteness and frozen-tree invariance pass (a learner recorded "not run" is a
registered outcome, not a smoke failure), 1 otherwise.

Usage (repo root):  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B generations/gen20_1/scripts/g201_smoke.py [--label smoke_S8]
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
G201 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(G201))

from gen201 import bootstrap  # noqa: E402
from gen201 import guard  # noqa: E402

guard.install()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gen201 import data as D  # noqa: E402
from gen201 import fitting as FI  # noqa: E402
from gen201 import folds as FO  # noqa: E402
from gen201 import learners as L  # noqa: E402
from gen201.run import Run  # noqa: E402
from gen201.seal import preregistration_state  # noqa: E402

SMOKE_SEED = FO.SEEDS[0]
TARGET_TRAIN_CELLS = 490
MAX_TEST_CELLS = 40
PLAN_TWO_PASS_HOURS = 12.0          # IMPLEMENTATION_PLAN.md §9: ~280-300 s per unit, ~12 h for 150 units, two passes
GATE_FACTOR = 2.0


def pick_folds(folds_bp, seed: int = SMOKE_SEED) -> tuple:
    """(giant fold = the one holding out sc009, the fold whose training set is closest to 490 cells)."""
    fs = [f for f in folds_bp if int(f.seed) == int(seed)]
    giant = [f for f in fs if D.SC009 in f.held_out_groups]
    if len(giant) != 1:
        raise AssertionError(f"expected one sc009 fold at seed {seed}, got {len(giant)}")
    rest = [f for f in fs if f is not giant[0]]
    big = min(rest, key=lambda f: (abs(len(f.train_index) - TARGET_TRAIN_CELLS), f.fold))
    return giant[0], big


def family_seconds(sel: dict) -> dict:
    return dict(sel["timing"]["seconds_by_family"])


#: the flags a smoke run must pass (C and C_PAIR may be recorded "not run" instead)
REQUIRED_FLAGS: tuple[str, ...] = ("coverage", "finite", "frozen_unchanged")


def run_with_fallback(fit, log=print) -> tuple[dict, dict, dict]:
    """``fit(include_c, include_c_pair) -> summary``, repeated without a learner whose fit failed.

    An ``fitting.ArmFailure`` of the C family or of C_PAIR switches that learner off (it is "not run") and repeats
    the whole run without it (the caller's ``fit`` refits every unit, ``resume=False``); any other exception, or a
    second failure of a learner already switched off, raises.  Returns (summary, include flags, failure records)."""
    include = {fam: True for fam in FI.SMOKE_GATED}
    failures: dict = {}
    while True:
        try:
            return fit(include["C"], include["C_PAIR"]), include, failures
        except FI.ArmFailure as e:
            fam = e.family
            if fam not in include or not include[fam]:
                raise
            include[fam] = False
            failures[fam] = {"arm": e.arm, "error": str(e),
                             "cause": None if e.__cause__ is None else repr(e.__cause__),
                             "traceback": "".join(traceback.format_exception(e))}
            log(f"S8: learner {fam} failed its smoke test ({e}); it is 'not run' (PRE_REGISTRATION.md "
                f"{'§4.2, O11' if fam == 'C' else '§4.5'}); repeating the smoke run without it")


def project(measured: list[dict], units: pd.DataFrame) -> dict:
    """Linear interpolation (extrapolation at the ends) in training rows of each family's seconds per unit, summed
    over ``units`` (columns design, split_seed, fold, n_train_rows, n_train_rows_nodmd)."""
    fams = sorted({k for m in measured for k in m["families"]})
    x = np.array([m["n_train_rows"] for m in measured], dtype=float)
    total, per_family = 0.0, {}
    for fam in fams:
        yv = np.array([m["families"].get(fam, 0.0) for m in measured], dtype=float)
        if len(x) >= 2 and x.max() > x.min():
            b = (yv[-1] - yv[0]) / (x[-1] - x[0])
            a = yv[0] - b * x[0]
        else:
            a, b = float(yv.mean()), 0.0
        s = 0.0
        for _, u in units.iterrows():
            for variant, col in (("primary", "n_train_rows"), ("nodmd", "n_train_rows_nodmd")):
                if fam == "C_PAIR" and not (u["design"] in FI.CPAIR_DESIGNS and variant == "primary"):
                    continue
                if fam == "A_PRIME" and not (u["design"] in FI.APRIME_DESIGNS and variant == "primary"):
                    continue
                s += max(a + b * float(u[col]), 0.0)
        per_family[fam] = s
        total += s
    return {"seconds_total": total, "hours_total": total / 3600.0,
            "hours_by_family": {k: v / 3600.0 for k, v in per_family.items()},
            "method": "per family: seconds per unit linear in the unit's training-row count through the two measured "
                      "units, summed over the 150 (design, seed, fold) units x {primary, nodmd}; C_PAIR only BP/B "
                      "primary, A_PRIME only BP primary; prediction time measured on <= 40 test cells"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", default="smoke_S8")
    ap.add_argument("--seed", type=int, default=SMOKE_SEED)
    ap.add_argument("--root", default=None, help="prediction root (default generations/gen20_1/predictions)")
    ap.add_argument("--manifest-dir", default=None, help="run-manifest directory (default generations/gen20_1/manifests)")
    ap.add_argument("--tiny", action="store_true", help=argparse.SUPPRESS)          # tests only: 4 / 2 trees, never in predictions/
    args = ap.parse_args(argv)
    root = bootstrap.assert_under_g201(args.root if args.root else bootstrap.PREDICTIONS)
    if args.tiny:
        if root == bootstrap.PREDICTIONS.resolve():
            print("STOP: --tiny is for tests and never writes under predictions/", file=sys.stderr)
            return 1
        L.N_ESTIMATORS, L.C_PAIR_N_ESTIMATORS = 4, 2
    if not args.label.startswith("smoke"):
        print("STOP: a smoke label must start with 'smoke'", file=sys.stderr)
        return 1
    if not L.registered_tree_counts() and not args.tiny:
        print("STOP: tree counts differ from the registered 400/200", file=sys.stderr)
        return 1
    before = guard.snapshot()
    status = 1
    with Run("g201_smoke", label=args.label, args=vars(args), seeds=[args.seed], manifest_dir=args.manifest_dir) as run:
        run.record("preregistration", preregistration_state(bootstrap.PREREG))
        bench = D.load(check=True)
        run.record("cohort", D.LOAD_RECORD)
        F = FO.build(bench, seeds=(args.seed,), designs=("BP",))["BP"]
        giant, big = pick_folds(F, args.seed)
        run.record("folds", [{"split_seed": int(f.seed), "fold": int(f.fold), "n_train": int(len(f.train_index)),
                              "n_test": int(len(f.test_index)), "held_out_groups": list(f.held_out_groups)}
                             for f in (giant, big)])
        ctx = FI.make_context(bench)
        ids = [(giant.seed, giant.fold), (big.seed, big.fold)]

        def fit_all(include_c: bool, include_c_pair: bool) -> dict:
            return FI.run_design(bench, "BP", seeds=(args.seed,), label=args.label, variants=FI.VARIANTS,
                                 resume=False, smoke=True, fold_ids=ids, max_test_cells=MAX_TEST_CELLS, folds_ctx=ctx,
                                 folds_list=F, check_plan=False, log=run.log, out_root=root, include_c=include_c,
                                 include_c_pair=include_c_pair)

        summ, include, failures = run_with_fallback(fit_all, log=run.log)
        run.record("run_design", summ)
        run.record("include", include)
        run.record("arm_failures", failures)
        mine = {(int(s), int(f)) for s, f in ids}
        sels = [s for s in (FI.load_selections(args.label, "BP", "primary", root=root)
                            + FI.load_selections(args.label, "BP", "nodmd", root=root))
                if (int(s["split_seed"]), int(s["fold"])) in mine]
        flags = {"coverage": True, "finite": True, "C": bool(include["C"]), "C_PAIR": bool(include["C_PAIR"])}
        for v in FI.VARIANTS:
            logd, logsf = FI.load_tables(args.label, "BP", v, root=root)
            for s in (x for x in sels if x["variant"] == v):
                d_arms, s_arms = s["arms_logd"], s["arms_logsf"]
                flags["finite"] &= bool(np.isfinite(logd[d_arms].to_numpy(float)).all()
                                        and np.isfinite(logsf[s_arms].to_numpy(float)).all())
                flags["coverage"] &= bool(s["coverage"]["logd_rows"] > 0 and s["coverage"]["manifests_checked"])
                flags["C"] &= bool("C" in s_arms and "C" in d_arms)
                if v == "primary":
                    flags["C_PAIR"] &= bool(s["c_pair"] is not None and s["c_pair"]["antisymmetric"])
        if include["C_PAIR"] and not flags["C_PAIR"]:
            failures["C_PAIR"] = {"arm": "C_PAIR", "error": "f(a,b) != -f(b,a): antisymmetry check failed",
                                  "cause": None, "traceback": None}
            run.record("arm_failures", failures)
        guard.assert_unchanged(before)
        flags["frozen_unchanged"] = True
        run.record("pass", flags)
        fits = [{"variant": s["variant"], "fold": s["fold"], **fit} for s in sels for fit in s["timing"]["fits"]]
        run.record("fits", fits)
        run.record("peak_rss_bytes", max((f["peak_rss"] or 0) for f in fits) if fits else None)
        measured = sorted(({"fold": s["fold"], "n_train_rows": s["n_train_rows"], "families": family_seconds(s)}
                           for s in sels if s["variant"] == "primary"), key=lambda m: m["n_train_rows"])
        run.record("measured_units", measured)
        tr = pd.read_parquet(bootstrap.MANIFESTS / "row_manifests" / "COH-TRAIN.parquet",
                             columns=["design", "split_seed", "fold", "is_dmd"])
        units = (tr.groupby(["design", "split_seed", "fold"])
                   .agg(n_train_rows=("is_dmd", "size"), n_train_rows_nodmd=("is_dmd", lambda s: int((~s).sum())))
                   .reset_index())
        proj = project(measured, units)
        proj["gate_hours"] = GATE_FACTOR * PLAN_TWO_PASS_HOURS
        proj["reduction_rule_applies"] = bool(proj["hours_total"] > proj["gate_hours"])
        proj["decision"] = ("reduction rule: designs run in separate later invocations (order BP, B, BR, BQ, BPG, A), "
                            "procedure unchanged, nothing dropped" if proj["reduction_rule_applies"]
                            else "within the S8 gate")
        run.record("projection", proj)
        run.log(f"pass flags {flags}; projected two-pass discovery {proj['hours_total']:.1f} h "
                f"(gate {proj['gate_hours']:.0f} h)")
        for fam in FI.SMOKE_GATED:
            if not flags[fam]:
                run.log(f"S8: {fam} is 'not run' in discovery and confirmation ({failures.get(fam, {}).get('error')})")
        status = 0 if all(flags[k] for k in REQUIRED_FLAGS) else 1
    return status


if __name__ == "__main__":
    sys.exit(main())
