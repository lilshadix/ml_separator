"""``scripts/g19_export_confirmation_views.py`` -- descriptive VIEW files of the FINISHED confirmation run.

The once-only runner writes ``evaluation/confirmation/decisions/confirmation.json``, ``tables/confirmation_*.csv`` and
``decisions/CONFIRMATION.md`` and nothing else, while the report (``report.INPUTS``) and the figures (F12 and the V6
panel of F13, ``g19_make_figures.confirmation_inputs``) read three files under ``evaluation/confirmation/`` that no
stage produced -- so after the run F12 would have been skipped as 'input missing' and Q3's per-system line would have
been 'not computed' (task X finding, 2026-09-26).  This script writes them, from what the run recorded and decided:

* ``v6_systems.csv`` -- ``s2.s2a.sign.per_system`` of the decision file, verbatim: ``system``,
  ``observed_median_logsf``, ``predicted_median_logsf``, ``sign_agrees`` (the seed-mean-of-per-seed-medians the run's
  S2(a) sign count was decided on; POST-HOC addendum 6 item 1);
* ``v6_rows.csv`` -- the hidden Pr(III) / Nd(III) rows of the 13 V6 systems with the S2 candidate's prediction, the
  SEED MEAN over the withheld seed indices of ``mean_logD`` / ``lower_80`` / ``upper_80`` (the addendum 6 pooling), joined
  to ``system``, ``metal_state`` and the observed ``log_D``;
* ``v6_pairs.csv`` -- the comparable Nd/Pr test-test pairs per system (section 2's pairs, as the run built them):
  ``observed_logsf`` and the seed-mean ``predicted_logsf``, with ``fold``, the two row ids and the acid stratum.

Every row carries the ``label`` column: a descriptive view of the single run that DECIDES NOTHING.  The verdicts are
the decision file's alone; this script re-scores no claim, re-fits no fold and adds no rule.

Gate: it refuses unless ``decisions/confirmation.json`` exists -- the run has finished and scored -- so it can never be
an early look at a confirmation record.  Records are read through ``g19_run_confirmation.read_seed_predictions`` (every
record verified against the registry entry of stage ``confirmation`` and against the live code digest) and paired
through ``v6_pair_inputs`` (the run's own scope guards).  No withheld seed value can enter: the records are scrubbed
(``confirmation.scrub_frame``) and the row attributes carry none.

Run from the repository root, after the run:
    PYTHONIOENCODING=utf-8 PYTHONPATH=generations/gen19_chem_transfer .venv/Scripts/python.exe \\
        generations/gen19_chem_transfer/scripts/g19_export_confirmation_views.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from gen19ct import paths  # noqa: E402
from gen19ct.evaluation import confirmation as CF  # noqa: E402
from gen19ct.evaluation import metrics as EM  # noqa: E402
from gen19ct.manifest import Run, git_head, write_csv  # noqa: E402

NAME = "g19_export_confirmation_views"
LABEL = ("descriptive view of the single confirmation run (seed mean over the withheld seed indices, POST-HOC addendum 6 "
         "item 1); decides nothing -- the verdicts are decisions/confirmation.json's alone")
VIEW_FILES: tuple[str, ...] = ("v6_systems.csv", "v6_rows.csv", "v6_pairs.csv")
ROW_COLS: tuple[str, ...] = ("row_id", "system", "metal_state", "log_D", "mean_logD", "lower_80", "upper_80")
PAIR_COLS: tuple[str, ...] = ("system", "fold", "row_id_a", "row_id_b", "acid_stratum", "observed_logsf", "predicted_logsf")


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def log(msg: str) -> None:
    print(f"[{pd.Timestamp.now():%H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------------------------------- #
# pure transforms (tested on tiny frames)
# --------------------------------------------------------------------------------------------- #

def systems_table(body: Mapping[str, Any]) -> pd.DataFrame:
    """``s2.s2a.sign.per_system`` as the report's ``v6_systems.csv``; empty (with the columns) when S2 has no sign block."""
    per = (((body.get("s2") or {}).get("s2a") or {}).get("sign") or {}).get("per_system") or []
    cols = ["system", "observed_median_logsf", "predicted_median_logsf", "sign_agrees"]
    df = pd.DataFrame(per, columns=cols) if per else pd.DataFrame(columns=cols)
    return df.assign(s2_status=str((body.get("s2") or {}).get("status")), label=LABEL)


def seed_mean_rows(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Per ``row_id`` over the seed indices: the mean of the prediction columns, the first of the row attributes and the
    number of seed indices that scored the row (every one, when the run was complete)."""
    if not frames:
        return pd.DataFrame(columns=[*ROW_COLS, "n_seeds", "label"])
    df = pd.concat(frames, ignore_index=True)
    g = df.groupby("row_id", sort=True)
    out = g.agg(system=("system", "first"), metal_state=("metal_state", "first"), log_D=("log_D", "first"),
                mean_logD=("mean_logD", "mean"), lower_80=("lower_80", "mean"), upper_80=("upper_80", "mean"),
                n_seeds=("seed_index", "nunique")).reset_index()
    return out[[*ROW_COLS, "n_seeds"]].assign(label=LABEL)


def seed_mean_pairs(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Per (``row_id_a``, ``row_id_b``) over the seed indices: the seed-mean predicted logSF beside the observed one."""
    if not frames:
        return pd.DataFrame(columns=[*PAIR_COLS, "n_seeds", "label"])
    df = pd.concat(frames, ignore_index=True)
    g = df.groupby(["row_id_a", "row_id_b"], sort=True)
    out = g.agg(system=("system", "first"), fold=("fold", "first"), acid_stratum=("acid_stratum", "first"),
                observed_logsf=("observed_logsf", "first"), predicted_logsf=("predicted_logsf", "mean"),
                n_seeds=("seed_index", "nunique")).reset_index()
    return out[[*PAIR_COLS, "n_seeds"]].assign(label=LABEL)


# --------------------------------------------------------------------------------------------- #
# the records of the S2 candidate, one seed index at a time
# --------------------------------------------------------------------------------------------- #

def candidate_frames(out_root: Path, body: Mapping[str, Any], *, rc: Any) -> dict[str, Any]:
    """The S2 candidate's V6 rows and pairs per seed index, through the runner's own verified reader and pair builder."""
    arm, ddir = rc.V6_CANDIDATE_ARM, rc.V6_DESIGN_DIR
    corpus = rc.confirmation_corpus()
    attrs = rc.pair_attributes()
    v6 = rc.v6_by_id(corpus)
    acid = rc.ACID_STRATUM_COL
    rows, pairs, missing = [], [], []
    for i in [int(k) for k in (body.get("seed_indices") or [])]:
        pred = rc.read_seed_predictions(out_root, arm, ddir, i)
        if pred is None:
            missing.append(i)
            continue
        cin = rc.v6_pair_inputs(pred, attrs, corpus, v6=v6, what=f"{NAME}/{arm}/i{i}")
        fr = cin["frame"]
        rows.append(pd.DataFrame({"row_id": fr["row_id"].astype(str).to_numpy(),
                                  "system": fr[EM.SYSTEM_COL].astype(str).to_numpy(),
                                  "metal_state": fr[EM.METAL_STATE_COL].astype(str).to_numpy(),
                                  "log_D": pd.to_numeric(fr[EM.Y_COL], errors="coerce").to_numpy(),
                                  "mean_logD": pd.to_numeric(fr["mean_logD"], errors="coerce").to_numpy(),
                                  "lower_80": pd.to_numeric(fr["lower_80"], errors="coerce").to_numpy(),
                                  "upper_80": pd.to_numeric(fr["upper_80"], errors="coerce").to_numpy(),
                                  "seed_index": i}))
        pr = cin["pairs"]
        same = pr[f"{acid}_a"].astype(str).to_numpy() == pr[f"{acid}_b"].astype(str).to_numpy()
        pairs.append(pd.DataFrame({"row_id_a": pr["row_id_a"].astype(str).to_numpy(),
                                   "row_id_b": pr["row_id_b"].astype(str).to_numpy(),
                                   "system": pr[EM.SYSTEM_COL].astype(str).to_numpy(),
                                   "fold": pr["fold"].astype(str).to_numpy(),
                                   "acid_stratum": [a if s else "mixed" for a, s in zip(pr[f"{acid}_a"].astype(str), same)],
                                   "observed_logsf": pd.to_numeric(pr["logsf_obs"], errors="coerce").to_numpy(),
                                   "predicted_logsf": pd.to_numeric(cin["logsf"], errors="coerce").to_numpy(),
                                   "seed_index": i}))
    return {"arm": arm, "design_dir": ddir, "rows": rows, "pairs": pairs, "seed_indices_missing": missing,
            "seed_indices_read": [int(k) for k in (body.get("seed_indices") or []) if int(k) not in missing]}


# --------------------------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------------------------- #

def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-root", default=str(paths.G19_ROOT))
    ap.add_argument("--no-manifest", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    ns = parse_args(argv)
    out_root = Path(ns.out_root)
    dec = CF.decisions_path(out_root)
    if not dec.exists():
        print(f"refused: {dec} does not exist -- the single confirmation run has not finished and scored; this script "
              "reads no confirmation record before it has (section 15)")
        return 2
    body = json.loads(dec.read_text(encoding="utf-8"))
    rc = _load("g19_run_confirmation")
    conf_dir = CF.conf_root(out_root)
    outs: list[Path] = []
    with (Run(NAME, args={k: v for k, v in vars(ns).items()}, seed=None,
              extra={"git_head": git_head(), "label": LABEL, "decides_nothing": True,
                     "decision_file_sha256": paths.digests(dec)["sha256"], "s2_status": (body.get("s2") or {}).get("status")})
          if not ns.no_manifest else _Null()) as run:
        sys_tab = systems_table(body)
        outs.append(write_csv(sys_tab, conf_dir / "v6_systems.csv"))
        log(f"v6_systems.csv: {len(sys_tab)} systems (S2 record set {(body.get('s2') or {}).get('status')})")
        cf = candidate_frames(out_root, body, rc=rc)
        rows, pairs = seed_mean_rows(cf["rows"]), seed_mean_pairs(cf["pairs"])
        outs.append(write_csv(rows, conf_dir / "v6_rows.csv"))
        outs.append(write_csv(pairs, conf_dir / "v6_pairs.csv"))
        log(f"v6_rows.csv: {len(rows)} hidden Pr/Nd rows; v6_pairs.csv: {len(pairs)} pairs; arm {cf['arm']}; seed indices "
            f"read {cf['seed_indices_read']}, missing {cf['seed_indices_missing']}")
        if run is not None:
            run.inputs(dec)
            run.outputs(*outs)
            run.extra.update({"arm": cf["arm"], "design_dir": cf["design_dir"], "seed_indices_read": cf["seed_indices_read"],
                              "seed_indices_missing": cf["seed_indices_missing"], "n_rows": int(len(rows)),
                              "n_pairs": int(len(pairs)), "n_systems": int(len(sys_tab))})
    return 0


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
