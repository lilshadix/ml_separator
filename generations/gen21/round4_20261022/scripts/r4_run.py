#!/usr/bin/env python
"""Run one round-4 configuration (or an exact round-3 reproduction ``R3:<ID>``) on one design over the five dev seeds.

    .venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/scripts/r4_run.py --config REF --design BP
    .venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/scripts/r4_run.py --config R3:R14 --design BP

Before an outer-scored round-4 configuration (not ``R3:``) runs, the stage-0 seal must verify
(manifests/stage0_seal.txt; scripts/r4_seal.py --verify); ``--technical-smoke`` runs one fold with every metric
suppressed (shape / plumbing only).  Checkpoints: predictions/_parts/<ID>_<design>/<seed>_<fold>.pkl; the key
includes the stage-0 seal so that a code change never reuses stale folds.  Outputs as in round 3:
  predictions/<ID>_<design>.parquet, results/<ID>/{summary,per_pair,contrasts,fitted,intervention}_<design>.*
and one append-only line per run in RUN_RECORD.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import donor as D4  # noqa: E402
from r4 import metrics as M4  # noqa: E402
from r4 import paths  # noqa: E402
from r4 import registry as RG  # noqa: E402
from r3 import harness as H  # noqa: E402
from r3 import metrics as MT  # noqa: E402
from g21 import cache as C  # noqa: E402
from g21 import evaluate as E  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.splits import folds_for  # noqa: E402


def _num(v):
    return "inf" if isinstance(v, float) and math.isinf(v) else v


def fitted_info(fitted: dict) -> dict:
    out = {}
    for k, v in (fitted or {}).items():
        if k == "hbk":
            out["ml_kernel"] = v["ml_kernel"]
        elif k.startswith("cor:"):
            out[k] = {"lambda": _num(v["lambda"]), "inner_scores": v["inner_scores"], "coef": [float(c) for c in v["coef"]],
                      "n_rows": v["n_rows"]}
        elif k.startswith("ref:"):
            out[k] = {"lambda": _num(v["lambda"]), "inner_scores": v["inner_scores"], "coef": [float(c) for c in v["coef"]],
                      "n_rows": v["n_rows"], "n_situations": v["n_situations"], "coverage": v["coverage"]}
        elif k.startswith("slot:"):
            out[k] = {"lambda": _num(v["lambda"]), "inner_scores": v["inner_scores"], "coef": [float(c) for c in v["coef"]],
                      "columns": v["columns"], "n_rows": v["n_rows"], "n_situations": v["n_situations"],
                      "class_rule": "donor" if v["class_map"] is None else "shuffled chemotype-block map"}
        elif k.startswith("b:"):
            out[k] = {"lambda": "inf" if v["n_trees"] == 0 else v["n_trees"], "n_trees": v["n_trees"],
                      "inner_scores": v["inner_scores"], "n_rows": v["n_rows"], "n_situations": v["n_situations"],
                      "geometry": v["mode"].get("geometry")}
        elif k.startswith("c:"):
            out[k] = {"lambda": "inf" if v["choice"] == "zero" else str(v["choice"]), "choice": v["choice"],
                      "best_net": v["best_net"], "inner_scores": v["inner_scores"], "an_counts": v["an_counts"],
                      "n_rows": v["n_rows"]}
        elif k == "pol_err" or k.startswith("pol_err_k"):
            out[k] = "table"
    return out


def record(line: str) -> None:
    with open(paths.R4 / "RUN_RECORD.md", "a", encoding="utf-8", newline="\n") as fh:
        fh.write(line.rstrip() + "\n")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def seal_tag(own: bool) -> str:
    """Checkpoint key.  Round-4 configurations: the verified stage-0 seal.  R3:<ID> reproductions: a digest of
    round 3's freeze seal and the round-4 wrapper code that builds their rows."""
    if not own:
        h = hashlib.sha256((paths.R3DIR / "manifests" / "freeze_seal.txt").read_text().split()[0].encode())
        for f in (Path(__file__), paths.R4 / "r4" / "registry.py", paths.R4 / "r4" / "paths.py"):
            h.update(paths.sha256_text_lf(f).encode())
        return "r3-" + h.hexdigest()[:10]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import r4_seal
    ok, msg, seal = r4_seal.verify()
    if not ok:
        raise SystemExit(f"stage-0 seal does not verify: {msg}")
    return seal[:12]


def block_c_route(cid: str, tag: str) -> str:
    """Sealed block-C routes (PROTOCOL section 12): S13 needs G-transfer; S14 / S15 / S16 need the decision's
    run_controls; S24 additionally its count gate.  The gate and decision files count only if they carry the current
    verified seal (tag).  Returns the refusal reason, or '' when the run is on route."""
    if cid not in ("S13", "S14", "S15", "S16", "S24"):
        return ""
    gate, dec = paths.RES / "gate_c.json", paths.RES / "blockC_decision.json"
    g = json.loads(gate.read_text()) if gate.exists() else {}
    d = json.loads(dec.read_text()) if dec.exists() else {}
    if g.get("seal") != tag:
        return f"{cid} is off route: results/gate_c.json is missing or was not computed under seal {tag}"
    if cid != "S13" and d.get("seal") != tag:
        return f"{cid} is off route: results/blockC_decision.json is missing or not written under seal {tag}"
    if cid == "S13" and g.get("G_transfer_passes") is not True:
        return "S13 is off route: G-transfer failed or not evaluated (results/gate_c.json)"
    if cid in ("S14", "S15", "S16", "S24") and d.get("run_controls") is not True:
        return f"{cid} is off route: block-C decision has run_controls != true (results/blockC_decision.json)"
    if cid == "S24" and g.get("S24_count_gate_passes") is not True:
        return "S24 is NOT RUN: its label-free count gate failed"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--design", required=True, choices=("BP", "BPG"))
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--technical-smoke", action="store_true", help="one fold, no metrics, nothing written but the log")
    a = ap.parse_args()
    bench = D.load_bench()
    cfg = RG.get(a.config, bench)
    own = not getattr(cfg, "r3", False)
    if a.technical_smoke:
        return smoke(bench, cfg, a.design)
    tag = seal_tag(own)
    why = block_c_route(cfg.cid, tag) if own else ""
    if why:
        record(f"- {now()} REFUSED {cfg.cid} {a.design} key={tag}: {why}")
        raise SystemExit(f"REFUSED: {why}")
    Y = RG.data_matrix(cfg.version, bench)
    parts_dir = paths.PRED / "_parts" / f"{cfg.cid}_{a.design}_{tag}"
    parts_dir.mkdir(parents=True, exist_ok=True)
    out = paths.RES / cfg.cid
    out.mkdir(parents=True, exist_ok=True)
    methods = {**RG.base_methods(bench, cfg), **cfg.methods}
    t0 = time.time()
    started = now()
    record(f"- {started} START {cfg.cid} {a.design} key={tag}")
    rows, fitted, info = [], {}, {}
    for f in folds_for(bench.frame, a.design, seeds=list(paths.DEV_SEEDS)):
        pf = parts_dir / f"{f.seed}_{f.fold}.pkl"
        if pf.exists():
            with open(pf, "rb") as fh:
                part = pickle.load(fh)
        else:
            rec = C.load(f.design, f.seed, f.fold)
            E.assert_test_isolation(bench, rec)
            blk = dict(rec["outer"], Y=Y, policy=cfg.policy, _rec=rec)
            blk["fitted"] = RG.fit(cfg, bench, rec, Y)
            r_ = []
            for cell in RG.cells_of(cfg, blk):
                r_ += H.cell_rows(rec, blk, cell, methods, policy=cfg.policy, ks=cfg.ks, Y=Y, version=cfg.version)
            part = {"rows": r_, "fitted": blk["fitted"], "info": fitted_info(blk["fitted"])}
            with open(pf.with_suffix(".tmp"), "wb") as fh:
                pickle.dump(part, fh)
            pf.with_suffix(".tmp").replace(pf)
        rows += part["rows"]
        fitted[(f.seed, f.fold)] = part["fitted"]
        info[f"{f.seed}_{f.fold}"] = part["info"]
        print(f"  {cfg.cid} {a.design} {f.seed} fold {f.fold}: {len(rows)} rows ({time.time() - t0:.0f}s)", flush=True)
    t = pd.DataFrame(rows)
    final = paths.PRED / f"{cfg.cid}_{a.design}.parquet"
    final.unlink(missing_ok=True)                       # A4: the predictions appear only after the checks below pass
    pending = final.with_name(final.name + ".pending")
    t.to_parquet(pending, index=False, compression="zstd")
    (out / f"fitted_{a.design}.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    arms = list(methods)
    kprim = 3 if 3 in cfg.ks else max(cfg.ks)
    def factory(b, cid=a.config):              # the scrambled pass rebuilds every method from the scrambled bench
        c2 = RG.get(cid, b)
        return {**RG.base_methods(b, c2), **c2.methods}
    iv = H.intervention(bench, a.design, methods, fitted, policy=cfg.policy, ks=cfg.ks, Y=Y, methods_factory=factory)
    iv["methods_check"] = ("scrambled pass: block Y, observe() input and every method rebuilt by registry.get on a bench "
                           "copy whose Y has every unread held-out value replaced; frame label columns / coef not scrambled")
    (out / f"intervention_{a.design}.json").write_text(json.dumps(iv, indent=1), encoding="utf-8")
    print("  intervention:", iv, flush=True)
    if not iv["passed"]:
        record(f"- {now()} {cfg.cid} {a.design} key={tag}: INTERVENTION FAILED {iv}")
        raise SystemExit("INTERVENTION FAILED")
    if own and getattr(cfg, "slot_cols", None):
        ident = slot_identity_check(t, info, cfg.slot_cols)
        (out / f"identity_{a.design}.json").write_text(json.dumps(ident, indent=1), encoding="utf-8")
        print("  slot identity:", {k: v for k, v in ident.items() if k != "folds"}, flush=True)
        if not ident["passed"]:
            record(f"- {now()} {cfg.cid} {a.design} key={tag}: SLOT IDENTITY CHECK FAILED")
            raise SystemExit("SLOT IDENTITY CHECK FAILED")
    if own and cfg.cid == "REF":
        ident = identity_check(t, info)
        (out / f"identity_{a.design}.json").write_text(json.dumps(ident, indent=1), encoding="utf-8")
        print("  identity:", {k: v for k, v in ident.items() if k != "folds"}, flush=True)
        if not ident["passed"]:
            record(f"- {now()} {cfg.cid} {a.design} key={tag}: IDENTITY CHECK FAILED")
            raise SystemExit("IDENTITY CHECK FAILED")
    pending.replace(final)
    tc = D4.add_donor(M4.add_clusters(t, bench.frame))
    M4.summary(tc, arms, K=cfg.K, k=kprim).to_csv(out / f"summary_{a.design}.csv", index=False)
    M4.per_publication(tc, arms, K=cfg.K, k=kprim).to_csv(out / f"per_publication_{a.design}.csv", index=False)
    M4.levels(tc, arms, K=cfg.K, k=kprim, replicates=a.replicates).to_csv(out / f"levels_{a.design}.csv", index=False)
    MT.per_pair(tc, arms, K=cfg.K, k=kprim).to_csv(out / f"per_pair_{a.design}.csv", index=False)
    M4.calibration_diag(tc, arms, K=cfg.K, k=kprim).to_csv(out / f"calibration_{a.design}.csv", index=False)
    (out / f"clusters_{a.design}.json").write_text(json.dumps(M4.cluster_counts(tc, K=cfg.K, k=kprim), indent=1), encoding="utf-8")
    pairs = list(cfg.contrasts) if own else [(c, r) for c in cfg.primary for r in ("PWL_R",) if c != r]
    if pairs:
        M4.contrasts(tc, pairs, K=cfg.K, k=kprim, replicates=a.replicates).to_csv(out / f"contrasts_{a.design}.csv", index=False)
    show = [x for x in (RG.SHOW if own else list(cfg.primary) + ["PWL_R", "PWL_Z", "HBK@ML"]) if x in arms]
    s = M4.summary(tc, show, K=cfg.K, k=kprim)
    s = s[s.subset == "all"]
    pd.set_option("display.width", 220)
    print(s[["arm", "R2", "MAE", "MAE_balanced", "RMSE", "R2_min_seed", "n_rows_per_seed", "n_cells_per_seed"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    secs = time.time() - t0
    record(f"- {now()} END {cfg.cid} {a.design} key={tag}: {len(t)} rows, intervention passed "
           f"({iv['n_predictions']} predictions), {secs:.0f}s")
    print(f"EXIT 0 ({secs:.0f}s)", flush=True)
    return 0


def identity_check(t: pd.DataFrame, info: dict) -> dict:
    """Per outer fold: B1 must equal round 3's R14 column exactly when its lambda is finite and the fold has no
    orphan training cells, and must equal B0 (PWL_R) exactly when lambda = infinity.  Folds with orphans (the
    round-4 orphan rule) are reported, not required to match R14."""
    folds, ok = [], True
    for (s, f), g in t.groupby(["split_seed", "fold"]):
        fi = info[f"{s}_{f}"]["ref:B1"]
        lam, orph = fi["lambda"], fi["coverage"]["orphan_cells"]
        d_r14 = float((g["B1"] - g["R14_REPRO"]).abs().max())
        d_b0 = float((g["B1"] - g["B0"]).abs().max())
        if lam == "inf":
            req = d_b0 == 0.0
        elif orph == 0:
            req = d_r14 == 0.0
        else:
            req = True
        ok &= bool(req)
        folds.append({"seed": int(s), "fold": int(f), "lambda": lam, "orphans": orph, "max_abs_B1_minus_R14": d_r14,
                      "max_abs_B1_minus_B0": d_b0, "passed": bool(req)})
    return {"passed": bool(ok), "n_folds": len(folds), "n_inf": sum(x["lambda"] == "inf" for x in folds),
            "n_orphan_folds": sum(x["orphans"] > 0 for x in folds), "folds": folds}


def slot_identity_check(t: pd.DataFrame, info: dict, cols: dict) -> dict:
    """Per outer fold and slot column: when the slot's lambda = infinity (block A) or its tree count = 0 (block B,
    recorded as lambda 'inf') its predictions must equal B1 exactly.  Block C (A4): when an auxiliary arm's outer choice
    is w_X = 0 with S11's choice being the same network, its predictions must equal the S11 column exactly."""
    folds, ok = [], True
    s11_col = next((c for c, p in cols.items() if p == "c:S11"), None)
    for (s, f), g in t.groupby(["split_seed", "fold"]):
        for col, part in cols.items():
            fi = info[f"{s}_{f}"]
            lam = fi[part]["lambda"]
            d = float((g[col] - g["B1"]).abs().max())
            req = (d == 0.0) if lam == "inf" else True
            rec = {"seed": int(s), "fold": int(f), "column": col, "lambda": lam, "max_abs_minus_B1": d}
            ch = fi[part].get("choice")
            if part.startswith("c:") and part != "c:S11" and s11_col is not None and isinstance(ch, dict) \
                    and ch.get("w_X") == 0.0 and fi["c:S11"]["choice"] == ch:
                d11 = float((g[col] - g[s11_col]).abs().max())
                rec["max_abs_minus_S11"] = d11
                req = req and d11 == 0.0
            ok &= bool(req)
            folds.append({**rec, "passed": bool(req)})
    fr = pd.DataFrame(folds)
    return {"passed": bool(ok), "n_checks": len(folds),
            "lambda_inf_by_column": fr.assign(inf=fr["lambda"] == "inf").groupby("column").inf.sum().astype(int).to_dict(),
            "folds": folds}


def smoke(bench, cfg, design) -> int:
    """Technical smoke: one fold, fits and predictions run, shapes / NaN patterns / support exactness checked;
    no target value is compared with any prediction and nothing is written."""
    f = folds_for(bench.frame, design, seeds=list(paths.DEV_SEEDS))[0]
    rec = C.load(f.design, f.seed, f.fold)
    E.assert_test_isolation(bench, rec)
    blk = dict(rec["outer"], Y=bench.Y, policy=cfg.policy, _rec=rec)
    blk["fitted"] = RG.fit(cfg, bench, rec, bench.Y)
    methods = {**RG.base_methods(bench, cfg), **cfg.methods}
    n = 0
    for cell in RG.cells_of(cfg, blk):
        for k in cfg.ks:
            seq = H.seq_for(blk, cell, cfg.policy, k)
            if k > len(seq):
                continue
            obs = E.observe(bench.Y[cell["ci"]], cell, seq, k)
            for name, fn in methods.items():
                v = np.asarray(fn(blk, cell, obs, k), float)
                assert v.shape == (14,), name
                for m in obs.read:
                    if name not in ("RIDGE2", "LINE", "HBK@ML", "G14@ZERO", "FLAT"):
                        assert v[m] == obs.y_masked[m], (name, "support not exact")
                n += 1
    print(f"technical smoke OK: {n} level vectors, fitted parts {sorted(blk['fitted'])}; "
          f"{json.dumps(fitted_info(blk['fitted']), default=str)[:600]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
