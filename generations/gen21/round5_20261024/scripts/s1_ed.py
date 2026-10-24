#!/usr/bin/env python
"""Stage-1 item 1.6 (CLAUDE_CODE_PROMPT.md 6.6; 6.0, 6.10, 8, 9): E-D read-budget ladder (M18), the 5th-read policy
envelope (M17) and the recipe-2 row.  EXPLORATORY (seen) throughout; the 4-read rung is E-A itself and every other
rung is a different read budget.  Frozen arm list, report rules and seeds: results/ED/ITEM_SPEC.md (--write-spec).

Order (brief 1.2): Stage-0 seal -> item code seal -> --repro (TECHNICAL SMOKE) -> the arms (rows, candidates, fits and
checks; exit 3 until the fit-poisoning check has passed) -> scripts/s1_ed_poison.py -> --from-rows (the tables, the
summary and ITEM_TABLE.md, written only when results/leakage/poison_ED_beta.json carries the current seal, every k,
50 fold-fits, all identical and matching the main run; the stored rows must carry the seal and the code digests).
Without --smoke the script refuses to run unless r5.sealcheck.require_seal() verifies, records START / END lines and
ends with the post-run seal check.  --smoke runs the construction path on real data for a few folds (identity and
leakage counters on real data), replaces y and every y-derived column by seeded noise before any table runs, prints
shapes, counts, NaN patterns and timings only, and writes nothing; --no-intervention / --no-lopo are smoke-only.

Writes (after the seal) under results/ED/: repro_<design>.json; rows_<design>.parquet (row-level LAD / ENV / REC
outputs), cand_<design>.parquet (ENV candidate SSEs), fits_<design>.csv (beta and lambda per fold and budget),
checks_<design>.json (leakage / support counters), one CSV per table (<table>_<design>.csv, every number labelled),
summary_<design>.json, loco_selection.csv, and ITEM_TABLE.md (the section-9.3 per-item table) when both designs ran.

    set PYTHONIOENCODING=utf-8
    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ed.py --smoke --design BPG
    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ed.py --repro --design both
    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ed.py --design both
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r5 import paths  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r5 import common as CM  # noqa: E402
from r5 import ed as ED  # noqa: E402
from gen20k import data as D  # noqa: E402

OUT = ED.OUT
ITEM = "ED"
CODE = ("r5/ed.py", "scripts/s1_ed.py", "scripts/s1_ed_poison.py", "r5/common.py", "r5/paths.py")


def code_digests() -> dict:
    return {c: paths.sha256_text_lf(paths.R5 / c) for c in CODE}


def smoke_surrogate(R: pd.DataFrame, cand: pd.DataFrame, design: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PRE-SEAL ENGINEERING CHECK (s0-mechanics-22): y and every y-derived column replaced by seeded noise before any table runs
    (p_* where finite, se_* = (p - y)^2, ae_* = |p - y|; expectation rows without p_ get noise se / ae; candidate SSEs
    noise), so no real-label metric is computed in memory."""
    rng = np.random.default_rng(paths.seed_for(f"ED|SMOKE_SURROGATE|{design}"))
    S = R.copy()
    ys = rng.normal(0.0, 0.3, len(S))
    S["y"] = ys
    for a in ED.ARMS:
        pa = S["p_" + a].to_numpy(float)
        ps = np.where(np.isfinite(pa), ys + rng.normal(0.0, 0.2, len(S)), np.nan)
        e = rng.normal(0.0, 0.2, len(S))
        S["p_" + a] = ps
        S["se_" + a] = np.where(np.isfinite(ps), (ps - ys) ** 2, np.where(np.isfinite(S["se_" + a].to_numpy(float)), e ** 2, np.nan))
        S["ae_" + a] = np.where(np.isfinite(ps), np.abs(ps - ys), np.where(np.isfinite(S["ae_" + a].to_numpy(float)), np.abs(e), np.nan))
    C = cand.copy()
    for c in [c for c in C.columns if c.startswith("sse_")]:
        C[c] = (rng.normal(0.0, 0.2, len(C)) ** 2) * C.n_q.to_numpy(float)
    return S, C


def poison_gate(tag: str) -> tuple[bool, str]:
    """s0-mechanics-13: the item's tables are written only after the 50-fold-fit poisoning check of the k-specific
    beta refits passed under the current seal for every k of ED.BETA_KS."""
    pf = ED.LEAK / "poison_ED_beta.json"
    if not pf.exists():
        return False, "results/leakage/poison_ED_beta.json missing (run scripts/s1_ed_poison.py)"
    q = json.loads(pf.read_bytes())
    ok = (q.get("seal") == tag and list(q.get("ks", [])) == list(ED.BETA_KS) and q.get("n_folds") == 50
          and q.get("all_identical") is True and q.get("matches_main_run") is True)
    return ok, ("passed" if ok else f"not valid for seal {tag} (seal {q.get('seal')}, ks {q.get('ks')}, folds "
                f"{q.get('n_folds')}, identical {q.get('all_identical')}, matches {q.get('matches_main_run')})")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _shape_report(name: str, df: pd.DataFrame) -> str:
    """Shapes and NaN pattern only (smoke): no value is printed."""
    if not isinstance(df, pd.DataFrame):
        return f"    {name}: {type(df).__name__}"
    num = df.select_dtypes("number")
    nan_cols = sorted(num.columns[num.isna().any()]) if len(num.columns) else []
    return (f"    {name}: {df.shape[0]} rows x {df.shape[1]} cols; numeric cols with NaN {len(nan_cols)}/{len(num.columns)}"
            + (f" {nan_cols}" if nan_cols else ""))


def _write_csv(df: pd.DataFrame, p: Path, written: list) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(df.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    written.append(p)


def _write_json(obj, p: Path, written: list) -> None:
    paths.write_text_lf(p, json.dumps(obj, indent=1, default=str) + "\n")
    written.append(p)


def cluster_map() -> dict:
    """cell_id -> cluster_cpg on the full C3 row set (round 4's 13 clusters); identical under BPG and BP."""
    m = {}
    for d in paths.DESIGNS:
        t = CM.load_rows(d)
        mm = t.groupby("cell_id").cluster_cpg.first().astype(str).to_dict()
        if m and mm != m:
            raise AssertionError("cluster_cpg differs between designs")
        m = mm
    return m


def pareto_outcome(pareto: pd.DataFrame, frame: str = "C3pop", var: str = "all") -> dict:
    """REGISTRY 1.6-R01 ('does any member ...') over the written label-free family only (ED.ENV_LABEL_FREE; DOPTQ is
    the reference); ORACLE_B1 [O] is reported as a bound (never a member) and the LOOSE rule as a sensitivity."""
    p = pareto[(pareto.frame == frame) & (pareto.variant == var)]
    if "role" in p.columns:
        p = p[~p.role.astype(str).str.startswith("answer")]
    dom = p.pareto_dominates.astype(bool)
    fam, orc, loose = (p.policy.isin(ED.ENV_LABEL_FREE), p.policy == "ORACLE_B1",
                       p.policy.astype(str).str.endswith("_LOOSE"))
    return {"family": [x for x in ED.ENV_LABEL_FREE if x != "DOPTQ"],
            "family_any_dominates": bool(dom[fam].any()) if fam.any() else None,
            "family_dominating": sorted(p[fam & dom].policy),
            "oracle_bound_dominates": bool(dom[orc].any()) if orc.any() else None,
            "loose_sensitivity_dominating": sorted(p[loose & dom].policy)}


def summary(T: dict, design: str, checks: dict, repro: dict, seconds: float) -> dict:
    """Key numbers of one design (post-seal only)."""
    lv, ct = T["ladder_levels"], T["ladder_contrasts"]

    def lev(frame, arm="B1", var="all"):
        r = lv[(lv.frame == frame) & (lv.arm == arm) & (lv.variant == var)]
        cols = [c for c in ("row", "R2", "per_seed", "seeds_ge_050", "P_boot_ge_050", "loco_min", "loco_max",
                            "relSSE_vs_B1_4reads", "level_note", "label") if c in lv.columns]
        return r.iloc[0][cols].to_dict() if len(r) else None
    pareto = T["envelope_pareto"]
    po = {v: pareto_outcome(pareto, "C3pop", v) for v in sorted(set(pareto.variant))}
    pa = po.get("all", {})
    sat = T["saturation_check"]
    c3 = lv[(lv.frame == "C3pop|r5") & (lv.arm == "B1") & (lv.variant == "all")]
    c4 = lv[(lv.frame == "C4|r5") & (lv.arm == "B1") & (lv.variant == "all")]
    mix = {"row": ED.row_name("C3pop|r5"), "rows_per_seed": float(c3.rows_per_seed.iloc[0]) if len(c3) else None,
           "rows_per_seed_keep_4_reads": float(c3.rows_fewer_reads_per_seed.iloc[0])
           if len(c3) and "rows_fewer_reads_per_seed" in c3 else None,
           "R2_B1_rows_keep_4_reads": float(c3.R2_rows_fewer_reads.iloc[0])
           if len(c3) and "R2_rows_fewer_reads" in c3 else None,
           "C4_r5_B1_like_for_like": float(c4.R2.iloc[0]) if len(c4) else None,
           "label": ED.SEEN + "; different read budget (the C3 5-read row is a mixture; C4 is the like-for-like "
                              "5-metal number)"}
    return {"label": ED.SEEN + " (E-D; the 4-read rung is E-A, 5-7 reads are a different read budget)",
            "design": design, "created_utc": _now(), "seconds": seconds, "repro": repro, "checks": checks,
            "c3_row": ED.C3_ROW, "c3_r5_mixture": mix,
            "ladder_B1": {f: lev(f) for f in sorted(set(lv.frame))},
            "ladder_B1_noC5": {f: lev(f, var="noC5") for f in sorted(set(lv.frame))},
            "rung_contrasts_B1": ct[(ct.variant == "all") & ct.frame.str.contains(": r")][
                [c for c in ("frame", "row", "delta_R2", "seeds_pos", "relSSE", "boot_lo", "boot_hi", "MDE80", "loco_min",
                             "loco_max", "label") if c in ct.columns]]
            .to_dict("records"),
            "envelope_width": T["envelope_width"].to_dict("records"),
            "pareto_question": ("1.6-R01 (descriptive): does any member of the written label-free family keep every "
                                f"block above {ED.PARETO_FLOOR} and improve one block with a cluster lower bound > 0 "
                                "(vs DOPTQ, C3 min(4, k_max), 5 reads)?  ORACLE_B1 [O] is a bound, never a member; "
                                "LOOSE is a sensitivity"),
            "pareto_family": pa.get("family", []),
            "pareto_any_dominates_C3pop": pa.get("family_any_dominates"),
            "pareto_dominating_policies": pa.get("family_dominating", []),
            "pareto_oracle_bound_dominates_C3pop": pa.get("oracle_bound_dominates"),
            "pareto_loose_sensitivity_dominating_C3pop": pa.get("loose_sensitivity_dominating", []),
            "pareto_by_variant_C3pop": po,
            "saturation_check": sat.to_dict("records"),
            "random_mc": T["envelope_random_mc"].to_dict("records"),
            "recipe_primary": T["recipe_contrasts"][T["recipe_contrasts"].primary.astype(bool)][
                ["frame", "arm", "variant", "delta_R2", "seeds_pos", "boot_lo", "boot_hi", "label"]].to_dict("records")}


def pass_seconds(tag: str) -> dict:
    """Wall-clock seconds of this item's completed sealed passes under ``tag``, from the START / END pairs that
    r5.sealcheck.run_sealed writes to RUN_RECORD.md: {'repro': s, 'run': s, 'from-rows': s} (the last completed pass
    per mode) and 'poison': [s, ...] (every s1_ed_poison.py pass)."""
    p = Path(CM.RUN_RECORD)
    if not p.exists():
        return {}
    start, out = {}, {}
    for ln in p.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^- (\S+) (START|END) (ED|ED_POISON) (.*)$", ln.rstrip())
        if not m:
            continue
        try:
            t = dt.datetime.fromisoformat(m.group(1))
        except ValueError:
            continue
        ev, it, rest = m.group(2), m.group(3), m.group(4)
        if ev == "START":
            ms = re.match(rf"^(.+) seal={re.escape(tag)}$", rest)
            if ms:
                start[(it, ms.group(1))] = t
            continue
        if not rest.startswith(f"seal={tag} "):
            continue
        for (it0, mode), t0 in list(start.items()):
            if it0 == it and f" {mode} status=" in rest:
                s = (t - t0).total_seconds()
                if it == "ED_POISON":
                    out.setdefault("poison", []).append(s)
                else:
                    out[mode] = s
                del start[(it0, mode)]
                break
    return out


def compute_line(S: dict, compute_s: float, tag: str, mode: str) -> str:
    """The compute field: this pass (timed here) and the item's earlier sealed passes under the same seal."""
    ps = pass_seconds(tag)
    this = f"{mode} pass {compute_s:.0f} s ({', '.join(f'{d} {S[d]['seconds']:.0f} s' for d in S)})"
    parts, tot = [], compute_s
    for key, name in (("repro", "reproduction (TECHNICAL SMOKE)"), ("run", "arms (rows, candidates, fits, checks)"),
                      ("poison", "fit poisoning of the k-specific beta refits")):
        if key == mode:
            continue
        v = ps.get(key)
        if v is None:
            parts.append(f"{name}: no completed pass under seal {tag}")
            continue
        s = float(sum(v)) if isinstance(v, list) else float(v)
        tot += s
        parts.append(f"{name} {s:.0f} s" + (f" ({len(v)} pass{'es' if len(v) > 1 else ''})" if isinstance(v, list) else ""))
    return f"{this}; earlier passes under seal {tag} (RUN_RECORD.md): " + ", ".join(parts) + f"; total {tot:.0f} s"


def next_registered(item_id: str = ED.ITEM_ID) -> str:
    """The next registry item after ``item_id`` (REGISTRY.csv rows of kind 'item'), read at writing time."""
    try:
        r = pd.read_csv(paths.R5 / "REGISTRY.csv", dtype=str, keep_default_na=False)
        it = r[(r.kind == "item") & r.id.str.fullmatch(r"\d+\.\d+")]
        key = lambda s: tuple(int(x) for x in s.split("."))          # noqa: E731
        nxt = sorted((x for _, x in it.iterrows() if key(x.id) > key(item_id)), key=lambda x: key(x.id))
    except (OSError, ValueError, KeyError, AttributeError):
        nxt = []
    if not nxt:
        return "n/a (no later registry item found in REGISTRY.csv)"
    x = nxt[0]
    return f"Registry item {x.id} ({x.arm}; {x.estimand}; registry status {x.status})"


def _designs(flags: dict) -> str:
    """'BPG and BP' / 'BPG only' / 'neither design' for the designs whose flag is true."""
    on = [d for d, v in flags.items() if v]
    return " and ".join(on) + (" only" if 0 < len(on) < len(flags) else "") if on else "neither design"


def changed_text(S: dict, T: dict) -> str:
    """'What changed scientifically', computed: the paired 5th read on the named C3 row and the like-for-like C4 level,
    the 1.6-R01 Pareto answer over the written family (the [O] bound and LOOSE beside it) and the 1.6-R02 outcome."""
    d54 = {d: next((r["delta_R2"] for r in S[d]["rung_contrasts_B1"] if r["frame"] == "C3pop: r5 - r4"), float("nan"))
           for d in S}
    mix = {d: S[d]["c3_r5_mixture"] for d in S}
    fam = {d: S[d]["pareto_dominating_policies"] for d in S}
    orc = {d: S[d]["pareto_oracle_bound_dominates_C3pop"] for d in S}
    loose = {d: S[d]["pareto_loose_sensitivity_dominating_C3pop"] for d in S}
    s1 = (f"Paired on E-A's identical queries of the row '{ED.C3_ROW}', the DOPT_Q-nested 5th read changes B1's pooled "
          "R2 by " + ", ".join(f"{d} {v:+.4f}" for d, v in d54.items())
          + " (a different read budget, EXPLORATORY seen, never an E-A value; at 5 reads "
          + ", ".join(f"{d} {m['rows_per_seed_keep_4_reads']:.0f} of {m['rows_per_seed']:,.0f}" for d, m in mix.items())
          + " rows per seed keep 4 reads), and the like-for-like C4 5-metal level is "
          + ", ".join(f"{d} {m['C4_r5_B1_like_for_like']:.4f}" for d, m in mix.items()) + ".")
    fam_txt = ("no member of the written label-free family Pareto-dominates DOPTQ ("
               + "; ".join(f"{d}: none" for d in S) + ")") if not any(fam.values()) else \
        ("a member of the written label-free family Pareto-dominates DOPTQ ("
         + "; ".join(f"{d}: {', '.join(v) if v else 'none'}" for d, v in fam.items()) + ")")
    s2 = (f"The pre-declared Pareto question (1.6-R01; vs DOPTQ, C3 min(4, k_max), 5 reads): {fam_txt}; the ORACLE_B1 "
          f"[O] bound (never a member) meets the criterion in {_designs(orc)}; LOOSE sensitivity: "
          + "; ".join(f"{d}: {', '.join(v) if v else 'none dominates'}" for d, v in loose.items()) + ".")
    sat = {}
    for d in S:
        sc = T[d]["saturation_check"]
        sc = sc[sc.variant == "all"]
        st = T[d]["saturation"]
        st = st[st.variant == "all"].set_index("frame")
        if len(sc):
            r = sc.iloc[0]
            low = [f for f in str(r.strata_low).split(",") if f and f != "nan"]
            sat[d] = (bool(r.rule_fires), ", ".join(f"{f} {st.R2.get(f, float('nan')):.4f}" for f in low), float(r.spread))
    s3 = ("The saturation density-invariance rule (1.6-R02; CONSISTENCY CHECK, not a kill) fires in "
          + _designs({d: v[0] for d, v in sat.items()}) + " ("
          + "; ".join(f"{d}: strata below {ED.SAT_FLOOR} with >= {ED.SAT_MIN_CLUSTERS} clusters "
                      f"{v[1] or 'none'}, spread {v[2]:.4f}" for d, v in sat.items()) + ").")
    return " ".join((s1, s2, s3))


def report_rule_sections(S: dict, T: dict) -> str:
    """Report-rule outcomes appended below the section-9.3 table: the 1.6-R01 answer rows (family, [O] bound, LOOSE),
    the 1.6-R02 saturation strata and rule, and the M17 envelope width with its RANDOM note."""
    pa = pd.concat([T[d]["envelope_pareto"] for d in S], ignore_index=True)
    ans = pa[pa.role.astype(str).str.startswith("answer") & (pa.frame == "C3pop")] if "role" in pa.columns else pa.iloc[:0]
    ans = ans[[c for c in ("design", "variant", "policy", "role", "pareto_dominates", "dominating_policies",
                           "keeps_every_block_above_floor_policies", "policies", "label") if c in ans.columns]]
    sa = pd.concat([T[d]["saturation"] for d in S], ignore_index=True)
    sa = sa[sa.variant == "all"][[c for c in ("design", "frame", "R2", "R2_B0", "MSE", "rows_per_seed", "cells", "clusters",
                                              "level_note", "label") if c in sa.columns]]
    sc = pd.concat([T[d]["saturation_check"] for d in S], ignore_index=True)
    sc = sc[[c for c in ("design", "variant", "rule", "strata", "strata_low", "spread", "rule_fires", "strata_values", "label")
             if c in sc.columns]]
    ew = pd.concat([T[d]["envelope_width"] for d in S], ignore_index=True)
    note = str(ew.note.dropna().iloc[0]) if "note" in ew.columns and ew.note.notna().any() else ""
    if note and "envelope_with_RANDOM_MC_mean" in ew.columns:
        w = ew[(ew.frame == "C3pop") & (ew.variant == "all")]
        note += " (C3 min(4, k_max), all rows: " + "; ".join(
            f"{r.design} exact {r.envelope:.6f}, with the Monte Carlo mean {r.envelope_with_RANDOM_MC_mean:.6f}, seen "
            f"{r.seen_envelope:.4f}" for r in w.itertuples()) + ")"
    ew = ew[[c for c in ("design", "frame", "variant", "envelope", "lowest", "highest", "envelope_with_RANDOM_MC_mean",
                         "seen_envelope", "per_seed_span_min", "per_seed_span_max", "label") if c in ew.columns]]
    out = ["", "## Report-rule outcomes", "",
           "### 1.6-R01: the pre-declared Pareto question (C3 min(4, k_max), 5 reads, vs DOPTQ)", "",
           "Evaluated over the written label-free family only (" + ", ".join(x for x in ED.ENV_LABEL_FREE if x != "DOPTQ")
           + "; DOPTQ is the reference).  ORACLE_B1 [O] is a bound on its own row, never a member; LOOSE legality is a "
             "labelled sensitivity on its own row.  Per-policy rows: envelope_pareto_<design>.csv.", "",
           CM.md_table(ans), "",
           "### 1.6-R02: density invariance at saturation (CONSISTENCY CHECK, not a kill)", "", CM.md_table(sa), "",
           CM.md_table(sc), "",
           "### M17 envelope width over the label-free family (5 reads)", "", CM.md_table(ew), ""]
    if note:
        out += [f"Note: {note}.", ""]
    return "\n".join(out)


def item_table(S: dict, T: dict, compute_s: float, *, tag: str = "SMOKE", mode: str = "from-rows",
               gate_ok: bool = True, gate_msg: str = "") -> str:
    """The section-9.3 per-item table over both designs (written only after both ran), with the report-rule outcomes
    below it."""
    lv = pd.concat([T[d]["ladder_levels"] for d in S], ignore_index=True)
    main = ["C3pop|r2", "C3pop|r3", "C3pop|r4", "C3pop|r5", "C4|r4", "C4|r5", "C5|r6", "C6|r7"]
    for c, v in (("level_note", ""), ("row", ""), ("relSSE_vs_B1_4reads", np.nan)):
        if c not in lv.columns:
            lv = lv.assign(**{c: v})
    met = lv[lv.frame.isin(main) & (lv.arm == "B1") & (lv.variant == "all")][
        ["design", "frame", "row", "rung_estimand", "level_note", "R2", "MSE", "MAE", "relSSE_vs_B1_4reads", "seeds_ge_050",
         "P_boot_ge_050", "loco_min", "loco_max", "lopo_min", "lopo_max", "label"]].rename(
        columns={"P_boot_ge_050": "P_boot_ge_050 (descriptive; never a posterior)"})
    c5 = lv[lv.frame.isin(main) & (lv.arm == "B1")].pivot_table(index=["design", "frame"], columns="variant", values="R2",
                                                                 dropna=False)
    c5.columns = [f"R2_{c}" for c in c5.columns]
    c5 = c5.reset_index()
    c5 = c5.sort_values("design", key=lambda s: s.map({d: i for i, d in enumerate(S)}), kind="stable").reset_index(drop=True)
    c5.insert(2, "row", c5.frame.map(ED.row_name))
    c5["level_note"] = np.where(c5.frame == ED.E_A_RUNG, ED.EA_NOTE, "")
    c5["label"] = [ED.join_labels(ED.SEEN, "C5-BTBP out = SENSITIVITY", "SENSITIVITY (design BP)" if d == "BP" else "",
                                  ED.budget_tag(ED.reads_of(f)))
                   for d, f in zip(c5.design, c5.frame)]
    ut = pd.concat([T[d]["unit_tests"] for d in S], ignore_index=True)
    ut = ut[ut.variant == "all"]
    ct = pd.concat([T[d]["ladder_contrasts"] for d in S] + [T[d]["envelope_contrasts"] for d in S]
                   + [T[d]["recipe_contrasts"] for d in S], ignore_index=True)
    if "row" not in ct.columns:
        ct = ct.assign(row="")
    keyc = ct[(ct.variant == "all") & ct.frame.isin(["C3pop: r5 - r4", "C4: r5 - r4", "C6: r7 - r4",
                                                      "C3pop: BISECT_R - DOPTQ", "C3pop: RANDOM - DOPTQ",
                                                      "C3pop: BISECT_R - RANDOM", "r2: LADDER - HULL", "r4: LADDER - HULL"])]
    lt = keyc[["design", "frame", "row", "delta_R2", "relSSE", "seeds_pos", "MDE80", "lopo_min", "lopo_max", "lopo_min_by",
               "lopo_max_by", "units_WLT", "top3_unit_share", "label"]]
    dup = pd.concat([T[d]["duplicates"] for d in S], ignore_index=True)
    chk = {d: S[d]["checks"] for d in S}
    poison = ED.LEAK / "poison_ED_beta.json"
    if poison.exists():
        q = json.loads(poison.read_bytes())
        pst = (f"all_identical {q.get('all_identical')}, matches_main_run {q.get('matches_main_run')}, ks {q.get('ks')}, "
               f"{q.get('n_folds')} fold-fits, seal {q.get('seal')} (gate: {gate_msg or 'n/a'})")
    else:
        pst = "PENDING (scripts/s1_ed_poison.py)"
    lp = ED.LEAK / "poison_ED_loco.json"
    pst = f"{pst}; LOCO static selection poisoning: " + (str(json.loads(lp.read_bytes()).get("passed")) if lp.exists()
                                                          else "PENDING")
    leak = "; ".join(f"{d}: isolation {c['isolation_ok']}/{c['folds']} folds, intervention failed {c['intervention_failed']}"
                     f"/{c['intervention_checked']}, peek caught {c['peek_control_caught']}/{c['peek_control_runs']}, "
                     f"both-ends-read {c['both_ends_read']}, illegal {c['illegal_or_unnested']}, DOPT 5th read not legal "
                     f"{c['dopt5_not_legal']}, static-set intervention failed {c['static_sets_intervention_failed']}/"
                     f"{c['static_sets_intervention_checked']}" for d, c in chk.items()) \
        + f"; fit poisoning of the k-specific beta refits: {pst}"
    b = {d: S[d]["ladder_B1"] for d in S}
    mix = {d: S[d]["c3_r5_mixture"] for d in S}
    per_seed = {}
    for d in S:
        per_seed[f"{d} {ED.row_name('C3pop|r5')}, B1, DOPT_Q-nested ({ED.RB})"] = (b[d].get("C3pop|r5") or {}).get("per_seed")
        per_seed[f"{d} {ED.row_name('C4|r5')}, B1, like-for-like 5-metal ({ED.RB})"] = (b[d].get("C4|r5") or {}).get("per_seed")
    budget = ("2-7 reads (code k = 1-6); 4 reads = the E-A rung; every other rung (2, 3, 5-7 reads) = a different read "
              f"budget.  The row '{ED.C3_ROW}' at 5 reads is a mixture: "
              + "; ".join(f"{d} {m['rows_per_seed_keep_4_reads']:.0f} of {m['rows_per_seed']:,.0f} rows per seed (the "
                          f"k_max = 3 cells) keep 4 reads, B1 R2 {m['R2_B1_rows_keep_4_reads']:.4f} on them"
                          for d, m in mix.items())
              + "; C4 (k_max >= 4) is the like-for-like 5-metal number (B1 at 5 reads "
              + ", ".join(f"{d} {m['C4_r5_B1_like_for_like']:.4f}" for d, m in mix.items()) + ")")
    item = {"item": "1.6 E-D: M18 read-budget ladder + M17 policy envelope + recipe-2 row",
            "population": {"row": f"{ED.C3_ROW} (E-A's queries)", **CM.population(CM.load_rows("BPG"))},
            "design": "BPG (primary) and BP (labelled sensitivity); identical test folds",
            "read_budget": budget,
            "arms": "B1 (beta refit per fold and budget), B0 (beta = 0), HIST_D7 (DIAGNOSTIC), B1_k3beta (SENSITIVITY); "
                    "5th-read policies: the written label-free family " + ", ".join(ED.ENV_LABEL_FREE)
                    + "; ORACLE_B1 [O] (bound only, never a member); LOOSE sensitivity MAXMIN_R_LOOSE, RANDOM_LOOSE; "
                      "recipe-2 " + ", ".join(ED.REC_POLICIES),
            "metrics": met, "per_seed": per_seed,
            "units": ut, "lopo_top3": lt, "c5": c5, "duplicates": dup, "leakage": leak,
            "compute": compute_line(S, compute_s, tag, mode),
            "label": ED.SEEN + "; reproduction run TECHNICAL SMOKE; BP and C5-BTBP out SENSITIVITY",
            "changed": changed_text(S, T),
            "next_step": next_registered() if gate_ok else
            f"scripts/s1_ed_poison.py, then scripts/s1_ed.py --from-rows (fit poisoning: {gate_msg})"}
    return CM.per_item_table(item, title="Item 1.6 (E-D): per-item table") + report_rule_sections(S, T)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--design", choices=("BPG", "BP", "both"), default="both")
    ap.add_argument("--smoke", action="store_true", help="technical smoke: metrics suppressed, nothing written")
    ap.add_argument("--smoke-folds", type=int, default=2, help="folds per design in --smoke (default 2)")
    ap.add_argument("--repro", action="store_true", help="only the reproduction targets of brief 6.0 (TECHNICAL SMOKE)")
    ap.add_argument("--from-rows", action="store_true", help="re-run the analysis from results/ED/rows_<design>.parquet")
    ap.add_argument("--no-intervention", action="store_true", help="skip the prediction-intervention re-runs (--smoke only)")
    ap.add_argument("--no-lopo", action="store_true", help="skip leave-one-publication-out (--smoke only)")
    ap.add_argument("--write-spec", action="store_true", help="write results/ED/ITEM_SPEC.md and exit (no data read)")
    ap.add_argument("--amend", action="store_true", help="--write-spec after the seal (an amendment and a reseal follow)")
    args = ap.parse_args(argv)
    from r5 import sealcheck
    if args.write_spec:
        sealcheck.refuse_if_sealed("--write-spec (results/ED/ITEM_SPEC.md is a sealed input)", args.amend)
        paths.write_text_lf(OUT / "ITEM_SPEC.md", ED.item_spec_text())
        print(f"wrote {OUT / 'ITEM_SPEC.md'}")
        return 0
    sealcheck.forbid_reduced_coverage(bool(args.smoke), no_intervention=bool(args.no_intervention),
                                      no_lopo=bool(args.no_lopo))
    if args.smoke:
        if args.from_rows:
            raise SystemExit("--from-rows reads written results; not a smoke")
        return body(args, "SMOKE")
    tag = sealcheck.require_seal()
    mode = "repro" if args.repro else ("from-rows" if args.from_rows else "run")
    return sealcheck.run_sealed(ITEM, mode, tag, lambda: body(args, tag))


def body(args, tag: str) -> int:
    from r5 import sealcheck
    t0 = time.time()
    written: list[Path] = []
    smoke = bool(args.smoke)
    mode = "smoke" if smoke else ("repro" if args.repro else ("from-rows" if args.from_rows else "run"))
    designs = ("BPG", "BP") if args.design == "both" else (args.design,)
    print(f"item 1.6 (E-D) {'PRE-SEAL ENGINEERING CHECK (--smoke; metrics suppressed; nothing written)' if smoke else 'seal ' + tag}; "
          f"designs {designs}; mode {mode}", flush=True)
    bench = D.load_bench()
    clusters = cluster_map()
    cid2ci = {c: i for i, c in enumerate(bench.frame.cell_id.astype(str))}
    c3 = sorted(cid2ci[c] for c in clusters)
    print(f"  bench {len(bench.frame)} cells; C3 cells {len(c3)}; clusters {len(set(clusters.values()))}", flush=True)
    gate_ok, gate_msg = (True, "smoke") if smoke else poison_gate(tag)
    static = selection = None
    if not args.repro and not args.from_rows:
        t1 = time.time()
        chk_static = ED.Checks()
        static = ED.static_tables(bench, c3, intervention=not args.no_intervention, checks=chk_static)
        selection = ED.loco_static_selection(static, clusters)
        nsets = {r: sum(len(s["sets"].get(r, {"S": []})["S"]) for s in static.values()) for r in ED.REC_READS}
        print(f"  static tables: {len(static)} cells; legal read sets per size {nsets}; cation-exchanger cells "
              f"{sum(s['cx'] for s in static.values())}; static-set intervention checked "
              f"{chk_static.c['static_sets_intervention_checked']}, failed {chk_static.c['static_sets_intervention_failed']}; "
              f"LOCO selection rows {len(selection)} ({time.time() - t1:.0f}s)", flush=True)
        if chk_static.c["static_sets_intervention_failed"]:
            raise SystemExit("static-set intervention check failed")
        t1 = time.time()
        pl = ED.poison_loco(bench, c3, clusters, static, selection)      # s0-mechanics-14
        pl.update({"label": "leakage check (fit poisoning of the recipe-2 LOCO static selection; brief 8 item 2)",
                   "seal": tag, "created_utc": _now(), "code_lf_sha256": code_digests()})
        print(f"  LOCO selection fit poisoning: {pl['clusters']} clusters, changed {pl['changed']}, leaky positive "
              f"control caught {pl['pc_caught']}/{pl['pc_runs']} -> passed {pl['passed']} ({time.time() - t1:.0f}s)",
              flush=True)
        if not smoke:
            _write_json(pl, ED.LEAK / "poison_ED_loco.json", written)
        if not pl["passed"]:
            if smoke:
                raise SystemExit("LOCO selection fit-poisoning check failed")
            sealcheck.stop(ITEM, mode, tag, "LOCO selection fit-poisoning check failed (results/leakage/poison_ED_loco.json)")
    elif args.from_rows and (OUT / "loco_selection.csv").exists():
        selection = pd.read_csv(OUT / "loco_selection.csv")
    S, T = {}, {}
    for design in designs:
        td = time.time()
        if args.from_rows:
            R = pd.read_parquet(OUT / f"rows_{design}.parquet")
            cand = pd.read_parquet(OUT / f"cand_{design}.parquet")
            checks = json.loads((OUT / f"checks_{design}.json").read_bytes())
            seals = set(R.seal.astype(str)) if "seal" in R.columns else set()
            if seals != {tag} or checks.get("seal") != tag or checks.get("code_lf_sha256") != code_digests():
                sealcheck.stop(ITEM, mode, tag, f"{design}: --from-rows refused: the stored rows / checks do not carry the "
                                                f"current seal {tag} and code digests (s0-mechanics-18)")
        else:
            res = ED.compute_design(
                bench, design, ks=(3, 4) if args.repro else ED.BETA_KS, folds=args.smoke_folds if smoke else None,
                families=("LAD",) if args.repro else ("LAD", "ENV", "REC"), max_reads=5 if args.repro else ED.MAX_READS,
                min_reads=4 if args.repro else 2,
                static=static, selection=selection, cluster_of=clusters, intervention=not args.no_intervention,
                log=lambda s: print(s, flush=True))
            R, cand, checks = res["rows"], res["cand"], dict(res["checks"])
            if static is not None:
                checks.update({k: v for k, v in chk_static.c.items() if k.startswith("static_")})
            fits = res["fits"]
            print(f"  {design}: rows {len(R)} (families {R.family.value_counts().to_dict()}); candidates {len(cand)}; "
                  f"fits {len(fits)}; NaN in se_B1 {int(R.se_B1.isna().sum())}; checks {checks}; {res['seconds']:.0f}s",
                  flush=True)
            k3 = fits[fits.k == ED.K_EA]
            if len(k3):
                print(f"  {design}: k = 3 refit vs stored round-4 beta: max |diff| {k3.abs_diff_stored.max():.3e}, lambda "
                      f"equal {int(k3.lambda_equal.sum())}/{len(k3)} (identity check)", flush=True)
        bad = [k for k in ("intervention_failed", "both_ends_read", "illegal_or_unnested", "dopt5_not_legal",
                           "dopt5_mismatch_cache", "loose_not_superset", "dopt_candidate_mismatch",
                           "cohort_largest_budget_closed_queries") if checks.get(k)]
        if checks.get("peek_control_caught", 0) != checks.get("peek_control_runs", 0):
            bad.append("peek_control_not_caught")
        if not args.no_intervention:           # s0-mechanics-5: full coverage of the intervention and its control
            if not (checks.get("intervention_checked", 0) > 0
                    and checks.get("intervention_checked") == checks.get("read_sets_checked")):
                bad.append("intervention_coverage")
            if checks.get("peek_control_runs") != checks.get("folds"):
                bad.append("peek_control_coverage")
        if bad:
            if smoke:
                raise SystemExit(f"{design}: leakage / support check failed: {bad} {checks}")
            sealcheck.stop(ITEM, mode, tag, f"{design}: leakage / support check failed: {bad}")
        rep = ED.repro_checks(R, design, scores=not smoke)
        if smoke:
            print(f"  {design} reproduction identity (partial folds; PRE-SEAL ENGINEERING CHECK): rows matched {rep['rows_matched']} of K3 "
                  f"{rep['rows_K3']}; max |diff| vs REF {rep['max_abs_diff']}; read sets equal share "
                  f"{rep['read_sets_equal_share']:.3f}; 5-read B0 vs R3-B0 max |diff| {rep['max_abs_diff_B0_r5_vs_R3B0']:.3e} "
                  f"(rows {rep['rows_matched_r5']}); R3-B0 policies {rep['R3B0_policies']}", flush=True)
            if not args.repro:
                Rs, Cs = smoke_surrogate(R, cand, design)       # no table ever sees a real label in the smoke
                Td = ED.analyse(Rs, Cs, design, clusters=sorted(set(clusters.values())), selection=selection,
                                lopo=not args.no_lopo, log=lambda s: print(s, flush=True))
                for k, v in Td.items():
                    print(_shape_report(k, v), flush=True)
                T[design] = Td
                S[design] = summary(Td, design, checks, rep, round(time.time() - td, 1))   # in memory only
                print(f"    summary_{design} (surrogate): {len(json.dumps(S[design], default=str))} characters (not "
                      "written)", flush=True)
            print(f"  {design} smoke done ({time.time() - td:.0f}s)", flush=True)
            continue
        rep["seal"] = tag
        rep["code_lf_sha256"] = code_digests()
        _write_json(rep, OUT / f"repro_{design}.json", written)
        failed = [k for k, v in rep.get("identity_checks", {}).items() if not v]
        print(f"  {design} reproduction (TECHNICAL SMOKE): identity {rep['identity_ok']} (failed: {failed or 'none'}); "
              f"targets {rep.get('targets')}; ok {rep.get('ok')}", flush=True)
        if not rep.get("ok"):
            sealcheck.stop(ITEM, mode, tag, f"{design}: reproduction mismatch -- the item stops (brief 6.0); see "
                                            f"results/ED/repro_{design}.json")
        if args.repro:
            continue
        if not args.from_rows:
            OUT.mkdir(parents=True, exist_ok=True)
            checks.update({"seal": tag, "code_lf_sha256": code_digests()})
            R.assign(label=ED.row_labels(R), seal=tag).to_parquet(OUT / f"rows_{design}.parquet", index=False)
            cand.assign(label=ED.label_for("B1", "", "all", design, "5th-read candidate SSE (RANDOM / ORACLE inputs)"),
                        seal=tag).to_parquet(OUT / f"cand_{design}.parquet", index=False)
            written += [OUT / f"rows_{design}.parquet", OUT / f"cand_{design}.parquet"]
            _write_csv(fits.assign(label="fitted parameter (training side; EXPLORATORY context)", seal=tag),
                       OUT / f"fits_{design}.csv", written)
            _write_json(checks, OUT / f"checks_{design}.json", written)
        if not gate_ok:                        # s0-mechanics-13: tables only after the beta-refit poisoning passed
            print(f"  {design}: tables not written: fit poisoning {gate_msg}; rerun after scripts/s1_ed_poison.py with "
                  "--from-rows", flush=True)
            continue
        Td = ED.analyse(R, cand, design, clusters=sorted(set(clusters.values())), selection=selection,
                        lopo=not args.no_lopo, log=lambda s: print(s, flush=True))
        for k, v in Td.items():
            if k == "recipe_loco_selection":
                continue
            _write_csv(v.assign(seal=tag) if len(v) else v, OUT / f"{k}_{design}.csv", written)
        T[design] = Td
        S[design] = summary(Td, design, checks, rep, round(time.time() - td, 1))
        S[design].update({"seal": tag, "code_lf_sha256": code_digests(), "fit_poisoning": gate_msg})
        _write_json(S[design], OUT / f"summary_{design}.json", written)
    if not smoke and not args.repro and selection is not None and not args.from_rows:
        _write_csv(selection.assign(label=ED.SEEN + "; LOCO selection by B0 SSE on the other clusters' cells", seal=tag),
                   OUT / "loco_selection.csv", written)
    if smoke and set(S) == {"BPG", "BP"}:
        txt = item_table(S, T, time.time() - t0, tag=tag, mode=mode, gate_ok=gate_ok, gate_msg=gate_msg)
        print(f"  per-item table built in memory on the surrogate: {len(txt.splitlines())} lines (not written)", flush=True)
    if not smoke and not args.repro and gate_ok and set(S) == {"BPG", "BP"}:
        paths.write_text_lf(OUT / "ITEM_TABLE.md", item_table(S, T, time.time() - t0, tag=tag, mode=mode, gate_ok=gate_ok,
                                                              gate_msg=gate_msg) + f"\nSeal: {tag}\n")
        written.append(OUT / "ITEM_TABLE.md")
    for p in written:
        print(f"wrote {p}")
    print(f"done in {time.time() - t0:.0f}s")
    if not smoke and not args.repro and not gate_ok:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
