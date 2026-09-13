"""L2 gate (PRE_REGISTRATION section 3, L2): is the honest curvature headroom still worth chasing?

    H = MAE(G14) - MAE(O_CURV_LPO)   on the standard pair tables, five designs, discovery seeds,
                                     chemotype-blocked paired bootstrap (gen13sep.inference).

OPEN if H >= 0.02 under BP and the 95 % interval excludes zero; otherwise CLOSED.  The gate is
INVALID if the in-sample O_CURV computed by the same loop does not reproduce gen15's 0.4272 +- 0.001
(or G14 its 0.5001).

Writes gen16_leads/results/L2/{gate_board.csv, gate_contrasts.csv, gate_perext_<design>.csv,
gate_checks.json, L2_GATE.md}.

Run from the repo root:
    PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 .venv/Scripts/python.exe gen16_leads/scripts/l2_gate.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen16 import bootstrap  # noqa: E402
from gen16 import l2_lpo as L  # noqa: E402
from gen15 import valuebench as V  # noqa: E402

OUT = bootstrap.RESULTS / "L2"
OUT.mkdir(parents=True, exist_ok=True)
MARGIN = 0.02
#: gen15 exp/labelerr/results/s4b_oracle_honesty.csv, one-seed all-cells frame, true sign
S4B = {"SIGN_OWNB": 0.3771356739661752, "SIGN_OWNB_LOPO": 0.42453252585145684}
COLS = ["design", "comparison", "point", "ci95_low", "ci95_high", "bca_low", "bca_high",
        "p_two_sided", "n_units", "units_improved", "seeds_positive", "loco_sign_stable",
        "passes_P1", "family"]


def main() -> None:
    t0 = time.time()
    bench = V.load()
    err = L.full_refit_error(bench)
    print(f"[L2 gate] bench loaded; max |s4_floor._fit(C[i]) - bench.coef[i]| = {err:.3e}", flush=True)
    if err > 1e-9:
        raise SystemExit("the copied refit does not reproduce bench.coef; stop")

    # BP first so the reproduction check is visible before the other designs run
    designs = ["BP"] + [d for d in V.DESIGNS if d != "BP"]
    B, C, perext = L.score_gate(bench, designs)
    B["lead"] = L.LEAD
    B.to_csv(OUT / "gate_board.csv", index=False)
    C.to_csv(OUT / "gate_contrasts.csv", index=False)
    for d, pe in perext.items():
        pe.to_csv(OUT / f"gate_perext_{d}.csv", index=False)

    checks = L.reproduction_check(B)
    valid = all(v["ok"] for v in checks.values())
    reg = C[(C.family == "registered") & (C.comparison == "OCURVLPO_vs_G14")].set_index("design")
    bp = reg.loc["BP"]
    H = float(bp["point"])
    ci_excl = bool(bp["ci95_low"] > 0)
    bca_excl = bool(bp["bca_low"] > 0)
    gate = "invalid" if not valid else ("open" if (H >= MARGIN and ci_excl) else "closed")
    all_five_sign = bool((reg["point"] > 0).all())

    wide = V.wide(B)
    gap = (wide.loc["O_CURV_LPO"] - wide.loc["O_CURV"])
    s4b_gap = S4B["SIGN_OWNB_LOPO"] - S4B["SIGN_OWNB"]

    summary = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "reproduction": checks, "gate_valid": valid, "gate": gate,
        "headroom_BP": {"point": H, "ci95_low": float(bp["ci95_low"]), "ci95_high": float(bp["ci95_high"]),
                        "bca_low": float(bp["bca_low"]), "bca_high": float(bp["bca_high"]),
                        "p_two_sided": float(bp["p_two_sided"]), "seeds_positive": int(bp["seeds_positive"]),
                        "n_seeds": int(bp["n_seeds"]), "loco_sign_stable": bool(bp["loco_sign_stable"]),
                        "n_units": int(bp["n_units"]), "units_improved": int(bp["units_improved"]),
                        "passes_P1": bool(bp["passes_P1"]), "ci_excludes_zero": ci_excl,
                        "bca_excludes_zero": bca_excl},
        "headroom_five_designs": {d: float(reg.loc[d, "point"]) for d in V.DESIGNS},
        "positive_in_all_five": all_five_sign,
        "macro_mae": {a: {d: float(wide.loc[a, d]) for d in V.DESIGNS} for a in L.ARMS},
        "insample_vs_lpo_gap": {d: float(gap[d]) for d in V.DESIGNS},
        "s4b_gap_one_seed_true_sign": s4b_gap,
        "seconds": time.time() - t0,
    }
    (OUT / "gate_checks.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pd.set_option("display.width", 220)
    print("\n=== extractant-macro MAE of log SF (5 discovery seeds) ===")
    print(wide.round(4).to_string())
    print("\n=== contrasts (positive favours the candidate) ===")
    print(C[COLS].round(4).to_string(index=False))
    print("\nreproduction:", json.dumps(checks, indent=1))
    print(f"\nGATE: {gate.upper()}   H(BP) = {H:+.4f}  CI [{bp['ci95_low']:+.4f}, {bp['ci95_high']:+.4f}] "
          f"BCa [{bp['bca_low']:+.4f}, {bp['bca_high']:+.4f}]  p = {bp['p_two_sided']:.4f}")

    write_md(summary, wide, C, reg)
    print(f"\n[L2 gate] total {time.time() - t0:.0f}s; wrote {OUT}")


def write_md(s: dict, wide: pd.DataFrame, C: pd.DataFrame, reg: pd.DataFrame) -> None:
    hb = s["headroom_BP"]
    gate = s["gate"]
    lines = [
        "# L2 gate — the honest (leave-pair-out) curvature headroom",
        "",
        f"*Discovery seeds (5), designs B / BR / BQ / A / BP, extractant-macro MAE of pairwise log SF, "
        f"n = 90 extractants, chemotype-blocked paired bootstrap (10 000 replicates). "
        f"Run {s['timestamp']}, {s['seconds']:.0f} s.*",
        "",
        f"## Verdict: **{gate.upper()}**",
        "",
        f"Registered rule: OPEN if H = MAE(G14) − MAE(O_CURV_LPO) ≥ {MARGIN} under BP and the 95 % CI excludes zero.",
        "",
        f"* H under BP = **{hb['point']:+.4f}**, percentile 95 % CI [{hb['ci95_low']:+.4f}, {hb['ci95_high']:+.4f}], "
        f"BCa [{hb['bca_low']:+.4f}, {hb['bca_high']:+.4f}], p = {hb['p_two_sided']:.4f}, "
        f"{hb['units_improved']}/{hb['n_units']} extractants improved, {hb['seeds_positive']}/{hb['n_seeds']} seeds positive, "
        f"LOCO sign stable = {hb['loco_sign_stable']}, passes P1 = {hb['passes_P1']}.",
        f"* H ≥ {MARGIN}: {hb['point'] >= MARGIN}.  CI excludes zero: {hb['ci_excludes_zero']} (BCa: {hb['bca_excludes_zero']}).",
        f"* H in the five designs: " + ", ".join(f"{d} {v:+.4f}" for d, v in s["headroom_five_designs"].items())
        + f"; same sign in all five: {s['positive_in_all_five']}.",
        "",
        "## Reproduction of the gen15 anchors through this loop (BP)",
        "",
        "| arm | gen15 | this run | abs diff | ok (≤ 0.001) |",
        "|---|---|---|---|---|",
    ]
    for arm, v in s["reproduction"].items():
        lines.append(f"| {arm} | {v['expected']:.6f} | {v['obtained']:.6f} | {v['abs_diff']:.2e} | {v['ok']} |")
    lines += [
        "",
        f"Gate valid (O_CURV reproduces 0.4272 ± 0.001 and G14 0.5001): **{s['gate_valid']}**.",
        "",
        "## Five-design table (extractant-macro MAE)",
        "",
        "| arm | " + " | ".join(V.DESIGNS) + " |",
        "|---|" + "---|" * len(V.DESIGNS),
    ]
    for arm in L.ARMS:
        lines.append(f"| {arm} | " + " | ".join(f"{wide.loc[arm, d]:.4f}" for d in V.DESIGNS) + " |")
    lines += [
        "",
        "## Contrasts (positive favours the candidate)",
        "",
        "| design | comparison | family | point | 95 % CI | BCa | p | improved | seeds | LOCO | P1 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in C.iterrows():
        lines.append(f"| {r.design} | {r.comparison} | {r.family} | {r.point:+.4f} | [{r.ci95_low:+.4f}, {r.ci95_high:+.4f}] | "
                     f"[{r.bca_low:+.4f}, {r.bca_high:+.4f}] | {r.p_two_sided:.4f} | {r.units_improved}/{r.n_units} | "
                     f"{r.seeds_positive}/{r.n_seeds} | {r.loco_sign_stable} | {r.passes_P1} |")
    gap = s["insample_vs_lpo_gap"]
    lines += [
        "",
        "## In-sample vs leave-pair-out gap of the own-curvature oracle",
        "",
        f"gen15 `s4b_oracle_honesty.csv` (one seed, all 521 cells in one frame, **true** sign, global constants): "
        f"SIGN_OWNB {S4B['SIGN_OWNB']:.4f} → SIGN_OWNB_LOPO {S4B['SIGN_OWNB_LOPO']:.4f}, gap {s['s4b_gap_one_seed_true_sign']:+.4f}.",
        "",
        "This run (5 discovery seeds, fold-wise pair tables, **G14's predicted** sign and training-fold constants): "
        + ", ".join(f"{d} {gap[d]:+.4f}" for d in V.DESIGNS) + ".",
        "",
    ]
    (OUT / "L2_GATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
