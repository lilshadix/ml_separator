"""Analysis of the gen11 `matched` stage, exactly as pre-registered.

Implements MATCHED_PREREGISTRATION.md (sha256 4d7b4021...):
  primary endpoint  eff(F) - eff(D), i.e. macro MAE(D) - macro MAE(F) on identical rows
  arithmetic check  |eff(G) - eff(F)| < 0.02
  diagnostics       Am dose-response, level/shape split, far band

Every interval is the repository's own paired chemotype bootstrap, imported.
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lanthanide_separation.gen11 import analysis as A  # noqa: E402

RUN = Path(__file__).resolve().parents[1] / "runs" / "gen11_transfer"
ARMS = RUN / "arms"
CTRL = "A_GEN10_CONTROL__JOINT__GENERAL__ANNOTATION_SAFE__HIERARCHICAL__lam1"
D_ARM = "D_LN_PLUS_NON_ACTINIDE__JOINT__GENERAL__ANNOTATION_SAFE__HIERARCHICAL__lam1"
C_ARM = "C_LN_PLUS_ACTINIDES__JOINT__GENERAL__ANNOTATION_SAFE__HIERARCHICAL__lam1"
E_ARM = "E_LN_PLUS_ALL__JOINT__GENERAL__ANNOTATION_SAFE__HIERARCHICAL__lam1"


def load(stem: str) -> pd.DataFrame:
    return pd.read_parquet(ARMS / f"oof_{stem}.parquet")


def draws(prefix: str) -> dict[str, str]:
    out = {}
    for p in sorted(ARMS.glob(f"oof_{prefix}__*.parquet")):
        stem = p.name[len("oof_"):-len(".parquet")]
        m = re.search(r"draw(\d+)$", stem)
        if m:
            out[f"{prefix[0]}{m.group(1)}"] = stem
    return out


def macro(frames, arm):
    w = A.wide_predictions(frames)
    err = (w[f"prediction_{arm}"] - w["log_D"]).abs()
    return err.groupby([w["split_seed"], w["ecfp_cluster"]]).mean() \
              .groupby(level=0).mean().mean()


def main() -> int:
    f_draws, g_draws = draws("F_ACTINIDES_ONLY_MATCHED"), draws("G_RANDOM_AUX_MATCHED")
    if not f_draws:
        print("no F arms on disk yet"); return 1
    print(f"F draws: {sorted(f_draws)}\nG draws: {sorted(g_draws)}\n")

    base = {"ctrl": load(CTRL), "D": load(D_ARM), "C": load(C_ARM), "E": load(E_ARM)}
    for k, s in {**f_draws, **g_draws}.items():
        base[k] = load(s)

    # ---- context: every arm against the design-matched control -----------------
    print("=" * 78)
    print("CONTEXT  eff(X) = macro(control) - macro(X)   [+ = better]")
    print("=" * 78)
    cands = ["C", "E", "D", *sorted(f_draws), *sorted(g_draws)]
    ctx = A.compare(base, reference="ctrl", candidates=cands)
    print(ctx.to_string(index=False))

    # ---- PRIMARY: eff(F) - eff(D) = macro(D) - macro(F) ------------------------
    print("\n" + "=" * 78)
    print("PRIMARY ENDPOINT   eff(F) - eff(D), reference = D  [+ = actinides better at equal n]")
    print("=" * 78)
    prim = A.compare(base, reference="D", candidates=sorted(f_draws))
    print(prim.to_string(index=False))
    col = "point_delta"
    for stat in ("mae", "offset_mae", "shape_mae"):
        sub = prim[prim["statistic"] == stat]
        if sub.empty:
            continue
        excl = ((sub["bca_low"] > 0) | (sub["bca_high"] < 0)).sum()
        print(f"\n{stat:10s} mean over {len(sub)} draws {sub[col].mean():+.4f}   "
              f"min {sub[col].min():+.4f}  max {sub[col].max():+.4f}   "
              f"draws with BCa excluding zero: {excl}/{len(sub)}")

    print("\n--- per split seed, each F draw vs D ---")
    for k in sorted(f_draws):
        ps = A.per_seed_direction(base, reference="D", candidate=k)
        print(f"{k}: " + "  ".join(f"{v:+.4f}" for v in ps["improvement"])
              + f"   seeds_positive {int(ps['improved'].sum())}/5")

    # ---- arithmetic check: G must track F --------------------------------------
    if g_draws:
        print("\n" + "=" * 78)
        print("ARITHMETIC CHECK   eff(G) - eff(D); G's pool is 91.2% actinide so G must track F")
        print("=" * 78)
        gch = A.compare(base, reference="D", candidates=sorted(g_draws))
        print(gch.to_string(index=False))
        gm = gch[gch["statistic"] == "mae"]["point_delta"].mean()
        fm = prim[prim["statistic"] == "mae"]["point_delta"].mean()
        print(f"\nmean G-vs-D {gm:+.4f}   vs   mean F-vs-D {fm:+.4f}"
              f"   |difference| {abs(gm-fm):.4f}  (pre-registered < 0.02)")

    # ---- diagnostic 2: level vs shape ------------------------------------------
    print("\n" + "=" * 78)
    print("DIAGNOSTIC  level vs shape (C's gain was offset +0.0492 / shape +0.0026)")
    print("=" * 78)
    for k in sorted(f_draws)[:2] + ["C"]:
        try:
            d = A.decompose(base, reference="D" if k != "C" else "ctrl", candidate=k)
            print(f"\n{k}:"); print(d.to_string(index=False))
        except Exception as exc:  # noqa: BLE001
            print(f"{k}: decompose unavailable ({exc})")

    # ---- diagnostic 3: far band -------------------------------------------------
    print("\n" + "=" * 78)
    print("DIAGNOSTIC  macro MAE by chemotype band (far = nn_train_tanimoto <= 0.4)")
    print("=" * 78)
    w = A.wide_predictions(base)
    w["band"] = A.band_of(w["nn_train_tanimoto"])
    for band in ("far", "mid", "near"):
        b = w[w["band"] == band]
        if b.empty:
            continue
        row = {}
        for arm in ["ctrl", "D", "C", *sorted(f_draws)]:
            e = (b[f"prediction_{arm}"] - b["log_D"]).abs()
            row[arm] = e.groupby(b["ecfp_cluster"]).mean().mean()
        fmean = sum(row[k] for k in f_draws) / len(f_draws)
        print(f"{band:5s} n={len(b):6d}  ctrl {row['ctrl']:.4f}  D {row['D']:.4f}  "
              f"C {row['C']:.4f}  F(mean) {fmean:.4f}   F-D {row['D']-fmean:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
