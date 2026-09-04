#!/usr/bin/env python
"""PHASE 8 — how publication-dependent is the shape result?

gen9's extractant-shape gain rests on 25 ligands and 155 distinct curves from 26
publications, and one publication contributes a third of the summed gain.  The
chemotype-blocked bootstrap treats curves from one campaign as exchangeable across
chemotypes, which they are not if the campaign's measurement style — window,
density, acid — is what the model learned.  So the interval is recomputed under
every clustering a sceptic would ask for:

1. leave-one-publication-out: the effect with each DOI removed in turn;
2. publication-blocked bootstrap: resample DOIs, not chemotypes;
3. chemotype-blocked bootstrap: gen9's own, for reference;
4. a conservative two-factor scheme: resample publications *and*, within each,
   chemotypes — the interval can only widen.

Per axis, per metric, per comparison (candidate vs frozen, candidate vs A0), on
the per-curve shape table gen9 wrote.  Positive = candidate better.  Every curve's
publication is the DOI its rows were recovered with (``nuisance__doi`` from the
gen7 recovered-variables table, joined on ``row_id`` through the curve membership).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

GEN9 = REPO_ROOT / "runs" / "gen9_shape"
GEN7 = REPO_ROOT / "runs" / "gen7_architecture"
OUT = REPO_ROOT / "runs" / "gen10_final" / "publication_sensitivity"
METRICS = {"shape_mae": -1.0, "slope_abs_error": -1.0, "row_mae": -1.0,
           "span_recovery_distance": -1.0, "spearman": 1.0}
AXES = ("extractant", "acid", "metal_series")
REPLICATES = 5000


def curve_publication(membership: pd.DataFrame, recovered: pd.DataFrame) -> pd.Series:
    """One DOI per curve: the modal DOI of its rows (ties -> first sorted)."""
    doi = recovered.set_index(recovered["row_id"].astype(str))["nuisance__doi"]
    work = membership.assign(doi=membership["row_id"].astype(str).map(doi))
    modal = work.groupby("curve_id")["doi"].agg(
        lambda s: s.dropna().astype(str).value_counts().sort_index().idxmax()
        if s.notna().any() else "unknown")
    return modal


def paired_deltas(table: pd.DataFrame, candidate: str, reference: str, metric: str) -> pd.DataFrame:
    """Per (curve, seed) candidate-vs-reference delta, signed so positive = better."""
    work = table.copy()
    if metric == "span_recovery_distance":
        work["span_recovery_distance"] = (work["span_recovery"] - 1.0).abs()
    a = work[work["model"] == candidate].set_index(["curve_id", "split_seed"])[metric]
    b = work[work["model"] == reference].set_index(["curve_id", "split_seed"])[metric]
    joined = pd.concat([a.rename("candidate"), b.rename("reference")], axis=1).dropna()
    sign = METRICS[metric]
    delta = sign * (joined["candidate"] - joined["reference"])
    meta = work.drop_duplicates("curve_id").set_index("curve_id")[["extractant", "tanimoto_cluster",
                                                                      "doi"]]
    out = delta.rename("delta").reset_index().merge(meta, left_on="curve_id", right_index=True)
    # one number per curve: mean over seeds, so a curve is one unit regardless of
    # how many seeds held it out
    return out.groupby("curve_id").agg(delta=("delta", "mean"), extractant=("extractant", "first"),
                                       tanimoto_cluster=("tanimoto_cluster", "first"),
                                       doi=("doi", "first")).reset_index()


def block_bootstrap(deltas: pd.DataFrame, block: str, *, replicates: int = REPLICATES,
                    seed: int = 20260821) -> dict:
    groups = deltas.groupby(block)["delta"].apply(lambda s: s.to_numpy())
    names = list(groups.index)
    arrays = [groups[n] for n in names]
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    for r in range(replicates):
        picks = rng.integers(0, len(names), size=len(names))
        draws[r] = np.concatenate([arrays[i] for i in picks]).mean()
    point = float(deltas["delta"].mean())
    return {"point": point, "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975)), "n_blocks": len(names),
            "ci_width": float(np.quantile(draws, 0.975) - np.quantile(draws, 0.025))}


def two_factor_bootstrap(deltas: pd.DataFrame, *, replicates: int = REPLICATES,
                         seed: int = 20260821) -> dict:
    """Resample publications; within each drawn publication resample its chemotypes.

    A deliberately conservative scheme: it treats a chemotype measured by two
    publications as two partly independent draws and a publication measuring five
    chemotypes as one block, so whichever factor carries the dependence is
    resampled.  The interval cannot be narrower than the wider single-factor one.
    """
    rng = np.random.default_rng(seed)
    # Precompute once: for each publication, the list of its chemotype delta arrays.
    # The replicate loop is then pure numpy indexing rather than a groupby per draw.
    nested = [[g["delta"].to_numpy(dtype=float) for _, g in block.groupby("tanimoto_cluster")]
              for _, block in deltas.groupby("doi")]
    draws = np.empty(replicates)
    for r in range(replicates):
        picked = rng.integers(0, len(nested), size=len(nested))
        parts = []
        for i in picked:
            chem = nested[i]
            picks = rng.integers(0, len(chem), size=len(chem))
            parts.extend(chem[j] for j in picks)
        draws[r] = np.concatenate(parts).mean()
    return {"point": float(deltas["delta"].mean()),
            "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975)),
            "ci_width": float(np.quantile(draws, 0.975) - np.quantile(draws, 0.025))}


def leave_one_publication_out(deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    full = float(deltas["delta"].mean())
    total_gain = float(deltas["delta"].sum())
    for doi, block in deltas.groupby("doi"):
        rest = deltas[deltas["doi"] != doi]
        rows.append({"doi": doi, "n_curves": int(len(block)),
                     "n_ligands": int(block["extractant"].nunique()),
                     "publication_mean_gain": float(block["delta"].mean()),
                     "share_of_summed_gain": float(block["delta"].sum() / total_gain)
                     if total_gain != 0 else np.nan,
                     "effect_without": float(rest["delta"].mean()) if len(rest) else np.nan,
                     "effect_full": full,
                     "curves_improved_without": float((rest["delta"] > 0).mean())
                     if len(rest) else np.nan})
    return pd.DataFrame(rows).sort_values("share_of_summed_gain", ascending=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve-shape", type=Path, default=GEN9 / "shape" / "curve_shape.parquet")
    parser.add_argument("--candidates", nargs="*", default=["GEN9_SHAPE_RECOMPOSED",
                                                            "GEN9_REL_MONOLITH"])
    parser.add_argument("--references", nargs="*", default=["REC_ecfp_plus_recovered",
                                                            "GEN9_A0_ROW_ONLY"])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    table = pd.read_parquet(args.curve_shape)
    membership = pd.read_parquet(GEN9 / "curves" / "curve_membership.parquet")
    recovered = pd.read_parquet(GEN7 / "cache" / "recovered_cells.parquet")
    publication = curve_publication(membership, recovered)
    table["doi"] = table["curve_id"].map(publication).fillna("unknown")
    coverage = {"n_curves": int(table["curve_id"].nunique()),
                "n_publications": int(table["doi"].nunique()),
                "curves_without_doi": int((table.drop_duplicates("curve_id")["doi"] == "unknown").sum())}

    intervals, lopo_tables = [], []
    for axis in AXES:
        block = table[table["axis_label"] == axis]
        for candidate in args.candidates:
            for reference in args.references:
                for metric in METRICS:
                    deltas = paired_deltas(block, candidate, reference, metric)
                    if len(deltas) < 5:
                        continue
                    base = {"axis": axis, "candidate": candidate, "reference": reference,
                            "metric": metric, "n_curves": int(len(deltas)),
                            "n_ligands": int(deltas["extractant"].nunique()),
                            "n_publications": int(deltas["doi"].nunique()),
                            "curves_improved": float((deltas["delta"] > 0).mean())}
                    for scheme, result in (
                            ("chemotype", block_bootstrap(deltas, "tanimoto_cluster",
                                                          replicates=args.replicates)),
                            ("publication", block_bootstrap(deltas, "doi",
                                                            replicates=args.replicates)),
                            ("ligand", block_bootstrap(deltas, "extractant",
                                                       replicates=args.replicates)),
                            ("two_factor", two_factor_bootstrap(deltas,
                                                                replicates=args.replicates))):
                        intervals.append({**base, "scheme": scheme, **result,
                                          "positive": bool(result["ci_low"] > 0)})
                    if metric in ("shape_mae", "slope_abs_error"):
                        lopo = leave_one_publication_out(deltas)
                        lopo.insert(0, "metric", metric)
                        lopo.insert(0, "reference", reference)
                        lopo.insert(0, "candidate", candidate)
                        lopo.insert(0, "axis", axis)
                        lopo_tables.append(lopo)
    intervals_table = pd.DataFrame(intervals)
    intervals_table.to_csv(args.out / "intervals_by_scheme.csv", index=False)
    lopo_table = pd.concat(lopo_tables, ignore_index=True)
    lopo_table.to_csv(args.out / "leave_one_publication_out.csv", index=False)

    # --- the headline: does the extractant shape gain survive every scheme? ---
    head = intervals_table[(intervals_table["axis"] == "extractant")
                           & (intervals_table["candidate"] == "GEN9_SHAPE_RECOMPOSED")
                           & (intervals_table["reference"] == "REC_ecfp_plus_recovered")]
    verdict = {}
    for metric, block in head.groupby("metric"):
        verdict[metric] = {
            "point": float(block["point"].iloc[0]),
            "positive_under_every_scheme": bool(block["positive"].all()),
            "widest_scheme": block.sort_values("ci_width").iloc[-1]["scheme"],
            "widest_ci": [float(block.sort_values("ci_width").iloc[-1]["ci_low"]),
                          float(block.sort_values("ci_width").iloc[-1]["ci_high"])],
            "chemotype_ci": [float(block[block["scheme"] == "chemotype"]["ci_low"].iloc[0]),
                             float(block[block["scheme"] == "chemotype"]["ci_high"].iloc[0])],
            "publication_ci": [float(block[block["scheme"] == "publication"]["ci_low"].iloc[0]),
                               float(block[block["scheme"] == "publication"]["ci_high"].iloc[0])],
        }
    lopo_head = lopo_table[(lopo_table["axis"] == "extractant")
                           & (lopo_table["candidate"] == "GEN9_SHAPE_RECOMPOSED")
                           & (lopo_table["reference"] == "REC_ecfp_plus_recovered")
                           & (lopo_table["metric"] == "shape_mae")]
    summary = {"coverage": coverage, "extractant_shape_recomposed_vs_frozen": verdict,
               "lopo_shape_mae": {
                   "min_effect_without_any_publication": float(lopo_head["effect_without"].min()),
                   "max_single_publication_share": float(lopo_head["share_of_summed_gain"].max()),
                   "publication_with_max_share": str(lopo_head.iloc[0]["doi"]),
                   "n_publications": int(len(lopo_head)),
                   "effect_stays_positive_under_every_removal": bool(
                       (lopo_head["effect_without"] > 0).all())}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("\nextractant, SHAPE_RECOMPOSED vs frozen, by scheme:")
    print(head[["metric", "scheme", "point", "ci_low", "ci_high", "ci_width", "n_blocks", "positive"]]
          .sort_values(["metric", "scheme"]).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
