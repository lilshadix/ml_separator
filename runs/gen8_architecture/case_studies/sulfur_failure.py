"""Reproduce every number in ``sulfur_failure.md`` (gen8 brief section 27).

Usage:  .venv/bin/python runs/gen8_architecture/case_studies/sulfur_failure.py

Reads only frozen artefacts:
  runs/gen7_architecture/cache/cohort.parquet
  runs/gen7_architecture/finalists/oof_predictions.parquet   (REC_ecfp_plus_recovered)
  runs/gen8_architecture/series/curve_membership.parquet
  runs/gen8_architecture/cross_series/one_shot_candidate_scores.parquet
  runs/gen8_architecture/active_acquisition/primary_detail.parquet
  dataset with 3D structures/dataset.parquet                 (extractant_name, raw D, safe_exp_id)

and, when it is present on this machine, the sibling dataset builder's raw exports
  ../lanthanide_dataset_builder/raw_data/*_SAFE.csv          (Solvent_Name, DOI)
which join 1:1 on ``safe_exp_id``.  Section 7 of the write-up depends on them; the
rest of the script runs without them.

No experimental quantity of a held-out ligand is ever used to build a feature or
pick a hyper-parameter; the k-shot numbers are exhaustive enumerations, not fits.
Every quantity derived from a held-out target (oracle level, oracle candidate choice,
the 1NN level probe, the group contrasts) is diagnostic and is labelled as such.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from lanthanide_separation.gen8.mechanism import (  # noqa: E402
    MECHANISM_COLUMNS, mechanism_features,
)
from lanthanide_separation.gen8.inference import paired_chemotype_bootstrap  # noqa: E402
from lanthanide_separation.gen8.report import md_table  # noqa: E402

MODEL = "REC_ecfp_plus_recovered"
TARGET = "CCCCOP(=S)(CP(=S)(OCCCC)OCCCC)OCCCC"          # TWE-24
ACID = "cond__acid_concentration_M"
PAIR_CHUNK = 5000                                        # keeps the k=2 sweep in RAM
BUILDER = Path("/Users/lilshadix/PycharmProjects/lanthanide_dataset_builder/raw_data")
#: the condition axes an acquisition policy is allowed to see (gen8.kshot.POLICY_AXES)
POLICY_AXES = ("massact__log10_cond__acid_concentration_M",
               "massact__log10_cond__extractant_concentration_M",
               "lanthanide_index", "cond__temperature_C")


# ---------------------------------------------------------------- helpers
def standardise(matrix: np.ndarray) -> np.ndarray:
    """gen8_mechanism_similarity.standardise, reproduced verbatim."""
    out = np.asarray(matrix, float).copy()
    median = np.nanmedian(out, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    out = np.where(np.isfinite(out), out, median)
    centre, scale = out.mean(axis=0), out.std(axis=0)
    return (out - centre) / np.where(scale > 1e-9, scale, 1.0)


def kshot_exhaustive(residual: np.ndarray) -> dict[str, float]:
    """Offset correction: measure k rows, add their mean residual, score the rest.

    k=1 matches ``gen8_calibration_geography.build_scores`` exactly
    (candidate i scores ``mean_{j!=i} |r_j - r_i|``).
    """
    n = residual.size
    diff = np.abs(residual[:, None] - residual[None, :])
    one = diff.sum(axis=1) / (n - 1)
    ii, jj = np.triu_indices(n, k=1)
    two = np.empty(ii.size, float)
    for start in range(0, ii.size, PAIR_CHUNK):
        sl = slice(start, min(start + PAIR_CHUNK, ii.size))
        offset = 0.5 * (residual[ii[sl]] + residual[jj[sl]])
        block = np.abs(residual[None, :] - offset[:, None])
        rows = np.arange(block.shape[0])
        total = block.sum(1) - block[rows, ii[sl]] - block[rows, jj[sl]]
        two[sl] = total / (n - 2)
    return {"one_rand": one.mean(), "one_best": one.min(), "one_worst": one.max(),
            "two_rand": two.mean(), "two_best": two.min(), "two_worst": two.max()}


def block_contrast(values_a: np.ndarray, blocks_a, values_b: np.ndarray, blocks_b,
                   *, replicates: int = 5000, seed: int = 8675309) -> tuple[float, float, float]:
    """Unpaired difference of two group means, resampling Tanimoto chemotypes.

    The two groups are disjoint sets of ligands, so nothing is paired and the
    interval is wide by construction; that is the point of reporting it.
    """
    rng = np.random.default_rng(seed)

    def draw(values: np.ndarray, blocks) -> np.ndarray:
        blocks = np.asarray(blocks, dtype=object)
        names = sorted(set(blocks))
        sums = np.array([values[blocks == b].sum() for b in names], float)
        sizes = np.array([(blocks == b).sum() for b in names], float)
        pick = rng.integers(0, len(names), size=(replicates, len(names)))
        return sums[pick].sum(1) / sizes[pick].sum(1)

    delta = draw(values_a, blocks_a) - draw(values_b, blocks_b)
    return (float(values_a.mean() - values_b.mean()),
            float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975)))


def central_pick(block: pd.DataFrame) -> int:
    """``gen8.kshot.policy_central`` on the full candidate pool: the row nearest the
    mean of the standardised condition axes.  It reads no target."""
    axes = block[[c for c in POLICY_AXES if c in block.columns]].to_numpy(float)
    axes = np.where(np.isfinite(axes), axes, 0.0)
    spread = axes.std(axis=0)
    axes = (axes - axes.mean(axis=0)) / np.where(spread > 1e-9, spread, 1.0)
    return int(np.argmin(np.linalg.norm(axes - axes.mean(axis=0), axis=1)))


def per_ligand(oof: pd.DataFrame, ligands) -> pd.DataFrame:
    rows = []
    for lig in ligands:
        block_all = oof[oof["extractant"] == lig]
        seeds = []
        for seed, block in block_all.groupby("split_seed"):
            r = (block["log_D"] - block["prediction"]).to_numpy(float)
            record = {"seed": seed, "n": r.size,
                      "zero": np.abs(r).mean(),
                      "oracle": np.abs(r - np.median(r)).mean(),
                      "mean_resid": r.mean(), "sd_resid": r.std(ddof=0)}
            record.update(kshot_exhaustive(r))
            with np.errstate(invalid="ignore"):
                record["spearman"] = pd.Series(block["prediction"].to_numpy(float)).corr(
                    pd.Series(block["log_D"].to_numpy(float)), method="spearman")
            seeds.append(record)
        out = pd.DataFrame(seeds).mean(numeric_only=True)
        out["extractant"] = lig
        rows.append(out)
    return pd.DataFrame(rows)


def curve_slopes(frame: pd.DataFrame) -> pd.DataFrame:
    out = []
    for cid, g in frame.groupby("curve_id"):
        g = g.sort_values("axis_value")
        xs = g["axis_value"].to_numpy(float)
        if len(set(xs)) < 2:
            continue
        out.append({"curve_id": cid, "axis": g["axis_label"].iloc[0], "n": len(g),
                    "true_slope": np.polyfit(xs, g["log_D"].to_numpy(float), 1)[0],
                    "pred_slope": np.polyfit(xs, g["pred"].to_numpy(float), 1)[0]})
    return pd.DataFrame(out)


def upstream(cohort: pd.DataFrame, oof: pd.DataFrame, name_of: dict) -> None:
    """What the primary export records about TWE-24's solvent, and what it costs.

    ``dataset.parquet`` carries ``safe_exp_id``, which joins 1:1 onto the sibling
    dataset builder's raw SAFE exports; those carry ``Solvent_Name`` and ``DOI``,
    both of which the modelling bundle flattens away.
    """
    if not BUILDER.exists():
        print("\n[upstream] dataset builder not on this machine -- section 7 skipped")
        return
    files = sorted(BUILDER.glob("*_SAFE.csv"))
    raw = pd.concat([pd.read_csv(f, low_memory=False).assign(stem=f.stem.removesuffix("_SAFE"))
                     for f in files], ignore_index=True)
    raw["safe_exp_id"] = raw["stem"] + "_SAFE:" + raw["exp_id"].astype(str)
    bundle = pd.read_parquet(ROOT / "dataset with 3D structures/dataset.parquet",
                             columns=["canonical_smiles", "extractant_name", "safe_exp_id"])
    joined = bundle.merge(raw[["safe_exp_id", "Solvent_Name", "DOI"]].drop_duplicates("safe_exp_id"),
                          on="safe_exp_id", how="left", validate="many_to_one")
    assert joined["Solvent_Name"].notna().all(), "the safe_exp_id join is no longer 1:1"

    twe = joined[joined["extractant_name"].isin(
        ["TWE-23", "TWE-24", "TWE-27", "TWE-28", "TWE-29", "TWE-30"])]
    print("\nrecorded solvent of the six P=S compounds (upstream, before flattening):")
    print(twe.groupby("extractant_name")["Solvent_Name"].agg(lambda s: sorted(set(s))).to_string())

    flags = cohort[cohort["extractant"] == TARGET]
    on = [c for c in flags.columns
          if (c.startswith(("cond__diluent__", "cond__additive__", "geom_cond__modifier_class",
                            "geom_cond__diluent_family")) and (flags[c] > 0).any())]
    print("what the model is told instead:", on)

    # the modifier's worth, measured inside the one campaign that used both
    campaign = raw[raw["DOI"].astype(str).str.contains("211267", na=False)].copy()
    value = pd.to_numeric(campaign["obsDvaluesValue"], errors="coerce")
    campaign["level"] = np.log10(value.where(value > 0))
    campaign["modified"] = campaign["Solvent_Name"].astype(str).str.lower().str.contains("octanol")
    # neat 1-octanol as the *diluent* is a different thing from octanol as a *modifier*
    # in a hydrocarbon diluent, which is TWE-24's case; keep the three classes apart.
    solvent = campaign["Solvent_Name"].astype(str).str.lower().str.strip()
    campaign["class"] = np.where(campaign["modified"] & solvent.str.contains(","),
                                 "hydrocarbon + octanol modifier",
                                 np.where(solvent == "1-octanol", "neat 1-octanol", "no octanol"))
    macro = campaign.groupby(["Extractant_Name", "class"])["level"].mean().reset_index()
    macro = macro[np.isfinite(macro["level"])]
    print("CORDIS 211267, ligand level by solvent class (one extractant one vote):")
    print(macro.groupby("class")["level"].agg(["size", "mean"]).round(2).to_string())
    rng = np.random.default_rng(0)
    a = macro.loc[macro["class"] == "hydrocarbon + octanol modifier", "level"].to_numpy(float)
    b = macro.loc[macro["class"] == "no octanol", "level"].to_numpy(float)
    draws = np.array([rng.choice(a, a.size, True).mean() - rng.choice(b, b.size, True).mean()
                      for _ in range(5000)])
    both = set(macro.loc[macro["class"] == "hydrocarbon + octanol modifier", "Extractant_Name"]) \
        & set(macro.loc[macro["class"] == "no octanol", "Extractant_Name"])
    target_level = float(macro.loc[macro["Extractant_Name"] == "TWE-24", "level"].iloc[0])
    print(f"modifier vs no octanol: {a.mean() - b.mean():+.2f} decades, bootstrap CI95 over "
          f"extractants [{np.quantile(draws, 0.025):+.2f}, {np.quantile(draws, 0.975):+.2f}]; "
          f"extractants measured both ways: {len(both)} (so this is between-ligand and confounded)")
    print(f"TWE-24's level {target_level:+.2f} is the {100 * (a < target_level).mean():.0f}th "
          f"percentile of the {a.size} modifier ligands and above all {b.size} unmodified ones")


# ---------------------------------------------------------------- main
def main() -> int:
    here = Path(__file__).resolve().parent
    cohort = pd.read_parquet(ROOT / "runs/gen7_architecture/cache/cohort.parquet")
    oof = pd.read_parquet(ROOT / "runs/gen7_architecture/finalists/oof_predictions.parquet")
    oof = oof[oof["model"] == MODEL].copy()
    member = pd.read_parquet(ROOT / "runs/gen8_architecture/series/curve_membership.parquet")
    scores = pd.read_parquet(
        ROOT / "runs/gen8_architecture/cross_series/one_shot_candidate_scores.parquet")
    raw = pd.read_parquet(ROOT / "dataset with 3D structures/dataset.parquet",
                          columns=["canonical_smiles", "extractant_name", "D", "log_D"])
    name_of = raw.groupby("canonical_smiles")["extractant_name"].agg(
        lambda s: sorted(set(s))[0]).to_dict()

    ligands = sorted(cohort["extractant"].unique())
    mech = mechanism_features(ligands)
    assert mech.attrs["n_unparsed"] == 0
    mech.insert(0, "extractant", ligands)
    mech = mech.set_index("extractant")

    # --- the two neighbourhoods ------------------------------------------
    mz = standardise(mech[list(MECHANISM_COLUMNS)].to_numpy(float))
    d_mech = np.linalg.norm(mz[:, None, :] - mz[None, :, :], axis=2)
    ecfp_cols = [c for c in cohort.columns
                 if c.startswith("ecfp_") and c != "ecfp_cluster"]
    first = cohort.drop_duplicates("extractant").set_index("extractant").loc[ligands]
    bits = (first[ecfp_cols].to_numpy() > 0).astype(np.int64)
    inter = bits @ bits.T
    count = bits.sum(1)
    union = count[:, None] + count[None, :] - inter
    tanimoto = np.where(union > 0, inter / union, 0.0).astype(float)
    index = {l: i for i, l in enumerate(ligands)}
    i0 = index[TARGET]

    sulfur = [l for l in ligands if mech.loc[l, "mech__n_S_donor"] > 0]
    assert not (mech.loc[sulfur, "mech__n_acidic_H"] > 0).any(), \
        "an S-donor with an exchangeable proton exists after all -- rewrite section 1"

    def donors(l: str) -> str:
        m = mech.loc[l]
        skip = {"mech__n_O_donor", "mech__n_N_donor", "mech__n_S_donor", "mech__n_P",
                "mech__n_donor_total", "mech__n_acidic_H", "mech__n_chelate_pairs"}
        return ", ".join(f"{int(m[c])} {c.removeprefix('mech__n_').replace('_', ' ')}"
                         for c in MECHANISM_COLUMNS
                         if c.startswith("mech__n_") and c not in skip and m[c] > 0)

    table = per_ligand(oof, ligands)
    table["name"] = table["extractant"].map(lambda l: name_of.get(l, "?"))
    table.to_csv(here / "per_ligand_kshot.csv", index=False)

    reference = table[table["n"] >= 4]
    print("cohort reference (143 ligands with >=4 rows):")
    print(reference[["zero", "oracle", "one_rand", "one_best", "one_worst",
                     "two_rand", "two_best", "two_worst"]].mean().round(3).to_dict())
    print("worst zero-shot ligand:",
          table.sort_values("zero", ascending=False).iloc[0][["name", "zero"]].to_dict())

    # --- neighbours of the target ----------------------------------------
    for label, order in (("ECFP tanimoto", np.argsort(-tanimoto[i0])[1:6]),
                         ("mechanistic", np.argsort(d_mech[i0])[1:6])):
        print(f"\ntop-5 {label} neighbours of {name_of.get(TARGET)}:")
        for j in order:
            l = ligands[j]
            print(f"  {name_of.get(l, '?'):12s} tan={tanimoto[i0, j]:.3f} "
                  f"d_mech={d_mech[i0, j]:.3f} softness={mech.loc[l, 'mech__softness_mean']:.3f} "
                  f"| {donors(l)}")

    # --- curves ----------------------------------------------------------
    pred = oof.groupby("row_id")["prediction"].mean().rename("pred")
    work = cohort[["row_id", "extractant", "series_id", "log_D", ACID]].merge(pred, on="row_id")
    work = work.merge(member[["row_id", "curve_id", "axis_label", "axis_value"]],
                      on="row_id", how="left")
    slopes = curve_slopes(work[work["curve_id"].notna()])
    slopes = slopes.merge(work.drop_duplicates("curve_id")[["curve_id", "extractant"]],
                          on="curve_id", how="left")
    slopes["name"] = slopes["extractant"].map(lambda l: name_of.get(l, "?"))
    slopes.to_csv(here / "curve_slopes.csv", index=False)
    for axis in ("acid", "metal_series", "extractant"):
        g = slopes[slopes["axis"] == axis]
        print(f"cohort {axis:12s} n={len(g):3d} median true slope {g.true_slope.median():+.3f} "
              f"median predicted {g.pred_slope.median():+.3f}")

    target_rows = work[work["extractant"] == TARGET].sort_values(ACID)
    target_rows = target_rows.assign(residual=target_rows["log_D"] - target_rows["pred"])
    print("\n" + md_table(target_rows, columns=[ACID, "log_D", "pred", "residual"]))

    # --- which row do the DEPLOYABLE central rules actually pick? ----------
    # CENTRAL/MEDOID/MID_ACID select on the condition axes only; none of them
    # reads a residual.  On the full six-row pool they all land mid-acid.
    pool = cohort[cohort["extractant"] == TARGET].sort_values(ACID).reset_index(drop=True)
    # exhaustive per-candidate 1-shot MAE, seeds averaged last (section 5's convention)
    block = oof[oof["extractant"] == TARGET]
    wide = (block.assign(r=block["log_D"] - block["prediction"])
            .pivot_table(index="row_id", columns="split_seed", values="r")
            .loc[pool["row_id"]].to_numpy(float))
    n_pool = wide.shape[0]
    target_one_shot = np.stack(
        [np.abs(wide[:, s][:, None] - wide[:, s][None, :]).sum(1) / (n_pool - 1)
         for s in range(wide.shape[1])], axis=1).mean(axis=1)
    print("\nexhaustive 1-shot MAE by candidate row (seeds averaged): "
          + ", ".join(f"{pool.loc[i, ACID]:.4f} M -> {target_one_shot[i]:.3f}"
                      for i in range(n_pool)))
    pick = central_pick(pool)
    print(f"\nCENTRAL (nearest the pool mean of the condition axes) picks "
          f"[HNO3] = {pool.loc[pick, ACID]:.4f} M -- candidate rank "
          f"{int(np.argsort(np.argsort(target_one_shot))[pick]) + 1} of {len(pool)} "
          f"(1 = best), its exhaustive 1-shot MAE {target_one_shot[pick]:.3f}")
    detail = pd.read_parquet(
        ROOT / "runs/gen8_architecture/active_acquisition/primary_detail.parquet")
    harness = detail[(detail["extractant"] == TARGET) & (detail["k"] == 1)
                     & (detail["adapter"] == "OFFSET_K1")].groupby("policy")["mae"].mean()
    print("harness k=1 OFFSET_K1 on this ligand: "
          + ", ".join(f"{p} {harness[p]:.3f}" for p in
                      ["RANDOM", "CENTRAL", "MEDOID", "MID_ACID", "FARTHEST_FROM_EXISTING",
                       "ORACLE[OFFSET_K1]"] if p in harness.index))

    # --- contamination probe: 1NN ligand-level lookup ---------------------
    level = cohort.groupby("extractant")["log_D"].mean()
    sim = tanimoto.copy()
    np.fill_diagonal(sim, -1.0)

    def nn_level(target: str, drop: str | None) -> tuple[float, str]:
        i = index[target]
        cand = [k for k, l in enumerate(ligands) if l != target and l != drop]
        j = cand[int(np.argmax(sim[i, cand]))]
        return float(level[ligands[j]]), name_of.get(ligands[j], "?")

    probe = []
    for l in ligands:
        if l == TARGET:
            continue
        with_p, with_n = nn_level(l, None)
        without_p, without_n = nn_level(l, TARGET)
        probe.append({"name": name_of.get(l, "?"), "changed": with_n != without_n,
                      "err_with": abs(level[l] - with_p),
                      "err_without": abs(level[l] - without_p)})
    probe = pd.DataFrame(probe)
    print(f"\n1NN ligand-level MAE  with {name_of.get(TARGET)}: {probe.err_with.mean():.4f}"
          f"  without: {probe.err_without.mean():.4f}"
          f"  ligands whose neighbour changes: {int(probe.changed.sum())}")
    print(probe[probe["changed"]].to_string(index=False))

    # --- paired bootstrap -------------------------------------------------
    per = scores.groupby(["extractant", "tanimoto_cluster", "split_seed"])[
        ["zero_shot_mae", "one_shot_mae"]].mean().reset_index()
    long = pd.concat([per.assign(arm="zero", mae=per["zero_shot_mae"]),
                      per.assign(arm="one", mae=per["one_shot_mae"])])
    print("\n" + paired_chemotype_bootstrap(
        long, {"all 143: zero-shot - one-shot": ("zero", "one")},
        value_column="mae").to_string(index=False))
    print(paired_chemotype_bootstrap(
        long[long["extractant"].isin(sulfur)],
        {"11 S-donors: zero-shot - one-shot": ("zero", "one")},
        value_column="mae").to_string(index=False))

    # --- UNPAIRED subgroup contrasts, chemotype-block bootstrap -----------
    # These are group-vs-group, not paired, and both subgroups were chosen after
    # seeing which ligands failed.  Exploratory: report the interval, not a verdict.
    block_of = oof.drop_duplicates("extractant").set_index("extractant")["tanimoto_cluster"]
    ps = [l for l in ligands if mech.loc[l, "mech__n_thiophosphoryl_S"] > 0]
    ref_index = list(reference["extractant"])
    complement = [l for l in ref_index if l not in sulfur]
    print(f"\nunpaired contrasts against the {len(complement)} non-S ligands with >=4 rows "
          "(chemotype-block bootstrap, 95%):")
    lookup = reference.set_index("extractant")
    for label, group, column in (
            ("10 S-donors excl. TWE-24, zero-shot", [l for l in sulfur if l != TARGET], "zero"),
            ("11 S-donors incl. TWE-24, zero-shot", sulfur, "zero"),
            ("5 P=S excl. TWE-24,      zero-shot", [l for l in ps if l != TARGET], "zero"),
            ("5 P=S excl. TWE-24,      k=1 random", [l for l in ps if l != TARGET], "one_rand"),
            ("6 P=S incl. TWE-24,      k=1 random", ps, "one_rand")):
        group = [l for l in group if l in lookup.index]
        point, low, high = block_contrast(
            lookup.loc[group, column].to_numpy(float), [block_of[l] for l in group],
            lookup.loc[complement, column].to_numpy(float), [block_of[l] for l in complement])
        print(f"  {label}: {lookup.loc[group, column].mean():.3f} vs "
              f"{lookup.loc[complement, column].mean():.3f}  diff {point:+.3f} "
              f"CI95 [{low:+.3f}, {high:+.3f}]")

    # --- upstream provenance: the solvent the bundle throws away ----------
    upstream(cohort, oof, name_of)

    # --- the compact S-donor table ---------------------------------------
    agg = slopes.groupby("name").agg(axes=("axis", lambda v: "/".join(sorted(set(v)))),
                                     true_slope=("true_slope", "mean"),
                                     pred_slope=("pred_slope", "mean")).reset_index()
    rows = []
    for l in sulfur:
        nm = name_of.get(l, "?")
        i = index[l]
        j = int(np.argmax(sim[i]))
        d = d_mech[i].copy()
        d[i] = np.inf
        k = int(np.argmin(d))
        r = table[table["extractant"] == l].iloc[0]
        g = agg[agg["name"] == nm]
        rows.append({"name": nm, "donors": donors(l),
                     "softness": float(mech.loc[l, "mech__softness_mean"]),
                     "n_rows": int(r["n"]),
                     "nn_ecfp": f"{name_of.get(ligands[j], '?')} ({sim[i, j]:.2f})",
                     "nn_mech": name_of.get(ligands[k], "?"),
                     "axis": g["axes"].iloc[0] if len(g) else "no usable curve",
                     "true_slope": float(g["true_slope"].iloc[0]) if len(g) else np.nan,
                     "pred_slope": float(g["pred_slope"].iloc[0]) if len(g) else np.nan,
                     "zero_shot": r["zero"], "oracle_level": r["oracle"],
                     "level_share": 1 - r["oracle"] / r["zero"],
                     "k1_rand": r["one_rand"], "k1_best": r["one_best"],
                     "k1_worst": r["one_worst"], "k2_rand": r["two_rand"]})
    compact = pd.DataFrame(rows).sort_values("zero_shot", ascending=False)
    compact.to_csv(here / "sulfur_compact.csv", index=False)
    print("\n" + md_table(compact))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
