"""Stage 4 -- reconstruct experimental series and curves.

The definitions mirror ``src/lanthanide_separation/gen8/series.py`` so that the
cleaned multi-metal table can be fed to the existing k-shot machinery without
re-deriving anything:

``series``  one extractant system x one setting of every *categorical* condition
            (acid identity, diluent, modifier, aqueous agents).  The metal and
            the continuous axes may vary inside a series.
``curve``   a maximal subset of one series in which exactly one axis varies and
            every other axis is held fixed.

Two deliberate rules carried over from gen8:

* concentration axes are read on the log10 scale, because mass action is linear
  in log concentration and a linear-scale slope would not be the stoichiometric
  coefficient;
* ``log_D`` is never read to decide series or curve membership.  Grouping is a
  function of the conditions and the metal alone, so no target information can
  leak into the task definition.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sae_paths import INTERMEDIATE_DIR, AUDIT_DIR, ensure_dirs  # noqa: E402

# gen8 compatibility constants.
MIN_CURVE_POINTS = 3
MIN_CURVATURE_POINTS = 4
GROUP_ROUND_DECIMALS = 6

#: Categorical conditions that define a series.  Two records with different
#: values here are different chemical systems, not two points on one curve.
SERIES_CATEGORICAL = (
    "extractant_system_key", "acid_signature", "solvent_key",
    "modifier_name", "complexant_smiles_canonical", "holdback_smiles_canonical",
)

#: Axes a curve may run along.  The first five are gen8's; the rest exist only
#: in the expanded archive and are reported separately so the gen8-comparable
#: counts stay clean.
GEN8_AXES = (
    "acid_concentration_M", "contact_time_min", "extractant_primary_concentration_M",
    "metal_concentration_M", "temperature_C",
)
EXTENDED_AXES = (
    "nitrate_concentration_M", "modifier_concentration_M",
    "complexant_concentration_M", "holdback_concentration_M", "phase_ratio_org_aq",
)
CURVE_AXES = GEN8_AXES + EXTENDED_AXES + ("metal",)

#: Axes read on the log10 scale.
LOG_AXES = frozenset({
    "acid_concentration_M", "extractant_primary_concentration_M", "metal_concentration_M",
    "nitrate_concentration_M", "modifier_concentration_M",
    "complexant_concentration_M", "holdback_concentration_M",
})

AXIS_LABEL = {
    "acid_concentration_M": "acid",
    "extractant_primary_concentration_M": "extractant",
    "metal_concentration_M": "metal_concentration",
    "temperature_C": "temperature",
    "contact_time_min": "contact_time",
    "nitrate_concentration_M": "nitrate",
    "modifier_concentration_M": "modifier_concentration",
    "complexant_concentration_M": "complexant_concentration",
    "holdback_concentration_M": "holdback_concentration",
    "phase_ratio_org_aq": "phase_ratio",
    "metal": "metal_series",
}


def series_id(frame: pd.DataFrame) -> pd.Series:
    view = frame[list(SERIES_CATEGORICAL)].to_numpy(dtype=object)
    rows = ("|".join("" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
                     for v in row) for row in view)
    return pd.Series([hashlib.sha1(r.encode()).hexdigest()[:16] for r in rows],
                     index=frame.index, name="series_id")


def axis_values(frame: pd.DataFrame, axis: str) -> np.ndarray:
    """Numeric coordinate of one axis, on the scale a slope should be taken in."""
    if axis == "metal":
        # Lanthanides keep gen8's contraction index; every other metal is
        # ordered by atomic number so that a mixed sweep still has an abscissa.
        index = pd.to_numeric(frame["lanthanide_index"], errors="coerce")
        atomic = pd.to_numeric(frame["atomic_number"], errors="coerce")
        return np.where(np.isfinite(index), index, atomic).astype(float)
    values = pd.to_numeric(frame[axis], errors="coerce").to_numpy(dtype=float)
    if axis in LOG_AXES:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(values > 0, np.log10(np.where(values > 0, values, np.nan)), np.nan)
    return values


def _held_key(frame: pd.DataFrame, held) -> pd.Series:
    parts = []
    for axis in held:
        if axis == "metal":
            parts.append(np.asarray(
                [f"{a}|{b}" for a, b in zip(frame["metal_symbol"].to_numpy(),
                                            frame["metal_oxidation_state"].to_numpy())],
                dtype=object))
        else:
            values = pd.to_numeric(frame[axis], errors="coerce").round(GROUP_ROUND_DECIMALS)
            parts.append(np.asarray([repr(v) for v in values.to_numpy(dtype=float)], dtype=object))
    if not parts:
        return pd.Series(["_"] * len(frame), index=frame.index)
    return pd.Series(["|".join(row) for row in zip(*parts)], index=frame.index)


def _n_distinct(frame: pd.DataFrame, axis: str) -> int:
    if axis == "metal":
        # `.astype(str).agg("|".join)` can leave real floats in a pandas-3
        # object array (the trap gen8/series.py documents), so build the
        # strings in plain Python.
        pairs = {f"{a}|{b}" for a, b in zip(frame["metal_symbol"].to_numpy(),
                                            frame["metal_oxidation_state"].to_numpy())}
        return len(pairs)
    return int(pd.to_numeric(frame[axis], errors="coerce")
               .round(GROUP_ROUND_DECIMALS).nunique(dropna=False))


def build_curves(frame: pd.DataFrame, min_points: int = MIN_CURVE_POINTS) -> pd.DataFrame:
    """One row per (record, curve) membership; a record may join several curves."""
    records = []
    for sid, block in frame.groupby("series_id", sort=True):
        for axis in CURVE_AXES:
            if _n_distinct(block, axis) < min_points:
                continue
            held = [a for a in CURVE_AXES if a != axis]
            keys = _held_key(block, held)
            for key, sub in block.groupby(keys, sort=True):
                if _n_distinct(sub, axis) < min_points:
                    continue
                coord = axis_values(sub, axis)
                finite = np.isfinite(coord)
                if finite.sum() < min_points:
                    continue
                digest = hashlib.sha1(f"{sid}|{axis}|{key}".encode()).hexdigest()[:16]
                n_points = int(finite.sum())
                for measurement_id, value, ok in zip(
                        sub["canonical_measurement_id"].to_numpy(), coord, finite):
                    if not ok:
                        continue
                    records.append({
                        "curve_id": digest, "series_id": sid, "axis": axis,
                        "axis_label": AXIS_LABEL[axis],
                        "canonical_measurement_id": measurement_id,
                        "axis_value": float(value), "n_points": n_points,
                    })
    columns = ["curve_id", "series_id", "axis", "axis_label",
               "canonical_measurement_id", "axis_value", "n_points"]
    return pd.DataFrame(records, columns=columns) if records else pd.DataFrame(columns=columns)


def curve_statistics(frame: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Slope / curvature / monotonicity of log D along each curve's axis.

    These are *descriptive statistics of an already-defined curve*.  The curve
    itself was defined without reading log_D, so computing them here introduces
    no leakage into the grouping.
    """
    values = frame.set_index("canonical_measurement_id")["log_D"]
    meta = frame.set_index("canonical_measurement_id")
    records = []
    for curve_id, block in membership.groupby("curve_id", sort=True):
        ids = block["canonical_measurement_id"].to_numpy()
        x = block["axis_value"].to_numpy(dtype=float)
        y = values.reindex(ids).to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        x_ok, y_ok = x[ok], y[ok]
        record = {
            "curve_id": curve_id,
            "series_id": block["series_id"].iloc[0],
            "axis": block["axis"].iloc[0],
            "axis_label": block["axis_label"].iloc[0],
            "n_members": int(len(block)),
            "n_points_with_target": int(ok.sum()),
            "axis_min": float(np.nanmin(x)) if np.isfinite(x).any() else None,
            "axis_max": float(np.nanmax(x)) if np.isfinite(x).any() else None,
        }
        sub = meta.reindex(ids)
        record["metals"] = sorted({m for m in sub["metal_symbol"].dropna()})
        record["n_metals"] = len(record["metals"])
        record["extractant_system_key"] = sub["extractant_system_key"].dropna().iloc[0] \
            if sub["extractant_system_key"].notna().any() else None
        record["solvent_key"] = sub["solvent_key"].dropna().iloc[0] \
            if sub["solvent_key"].notna().any() else None
        record["acid_signature"] = sub["acid_signature"].dropna().iloc[0] \
            if sub["acid_signature"].notna().any() else None
        record["system_component_class"] = sub["system_component_class"].dropna().iloc[0] \
            if sub["system_component_class"].notna().any() else None
        record["dois"] = sorted({d for d in sub["doi_primary"].dropna()})

        if len(x_ok) >= MIN_CURVE_POINTS and np.ptp(x_ok) > 0:
            order = np.argsort(x_ok)
            xs, ys = x_ok[order], y_ok[order]
            slope, intercept = np.polyfit(xs, ys, 1)
            residual = ys - (slope * xs + intercept)
            ss_tot = float(((ys - ys.mean()) ** 2).sum())
            record["slope"] = float(slope)
            record["r2"] = float(1.0 - (residual ** 2).sum() / ss_tot) if ss_tot > 0 else None
            record["monotonic"] = bool(np.all(np.diff(ys) >= 0) or np.all(np.diff(ys) <= 0))
            record["log_D_range"] = float(ys.max() - ys.min())
            if len(np.unique(xs)) >= MIN_CURVATURE_POINTS:
                record["curvature"] = float(np.polyfit(xs, ys, 2)[0])
            else:
                record["curvature"] = None
            record["usable"] = True
        else:
            record.update(slope=None, r2=None, monotonic=None,
                          log_D_range=None, curvature=None, usable=False)
        records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    ensure_dirs()
    frame = pd.read_parquet(INTERMEDIATE_DIR / "records_grouped.parquet")
    frame = frame.reset_index(drop=True)
    frame["series_id"] = series_id(frame)

    # Curves are built from canonical rows only.  Including the A/B duplicates
    # that stage03 already folded away put 3,819 redundant membership rows into
    # 287 curves, duplicating points at the same abscissa and silently
    # double-weighting them for any k-shot or slope fit downstream.
    canonical = frame[frame["is_canonical_row"]].copy()
    membership = build_curves(canonical)
    frame_for_stats = canonical
    stats = curve_statistics(frame_for_stats, membership)

    frame.to_parquet(INTERMEDIATE_DIR / "records_with_series.parquet", index=False)
    membership.to_parquet(INTERMEDIATE_DIR / "curve_membership.parquet", index=False)
    stats.to_parquet(AUDIT_DIR / "curve_inventory.parquet", index=False)

    usable = stats[stats["usable"]] if len(stats) else stats
    summary = {
        "records": int(len(frame)),
        "canonical_records_used_for_curves": int(len(canonical)),
        "series": int(frame["series_id"].nunique()),
        "curves_total": int(len(stats)),
        "curves_usable": int(len(usable)),
        "records_on_at_least_one_curve": int(membership["canonical_measurement_id"].nunique())
        if len(membership) else 0,
        "curves_by_axis": (usable["axis_label"].value_counts().to_dict() if len(usable) else {}),
        "points_per_curve": {
            "mean": float(usable["n_points_with_target"].mean()) if len(usable) else None,
            "median": float(usable["n_points_with_target"].median()) if len(usable) else None,
            "max": int(usable["n_points_with_target"].max()) if len(usable) else None,
        },
        # Bucketed rather than truncated: the previous ".head(15)" silently
        # dropped every series larger than 15 records, hiding that the largest
        # holds over a thousand.
        "series_size_distribution": {
            label: int(count) for label, count in
            pd.cut(frame["series_id"].value_counts(),
                   bins=[0, 1, 2, 5, 10, 25, 50, 100, 250, 500, 10 ** 9],
                   labels=["1", "2", "3-5", "6-10", "11-25", "26-50", "51-100",
                           "101-250", "251-500", ">500"])
            .value_counts().sort_index().items()},
        "series_size_max": int(frame["series_id"].value_counts().max()),
        "series_size_median": float(frame["series_id"].value_counts().median()),
        "min_curve_points": MIN_CURVE_POINTS,
    }
    (AUDIT_DIR / "stage04_series_report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
