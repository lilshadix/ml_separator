"""Stage 1 -- ingest the immutable raw export and collapse the per-metal fan-out.

The archive was exported as one CSV per *queried* metal.  A single archive
record therefore appears once in every file whose metal was present in that
experimental system, which is why 48,471 CSV rows carry only 16,770 distinct
``exp_id`` values.

This stage proves that the fan-out is a pure export artefact (every one of the
37 raw columns is constant inside an ``exp_id`` group) before collapsing it, and
records the full file membership so any raw row can be recovered.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sae_paths import RAW_DIR, INTERMEDIATE_DIR, AUDIT_DIR, ensure_dirs  # noqa: E402

RAW_COLUMNS_EXPECTED = 37


def load_raw() -> pd.DataFrame:
    """Read every raw CSV as text, with a stable per-row identifier."""
    frames = []
    for path in sorted(RAW_DIR.glob("*.csv")):
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])
        if len(frame.columns) != RAW_COLUMNS_EXPECTED:
            raise ValueError(f"{path.name}: expected {RAW_COLUMNS_EXPECTED} columns, got {len(frame.columns)}")
        frame["source_file"] = path.name
        # 1-based line number in the CSV *including* the header, so a reviewer
        # can open the file and jump straight to the row.
        frame["source_line_number"] = range(2, len(frame) + 2)
        frame["raw_row_id"] = [f"{path.stem}:{n}" for n in frame["source_line_number"]]
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no CSV files under {RAW_DIR}")
    raw = pd.concat(frames, ignore_index=True)
    raw["source_record_id"] = raw["exp_id"].astype(str).str.strip()
    return raw


def verify_fanout_is_metadata_only(raw: pd.DataFrame) -> dict:
    """Fail loudly if any *scientific* column varies inside an exp_id group.

    Collapsing the fan-out is only legitimate because the duplicated rows are
    byte-identical outside the export metadata.  This is checked, not assumed.
    """
    payload = [c for c in raw.columns
               if c not in ("source_file", "source_line_number", "raw_row_id", "source_record_id")]
    varying = raw.groupby("source_record_id")[payload].nunique(dropna=False)
    offenders = varying.columns[(varying > 1).any()].tolist()
    report = {
        "raw_rows": int(len(raw)),
        "distinct_exp_ids": int(raw["source_record_id"].nunique()),
        "payload_columns_checked": len(payload),
        "columns_varying_within_exp_id": offenders,
        "exp_id_groups_with_variation": int((varying > 1).any(axis=1).sum()),
    }
    if offenders:
        raise AssertionError(
            "The per-metal fan-out is NOT metadata-only: these columns differ inside an "
            f"exp_id group and must be treated as distinct measurements: {offenders}"
        )
    return report


def collapse(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(records, mapping)``.

    ``records`` has one row per archive record; ``mapping`` keeps every raw row
    with the canonical record it was folded into.
    """
    raw = raw.sort_values(["source_record_id", "source_file", "source_line_number"], kind="stable")

    grouped = raw.groupby("source_record_id", sort=True)
    records = grouped.head(1).copy()

    files = grouped["source_file"].apply(lambda s: sorted(set(s)))
    lines = grouped["raw_row_id"].apply(lambda s: sorted(set(s)))
    records = records.set_index("source_record_id")
    records["export_source_files"] = files
    records["export_metals_queried"] = files.apply(lambda fs: [f[:-4] for f in fs])
    records["raw_row_ids"] = lines
    records["export_fanout_size"] = lines.apply(len)
    records = records.reset_index()

    records["canonical_measurement_id"] = "SAE:" + records["source_record_id"]

    mapping = raw[["raw_row_id", "source_file", "source_line_number", "source_record_id"]].copy()
    mapping["canonical_measurement_id"] = "SAE:" + mapping["source_record_id"]
    mapping["fold_reason"] = "EXPORT_FANOUT_PER_METAL_FILE"
    first = set(records["raw_row_id"])
    mapping["is_representative_raw_row"] = mapping["raw_row_id"].isin(first)
    return records, mapping


def main() -> None:
    ensure_dirs()
    raw = load_raw()
    fanout_report = verify_fanout_is_metadata_only(raw)
    records, mapping = collapse(raw)

    fanout_report["records_after_fanout_collapse"] = int(len(records))
    fanout_report["raw_rows_folded_away"] = int(len(raw) - len(records))
    fanout_report["fanout_size_histogram"] = {
        str(k): int(v) for k, v in records["export_fanout_size"].value_counts().sort_index().items()
    }

    raw.to_parquet(INTERMEDIATE_DIR / "raw_rows.parquet", index=False)
    records.to_parquet(INTERMEDIATE_DIR / "records_collapsed.parquet", index=False)
    mapping.to_parquet(INTERMEDIATE_DIR / "raw_row_mapping.parquet", index=False)
    (AUDIT_DIR / "stage01_fanout_report.json").write_text(
        json.dumps(fanout_report, indent=2, sort_keys=True) + "\n")

    print(json.dumps({k: v for k, v in fanout_report.items() if k != "fanout_size_histogram"}, indent=2))
    print("fanout histogram:", fanout_report["fanout_size_histogram"])


if __name__ == "__main__":
    main()
