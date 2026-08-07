# Final lanthanide 3D dataset

This directory is self-contained. All asset paths are relative to this directory.

- `dataset.parquet`: all deduplicated experiments; use `geometry_ok` to gate 3D inputs.
- `dataset_geometry_available.parquet`: only rows with accepted 3D geometry.
- `geometries/`: one current dual-QC accepted XYZ per scientific complex.
- `accepted_geometries.csv`: complex-to-XYZ provenance and SHA-256 values.
- `excluded_geometries.csv`: complexes without an accepted geometry and the reason.
- `features/`: tabular 3D blocks, persistence images, and Vietoris-Rips inputs.
- `row_geometry_map.csv`: row-to-geometry and row-to-asset crosswalk.

Minimal Python usage:

```python
from pathlib import Path
import pandas as pd

root = Path("final_dataset_3d")
df = pd.read_parquet(root / "dataset.parquet")
rows_3d = df[df["geometry_ok"]].copy()
rows_3d["xyz_file"] = rows_3d["xyz_path"].map(root.__truediv__)
```
