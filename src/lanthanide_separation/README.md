# lanthanide_separation — the gen2 to gen11 library

Installed as the package `lanthanide-separation` (`pip install -e .`, see `pyproject.toml`); the
scripts under `scripts/` and the tests under `tests/` import it. gen12 onwards do not live here;
each has its own package under `generations/<gen>/` and imports this one through its `paths` or
`bootstrap` module. Modules in the order the programme built them:

## Pair target (`log SF`), gen2–gen4

| module | purpose |
|---|---|
| `pairs.py` | build leakage-auditable lanthanide selectivity pairs from the bundle |
| `cohort.py` | cohort and missingness accounting for the extended benchmark |
| `feature_registry.py` | explicit, leakage-auditable feature families (A0–A6) |
| `ligand_descriptors.py` | extended 2D ligand descriptors for amide-type extractants |
| `geometry_descriptors.py` | invariant metal-site descriptors from the bundled VR coordinates |
| `electronic.py` | provenance-audited electronic and energetic descriptors |
| `pair_response.py` | explicit pair-response geometry (how one cavity answers a metal swap) |
| `novelty.py` | structural-novelty sensitivity for leave-extractant-out evaluation |
| `evaluation.py` | nested group-held-out evaluation for paired 2D and Delta3D models |
| `ablation.py` | the leakage-safe all-pairs feature-family ablation |
| `simplicial.py`, `deep_evaluation.py` | metal-centred simplicial neural model and its guarded evaluation |
| `gen3_protocol.py`, `gen3_data.py`, `gen3_models.py`, `gen3_metrics.py`, `gen3_evaluation.py` | the frozen gen3 protocol layer over the unchanged gen2 cohort |
| `gen4_candidates.py` | gen4 candidate estimators on top of the frozen A2 champion |
| `kshot_calibration.py` | the first k-shot study: per-extractant calibration of OOF pair predictions |

## Level target (`log D`), gen5–gen11

| package | purpose |
|---|---|
| `levels.py` | gen5: level cohort, features and estimator (`DropAllNaNColumns` lives here) |
| `gen6/` | the diversity generation: `cohorts` (BASE91 vs EXPANDED152, the seeded group k-fold every later generation uses), `chemistry` (the frozen chemistry map), `provenance`, `metrics` (level / shape / offset), `hierarchical`, `acquisition`, `manifest` |
| `gen7/` | the architecture generation: `harness` (the evaluation contract), `contenders`, `decompositions`, `hierarchical_nn`, `kernels`, `pairwise`, `ligand_physics`, `recovered`, `quality`, `suites` |
| `gen8/` | few-shot calibration: `adapters`, `protocols`, `evaluate`, `kshot`, `curve_baselines`, `cnp`, `physics_latent`, `slope_restore`, `mechanism`, `series`, `data_v2`, `inference` (the chemotype-blocked paired bootstrap and BCa reused by gen13–gen20), `report` |
| `gen9/` | curve shape: `curves`, `relative`, `objective` (`CurveBoost`), `train`, `metrics`, `series_adapter`, `acquisition`, `audit` |
| `gen10/` | finalisation: `architectures` (fit once, predict a query), `runner`, `features`, `querycurves`, `perturb`, `consistency`, `setcontext`, `adaptation` |
| `gen11/` | multi-metal transfer: `pools`, `overlap`, `leakage_audit`, `auxfeatures`, `metalrep`, `arms`, `transfer`, `composition`, `negative`, `analysis`, `runner` |
