# gen14_direction

Gen14: the lanthanide separation curve predicted from **one bit and one scalar** — the direction of
selectivity, called from donor topology, times a constant magnitude.  See `GEN14_REPORT.md`.

Headline (design BP, publication-masked, the design deployment is chosen under):

* direction of selectivity: **0.821** macro accuracy over 82 extractants, against 0.768 for gen13's
  locked estimator on the same 39 columns and 0.559 for the constant rule;
* pairwise log SF: **0.500** extractant-macro MAE, against 0.552 for gen13's 209-column regression,
  0.536 for gen13's deployed bag and 0.622 for the corpus mean curve;
* the same model scores 0.491–0.500 under *all five* hold-out designs — the regression spans
  0.437–0.552.

Run:

    .venv/Scripts/python.exe gen14_direction/scripts/g14_baseline.py     # reproduce + five designs
    .venv/Scripts/python.exe gen14_direction/scripts/g14_sweep.py        # candidate round 1
    .venv/Scripts/python.exe gen14_direction/scripts/g14_sweep2.py       # candidate round 2
    .venv/Scripts/python.exe gen14_direction/scripts/g14_value.py        # the MAE endpoint
    .venv/Scripts/python.exe gen14_direction/scripts/g14_magnitude.py BP # amplitude priors
    .venv/Scripts/python.exe gen14_direction/scripts/g14_ceiling.py BP   # ceilings
    .venv/Scripts/python.exe gen14_direction/scripts/g14_steric.py       # the falsified post-hoc idea
    .venv/Scripts/python.exe gen14_direction/scripts/g14_tables.py       # regenerate results/TABLES.md

Predict from a SMILES string:

    .venv/Scripts/python.exe gen14_direction/scripts/g14_predict.py fit
    .venv/Scripts/python.exe gen14_direction/scripts/g14_predict.py predict --smiles "CCN(CC)C(=O)COCC(=O)N(CC)CC"

`cache/bench.pkl` is a disk cache of the frozen gen13 bench; delete it to rebuild.
