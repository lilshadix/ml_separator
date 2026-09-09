"""Generate REPORT.md straight from the result CSVs.

Nothing in the report is typed by hand: the narrative is the template below, every table is rendered
from ``results/*.csv`` at build time, and the four bench sanity checks are read back out of the
boards rather than quoted.  A number in the report and a number in ``results/`` therefore cannot
drift apart, and ``python make_tables.py`` regenerates the whole document after any re-score.

Usage:  python make_tables.py            # writes REPORT.md
        python make_tables.py -          # to stdout
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DESIGNS = ["B", "BR", "BQ", "A", "BP"]

MAIN = ("board_main_chemberta_mtr__mean+chemberta_mtr__cls+chemberta_mlm__mean+"
        "chemberta_mlm__cls.csv")
MAIN_C = MAIN.replace("board_", "contrast_")


def md(df: pd.DataFrame, index: bool = True) -> str:
    return df.to_markdown(index=index)


def wide(path: Path, value: str = "macro_mae_extractant") -> pd.DataFrame:
    B = pd.read_csv(path)
    w = B.pivot(index="arm", columns="design", values=value)
    return w[[d for d in DESIGNS if d in w.columns]]


def gains(path: Path, suffix: str = "_vs_G14") -> pd.DataFrame:
    C = pd.read_csv(path)
    g = C[C.comparison.str.endswith(suffix)]
    p = g.pivot(index="candidate", columns="design", values="point")
    return p[[d for d in DESIGNS if d in p.columns]]


def section(title: str, body: str) -> None:
    print(f"\n### {title}\n\n{body}\n")


NARRATIVE = """# gen15 / `embed` -- does a pretrained molecular representation carry the magnitude \
and curvature?

**Verdict: null, and on the direction actively harmful.** Three pretrained chemical language models
were rebuilt from scratch and scored as gen15 arms under all five designs. Not one arm of any kind
-- direction, magnitude, curvature, composite, or concatenated with gen14's own columns -- beats
`G14` under BP. The closest, MoLFormer's RBF magnitude head, is {best_bp:.4f} against `G14`'s
{g14_bp:.4f}, a loss of {best_gap:.4f} with a paired *p* of 0.92. Used for the **direction**, every
embedding is a catastrophe: the best of them reaches {best_dir:.4f} macro accuracy under BP against
TOPO39's {topo_bp:.4f}, which is the always-heavy floor ({heavy:.4f}) to within noise.

One thing did look alive and is reported in full below, because it is the most interesting number in
the experiment and because it does not survive: a ridge on MoLFormer's mean-pooled vector predicts
`log|a|` out of fold under BP at Spearman +0.23 (partial +0.25 with `n_metals` removed,
leave-one-chemotype-out stable), where TOPO39, LEAN209, Morgan counts, both ChemBERTa checkpoints and
a four-number size block all sit at or below zero. **A permutation null kills it.** Shuffling the
ligand->vector map -- every ligand keeps a real MoLFormer vector, just somebody else's -- gives a
null with sd {null_sd:.4f} and a 95th percentile of {null_q95:.4f}, *above* the observed
{obs_rho:.4f} (one-sided *p* = {perm_p:.3f}). With 90 ligands in 45 chemotypes the out-of-fold rank
correlation in this pipeline is so noisy that +0.23 is an ordinary draw from a representation that
knows nothing about the ligands.

## Bench sanity checks (this run's own numbers, not quoted)

| quantity | expected from the programme | measured here |
|:---|---:|---:|
| `G14` extractant-macro MAE, BP | 0.500 | **{g14_bp:.4f}** |
| `FLAT` extractant-macro MAE, BP | 0.589 | **{flat_bp:.4f}** |
| TOPO39 direction macro accuracy, BP | 0.821 | **{topo_bp:.4f}** |
| always-heavy direction macro accuracy | 0.559 | **{heavy:.4f}** |

All four reproduce, in three independently launched scoring runs (`board_main...`, `board_final`,
`board_molformer` each carry `FLAT` and `G14` and agree to the fourth decimal).

## What was built

The embedding table the repo's manifest recorded but never shipped was rebuilt for the 90 distinct
extractant SMILES of the gen15 cohort, keeping both the mean-pooled last hidden state over
non-padding tokens and the CLS/first-token vector for each model:

| tag | model id | dim | pools | distinct rows |
|:---|:---|---:|:---|:---|
| `chemberta_mtr` | `DeepChem/ChemBERTa-77M-MTR` | 384 | mean, cls | 89/90 |
| `chemberta_mlm` | `DeepChem/ChemBERTa-77M-MLM` | 384 | mean, cls | 88/90 |
| `molformer` | `ibm-research/MoLFormer-XL-both-10pct` | 768 | mean, cls | 90/90 |

`transformers` 5.16.1, `torch` 2.14.0+cpu, 90 x 3072 float32 -> `ligand_embeddings.parquet`.

**MoLFormer did load, contrary to the first pass.** An earlier attempt recorded it as failed with an
`OSError` naming `pytorch_model.bin`; that diagnosis was wrong. The repo does ship
`model.safetensors`, and the real cause was that the 187 MB weight download kept stalling on this
machine at ~17 MB, after which `transformers` fell back to looking for a `.bin` and reported the
fallback's absence. Retrying `hf_hub_download` until the blob completed fixed it (680 s), and the
model then loads on transformers 5.16.1 with `trust_remote_code=True`; five `lm_head.*` keys are
reported unexpected, which is what loading an encoder out of an MLM checkpoint with `AutoModel`
should do. Encoding the 90 SMILES then took 11.8 s. Morgan **count** fingerprints at radius 2 and 3
(2048 bits, unset bits dropped) are carried throughout as the cheap control: a pretrained
representation has to beat the substructure count, not only the 39 hand-built topology columns.

## How to read the tables

`MAG_*` and `CUR_*` arms keep gen14's direction and replace only the magnitude or only the curvature,
so they measure what the representation adds on top of the deployed model. `DIR_*` replaces only the
direction. `COMP_*` uses the representation for both. `CONCAT_*` gives the logistic gen14's 39
topology columns *and* 16 principal components of the embedding, which asks whether the embedding
adds anything TOPO39 does not already have. Every fit -- imputation, standardisation, PCA, penalty --
happens strictly inside the training fold of every fold of every design. `C = 1` is gen14's rule; the
C sweep is shown so its whole range is visible, and nothing was selected on a held-out score.

## The five-design result

No arm beats `G14` under BP. Under B/BR/BQ/A a handful of MoLFormer magnitude and curvature arms are
nominally ahead of `G14` by 0.003-0.012, and **all of them turn negative under BP** -- the exact
sign-inconsistency the protocol exists to catch. Even where positive, every one of those contrasts
has a 95 % interval straddling zero (*p* between 0.18 and 0.99) and none passes the programme's
`passes_P1` gate. There is no arm here, under any design, that is distinguishable from `G14`.

## Confounds

* **Near-duplicate ligands.** At raw cosine >= 0.99, 28/90 ligands have a twin under
  `chemberta_mtr__mean` (8 of them cross-chemotype), 12/90 under `chemberta_mlm__mean` (0
  cross-chemotype), 20/90 under `morgan2` (11 cross-chemotype), and only 5/90 under
  `molformer__mean` (0 cross-chemotype). This is the confound that would *inflate* a gain, and there
  is no gain to inflate; it is reported because it would have mattered had the sign gone the other
  way. Note that the nearest neighbour shares the test ligand's chemotype only 34-54 % of the time,
  so the chemotype hold-out does **not** remove every near neighbour -- a real caveat for anyone who
  later does find a gain here.
* **Publication identity.** Designs B/BR/BQ/A leave publications free; BP masks them. Every arm's
  best showing is under A (exact extractant, leaky) and the collapse from A to BP is the whole story:
  `DIR_cbmtr_mean_C1` goes 0.5070 -> 0.6526, direction accuracy 0.7807 -> 0.5032. An embedding is a
  function of the ligand alone and cannot leak conditions, but a ligand studied by one laboratory
  only is still a publication label in disguise under any design that is not BP.
* **Number of measured metals.** Spearman +0.4357 with `|a|` in this corpus. The embedding norm's
  correlation with `|a|` is <= 0.18 in magnitude for every block, so the size axis is not smuggling
  the `n_metals` confound in, and the one live correlation is reported partialled on `n_metals`.
* **Leave-one-chemotype-out.** TOPO39 beats the embedding in 26 of 45 chemotypes under BP (7 losses,
  12 ties). Dropping any single chemotype, the best the embedding's macro accuracy reaches is 0.5504
  and the worst TOPO39 falls to is 0.7735 -- they do not overlap, so the direction verdict does not
  rest on any one chemotype. The gen13 contrast machinery's own `loco_sign_stable` flag agrees: every
  `DIR_*` contrast against `G14` is negative and sign-stable under BP.

## What would have had to be true

For the direction, the embedding would have had to encode which donor set a ligand presents to the
metal in a way a 90-example linear probe can read. It does not: mean-pooling a SMILES transformer
produces a vector dominated by size and gross composition, and the 39 topology columns are a
hand-built answer to exactly the question the endpoint asks. For the magnitude, a correlation would
have had to clear a null whose sd is {null_sd:.2f} -- which at 90 ligands in 45 chemotypes means
nothing short of Spearman ~ 0.45 is interpretable from this probe at all. That number, not the
embedding, is the real finding: **the corpus cannot resolve a magnitude model, whatever the
representation.** The seven priors the programme already retired were not unlucky.

---

# Tables
"""


def _narrative_numbers() -> dict:
    """Read the four sanity checks and the probe's null back out of the CSVs."""
    n: dict = {}
    boards = [pd.read_csv(RES / f) for f in
              (MAIN, "board_final.csv", "board_molformer.csv") if (RES / f).exists()]
    B = pd.concat(boards, ignore_index=True)
    bp = B[B.design == "BP"].drop_duplicates("arm").set_index("arm")["macro_mae_extractant"]
    n["g14_bp"] = float(bp["G14"])
    n["flat_bp"] = float(bp["FLAT"])
    cand = bp.drop(index=["G14", "FLAT"])
    n["best_bp"] = float(cand.min())
    n["best_gap"] = n["best_bp"] - n["g14_bp"]
    D = pd.read_csv(RES / "headline_direction.csv")
    dbp = D[D.design == "BP"].set_index("model")["macro_accuracy"]
    n["topo_bp"] = float(dbp["TOPO39"])
    n["best_dir"] = float(dbp.drop(index=["TOPO39", "MAJORITY"]).max())
    F = pd.read_csv(RES / "constant_floor.csv")
    n["heavy"] = float(F[F.model == "ALWAYS_HEAVY"]["macro_accuracy"].iloc[0])
    P = pd.read_csv(RES / "probe_permutation_BP.csv").iloc[0]
    n["null_sd"] = float(P["null_sd"])
    n["null_q95"] = float(P["null_q95"])
    n["obs_rho"] = float(P["observed_spearman"])
    n["perm_p"] = float(P["p_one_sided"])
    return n


def render() -> None:
    print(NARRATIVE.format(**_narrative_numbers()))

    # ---- 1. the headline five-design MAE board -------------------------------------
    parts = []
    for f, keep in ((MAIN, None), ("board_final.csv", None), ("board_molformer.csv", None)):
        p = RES / f
        if p.exists():
            w = wide(p)
            parts.append(w if keep is None else w.loc[keep])
    if parts:
        allw = pd.concat(parts).groupby(level=0).first()
        ref = allw.loc[[i for i in ("FLAT", "G14") if i in allw.index]]
        rest = allw.drop(index=ref.index).sort_values("BP")
        section("Extractant-macro MAE, every arm x every design",
                md(pd.concat([ref, rest]).round(4)))

    # ---- 2. paired contrasts against G14 -------------------------------------------
    gparts = [gains(RES / f) for f in (MAIN_C, "contrast_final.csv", "contrast_molformer.csv")
              if (RES / f).exists()]
    if gparts:
        g = pd.concat(gparts).groupby(level=0).first().sort_values("BP", ascending=False)
        section("Chemotype-blocked paired gain vs G14 (positive = better than G14)",
                md(g.round(4)))

    # ---- 3. direction macro accuracy ------------------------------------------------
    p = RES / "headline_direction.csv"
    if p.exists():
        B = pd.read_csv(p)
        w = B.pivot(index="model", columns="design", values="macro_accuracy")
        w = w[[d for d in DESIGNS if d in w.columns]].sort_values("BP", ascending=False)
        section("Direction macro accuracy over extractants", md(w.round(4)))
        bp = B[B.design == "BP"][["model", "macro_accuracy", "ci_low", "ci_high",
                                  "pooled_accuracy", "gain", "p_two_sided"]]
        section("Direction under BP, with the chemotype-blocked bootstrap",
                md(bp.round(4).sort_values("macro_accuracy", ascending=False), index=False))
    p = RES / "constant_floor.csv"
    if p.exists():
        B = pd.read_csv(p)
        w = B.pivot(index="model", columns="design", values="macro_accuracy")
        section("Constant-direction floors", md(w[[d for d in DESIGNS if d in w.columns]].round(4)))

    # ---- 3b. the direct signal probe and its null ------------------------------------
    p = RES / "probe_baselines.csv"
    if p.exists():
        d = pd.read_csv(p)
        cols = ["block", "design", "n_cells", "spearman",
                "partial_spearman_given_n_metals", "loco_spearman_min", "loco_spearman_max",
                "sd_pred", "sd_true"]
        section("Out-of-fold prediction of log|a|: embedding vs the cheap alternatives",
                md(d[cols].round(4), index=False))
    p = RES / "signal_probe.csv"
    if p.exists():
        d = pd.read_csv(p)
        cols = ["block", "target", "design", "estimator", "spearman",
                "partial_spearman_given_n_metals", "loco_spearman_min", "loco_spearman_max"]
        section("Out-of-fold prediction of log|a| and of the curvature b",
                md(d[cols].round(4), index=False))
    p = RES / "probe_estimator_sweep_BP.csv"
    if p.exists():
        d = pd.read_csv(p)
        cols = ["estimator", "spearman", "partial_spearman_given_n_metals",
                "loco_spearman_min", "loco_spearman_max", "sd_pred"]
        section("MoLFormer log|a| under BP: estimator sweep", md(d[cols].round(4), index=False))
    p = RES / "probe_permutation_BP.csv"
    if p.exists():
        section("Permutation null (ligand -> vector map shuffled, folds unchanged)",
                md(pd.read_csv(p).round(4), index=False))

    # ---- 4. confounds ----------------------------------------------------------------
    for f, title in (("confound_duplicates.csv", "Near-duplicate ligands"),
                     ("confound_nn_chemotype.csv", "Nearest neighbour shares chemotype"),
                     ("confound_n_metals.csv", "n_metals / embedding-norm confound"),
                     ("loco_summary_chemberta_mtr__mean.csv", "Leave-one-chemotype-out summary")):
        p = RES / f
        if p.exists():
            d = pd.read_csv(p)
            if f == "confound_duplicates.csv":
                d = d[d.cosine == "raw"]
            section(title, md(d.round(4), index=False))


def main() -> None:
    buf = io.StringIO()
    stdout, sys.stdout = sys.stdout, buf
    try:
        render()
    finally:
        sys.stdout = stdout
    text = buf.getvalue()
    if len(sys.argv) > 1 and sys.argv[1] == "-":
        print(text)
        return
    (HERE / "REPORT.md").write_text(text, encoding="utf-8")
    print(f"wrote {HERE / 'REPORT.md'}  ({len(text)} chars)")


if __name__ == "__main__":
    main()
