# Figure refinement

Publication-quality rebuilds of the research figures from two repositories, one figure per
folder. **Nothing here overwrites an original.** The source renders stay where they were
(`ml_separator/runs/*/figures/`, `lanthanidestrain/automl/reports/figures/`); this tree is
additive.

```
figure_refinement/
├── common/
│   ├── pubstyle.py         palette, typography, panel letters, export + layout linter
│   ├── mlsep_data.py       ml_separator data access (paths, derived tables, bootstraps)
│   └── verify_exports.py   checks every export: PNG dpi, PDF vector-ness, PNG/PDF agreement
├── ml_separator/
│   ├── SUMMARY.md
│   ├── SCIENTIFIC_DEFECTS.md
│   └── figure_NN_<slug>/
│       ├── figure.png       600 dpi
│       ├── figure.pdf       vector, same bounding box as the PNG
│       ├── figure_script.py runs standalone from anywhere; regenerates this figure
│       ├── figure.lint.json layout-linter report for the committed render
│       ├── values.json      every number the figure plots, with its cohort
│       └── notes.md         source / problem / changes / scientific integrity / role / message
└── lanthanidestrain/
    ├── SUMMARY.md
    ├── SCIENTIFIC_DEFECTS.md
    ├── REPRODUCIBILITY.md   what regenerates from the upstream repo, and what does not
    ├── data/                the upstream result tables, vendored (PROVENANCE.json records
    │                        the commit) so each figure_script.py runs without re-cloning
    └── figure_NN_<slug>/
        ├── figure.png / figure.pdf / figure_script.py / figure.lint.json / values.json
        ├── notes.md
        └── original.png     the source render, kept for the before/after record
```

**Read these first.** Neither repository's story is only about layout:

* `ml_separator/SCIENTIFIC_DEFECTS.md` — a transfer matrix whose directional title its own
  estimator cannot support, and a waterfall that double-counts three split seeds.
* `lanthanidestrain/SCIENTIFIC_DEFECTS.md` — a headline comparison against a baseline the
  project itself has superseded, and a committed figure asserting a result the study
  withdrew.
* `lanthanidestrain/REPRODUCIBILITY.md` — 19 of 37 figures regenerate from what is
  committed; 18 do not.

Where the audit found a genuine implementation defect it is **documented, not silently
corrected**. Every claim in those three files was re-derived by hand from the repositories'
own artefacts; none rests on an audit report alone.

## The layout linter

The recurring failure in this material is a figure that renders without error and is still
unreadable: an annotation on top of a line, a legend over the data, a value label clipped by
its own bar, a panel letter on a tick label. Eyeballing thirty figures misses these.

`pubstyle.lint(fig)` reads the drawn artist geometry and reports three decidable defects:

| check | what it catches |
|---|---|
| `text-overlap` | two visible text artists whose boxes intersect by more than 1.5 pt² |
| `clipped` | a text artist with clipping on whose box escapes the axes that clips it |
| `layout-overflow` | the drawn extent exceeds the declared figure size by more than 14 %, i.e. the panels will be smaller than intended once the figure is placed in a column |

It excludes tick labels that matplotlib creates for locator positions outside the view
limits — they are never drawn but sit at arbitrary positions and would otherwise produce a
stream of false collisions.

`pubstyle.save(fig, folder, "figure")` lints, then writes the PNG at 600 dpi and the PDF from
the same figure object, and records the report in `figure.lint.json`. Every finalised figure
in this tree reports `ok: true`.

A figure still has to be **looked at**: the linter cannot see text sitting on a line, a
legend covering a trend, or a series you cannot tell from its neighbour. Every figure here
went through at least two render-inspect-fix cycles on top of the mechanical check.

## Regenerating

Each `figure_script.py` is standalone:

```bash
.venv/bin/python figure_refinement/ml_separator/figure_02_few_shot_frontier/figure_script.py
```

To check every export at once:

```bash
.venv/bin/python figure_refinement/common/verify_exports.py
```

## House rules applied throughout

* Reader-facing names in the artwork — no `GEN9_SHAPE_RECOMPOSED`, no `MAX_ENSEMBLE_SD` —
  unless the figure's subject *is* the arm registry. Mappings are recorded in `notes.md`.
* Every axis label carries its unit; every sign convention is stated.
* Oracle and other non-deployable arms are visually distinct and carry the words
  "not deployable" in the artwork, not only in the caption.
* No figure titles: the paper caption carries the explanation. A short grey strapline
  orients a panel where one is needed, and never states a claim.
* Legends do not cover data; where a legend needs room, the axis limit reserves it
  explicitly rather than leaving it to chance.
* Sample size is visible wherever a reader could otherwise be misled by a small cell.
* Uncertainty is shown wherever the comparison being drawn is of the same order as the
  spread behind it.
* Palette is Okabe-Ito, colour-blind safe; identity never rests on colour alone.
* Aesthetic changes stay aesthetic. Where the audit found a genuine implementation defect,
  it is documented in `SCIENTIFIC_DEFECTS.md` and **not** silently corrected.
