# gen7 literature review — architectures for absolute log D under a chemotype hold-out

**Scope.** Four topics were surveyed against one problem: predict absolute `log_D` for
Ln(III) solvent extraction on the frozen gen6/gen7 cohort (5,248 rows / 152 extractants /
131 ECFP clusters / 79 Tanimoto-0.7 chemotypes, `runs/gen7_architecture/oracles/summary.json`
cohort fingerprint `bed178ec1a7a82b0`), evaluated by macro MAE over ECFP clusters under a
chemotype hold-out. Incumbent to beat: **1.0468** (gen6 `MONO_ET`,
`runs/gen6_expC_5seed/arm_metrics.csv`); best number measured anywhere in gen7 so far:
**0.9976** (`TREE_MC_ecfp_massaction`, 3 seeds, `runs/gen7_architecture/learners/leaderboard.csv`).

Every entry carries a decision: **TEST** (worth a gen7 slot) or **DONT_TEST** (read, do not
build). Decisions were made on evidence quality *under honest splits*, not on headline
numbers. Hardware constraint throughout: 8 CPUs, no GPU, Python 3.13 / torch 2.13.

**A standing caution that governs the whole review.** Almost every headline in this
literature is a random-split number. Fooladi et al. measured the correlation between
in-distribution and OOD performance collapsing from *r* ≈ 0.93–0.98 (random/scaffold) to
*r* ≈ 0.16–0.60 (similarity clustering). A model or hyperparameter selected on an easy
inner split is being selected on a signal nearly uncorrelated with the number we report.

---

## Topic 1 — Small-data molecular property prediction and chemotype/OOD extrapolation

### 1.1 Fooladi, Vu, Mathea, Kirchmair (2025). *Evaluating ML Models for Molecular Property Prediction: Performance and Robustness on Out-of-Distribution Data.* JCIM 65(19), 9871–9891. — **TEST**
- **URL** https://pmc.ncbi.nlm.nih.gov/articles/PMC12529777/
- **Problem** What counts as OOD; 14 models × 10 splitting strategies.
- **Dataset** 8 binary classification sets, 4,980–12,890 molecules each (CYP isoforms, HIV, AMES, hERG); 10 qualifying splits per strategy-dataset with SDs.
- **Split** Random, Bemis-Murcko scaffold, generic scaffold, MW hi/lo, logP, ECFP4 k-means, max-dissimilarity, UMAP-agglomerative, Lo-Hi (ILP), DataSAIL (ILP).
- **Architecture** Not a model — an evaluation protocol.
- **Why it transfers** Two results bite. (a) Scaffold splits are near-free (median ΔROC-AUC 0.02–0.04) while ECFP4-similarity and UMAP clustering are the genuinely hard family — our Tanimoto-0.7 chemotype split is in the hard family, which validates our protocol. (b) The ID→OOD correlation collapse means any tuning done on an easier inner split is selection on the wrong signal. If gen3–gen6 arm selection ever ran on a non-chemotype inner CV, some negative results are selection artefacts.
- **Why it may fail** All 8 datasets are single-molecule binary classification with 5k–13k *distinct* molecules. We have ~190 ligands — 30–60× smaller on the axis that matters — plus a conditions axis they lack. Their "GNNs are more robust under clustering splits" must **not** be read as "use a GNN".
- **Difficulty** Low. Add matched-difficulty (nested chemotype) inner CV for all model selection; audit current tuning. ~1 day, no new dependency.

### 1.2 Steshin (2023). *Lo-Hi: Practical ML Drug Discovery Benchmark.* NeurIPS D&B. arXiv:2310.06399 — **TEST**
- **URL** https://arxiv.org/abs/2310.06399
- **Problem** Splits where no test molecule resembles any training molecule (ECFP4 Tanimoto < 0.4), via Balanced Vertex Minimum k-Cut.
- **Dataset** ChEMBL target sets, ~1k–10k molecules.
- **Split** Hi split: graph cut until no cross-split edge ≥ 0.4 remains — a **guaranteed** threshold, not a clustering heuristic.
- **Architecture** `lohi_splitter` (pip); baselines GBDT+ECFP4, KNN-Tanimoto, Chemprop.
- **Why it transfers** Tanimoto-0.7 *clustering* does not guarantee every test ligand is distant from every training ligand — single-linkage routinely leaves cross-cluster pairs above threshold. Given our own near-neighbour finding (edge −0.04 below Tanimoto 0.4, +0.66 above 0.8), the exact tail of the train-test similarity distribution determines the headline. GBDT beat Chemprop on the small KDR-Hi task.
- **Why it may fail** Classification tasks; the GBDT>Chemprop result is one dataset. With 190 ligands a strict <0.4 cut may leave a test set too small for stable macro MAE — the same trap as gen5's `min_rows=10` filter.
- **Difficulty** Low. Run on 190 SMILES, compare max cross-split Tanimoto against current folds. Half a day.

### 1.3 Joeres, Blumenthal, Kalinina (2025). *Data splitting to avoid information leakage with DataSAIL.* Nat. Commun. 16. — **TEST**
- **URL** https://www.nature.com/articles/s41467-025-58606-8
- **Problem** Leakage-reduced splitting as combinatorial optimisation (NP-hard), clustering + ILP; supports two heterogeneous entity types simultaneously with stratification.
- **Dataset** Method paper; DTI and protein/molecule sets.
- **Split** 1-D (cold molecule) and 2-D (cold molecule AND cold target) similarity-aware.
- **Architecture** Python splitter, MILP solver.
- **Why it transfers** Ours is a genuine two-dimensional interaction problem (ligand × lanthanide) with a third nuisance axis (publication). The 2-D split is the principled way to enforce chemotype-clean **and** publication-clean folds at once, instead of layering ad-hoc filters — directly relevant given the 26.32% publication mixing in gen5's `unseen_series`.
- **Why it may fail** 190 ligands × 105–115 publications with balanced fold sizes may be infeasible; the solver returns degenerate folds carrying almost no distinct chemistry. Risk: unbiased but uselessly high-variance estimate.
- **Difficulty** Low-medium. ~1 day to wire in.

### 1.4 Ramani, Arora, Kuchhal, Tatarin, Krasnov, Ranu, Karmakar (2026). *SC3: The Multi-Solvent Solubility Challenge and Benchmark.* arXiv:2606.07656 — **TEST**
- **URL** https://arxiv.org/abs/2606.07656
- **Problem** Solubility as f(solute, solvent, temperature) — the closest published analogue to our (ligand, metal, acid, conc., diluent, T) problem. Includes a recalibrated aleatoric floor and an explicit solvent-extrapolation split.
- **Dataset** 101,535 measurements, 1,327 solutes, 206 solvents. Aleatoric floor 0.106 log S from 481 multi-source pairs via Apelblat/van't Hoff fits.
- **Split** Eval (new solute-solvent pairs, seen solvents), OOD (161 long-tail solvents with zero training overlap), Gold/Silver/Bronze consensus tiers.
- **Architecture** 31 models over 6 families: LightGBM/XGBoost/CatBoost/RF on RDKit+Morgan; FastProp/FastSolv/MLP; Chemprop/GCN/GAT/GIN/Solvaformer; Uni-Mol2/SolTranNet/ChemFM; UNIFAC and Abraham LFER.
- **Why it transfers** (1) LightGBM on RDKit descriptors wins ID (RMSE 0.493) and holds OOD (0.611); no deep alternative closes the gap, ChemFM included. Strong independent support for the GBDT champion. (2) **Their aleatoric-floor method is directly reproducible on our data** — we have 105+ publications and titration series, so cross-publication duplicate cells give an inter-lab log D noise estimate. We do not currently know whether 1.047 is 1.2× or 6× the floor. (3) Their power-law fits asymptote *above* the noise floor, attributing residual error to representation, not data.
- **Why it may fail** Point (3) contradicts gen6 Phase 0's "coverage, not capacity". Both can be true in different regimes, but SC3 concluded "representation bottleneck" with 1,327 solutes and 206 solvents — orders of magnitude more coverage than our 152 extractants. We are still coverage-limited; importing their conclusion is a category error. Their OOD gap (+24%) is also far gentler than ours.
- **Difficulty** Low for the aleatoric-floor replication (~1 day). High and not recommended for their transfer-learning arm (needed ~10⁶ QM solvation energies).

### 1.5 Chuang & Keiser (2018). *Comment on "Predicting reaction performance in C–N cross-coupling using machine learning."* Science 362, eaat8603. — **TEST (already partly executed; see diagnosis)**
- **URL** https://www.science.org/doi/full/10.1126/science.aat8603
- **Problem** Classical controls: a random forest on DFT descriptors was matched by the same forest on random-valued features or plain one-hot component encodings.
- **Dataset** Ahneman HTE plates, ~4,600 reactions.
- **Split** Out-of-sample additive splits by plate, plus alternative plate-line splits the original authors did not run.
- **Architecture** Control experiments, not a model.
- **Why it transfers** Our exact design — a discrete component (extractant) crossed with continuous conditions — and our exact exposure. Under a chemotype hold-out a one-hot ligand baseline *cannot* predict an unseen ligand, so its macro MAE is the pure conditions-only floor and the gap to 1.047 is the total value our chemistry representation adds. **We have now measured this**: `NULL_metal_cond` = 1.0882 vs best tree 0.9976 → all ligand chemistry is worth **0.0906 macro MAE** (`runs/gen7_architecture/oracles/leaderboard.csv`).
- **Why it may fail** Nothing methodological. The original authors' rebuttal (Science 2018, aat8763) argues the random-feature result partly reflects the factorial HTE design; our data is not factorial, so the diagnostic is cleaner for us.
- **Difficulty** Very low. Hours. Extend with a random-feature-per-ligand arm to complete the control.

### 1.6 Muckley, Saal, Meredig et al. (2023). *Interpretable models for extrapolation in scientific machine learning.* Digital Discovery 2, 1425. — **TEST**
- **URL** https://pubs.rsc.org/en/content/articlehtml/2023/dd/d3dd00082f
- **Problem** Whether interpolation-vs-extrapolation changes which model class you should use: random k-fold CV vs leave-one-cluster-out.
- **Dataset** Multiple materials/molecular sets, mostly 10²–10⁴.
- **Split** Random CV vs LOCO-CV — the same contrast as our pooled vs chemotype regimes.
- **Architecture** Regularised linear regression vs RF/GBDT/NN.
- **Why it transfers** Under random CV linear models had 2× the error of black boxes; under LOCO-CV they were only 5% worse on average and **beat** the black box in ~40% of tasks. Our regime is LOCO by construction. A heavily regularised low-variance model on a handful of physically motivated terms (mass-action logs, ionic radius, donor counts) is a serious contender, not a strawman.
- **Why it may fail** Their datasets are single-entity tables; ours has a strongly non-linear conditions response — though the mass-action log transform is exactly the change of variables that linearises it. Also, "beats in 40%" means it loses 60% of the time. **Our gen7 data already argues against**: `LRN_ridge` 1.4563 and `LRN_elasticnet` 1.5125, both far worse than any tree and close to `NULL_global_mean` 1.5525.
- **Difficulty** Very low, and already largely run. Retest only with the mass-action-linearised feature set and missing indicators.

### 1.7 Sagawa, Koh, Hashimoto, Liang (2020). *Distributionally Robust Neural Networks for Group Shifts.* ICLR. arXiv:1911.08731 — **TEST**
- **URL** https://arxiv.org/abs/1911.08731
- **Problem** Minimising worst-group rather than average loss; group DRO only works with much stronger regularisation/early stopping.
- **Dataset** Vision/NLP; the mechanism is architecture-agnostic.
- **Split** Predefined groups with known subgroup shift.
- **Architecture** Training objective. Tabular analogue: per-group sample weighting inside the tree, optionally with iterative upweighting of high-loss extractants.
- **Why it transfers** We report macro MAE (equal weight per cluster) — if training is pooled we optimise an objective that does not match the metric. Our gen2 ablation is emphatic that this matters: pooled and equal-extractant metrics ranked arms in *opposite* orders. Note the current harness *does* apply `group_balanced_weights(ecfp_cluster)`, so the minimal version is already in place; the DRO escalation (exponentiated-gradient upweighting of worst clusters) is untested.
- **Why it may fail** Our worst extractants may be worst because their labels are noisy or their publications are outliers — DRO then chases noise. Sagawa's own message is that group DRO without heavy regularisation overfits the worst group, and tree regularisation knobs are coarse.
- **Difficulty** Very low for reweighting variants; low-medium for iterative upweighting (~50 lines). Hours to a day.

### 1.8 Boldini, Grisoni, Kuhn, Friedrich, Sieber (2023). *Practical guidelines for the use of gradient boosting for molecular property prediction.* J. Cheminform. 15, 73. — **TEST**
- **URL** https://pmc.ncbi.nlm.nih.gov/articles/PMC10464382/
- **Problem** XGBoost/LightGBM/CatBoost on molecular data with explicit hyperparameter-sensitivity analysis.
- **Dataset** 16 datasets, ~1.4M compounds, 94 endpoints; 2k–330k per set; 50 replicates on MoleculeNet.
- **Split** 80:20 random ×50; scaffold on MolData; 70:15:15 regression.
- **Architecture** GBDT over ECFP(r=2,1024), MACCS, 207 RDKit descriptors.
- **Why it transfers** (1) Tuning priority: learning rate, minority/sample weighting, minimum split gain dominate — and **one-factor interactions matter more than individual parameters**, so a coordinate-wise sweep systematically under-tunes. If gen3–gen6 arms were tuned one parameter at a time, some negative results are under-tuned rather than negative. (2) CatBoost is slowest for small/medium data; LightGBM and CatBoost both beat XGBoost under 10k compounds. A champion stable across two GBDT implementations is a safer champion.
- **Why it may fail** Almost all endpoints are classification on random splits at 1–2 orders more molecules. Transfer the tuning priority, not the implementation ranking. Their claim that categorical handling is irrelevant is **not** true for us — metal, acid type and diluent are genuine categoricals.
- **Difficulty** Very low. Joint random search over the three priority parameters; add LightGBM. ~1 day.

### 1.9 Adamczyk, Ludynia, Czech (2026). *Molecular Fingerprints Are Strong Models for Peptide Function Prediction.* Bioinformatics 42(5), btag179. — **TEST**
- **URL** https://academic.oup.com/bioinformatics/article/42/5/btag179/8653978
- **Problem** Whether count-based classical fingerprints + LightGBM match GNNs and pretrained transformers.
- **Dataset** 132 datasets over 6 benchmarks, typically 200–13,000, many with <1,000 positives.
- **Split** Mixed: random 80/20 with CD-HIT 40% dedup, homology-clustered OOD, 10×5-fold CV, leave-one-dataset-out.
- **Architecture** Count ECFP + Topological Torsion + RDKit fingerprints concatenated → LightGBM at default hyperparameters.
- **Why it transfers** The cheap, testable claim is that **count**-based fingerprints and the Topological Torsion / atom-pair families add signal over binary ECFP. Our extractants are large flexible molecules where the *number* of P=O or amide donors and the through-bond distance between them is the chemistry that sets extraction strength — exactly what counts and torsions encode and what a binary bit discards.
- **Why it may fail** Be sceptical: LightGBM ran at defaults (no SDs), no baseline was tuned, no significance tests; the margin over the best GNN on Peptides-func was 1.5% AUPRC. Tasks are overwhelmingly classification. Our own gen4 record ("extended 2D descriptors failed") gives "more per-ligand 2D features" a bad prior, and gen7 shows `REP_lig2d` 1.1403 losing to `REP_donors` 1.0535.
- **Difficulty** Very low. RDKit already computes all three. Hours.

### 1.10 Tynes, Gao, Burrill, Batista, Perez, Yang (2021). *Pairwise Difference Regression (PADRE).* JCIM 61(8), 3846–3857. — **TEST (highest-value new method)**
- **URL** https://pubs.acs.org/doi/10.1021/acs.jcim.1c00670
- **Problem** Meta-algorithm over any regressor: train on concatenated *pairs* to predict the property **difference**; at inference pair the test point with training points and average `anchor_value + predicted_delta`, using the spread as uncertainty.
- **Dataset** Multiple chemical sets; pairs scale as N², so the augmentation argument is strongest at small N. Follow-on (Fralish & Reker review) reports mean +0.04 Pearson *r* specifically for divergent scaffolds; DyAb turned 500 points into ~125,000 pairs.
- **Split** Standard chemical-search splits; pairing **after** splitting so no molecule appears in both train and test pairs.
- **Architecture** Meta-algorithm over any base learner — works directly over our tree. SQRL restricts pairing to chemically similar pairs to control cost.
- **Why it transfers** Best-matched method on the list to our diagnosed failure mode. gen6 says all error is in the ligand offset; gen7's `ORACLE_level` says removing that offset takes 0.9976 → 0.4974. PADRE never predicts an absolute level — only deltas from an **observed** training label — so systematic per-ligand and per-publication offsets cancel by construction. That also addresses publication leakage, since same-publication anchors cancel lab bias. The anchor distribution gives free, well-motivated uncertainty. Los Alamos authorship (Batista, Perez, Yang) is f-element adjacent.
- **Why it may fail** Three risks. (1) gen6's **two-stage residual trap**: Stage B on a cross-fitted residual learned level error and hurt (1.25 vs 1.08); only centring on the *true* cell mean helped (1.04). PADRE anchors on an observed label rather than a predicted residual — the version that works — but the distinction must be enforced in code, not assumed. (2) Under a chemotype hold-out every anchor is chemically distant, the regime where the review reports only +0.04. (3) Cost: naive pairing gives ~18M pairs; restrict to pairs sharing (metal, acid type, diluent) with condition deltas as features, or SQRL-style to K nearest anchors → ~2–5×10⁵ pairs.
- **Difficulty** Medium. ~200 lines around the existing tree plus split-then-pair discipline and an anchor-subsampling policy. Memory is the binding constraint. 2–4 days.

### 1.11 Hollmann, Müller, Purucker, Krishnakumar, Körfer, Hoo, Schirrmeister, Hutter (2025). *Accurate predictions on small data with a tabular foundation model (TabPFN).* Nature 637, 319–326. — **TEST**
- **URL** https://www.nature.com/articles/s41586-024-08328-6
- **Problem** A transformer pretrained on synthetic tabular tasks doing supervised learning by in-context inference; reported to beat tuned GBDTs up to ~10,000 samples.
- **Dataset** Envelope ≤10,000 samples, ≤500 features. Our 5,248 rows fit; our 2,000+ fingerprint columns do not without compression.
- **Split** Mostly standard tabular benchmarks (random). Chemistry follow-ups add OOD.
- **Architecture** Prior-data-fitted transformer, in-context learning at inference.
- **Why it transfers** Our data is genuinely tabular — heterogeneous categoricals (lanthanide, acid, diluent), continuous conditions, and a molecular block — TabPFN's home turf rather than a GNN's. Three chemistry evaluations report the same tilt: on-par in classification, clear and stable advantages in regression, strongest on small/medium data and under OOD.
- **Why it may fail** The structural worry is specific: **TabPFN attends over rows**. We have 5,248 rows but only 152 independent ligands, and within a titration series rows are near-duplicates. In-context attention will exploit that near-duplicate structure — worth a lot on a random split, worth zero under a chemotype hold-out where no row from the test ligand is in context. Published wins are one-row-one-molecule; our ratio is ~35:1. The ≤500-feature ceiling forces PCA on the ECFP block, and compressing sparse binary fingerprints blurs the distinctions we need (Guan et al. observed PCA-500 harming TabICL). CPU limits are cited near ~5,000 samples, so 5,248 is at the edge.
- **Difficulty** Medium. `pip install tabpfn`; needs a ≤500-column feature-compression step; tens of minutes per fold on CPU. 2–3 days.

### 1.12 Ben Hicham, Rittig, Grohe, Mitsos (2026). *Tabular foundation models for in-context prediction of molecular properties.* arXiv:2604.16123 — **TEST**
- **URL** https://arxiv.org/abs/2604.16123
- **Problem** TabPFN-2.5 / TabICL in-context over molecular features (RDKit2d, Mordred, Morgan, plus frozen CheMeleon / SMI-TED / CLAMP embeddings).
- **Dataset** Polaris 28 tasks (100–6,000 samples — our 5,248 sits at the top), MoleculeACE 30 tasks, 11 engineering datasets.
- **Split** Expert-curated fixed train/test splits, 5 repetitions. **Not** chemotype hold-outs.
- **Architecture** Frozen prior-fitted transformer, two-way row/column attention.
- **Why it transfers** TabPFN-CheMeleonFP wins 86.2% across 58 tasks vs CatBoost-Mordred 19.0%, and 100% on MoleculeACE; TabPFN-RDKit2d alone 56.9%. Reported 4.8×–27.3× CPU speedups. It is the only candidate that could plausibly beat the tree on the **mean** rather than only the variance.
- **Why it may fail** Their splits say nothing about extrapolation to Tanimoto-distant ligands. Their rows are one-molecule-one-label; ours have ~35 rows per ligand with strong (ligand, series) level structure, so grouped CV and macro weighting must be enforced or the in-context prior memorises cell means — the gen2 pooled-vs-macro inversion again. Licensing of TabPFN weights needs checking.
- **Difficulty** Low-medium; main work is grouped folds (already in the harness) and feature reduction.

### 1.13 Burns, Zalte, Abreu, Sieg, Feldmann, Mathea, Green (2025/2026). *Deep Learning Foundation Models from Classical Molecular Descriptors (CheMeleon).* arXiv:2506.15792 — **TEST**
- **URL** https://arxiv.org/abs/2506.15792
- **Problem** A D-MPNN pretrained to regress 1,613 Mordred descriptors on 1M PubChem molecules; fine-tuned or used as a 2048-d frozen representation. Zenodo 10.5281/zenodo.15426600, CC-Zero, shipped in chemprop ≥ 2.2.0.
- **Dataset** Pretrain 1M PubChem; benchmarks Polaris (28 tasks, ~100–few thousand), MoleculeACE (29 assays), ToxCast (20 endpoints).
- **Split** Fixed creator-defined test sets, 5 seeds, **Tukey HSD** (α=0.05) family-wise control. Not chemotype hold-outs.
- **Architecture** Chemprop D-MPNN, message dim 2048, 6 steps, 12.9M params.
- **Why it transfers** The only recent encoder that beats a Morgan+RDKit RF with a real statistical test: Polaris 75% vs 68%, MoleculeACE 97% vs 50%. Pretraining on noise-free descriptors rather than MLM is the mechanistic reason — the same "give the model the right functional form" lever that produced our only real gain (MASSACTION). It is also the only pretrained representation with an independent positive result on a hard OOD split (Guan et al.: ChemProp+CheMeleon best on DrugOOD, 0.701).
- **Why it may fail** The pretraining target *is* Mordred — an expanded version of our 2D block — so the credible gain is "smoother nonlinear encoding of descriptors we already have", and gen7 already shows `REP_lig2d` 1.1403 losing to `REP_donors` 1.0535. Its biggest margin comes from MoleculeACE, a within-congeneric-series activity-cliff benchmark that is the inverse of our regime. No OOD-extrapolation evidence exists for CheMeleon itself.
- **Difficulty** Medium. chemprop ≥2.2 pulls lightning; py3.13/torch 2.13 wheel is the only real risk. Fallback: load the Zenodo state_dict directly. Budget 1–2 h.

### 1.14 Kaduk, Masters, Zhu, Kwiatkowski, Beaini et al. (2024). *MiniMol.* arXiv:2404.14986 — **TEST**
- **URL** https://arxiv.org/abs/2404.14986
- **Problem** 10M-parameter GNN pretrained by **supervised** multi-task learning on ~3,300 quantum and biological tasks (Graphium LargeMix, 6M molecules, 526M labels); emits a 512-d fingerprint.
- **Dataset** Pretrain 6M/526M; downstream TDC ADMET (22 tasks, 500–13k).
- **Split** TDC scaffold 5-fold; also Polaris/MoleculeACE fixed splits in Burns et al.
- **Architecture** GNN over Graphium featurisation, 512-d readout used frozen.
- **Why it transfers** Second-best in the Burns Tukey-HSD benchmark (Polaris 71% vs RF 68%) at an order of magnitude fewer parameters; beats the previous best foundation model on 17/22 TDC tasks. Its pretraining signal is **supervised multi-task labels including quantum properties** — a genuinely different inductive bias from both MLM and descriptor regression, so it is the least redundant with our ECFP+2D block. 512-d compresses well to PCA-32. Designed to be used frozen.
- **Why it may fail** Pretraining labels are drug-discovery bio-assays; branched C8 diglycolamides and trialkylphosphine oxides are far outside that space, and multi-task ADMET has no mechanism to encode Ln(III)-donor affinity. MoleculeACE win rate only 40%. No OOD evidence. Heaviest dependency stack after Uni-Mol.
- **Difficulty** Medium-high. torch-geometric + graphium wheels are the risk. Timebox the install to 1 h and abandon if it fights.

### 1.15 Janet & Kulik (2017). *Resolving Transition Metal Chemical Space: RACs.* JPCA 121(46), 8939–8954. — **TEST**
- **URL** https://arxiv.org/abs/1708.06017
- **Problem** Revised autocorrelation descriptors for metal complexes: sums of products/differences of atomic properties at bond-path depth *d*, computed **starting from the metal and the coordinating atoms** rather than the whole molecule.
- **Dataset** Thousands of octahedral TM complexes; 155 non-trivial standard RACs. Spin-splitting and redox to 1–4 kcal/mol of DFT; M–L bond lengths to 0.01–0.03 Å.
- **Split** DFT-labelled, standard splits; the contribution is the featurisation.
- **Architecture** Graph-based descriptors (no 3D needed) + feature selection → conventional regressors.
- **Why it transfers** The most chemically motivated feature idea available, and orthogonal to everything tried. A whole-molecule ECFP treats a diglycolamide's long solubilising tails and its P=O/amide donor set with equal weight, diluting the coordinating environment — the only part of the ligand that touches the metal — across thousands of bits. RACs anchor autocorrelations at the donor atoms. A 2D analogue is buildable in pure RDKit: SMARTS-match donors (phosphoryl O, amide/carbonyl O, ether O, N, S), then depth-0..3 product/difference autocorrelations of five atomic properties. ~40–60 interpretable columns encoding donor identity, count, bite distance and chelate-ring electronics. **gen7 evidence supports the direction**: `REP_donors` (1.0535) already beats `REP_ecfp` (1.1159) and `REP_lig2d` (1.1403).
- **Why it may fail** gen4 found extended 2D descriptors failed and gen6 concluded representation is not the bottleneck; 50 more columns from the same graph could be gen4 under a new name. Donor assignment by SMARTS is a guess for ambidentate ligands. With 152 ligands, 50 extra columns is a real variance cost that must clear a 5-seed CI.
- **Difficulty** Medium. Pure RDKit, ~150–250 lines. 2–3 days.

### 1.16 Abarbanel & Hutchison (2024). *QupKake* (underlying: Bannwarth, Ehlert, Grimme, GFN2-xTB, JCTC 2019). — **TEST**
- **URL** https://pmc.ncbi.nlm.nih.gov/articles/PMC11325546/
- **Problem** Whether cheap semiempirical (GFN2-xTB) quantum features improve prediction of an experimental property. QupKake reaches RMSE 0.5–0.8 pKa units.
- **Dataset** Experimental micro-pKa sets; one GFN2-xTB single point per molecule suffices.
- **Split** Experimental held-out test sets.
- **Architecture** GNN + QM features. The transferable part is the ablation: atomic polarisabilities, coordination numbers and Wiberg bond orders carry the signal.
- **Why it transfers** Extraction thermodynamics is electronic — donor basicity, charge on the phosphoryl/amide oxygen, polarisability, protonation state — none recoverable from a topological fingerprint. Compute is bounded by the number of **ligands**, not rows: 152–190 extractants at seconds-to-minutes each is an afternoon on 8 CPUs. A genuinely different information channel from ECFP and 2D descriptors.
- **Why it may fail** (1) Same prior as RACs; the only lever that ever worked (MASSACTION) worked by supplying a *functional form*, not more features. (2) Conformer dependence — flexible polydentates have many minima; a CREST/ETKDG ensemble multiplies cost 10–50×. (3) The properties that matter are arguably those of the **complex**, not the free ligand, and xTB cannot cheaply treat Ln(III) (poor f-element parameterisation). We would be featurising a proxy. **Note**: the bundle already has GFN2-xTB scalars for 5,365/5,992 rows on the *complex*, and `REP_3d_complexphys` is the **worst** arm at 1.1732.
- **Difficulty** Medium. `xtb` installs cleanly; 190 molecules with ETKDG + GFN2 optimisation is hours. 3–5 days including the conformer decision.

### 1.17 Svensson, Aniceto, Norinder, Cortes-Ciriano, Spjuth, Carlsson, Bender (2018). *Conformal Regression for QSAR — Quantifying Prediction Uncertainty.* JCIM 58(5), 1132–1140. — **TEST**
- **URL** https://discovery.ucl.ac.uk/10060460/1/Svensson_etal_r1.pdf
- **Problem** Distribution-free prediction intervals with guaranteed marginal coverage, nonconformity normalised by an ensemble-based difficulty estimate.
- **Dataset** Multiple QSAR regression sets; works with a few hundred calibration points.
- **Split** Critical caveat from the follow-up literature: conformal coverage degrades under non-random (temporal, cluster, scaffold) splits because calibration and test are no longer exchangeable.
- **Architecture** Inductive/Mondrian conformal regression wrapped around any point predictor.
- **Why it transfers** We know error is a strong monotone function of nearest-train Tanimoto. That is exactly the setting for **Mondrian** conformal: bin the calibration set by nearest-train Tanimoto and calibrate per bin, restoring conditional coverage a global interval destroys. Deliverable: an interval that honestly widens for novel chemistry, plus a defer rule ("measure 2–3 pairs first") connecting to our validated k-shot result. For a screening tool this is worth more than 0.02 on macro MAE.
- **Why it may fail** Exchangeability is exactly what a chemotype hold-out breaks; the Mondrian fix mitigates but does not repair it, and the highest-novelty bin may hold too few calibration ligands. Expect under-coverage where it matters most — validate empirically per bin rather than trusting the theory. Our rows are not exchangeable within a ligand, so the conformal unit must be the ligand or (ligand, series), shrinking the calibration set to tens of units.
- **Difficulty** Low. MAPIE or ~80 lines of hand-rolled Mondrian conformal. 1–2 days.

### 1.18 Parrondo-Pizarro, Lanini, Rodríguez-Pérez (2026). *Uncertainty Quantification in Molecular ML under Data Shifts.* JCIM 66(2), 923–935. — **TEST**
- **URL** https://pmc.ncbi.nlm.nih.gov/articles/PMC12848971/
- **Problem** Benchmarks UQ estimators for ADME under four realistic shifts: feature (distance to 5-NN), label, discontinuity/activity-cliff, error shift.
- **Dataset** 293,569 in-house Novartis compounds over 7 endpoints; plus 3,517 public compounds over 5.
- **Split** In-house **temporal** (2022 train / 2023 calibration / 2024-25 test) — a genuine prospective shift. Public: scaffold.
- **Architecture** GNN ensembles; estimators = distance to 5-NN, ensemble variance, hybrids, and a **learned error model** (small RF predicting |error| from distance + variance + prediction, fitted on the calibration split).
- **Why it transfers** Directly actionable and cheap. Raw distance gave Spearman 0.06–0.20 with true error, raw ensemble variance 0.07–0.26, the **learned error model 0.06–0.46**, staying stable across the whole distance range where the others degraded. We can build this today on the existing tree: features = max Tanimoto to train, cluster size, mass-action values, ensemble spread (already emitted as `prediction_sd` in `contenders.py:LevelTree`), trained on a held-out chemotype calibration fold. It gives a defensible trust/don't-trust flag without touching the mean model.
- **Why it may fail** Their base is a GNN ensemble on 100k+ compounds; our calibration fold has a handful of chemotype clusters, so the error model may just learn cluster identity. Spearman ceiling 0.46 — useful for ranking, not for a per-prediction interval. And an error model trained on seen chemotypes may itself fail to extrapolate — the coverage problem one level up.
- **Difficulty** Low. sklearn RF over ~10 engineered features nested in the existing harness. 1–2 days.

### 1.19 Laghuvarapu, Lin, Sun (2023). *CoDrug: Conformal Drug Property Prediction with Density Estimation under Covariate Shift.* NeurIPS. — **TEST**
- **URL** https://arxiv.org/abs/2310.12033
- **Problem** Distribution-free prediction sets valid under covariate shift, reweighting conformal scores by a density ratio between calibration and test distributions.
- **Dataset** Small-molecule tasks with realistic drifts.
- **Split** Explicit covariate-shift splits; the related **LDMO** (Leave Dissimilar Molecules Out) cross-conformal variant clusters molecules first — the direct analogue of our Tanimoto-0.7 clustering.
- **Architecture** Model-agnostic wrapper.
- **Why it transfers** Wraps the existing tree with no change to the mean. LDMO-style cluster-wise calibration matches our split design exactly. Given our honest headline, an interval with demonstrated 90% coverage on held-out chemotypes is a more credible deliverable than another 2% MAE.
- **Why it may fail** Exchangeability is what a chemotype hold-out breaks, and the density ratio must be estimated in fingerprint space from ~152 molecules — the estimator will be noise. Expect wide and possibly still under-covering intervals on the most distant clusters.
- **Difficulty** Low for plain split/cross-conformal at cluster level (~50 lines). Skip the density-ratio reweighting. 1–2 days.

### 1.20 Nael, Alakonda, Elokely (2026). *Defining the Data set Defines the QSAR Claim.* JCIM 66(6), 2951–2954. — **TEST**
- **URL** https://pubs.acs.org/doi/10.1021/acs.jcim.6c00514
- **Problem** Predictive claims hinge on undocumented choices — standardisation, endpoint definition, aggregation, split construction, leakage diagnostics. Proposes executable, auditable **dataset contracts**.
- **Dataset / split / architecture** N/A — a perspective implementable with current tooling.
- **Why it transfers** Directly on our sore spot. We have already discovered the hard way that gen5's `unseen_series` folds mixed publications in 26% of rows, that `min_rows=10` discarded 37 of 47 chemically novel extractants, that `within_ligand_r2` was broken, and that a two-stage arm measured level error. Each is a contract violation an executable pre-registration would have caught *before* a run. Concretely: a `contract.yaml` plus assertions run before every job — standardisation rules, replicate aggregation, split definition, min-cells policy, max cross-split Tanimoto, publication-overlap rate — failing the run on violation.
- **Why it may fail** An editorial, not a validated method; it produces no metric improvement, only prevents false results. Keep it to a few dozen lines of assertions rather than a framework.
- **Difficulty** Low. 1–2 days.

### 1.21 Virany & Tripp (2025). *Hash Collisions in Molecular Fingerprints.* NeurIPS AI4Science Workshop. arXiv:2511.17078 — **TEST**
- **URL** https://arxiv.org/abs/2511.17078
- **Problem** Quantifies how folding ECFP into a fixed bit vector causes collisions that systematically **overestimate** similarity, and the downstream effect on GP regression and BO.
- **Dataset** Five DOCKSTRING benchmarks (250k scale).
- **Split** Standard; the contribution is a controlled exact-vs-hashed ablation.
- **Architecture** Exact (dictionary) vs hashed fingerprints; Tanimoto-kernel GPs downstream.
- **Why it transfers** Directly relevant where the kernel *is* fingerprint similarity: collision-inflated similarity biases toward "this new ligand looks closer to training than it is", corrupting both the posterior mean and the applicability domain — and inflating our Tanimoto-0.7 cluster boundaries. With 190 unique extractants, exact fingerprints are trivially affordable. Also a cheap count-vs-binary (MinMax vs Tanimoto) ablation. Note the bundle already collapses **190 distinct SMILES to 164 distinct 2048-bit fingerprints** (18 collision groups; 44 ligands have a Tanimoto-1.0 partner).
- **Why it may fail** The authors' own result is that gains were "small but consistent" and did not translate into BO improvement — hygiene, not MAE. It also changes the clusters and therefore the benchmark: **freeze the split definition on the current fingerprint** so regimes stay comparable across generations.
- **Difficulty** Very low. Half a day.

### DONT_TEST — Topic 1

| Method | One-line reason |
|---|---|
| **Guo, Hernández-Hernández, Ballester (2024), scaffold splits overestimate VS, arXiv:2406.00873** | We already use similarity clustering *and* report performance stratified by nearest-train Tanimoto, which is strictly more informative than another aggregate. A UMAP embedding of 190 points would be unstable. |
| **Fralish, Chen, Skaluba, Reker (2023), DeepDelta, J. Cheminform. 15, 101** | Same idea as PADRE but with a D-MPNN over ~10⁵–10⁶ pairs per fold — days per fold on CPU. The review's own note: with similarity thresholding, no neural network outperformed tree baselines. Run the meta-algorithm over the tree instead. |
| **Vermeire & Green (2021), transfer learning for solvation free energies, Chem. Eng. J. 418, 129307** | The right shape of answer, unavailable at our budget: there is no CombiSolv-QM analogue for Ln extraction, and generating 10⁵–10⁶ computed points needs f-element DFT that xTB cannot do. |
| **Gulrajani & Lopez-Paz (2021), In Search of Lost Domain Generalization, ICLR** | Headline: a well-tuned ERM baseline matches or beats every specialised DG algorithm under fair model selection. Invariance objectives are gradient-based with no tree implementation; adopting one means abandoning the champion. The carve-out (GroupDRO-style reweighting) is entry 1.7. |
| **van Tilborg, Alenicheva, Grisoni (2022), MoleculeACE, JCIM 62(23), 5938** | Measures the *opposite* axis: activity cliffs are high-similarity/high-delta; ours is low-similarity extrapolation. Flag for CheMeleon and TabPFN — both post their most spectacular win rates on this benchmark; discount those numbers. |
| **Stanley et al. (2021), FS-Mol, NeurIPS D&B** | Meta-learning needs thousands of related tasks (FS-Mol has 4,938). We have ~152 extractants — two orders short. Also binary classification, and our own k-shot memory says from k≈2 the model adds nothing over a no-model ΔZ fit. |
| **Bjerrum (2017), SMILES enumeration, arXiv:1703.07076** | A literal no-op for a fingerprint pipeline (canonicalisation-invariant features). Follow-up work reports randomised-SMILES pretraining *hurts* regression. |
| **Dou et al. (2023), Small-Data Challenges in Molecular Science, Chem. Rev. 123(13), 8736** | A taxonomy, not an experiment; every technique it recommends is covered more rigorously above. |
| **Antoniuk, Zaman et al. (2025), BOOM, NeurIPS D&B, arXiv:2505.01912** | Read for calibration (best model ~3× higher error OOD; ID accuracy does not predict OOD; MLM pretraining fails or degrades OOD — so our ~2.5–3× degradation is *normal*), but its one actionable lever (add a few OOD molecules) is exactly what gen6 Phase 0/F already established more specifically. |
| **Praski, Adamczyk, Czech (2025), Benchmarking Pretrained Molecular Embeddings, arXiv:2508.06199** | Establishes the prior, is not a method: 25 models × 25 datasets, hierarchical Bayesian testing, "nearly all neural models show negligible or no improvement over ECFP"; the only significant winner (CLAMP) is itself fingerprint-based. |

---

## Topic 2 — Pretrained molecular encoders as frozen embeddings

**Status: partly executed.** `runs/gen7_architecture/screen_embeddings/leaderboard.csv` already
contains 16 embedding arms at **1 seed**. Best is `EMB_chembertamlm_hybrid_pca32` at **1.0970**,
worse than the incumbent 1.0475 and far worse than 0.9976. Every raw (uncompressed) arm is
worse than its PCA-32 twin, and every CatBoost variant is worse than its ExtraTrees twin.

### 2.1 Ahmad, Simon, Chithrananda, Grand, Ramsundar (2022). *ChemBERTa-2.* arXiv:2209.01712 — **TEST (done; negative)**
- **Problem** SMILES MLM and multi-task-regression pretraining. `DeepChem/ChemBERTa-{5M,10M,77M,100M}-{MLM,MTR}`. Verified config: hidden 384, 3 layers, vocab 600, max_pos 515 — the "77M" is molecules, not parameters. MIT via the upstream repo; the DeepChem cards carry **no explicit licence tag**.
- **Dataset / split** Pretrain 77M PubChem; MoleculeNet scaffold splits, single split, **no error bars anywhere**.
- **Architecture** RoBERTa, 384-d (768 concatenating mean+CLS — what our parquet stores). MTR head regresses 200 RDKit-computable descriptors.
- **Why it transfers** Zero marginal cost: `dataset with 3D structures/ligand_pretrained_embeddings.parquet` already contains chemberta (MTR) and chemberta_mlm mean+CLS for all 190 extractants; arms are wired in `gen7/suites.py`. One of only four models (with CLAMP, R-MAT, MolBERT) that beat ECFP in the Praski Bayesian Bradley-Terry benchmark.
- **Why it may fail** Three verified problems. (1) **The tokenizer silently drops stereochemistry** — `tokenize` is identical for `C(C)OC(C)` and `[C@H](C)O[C@@H](C)`, zero UNKs, no `@` tokens in a 591-token vocab; our 190 SMILES collapse to **188** distinct vectors. (2) MTR pretraining is literally distillation of 200 RDKit descriptors — redundant with our 2D block. (3) Embeddings are nearly collinear on our ligands: **median max cosine 0.983**. BOOM: ChemBERTa's binned OOD R² drops 39%.
- **Difficulty** Trivial. Already run at 1 seed; re-run at 5.

### 2.2 Ross, Belgodere, Chenthamarakshan, Padhi, Mroueh, Das (2022). *MoLFormer-XL.* Nat. Mach. Intell. 4:1256–1264. — **TEST (done; negative)**
- **Problem** Linear-attention + rotary SMILES transformer. `ibm-research/MoLFormer-XL-both-10pct`; verified config hidden 768, 12 layers, vocab 2362, max_pos 202; 46.8M params; **Apache-2.0, explicitly tagged**; needs `trust_remote_code=True`.
- **Dataset / split** Pretrain 10% ZINC + 10% PubChem. Scaffold splits in the paper and in Praski et al.; fixed creator splits in Burns et al.; KDE tail-value OOD in BOOM.
- **Why it transfers** Best-behaved encoder on *our* molecules: 190/190 distinct vectors, stereo tokens present, median max cosine 0.957 (more spread than ChemBERTa), already cached, and our SMILES (20–174 chars) fit inside 202 tokens.
- **Why it may fail** Honest benchmarks are consistently unkind. Burns (Tukey HSD, 58 tasks): 39% on Polaris vs 68% for a Morgan+RDKit RF, 17% vs 50% on MoleculeACE — described as "surprisingly poorly". BOOM: binned OOD R² falls 53%; "all three foundation models do not show any significant improvement in OOD performance due to language modeling pretraining". Domain mismatch is real: ZINC/PubChem are drug-like; trialkylphosphine oxides and dioctyl-diglycolamides are tail chemistry. 768 dims over 152 ligands is a variance disaster without PCA. **Our measurement**: `EMB_molformer_pca32` 1.1627, `EMB_molformer_raw` 1.2449.
- **Difficulty** Trivial; already computed.

### 2.3 CheMeleon — see 1.13. **TEST** (the one encoder worth the install.)

### 2.4 MiniMol — see 1.14. **TEST** (least redundant pretraining signal.)

### DONT_TEST — Topic 2

| Method | One-line reason |
|---|---|
| **Zhou et al. (2023), Uni-Mol, ICLR** | Takes **one conformer** of the free ligand; our extractants have 15–25 rotatable bonds, so a single ETKDG conformer is conformer noise, and the bound conformation is set by the metal. gen7's `REP_3d_complexphys` is already the **worst** arm (1.1732). Uni-Core build on py3.13 CPU-only is a fight. |
| **Ji et al. (2024), Uni-Mol2, NeurIPS** | Same conformer problem ×10 in model size. Its 27%/14% gains are on QM9 and COMPAS-1D — quantum benchmarks on small rigid molecules with near-random splits, where geometry *is* the label-generating process. |
| **Rong et al. (2020), GROVER, NeurIPS** | No positive evidence under any honest split (below ECFP in Praski). Dead stack: torch 1.x, vendored chemprop, a pickled argparse Namespace that torch ≥2.6 `weights_only=True` refuses to load. |
| **Wang et al. (2022), MolCLR, Nat. Mach. Intell. 4:279** | Worst modern encoder in every re-evaluation (Burns: 14% Polaris, **0%** MoleculeACE vs RF 68%/50%). Self-defeating mechanism: its augmentations (delete a bond, mask an atom) treat exactly the donor substitutions that flip Ln(III) affinity as label-preserving noise. |
| **Jaeger, Fulle, Turk (2018), Mol2vec, JCIM 58(1), 27** | Vocabulary *is* the Morgan identifier set — information-equivalent to our ECFP block, and unseen radius-1 identifiers are OOV at exactly the point of interest. `model_300dim.pkl` is a gensim-3.x pickle gensim 4.x cannot load. |
| **Winter, Montanari, Noé, Clevert (2019), CDDD, Chem. Sci. 10:1692** | Conceptually the most honest objective (SMILES-invariance), but a TensorFlow 1.10 / py3.6 graph with no TF2 or torch port and weights behind a partially bit-rotted Drive script. Does not beat ECFP in Praski anyway. |
| **Frey et al. (2023), ChemGPT, Nat. Mach. Intell. 5:1297** | Generative decoder; causal hidden states pool poorly. Ranks near the bottom in Praski. Licence **unspecified** on the HF cards. Silent-failure trap: vocabulary pinned to `selfies==1.0.4`, and 2.x produces different tokens that still run. |

---

## Topic 3 — Gaussian processes, deep kernel learning, and kernel methods

**Status: partly executed.** `LRN_kernelridge_rbf` = **1.2684** (3 seeds), well behind every tree
and behind `NULL_metal_cond` (1.0882). `runs/gen7_architecture/screen_kernels/` is empty — the
kernel suite exists in `gen7/kernels.py` but has not been run.

### 3.1 Griffiths, Klarner, Moss, Ravuri et al. (2023). *GAUCHE: A Library for GPs in Chemistry.* NeurIPS. — **TEST**
- **URL** https://proceedings.neurips.cc/paper_files/paper/2023/hash/f2b1b2e974fa5ea622dd87f22815f423-Abstract-Conference.html
- **Problem** Exact GP regression with kernels over bit-vectors/strings/graphs, with UQ and BO. Provides the Tanimoto kernel k(x,x′)=⟨x,x′⟩/(⟨x,x⟩+⟨x′,x′⟩−⟨x,x′⟩) and "fragprints" (Morgan bits + RDKit fragment counts).
- **Dataset** Photoswitch 392; ESOL 1,128; FreeSolv 642; Lipophilicity 4,200; Buchwald-Hartwig 3,955; Suzuki-Miyaura 5,760 — all within one order of magnitude of our 5,248 rows.
- **Split** **80/20 random, 20 trials.** No scaffold or cluster split anywhere in the paper. Reports RMSE, NLPD, MSLL, quantile coverage error.
- **Architecture** Exact zero-mean GP, Tanimoto kernel on ECFP, marginal-likelihood hyperparameters. No validation set needed.
- **Why it transfers** Our ligand-level effective sample size is ~152 molecules — the regime (n=400–4,000) where Tanimoto GPs are competitive with GNNs and BNNs. Our model already behaves as a near-neighbour lookup; a Tanimoto GP makes that explicit and, uniquely, reports a posterior variance that grows as the test ligand leaves the manifold. Exact Cholesky at n=5,248 is ~10¹¹ flops — tens of seconds on 8 CPUs; the ligand-only kernel is rank ~152.
- **Why it may fail** Every headline is in-distribution. Under our split k(test,train) is bounded near the cluster threshold, so the posterior mean shrinks to the prior — the GP will *reproduce*, not fix, the coverage ceiling. GAUCHE also has no story for continuous non-molecular inputs; conditions must be added by hand via a composite kernel or all condition variance goes into the noise term.
- **Difficulty** Low. **Do not take a dependency on the `gauche` package** — the kernel is ~6 lines of numpy and an exact GP fits with `scipy.optimize`. 1–2 days including composite-kernel design.

### 3.2 Sigrist (2022). *Gaussian Process Boosting.* JMLR 23(232):1–46. — **TEST (highest structural fit)**
- **URL** https://www.jmlr.org/papers/v23/20-322.html
- **Problem** Joint estimation of y = F(X) + Zb + ε where F is a boosted ensemble and Zb is a GP and/or grouped/crossed/nested random effects. Removes both the linear-mean assumption of GPs and the conditional-independence assumption of boosting.
- **Dataset** Simulated n=5,000 with m=500 groups; n=500 spatial; plus real data. ~1 s on the grouped setting.
- **Split** Explicit **interpolation (existing groups) vs extrapolation (NEW groups)** — the closest published analogue to our unseen-extractant regime.
- **Architecture** Coordinate-descent alternation between covariance parameters and tree additions. `gpboost` (C++/LightGBM-derived). Vecchia approximation unnecessary at our n.
- **Why it transfers** The most direct structural encoding of what we have measured. gen6 Phase 0: all error is in the ligand offset. gen6 Phase 2: the level is a (ligand, series) quantity. gen7 `ORACLE_batch` (0.3502) vs `ORACLE_level` (0.4974): there is a further per-(ligand, publication-table) term worth 0.1472. GPBoost lets us write exactly that — a tree mean over conditions + mass-action, a grouped random intercept for (ligand, series) and for publication batch, and a GP over the fingerprint. Because mean and random effects are estimated **jointly**, it structurally avoids the two-stage residual trap. For a new ligand the random effect falls back to prior mean zero with a variance — a principled zero-shot fallback — and updates in closed form for k-shot, unifying the two models we currently run separately.
- **Why it may fail** Sigrist's own numbers do not promise a win: on **new groups** GPBoost is 1.458 vs CatBoost 1.464 — a 0.4% gap, indistinguishable from noise. The large gains (1.100 vs 1.183) are interpolation into seen groups — our easy regime. Simulations have 500 groups from the assumed additive generative model; we have 152 ligands with no guarantee the offset is additive-Gaussian in fingerprint space. Publication random effects risk absorbing real chemistry. No ordered-target-statistics, so categorical handling regresses.
- **Difficulty** Medium. `pip install gpboost` (verify py3.13 wheel; else a side venv). The modelling — which factors are random, and getting the Tanimoto GP alongside them — is the real cost. 3–5 days.

### 3.3 Ustimenko, Beliakov, Prokhorenkova (2022). *Gradient Boosting Performs Gaussian Process Inference.* arXiv:2206.05608 (CatBoost team). — **TEST**
- **URL** https://arxiv.org/abs/2206.05608
- **Problem** Proves boosting over oblivious trees is a kernel method converging to a kernel-ridge solution whose limit is a GP posterior mean; converts boosting into a sampler for epistemic uncertainty.
- **Dataset / split** Standard tabular benchmarks **plus explicit out-of-domain detection** — one of the few UQ papers here that does.
- **Architecture** SGLB + "virtual ensembles": a single CatBoost model yields a posterior sample without training an ensemble. Already shipped in CatBoost.
- **Why it transfers** Near-zero marginal cost, and it settles a question a GP experiment would otherwise confound: is the value of a GP the calibrated variance, or the kernel prior? If SGLB virtual ensembles already give usable knowledge uncertainty, the only remaining reason to build a Tanimoto GP is the prior itself.
- **Why it may fail** The uncertainty is over tree-partition space, not chemistry space: two chemically distant ligands landing in the same leaves for the *conditions* features look confidently in-domain — the exact failure our chemotype hold-out is designed to expose. SGLB also perturbs boosting dynamics, so the mean needs re-validation.
- **Difficulty** Low. Enable `posterior_sampling`/`langevin`, call virtual ensembles at predict time. Half a day plus a 5-seed re-run.

### 3.4 Cortés-Ciriano, van Westen, Bender, Malliavin (2014). *Proteochemometric modeling in a Bayesian framework.* J. Cheminform. 6:35. (Pairwise machinery: Airola & Pahikkala, arXiv:2009.01054.) — **TEST**
- **URL** https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-6-35
- **Problem** Bioactivity as a function of **both** a small molecule and a target, using a GP whose kernel combines a ligand kernel and a target kernel: K((d,t),(d′,t′)) = K_D(d,d′)·K_T(t,t′).
- **Dataset** ~11,000 ligand-receptor pairs over 8 adenosine receptors; plus 4 dengue NS3 proteases × 56 peptides.
- **Split** CV over pairs; the pairwise-kernel literature formalises exactly the four settings — both seen, new ligand, new target, both new — that our four regimes instantiate.
- **Architecture** GP with normalised-polynomial/radial kernels; accepts a per-datapoint experimental error as observation noise.
- **Why it transfers** Our system **is** a proteochemometric problem with the target replaced by a lanthanide. The Kronecker structure k_ligand ⊗ k_metal ⊗ k_conditions is the kernel-space version of the lesson that produced MASSACTION: supply the physical functional form. The metal kernel can be built on ionic radius / f-electron count rather than one-hot, giving free interpolation across the contraction. The paper also frames GP variance as an applicability domain and accepts per-point error — directly usable for our publication-heterogeneous noise.
- **Why it may fail** Adenosine receptors are 8 homologous proteins with dense pair coverage; our 152×14 grid is sparse and many extractants have a single metal, so the coregionalisation machinery has little to learn. Splits are not chemotype hold-outs. Our data is not a complete grid, so fast Kronecker algebra does not apply — but a plain n=5,248 Cholesky is affordable, so skip the vec trick.
- **Difficulty** Medium. No new dependency. Main work is designing the metal kernel and beating one-hot. 2–4 days. **Note** `gen7/suites.py:_metal` already defines `MET_continuous / MET_onehot / MET_physical / MET_all` arms; `runs/gen7_architecture/metal/` is empty.

### 3.5 Duvenaud, Nickisch, Rasmussen (2011). *Additive Gaussian Processes.* NIPS. — **TEST**
- **URL** https://arxiv.org/abs/1112.4394
- **Problem** A GP prior over functions decomposing as a sum of low-order interaction terms, with one variance learned per interaction order — kernel-space ANOVA.
- **Dataset / split** UCI-scale regression, standard random CV. No OOD claim.
- **Architecture** k_add = Σ_r σ_r² e_r(k_1,…,k_D), e_r the elementary symmetric polynomial.
- **Why it transfers** A principled way to write the model our own results argue for: a first-order ligand-offset term, first-order condition terms, and controlled second-order ligand×metal and ligand×acid interactions with the interaction variances **fitted** rather than assumed. The learned per-order variances are directly interpretable as "how much of log D is a pure ligand offset vs a genuine interaction" — a measurement we want independent of any MAE gain. Additive structure also extrapolates better: for a distant ligand only the ligand term collapses to prior, while condition terms still predict.
- **Why it may fail** Evidence-free transfer — no molecular paper in this scan uses additive-order kernels on fingerprints. With 152 ligands there is little signal to identify high-order variances, and many-variance marginal-likelihood optimisation is the overparameterised regime Ober et al. show overfits. Our tree already learns arbitrary-order interactions, so the additive prior is a restriction that helps only if true.
- **Difficulty** Low-medium. Pure numpy on top of 3.1. 1–2 days.

### 3.6 Griffiths, Greenfield, Thawani et al. (2022). *Data-driven discovery of molecular photoswitches with multioutput GPs.* Chem. Sci. 13:13541. — **TEST (low priority)**
- **URL** https://pubs.rsc.org/en/content/articlelanding/2022/sc/d2sc04306h
- **Problem** Multitask GP over four correlated wavelengths via the intrinsic model of coregionalisation.
- **Dataset** 405 photoswitches, 4 tasks, heavily incomplete coverage; external screen over 7,265 molecules.
- **Split** 80/20 random ×20; LOO for the TD-DFT comparison. **No scaffold split.** The one honest OOD number is the external screen.
- **Architecture** ICM: k((x,i),(x′,j)) = k_Tanimoto(x,x′)·B[i,j], B PSD via Cholesky.
- **Why it transfers** Structurally our closest match on the metal axis: 14 lanthanides = 14 correlated tasks, and B would be forced to recover the monotone lanthanide-contraction trend — a physically checkable output. Task coverage there is as sparse as ours. It also confirms fragprints beat Morgan or RDKit alone, validating our ECFP+2D block.
- **Why it may fail** The paper's own multitask benefit over single-task GP is statistically weak, and the external screen degrades MAE 15.5 → 22.7 nm (~46%) — a sober prior. Learning a full 14×14 B from 152 ligands with many single-metal extractants is close to unidentifiable; a rank-1 or ionic-radius-parameterised B collapses to the metal-kernel idea in 3.4.
- **Difficulty** Medium. 2–3 days.

### 3.7 Griffiths, Aldrick, Garcia-Ortegon, Lalchand, Lee (2022). *Heteroscedastic Bayesian optimisation.* MLST 3:015004. — **TEST (cheap version only)**
- **URL** https://iopscience.iop.org/article/10.1088/2632-2153/ac298c
- **Problem** GP with an input-dependent noise process so high-aleatoric regions are modelled rather than absorbed into the signal.
- **Dataset / split** FreeSolv (642) and synthetics; random splits.
- **Architecture** Most-likely-heteroscedastic GP: alternating mean-GP and noise-GP fits.
- **Why it transfers** Our rows come from 105–115 publications and the noise is manifestly publication-structured. A homoscedastic GP inflates one global noise term to cover the worst publications, over-smoothing the clean ones. Modelling noise as a function of publication metadata should improve NLPD sharply even if MAE moves little — and calibration is arguably the deliverable that matters.
- **Why it may fail** MAE gains are usually nil; the win is in the likelihood. Alternating fits are unstable at small n. **Risk specific to us**: if noise level correlates with chemistry (families studied only by one noisy lab), the noise GP down-weights exactly the chemistry we care about. A far cheaper 80% solution is a per-publication learned noise scale — a diagonal noise vector.
- **Difficulty** Low for per-publication noise variances (one line). High for the full alternating fit. **Do the cheap version first: 1 day.**

### 3.8 Jiang, Wu, Deng et al. (2021). *Could GNNs learn better molecular representation for drug discovery?* J. Cheminform. 13:12. — **TEST (done; negative)**
- **URL** https://link.springer.com/article/10.1186/s13321-020-00479-8
- **Problem** 4 descriptor models (SVM, XGBoost, RF, DNN) vs 4 graph models across 11 endpoints.
- **Dataset / split** 11 datasets, hundreds to tens of thousands; repeated random splits with per-model hyperparameter search.
- **Architecture** SVR/SVM with RBF on descriptors is the relevant arm.
- **Why it transfers** Descriptor models beat graph models on accuracy and compute, with SVM generally best for **regression** — the cheapest possible sanity check on whether kernel smoothing per se buys anything over trees, before investing in GP infrastructure.
- **Why it may fail** SVR gives no uncertainty, which is most of the reason to look at kernels. RBF on a heterogeneous block (sparse bits + continuous logs + one-hots) needs careful scaling. **Our measurement settles it**: `LRN_kernelridge_rbf` 1.2684 vs trees at 0.998–1.11 — kernel smoothing on the current features is not close, so the kernel-prior hypothesis is weak and 3.1/3.5/3.6 are less likely to pay.
- **Difficulty** Very low; already run.

### DONT_TEST — Topic 3

| Method | One-line reason |
|---|---|
| **Singh & Hernández-Lobato (2024), DKL for reaction outcome, Commun. Chem. 7:136** | The 8.58→4.87 RMSE jump conflates "deep" with "right kernel": the baseline was a Matérn-5/2 over descriptors on a fully-crossed HTE grid under a 70/20 random split, where nearly every component is in training. DKL(Morgan) is statistically indistinguishable from a plain GNN (4.86 vs 4.89) — the GP layer added calibration, not accuracy. 400-epoch training × 5 seeds × k folds × regimes on 8 CPUs with no GPU. |
| **Ober, Rasmussen, van der Wilk (2021), Promises and Pitfalls of DKL, UAI** | The governing paper for the entry above: marginal-likelihood DKL correlates *all* data points rather than informative ones, and the pathology **intensifies** in the overparameterised small-data regime — us. The prescribed fix (fully Bayesian DKL via SGLD/HMC) multiplies cost by the sample count; not runnable here. |
| **Tripp, Bacallado, Singh, Hernández-Lobato (2023), Tanimoto Random Features, NeurIPS** | Solves a problem we do not have: exact Cholesky at n=5,248 is tens of seconds. Their own table shows random-feature GPs *underperform* SVGP. Take the MinMax kernel definition and nothing else. |
| **Moss & Griffiths (2020), FlowMO, NeurIPS ML4Molecules workshop** | No published evidence the SMILES subsequence kernel beats Tanimoto; the same authors' later GAUCHE puts Tanimoto-fragprints on top. O(n²L²) with long alkyl chains, and it is a second numerical stack (GPflow/TensorFlow) beside torch 2.13. Curated SMARTS motif counts get the same signal for free. |
| **Li, Kong, Du et al. (2024), MUBen, TMLR** | Internalise two findings — deep ensembles best on RMSE, BBP/SGLD best-calibrated (77/88 top ranks) at a cost in accuracy; and a plain DNN on 200 RDKit descriptors stays competitive with heavyweight backbones — but every arm needs a neural backbone we cannot train, and the datasets have no conditions axis. |
| **Hirschfeld, Swanson, Yang, Barzilay, Coley (2020), UQ with NNs for molecular property prediction, JCIM 60(8), 3770** | Its own conclusion is decisive: none of the methods is unequivocally superior and none produces a reliable error ranking across datasets. The "GP head on frozen features" recipe would spend the GP's 152-molecule budget on a drug-like-pretrained embedding we cannot validate. |
| **Holzenkamp, Lyu, Kleinekathöfer, Zaspel (2024), UQ for GPR-based MLIPs, arXiv:2410.20398** | Not a method to adopt — a warning to obey. Predictions with increasing σ carry increasing **systematic bias** the uncertainty does not capture, so GPR intervals "can be highly overconfident"; and uncertainty-greedy acquisition produced *worse* test error than random. This directly qualifies gen6 Phase 2's level-uncertainty acquisition arm. **Free diagnostic: plot signed bias vs predicted σ bin, not just \|error\| vs σ.** |
| **Jamali, Cheng, Vargas-Hernández (2026), Spectral Analysis of Molecular Features, arXiv:2510.14217** | One useful positive (only ECFP kernels show a positive spectral-richness/performance correlation; local 3D descriptors are consistently negative) but as a method it argues *against* richer 3D or embedding kernels — matching gen4's failed extended descriptors and gen7's `REP_3d_complexphys` at 1.1732. Purely in-distribution. |

---

## Topic 4 — ML for lanthanide extraction and metal–ligand binding

This is the domain-adjacent literature. It contains our dataset's ancestor, the only external
grouped-CV replication of our regime, and the two largest labelled metal–ligand corpora in
existence.

### 4.1 Liu, Johnson, Jansone-Popova, Jiang (2022). *Advancing Rare-Earth Separation by Machine Learning.* JACS Au 2(6), 1428–1434. — **DONT_TEST (but cite; it is our dataset's ancestor)**
- **URL** https://pubs.acs.org/doi/10.1021/jacsau.2c00122
- **Problem** Literally our task: absolute log D for Ln(III) extraction from ligand SMILES + lanthanide + conditions.
- **Dataset** 1,202 log D values; 109 ligands; 14 lanthanides; phosphine oxides, amides, N-heterocycles.
- **Split** **Random row-level.** 1,085 points, 80% resampled per epoch, 117 "validation". Plus a genuinely prospective test: 4 newly synthesised DGA variants measured in the lab (MAE 0.21–0.41).
- **Architecture** MLP, 2,291 inputs = 2,048 ECFP + 208 RDKit + 14 Ln descriptors + conditions; 512-128-16, PReLU, L1 loss, SGD lr 1e-5, wd 0.01, 15,000 epochs.
- **Why it transfers** Same target, same feature families, same metal set. Their prospective 4-ligand test is the only true out-of-sample log D evidence in the literature.
- **Why it may fail** The headline R²=0.85 / MAE=0.34 is a random row split at ~11 rows per ligand — nearly every test row's ligand is in training, so it measures interpolation over conditions. All four prospective ligands are DGA analogues of the best-covered training family — a near-neighbour lookup in our terms. **Anyone comparing our 1.047 macro MAE to their 0.34 is comparing different problems, and any writeup must say so explicitly.** An MLP at n≈1.2k has no reason to beat a tree at n≈5.2k; our model is a strict superset.
- **Difficulty** Trivial to reimplement, pointless.

### 4.2 Harish (2026). *Objective-Dependent Active Learning and Calibrated Uncertainty for Sample-Efficient Discovery of Rare-Earth Extractants.* ACS Omega 11(32), 48692. — **TEST (highest-priority external replication)**
- **URL** https://pubs.acs.org/doi/10.1021/acsomega.6c07482
- **Problem** Re-benchmarks the Liu 2022 dataset under leakage-controlled splits; compares AL acquisition; calibrates with conformal prediction. The closest external analogue of our whole programme.
- **Dataset** 1,202 log D rows, 93 ligands, 14 lanthanides, 45 literature sources; plus an independent 4,200-compound log D benchmark and 4 prospective ligands.
- **Split** **Both**: the published row-level random split, and grouped CV forbidding a ligand, a source publication, or a structural family from appearing on both sides.
- **Architecture** Random forest (plus five model/uncertainty estimators), split conformal and **Mondrian (group-conditional) conformal**, several AL acquisition functions.
- **Why it transfers** Directly validates our regime numbers from outside: RF gets R²=0.94 / RMSE=0.33 on the row split but **grouped-CV R² collapses to 0.20 … −0.03**. Their source-disjoint split is the same publication-leakage control our provenance audit built. Two concretely testable transfers: (i) conformal calibration gave an 8-fold in-distribution error reduction in calibration and is model-agnostic; (ii) Mondrian conformal reduces subgroup imbalance under shift — what a per-chemotype abstention rule needs.
- **Why it may fail** Also a challenge, not just support: they report AL gives **no advantage over random selection** for building a predictive model, and that exploitation-style acquisition finds top extractants but yields a biased model. gen6 Experiment F concluded max-min Tanimoto beats random. Both can be true (max-min is exploration, not exploitation; our metric is macro MAE on hard chemistry, theirs includes discovery recall), but **the discrepancy must be resolved before we act on F**. Single-author ACS Omega paper — treat numbers as indicative until the released code is checked.
- **Difficulty** Low. Conformal wrappers are ~50 lines around the existing tree with a calibration fold; Mondrian conditions the quantile on the chemotype cluster. A direct replication is a day's work.

### 4.3 Zahariev, Ash, Karunaratne, Stender, Gordon, Windus, Pérez García (2024). *Prediction of stability constants of metal–ligand complexes by ML.* J. Chem. Phys. 160(4), 042502. — **TEST**
- **URL** https://www.osti.gov/servlets/purl/2305609
- **Problem** Predict metal–ligand log K₁ across the periodic table with Chemprop; rank ligands by metal-ion selectivity (LOGKPREDICT + HostDesigner).
- **Dataset** ~1,600 NIST log K points (ligands with ≥5 metals each, up to 50 metals); plus two lanthanide-only IUPAC subsets.
- **Split** 5-fold CV for headline (RMSE 0.629 ± 0.044, R² 0.960 ± 0.006) **plus a genuine held-out-ligand study: six ligands and all their metal rows removed before training.**
- **Architecture** Chemprop D-MPNN with the metal encoded as an atom via **dative bonds** to donor atoms. "D-MPNN+" adds 12 tabulated features: log K at zero ionic strength, ionic strength, ligand charge, metal formal charge, rotations restricted on complexation, effective ionic radius, most common coordination number, MM3 strain energy, metal hydration free energy, and the Hancock–Marsicano electrostatic (rdhE) and covalent (rdhC) descriptors.
- **Why it transfers** **Strongest external corroboration of our level finding, with a number attached.** On six held-out ligands, absolute RMSE is 0.53 (D-MPNN) / 0.46 (D-MPNN+), but after subtracting the single best constant shift per ligand it drops to **0.31 / 0.25**. Roughly half the error on a chemically unseen ligand is a constant per-ligand offset; the metal-to-metal *shape* is already good. Independently reproduces "the level carries the error". Two cheap transfers: (a) the metal-descriptor block (rdhE, rdhC, ionic radius, CN, ΔG_hyd) is tabulated and free; (b) report a shift-invariant metric alongside macro MAE so shape progress is not masked by level error.
- **Why it may fail** The 0.25 "relative" number is an **oracle** — one free parameter per test ligand fitted on test labels, the same trap as our `ORACLE_level` (0.4974). It is a diagnosis, never a performance claim. Their metal descriptors will also buy little: our own metal oracle bounds that whole axis. Dative-bond SMILES needs a known coordination mode, which we do not have for 190 extractants.
- **Difficulty** Low for the descriptor block (a 14-row Ln³⁺ lookup; rdhE/rdhC from Hancock & Marsicano 1978) and the shift-invariant metric (~20 lines). Medium for Chemprop on CPU.

### 4.4 Karunaratne, Zahariev, Pérez García (2025). *A Comprehensive ML Model for Metal−Ligand Binding Prediction.* JCIM 65, 11532–11542. — **TEST**
- **URL** https://www.osti.gov/pages/servlets/purl/3001807
- **Problem** Predict log K₁ for 102 metal ions across 73 elements from ligand SMILES + metal + conditions.
- **Dataset** **32,459 stability constants over 3,585 ligands and 102 metal ions**, plus 11,521 protonation constants over 3,864 ligands (IUPAC SC-Database).
- **Split** An external test set of 20–40 **ligands** with all their metal rows removed up front; remainder 90:10 with 5-fold CV inside. Best model M4: test MAE 0.599 ± 0.004; external MAE 0.834 / RMSE 1.285 / R² 0.942.
- **Architecture** Chemprop D-MPNN with the metal as a **disconnected SMILES component** (`NCCNCCN.[Ni+2]`) — an isolated graph node, no coordination edges. Optional metal descriptors, charges, conditions, 40 f-regression-selected RDKit descriptors.
- **Why it transfers** (1) On held-out ligands D-MPNN beat RF, GB, KNN, Lasso and linear regression on engineered features, **and** beat fine-tuned ChemBERTa and MolBERT on identical splits. Two independent groups now find D-MPNN > tree-on-fingerprints for metal–ligand binding — a real architecture signal we have never tested. (2) The 32k-point IUPAC corpus is the largest chemically adjacent labelled resource in existence and covers lanthanides — a credible pretraining source for a transfer/multitask log D model, which is the one lever gen6 Phase 0 said matters.
- **Why it may fail** Their external ligands are 20–40 hand-picked molecules, not a Tanimoto-disjoint partition, so 0.834 is an easier regime than ours. log K₁ is aqueous 1:1 complexation; log D is a two-phase partition with stoichiometry, acid competition, aggregation and diluent effects — the representation may transfer, the label scale certainly does not. Their own ablation shows Morgan fingerprints and RDKit descriptors added **nothing** to the D-MPNN, so bolting a D-MPNN onto our stack risks paying for capacity gen6 already ruled out.
- **Difficulty** Medium. Chemprop v2 on 8 CPUs at ~5k rows: 5-fold ≈ 1–3 h; the 32k pretrain is an overnight CPU job. Metal-as-disconnected-component is a one-line SMILES concatenation. Budget half a day for the environment.

### 4.5 Chaube, Goverapet Srinivasan, Rai (2020). *Applied ML for predicting the lanthanide-ligand binding affinities.* Sci. Rep. 10, 14322. — **DONT_TEST**
- **URL** https://www.nature.com/articles/s41598-020-71255-9
- **Dataset** 6,583 log K₁ values, 698 ligands, 15 Ln cations, 8 solvent media — almost exactly our row count.
- **Split** **Random** 5,266/1,317 with tenfold CV. No ligand-disjoint evaluation. Applicability domain by bounding box only.
- **Architecture** Six classical learners on 102 features (83 ligand + 14 metal + 3 solvent-medium + T + ionic concentration). AdaBoost best: MAE 0.39, RMSE 0.91, R² 0.98.
- **Why it transfers** Its feature schema confirms which blocks the field considers necessary — a one-hour audit that our diluent/medium block is at least as rich as their 3 solvent properties is worth doing, since that is the block gen6 never interrogated (and gen7's `RECOVERED` now addresses).
- **Why it may fail** MAE 0.39 / R² 0.98 is a pure random split at ~9.4 rows per ligand — an interpolation number widely cited as evidence that lanthanide binding is nearly solved. It is not. AdaBoost on 102 descriptors is strictly weaker than our tree. The 71-million-PubChem extrapolation is a bounding-box claim with no held-out chemistry.
- **Difficulty** Trivial to reproduce; teaches nothing new.

### 4.6 Kanahashi, Urushihara, Yamaguchi (2022). *ML-based analysis of overall stability constants of metal–ligand complexes.* Sci. Rep. 12, 11159. — **DONT_TEST (but use as the reality anchor)**
- **URL** https://www.nature.com/articles/s41598-022-15300-9
- **Dataset** 19,810 points: 13,559 β₁ and 6,251 βₙ; 57 cations, 2,706 ligands.
- **Split** Standard CV plus a custom validation on 20 ligands not in training, chosen by an applicability-domain criterion.
- **Architecture** GPR, Matérn-3/2 with ARD. 12 metal descriptors + Mordred + 2 conditions, reduced to 59 (β₁) / 25 (βₙ). **β₁ MAE 1.31 / R² 0.84; βₙ MAE 1.30 / R² 0.92.**
- **Why it transfers** Best available reality anchor: with 13.5k points and 2,706 ligands, best-case β₁ MAE is **1.31 log units**. Our 1.047 macro MAE on Tanimoto-0.7-disjoint chemotypes is, on that scale, not a weak result — useful framing for a paper. Their ARD ranking (metal *and* ligand electronegativity dominate β₁) is a cheap sanity check against our attributions.
- **Why it may fail** Their second finding — predicted β₁ is the best single predictor of βₙ — is the anchor-then-map structure we already tested and that failed: gen6 Phase 2 found the two-stage model worse than the monolith (1.25 vs 1.08; 1.04 only with an oracle cell mean). Exact GPR is O(n³), and our model already behaves like a near-neighbour lookup — a Matérn GP on descriptors would formalise that without adding chemistry.
- **Difficulty** Low, but the payoff is a re-run of a known negative.

### 4.7 Liu, Ding, Yuan, Fang, Wang, Li, Wang, Qi, Hu (2026). *ML-assisted performance prediction and elucidation of governing factors in rare-earth solvent extraction systems.* Sep. Purif. Technol. — **TEST (for the data, not the method)**
- **URL** https://doi.org/10.1016/j.seppur.2026.138089
- **Dataset** **3,343 curated REE extraction data points** — the largest published REE log D set before ours (we have 5,992).
- **Split** Not stated in any accessible text; the reported R² = 0.96 / MAE = 0.14 is only consistent with a random row split.
- **Architecture** CatBoost on extractant descriptors + metal properties + conditions, with SHAP.
- **Why it transfers** **The value is the data, not the method.** gen6 Phase 0 measured that chemical coverage, not capacity, is the bottleneck (+0.163 macro MAE from breadth alone, 2.8× on distant chemistry), and BOOM independently reports a handful of OOD molecules fixes 7/8 tasks. A separately curated 3,343-point database is the highest-expected-value acquisition available: even with substantial overlap, the non-overlapping extractants extend exactly the coverage we have measured to be binding. Contact the authors or check the SI.
- **Why it may fail** Treat every number with suspicion. MAE 0.14 log units on literature-curated extraction data is below any plausible inter-laboratory floor — SC3 needed careful multi-source curve fitting to defend 0.106 log S, and log D across heterogeneous acid/diluent systems is noisier. That implies a random split with the same extractant on both sides, i.e. memorisation of titration series. Their SHAP headline — conditions matter more than extractant properties — is exactly the artefact a random split produces (conditions vary within a series and are learnable; ligand identity is constant and looks redundant) and contradicts what a chemotype hold-out shows. **Cite as a cautionary contrast, not a result.**
- **Difficulty** Low-medium: obtain the SI database, deduplicate against our 5,992 rows by (SMILES, metal, acid, concentrations, diluent, T), re-derive provenance. 2–4 days of curation, no compute.

---

## Cross-topic synthesis

Four independent lines converge on the same three conclusions.

**1. The tree champion is right, and the burden of proof is on anything replacing it.**
SC3 — the closest published analogue (solute × solvent × temperature, explicit solvent-extrapolation
split, 31 models) — has LightGBM on RDKit descriptors beating every GNN and foundation model, ID and
OOD. Praski et al. find nearly all pretrained embeddings statistically indistinguishable from ECFP.
BOOM finds MLM pretraining improves ID and sometimes *degrades* OOD. MoleculeACE finds descriptor ML
beating graph/SMILES DL at our exact scale. Our own gen7 screens agree: the best learner is a tree at
0.9976, kernel ridge is 1.2684, the MLP is 1.3599, ridge is 1.4563, and every one of 16 pretrained
embedding arms is worse than the incumbent.

**2. The highest-value work is measurement discipline, not new methods.**
Fooladi's ID→OOD correlation collapse (r ≈ 0.95 → 0.16–0.60) means model selection on an easy inner
split is selection on the wrong signal. Harish 2026 independently reproduces our regime's collapse on
the ancestor dataset (R² 0.94 → 0.20…−0.03) and reports that active learning gives **no** advantage
over random — which directly contradicts gen6 Experiment F and must be resolved. SC3's aleatoric-floor
method is reproducible on our data in a day and would tell us whether 1.047 is 1.2× or 6× the floor —
a ratio that determines whether further modelling is worth doing at all.

**3. Of the genuinely new methods, three are well matched.**
*Anchored pairwise-difference regression (PADRE)* formalises what we already know works: it predicts
only deltas from **observed** training labels, so per-ligand and per-publication offsets cancel by
construction — and gen7's `ORACLE_batch` now says a per-(ligand, batch) offset is worth 0.147 on top
of the level. *GPBoost* writes the same structure as an explicit random-effects model estimated
jointly with the tree mean, which is the version of the two-stage idea that does not fall into the
gen6 residual trap. *Mondrian conformal + a learned error model* delivers the abstention rule the
project actually needs; the Parrondo-Pizarro result (learned error model Spearman up to 0.46 vs ≤0.26
for raw distance or variance) is the cheapest UQ win available, and Holzenkamp's warning tells us to
report signed bias per σ bin so we do not over-claim.

---

## Shortlist to implement — ordered by expected value

Expected value = (probability it moves a number we report) × (size of the move) ÷ cost.
"Measured" means a gen7 artefact already contains evidence for it.

| # | Item | Topic ref | Cost | Expected effect | Evidence class |
|---|---|---|---|---|---|
| 1 | **Turn missing-value indicators on everywhere, and audit whether the gain is batch inference** | 1.5, 1.20 | hours | **+0.0614 macro MAE, measured** (`IND_none` 1.108949 → `IND_all` 1.047537, bit-identical to `LRN_extratrees` and `TREE_MC_lig2d_ext_massaction`) — larger than the entire lig2d ligand block. But `cond__contact_time_min` is 39.95% NaN and curator-batch-structured, so part of the gain may be study inference. | measured + leakage risk |
| 2 | **Re-run every gen7 screen at 5 seeds with the paired chemotype-block bootstrap** | 1.1, 1.20 | 1–2 d CPU | No arm ordering currently in `runs/gen7_architecture/` is believable: learners/oracles/indicators ran at 3 seeds, representations and embeddings at **1**. Refit noise alone is up to 0.0101 macro MAE (`runs/gen6_phase0/summary.json`). | protocol |
| 3 | **Fix the learner/representation confound: `Tabular` defaults `add_indicator=False` while `LevelTree` imputes with indicators** | 1.5 | hours | `LRN_*` and `REP_*` are handicapped by 0.0614 relative to `TREE_*`. The learner leaderboard is not like-for-like and must be re-run before any "learner does not matter" claim. | measured harness defect |
| 4 | **Batch/level deconfounding: model the (ligand, batch) offset explicitly (GPBoost random effects, or PADRE anchoring)** | 3.2, 1.10, 4.3 | 3–5 d | `ORACLE_level` 0.4974 → `ORACLE_batch` 0.3502 = **0.1472**, and shape falls 0.5086 → 0.3634 (−28.5%). ~29% of the "immovable" shape error is study-to-study calibration, not chemistry. This is the largest untouched structured signal in the data. | measured oracle |
| 5 | **Aleatoric-floor estimate from cross-publication duplicate cells** | 1.4 | 1 d | We do not know whether 1.047 is 1.2× or 6× the floor. gen6 Phase 0 already showed the published 0.189 floor is contaminated (26 cells with *no* recoverable difference scatter 0.875, more than the 79 with a physical axis). | protocol / decisive |
| 6 | **Complete the recovered-variable block and confirm at 5 seeds** | 4.5, 1.20 | 1–2 d | **+0.0310 / +0.0499 measured at 3 seeds** (`REC_base` 1.0359 → `REC_plus_recovered` 1.0049; `REC_champion` 1.1089 → 1.0590). The aqueous complexant explains 56.7% of within-repeated-cell SS and is entirely absent downstream. | measured |
| 7 | **Anchored pairwise-difference regression (PADRE) over the tree** | 1.10 | 2–4 d | Directly attacks the 0.5002 offset headroom. Must anchor on observed labels, never on a predicted residual (gen6 trap: 1.25 vs 1.08). Restrict pairs by shared (metal, acid, diluent). | literature, well-matched |
| 8 | **Mondrian conformal + learned error model (abstention rule)** | 1.17, 1.18, 4.2 | 1–2 d | Changes the deliverable rather than the metric: a per-cluster interval and a defer rule. Learned error model reaches Spearman 0.46 vs ≤0.26 for raw distance/variance. Report signed bias per σ bin (Holzenkamp). | literature, cheap |
| 9 | **Lo-Hi / DataSAIL split audit: is any cross-fold ligand pair above Tanimoto 0.7?** | 1.2, 1.3 | half a day | Determines whether 1.047 is measured on genuinely novel chemistry. Freeze the current split for comparability; report the audit separately. | protocol |
| 10 | **Donor-anchored RAC-style 2D autocorrelations** | 1.15 | 2–3 d | `REP_donors` (1.0535) already beats `REP_ecfp` (1.1159) and `REP_lig2d` (1.1403) — donor-centric information is the only ligand channel that is winning. RACs are the principled extension. Gate behind a 5-seed CI. | literature + measured direction |
| 11 | **Count/unhashed fingerprints and Topological Torsion** | 1.9, 1.21 | half a day | 190 SMILES collapse to 164 distinct 2048-bit fingerprints (18 collision groups; 44 ligands have a Tanimoto-1.0 partner). Hygiene with a plausible small gain. Freeze the split definition. | literature, near-free |
| 12 | **Acquire the SPT 2026 3,343-point REE database** | 4.7 | 2–4 d curation | Coverage is the measured bottleneck (+0.163 from breadth in gen6 Exp A). Highest-EV acquisition on the list; ignore their metrics entirely. | measured bottleneck |
| 13 | **CheMeleon frozen 2048-d representation** | 1.13 | 1–2 h | The only pretrained encoder with a Tukey-HSD win over a Morgan+RDKit RF and an independent positive OOD result. But its pretraining target is Mordred — likely redundant with our 2D block. | literature, cheap, low prior |
| 14 | **TabPFN as a learner swap (PCA-compressed features, grouped folds)** | 1.11, 1.12 | 2–3 d | The only candidate that could beat the tree on the mean. Structural worry: it attends over rows and we have ~35 rows per independent ligand. Hard time budget before starting. | literature, uncertain |
| 15 | **CatBoost SGLB virtual ensembles for epistemic variance** | 3.3 | half a day | Settles whether a GP's value is the variance or the kernel prior, at near-zero cost. Uncertainty lives in tree-partition space, not chemistry space — validate against nearest-train Tanimoto. | literature, cheap |
| 16 | **Chemprop D-MPNN with the metal as a disconnected node, optionally pretrained on the 32k IUPAC log K corpus** | 4.4, 4.3 | 1–3 h/fold + overnight pretrain | Two independent groups find D-MPNN > tree-on-fingerprints for metal–ligand binding — the one architecture signal from the *domain* literature we have never tested. But their own ablation shows fingerprints/descriptors add nothing to it, and log K ≠ log D. | literature, domain-specific |
| 17 | **Exact Tanimoto GP with a composite k_ligand × k_metal × k_conditions kernel** | 3.1, 3.4, 3.5 | 1–2 d | Do **not** install `gauche`; the kernel is ~6 lines and exact Cholesky at n=5,248 is seconds. Prior is weak: `LRN_kernelridge_rbf` is already 1.2684. Value is the variance and the additive-order decomposition, not the mean. | literature, low prior |
| 18 | **Metal-descriptor block (ionic radius, rdhE, rdhC, CN, ΔG_hyd) + shift-invariant metric** | 4.3, 3.4 | hours | `runs/gen7_architecture/metal/` is empty and the `MET_*` suite is already defined. Our metal oracle bounds the whole axis at ~0.2, so expect little on MAE — but the shift-invariant metric is needed so shape progress is not masked by level error. | literature, cheap, bounded |
| 19 | **GFN2-xTB free-ligand electronic descriptors** | 1.16 | 3–5 d | A genuinely different information channel. But the bundle already has complex-level xTB scalars and `REP_3d_complexphys` is the worst arm (1.1732), and the free-ligand proxy is conformer-noisy. | literature, low prior |
| 20 | **Executable dataset contract (`contract.yaml` + pre-run assertions)** | 1.20 | 1–2 d | Produces no metric gain; prevents the next `min_rows`/`within_ligand_r2`/publication-mixing class of error. Keep to a few dozen lines of assertions. | protocol / insurance |

**Explicitly not on the list**, with reasons in the DONT_TEST tables above: end-to-end deep kernel
learning, GROVER, MolCLR, Uni-Mol/Uni-Mol2, Mol2vec, CDDD, ChemGPT, FlowMO string kernels, Tanimoto
random features, DeepDelta's neural pairing, meta-learning (FS-Mol), SMILES enumeration, invariance
objectives (IRM/CORAL/DANN), and QM-to-experiment transfer pretraining.
