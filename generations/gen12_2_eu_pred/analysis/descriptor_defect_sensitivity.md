# Post-hoc: do the descriptor-definition defects change the answer?

*Found by adversarial review **after** the locked evaluation. The frozen v1.1.0 matrix still reproduces bit-identically and remains the basis of every headline number; this page is a sensitivity and is labelled as one.*

The corrections change **52 of 114** columns and add 1 (coord__donor__n_O_ether_aromatic_ring).

## Level MAE, zero-shot, design B, full cohort

|                 |   posthoc_spec |
|:----------------|---------------:|
| L1_ECFP         |         1.08   |
| L3_ECFP_GENERIC |         1.0624 |
| L4_COORD        |         0.9744 |
| L5_ECFP_COORD   |         1.0106 |
| L7_ALL          |         1.0165 |

## The contrasts, frozen against post-hoc

| contrast       |   frozen_delta | frozen_bca                                |   frozen_p |   posthoc_delta | posthoc_bca                                 | posthoc_ci95                                |   posthoc_p | sign_unchanged   |
|:---------------|---------------:|:------------------------------------------|-----------:|----------------:|:--------------------------------------------|:--------------------------------------------|------------:|:-----------------|
| PRIMARY_G_vs_D |         0.0343 | [0.0043671172015991, 0.0746883602416177]  |     0.092  |          0.0459 | [0.010146143995493781, 0.09749864659093183] | [0.0031497549152824174, 0.0823013835397861] |      0.02   | True             |
| E_vs_A         |         0.0595 | [0.0202073243378065, 0.0949379799941126]  |     0.0074 |          0.0694 | [0.01728857415996161, 0.13214646391070872]  | [0.003158569150748934, 0.11708688808430771] |      0.0348 | True             |
| C_vs_D         |         0.1001 | [-0.0598669033236999, 0.2960524093865563] |     0.4644 |          0.0879 | [-0.08293675318318937, 0.30907226749809535] | [-0.12157235408649297, 0.2514645797067629]  |      0.5874 | True             |
