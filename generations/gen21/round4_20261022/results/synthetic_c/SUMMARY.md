# Block C synthetic discrimination study, run 4 (the sealed code)

6 paired replicates on two BPG folds (seed 130363, folds 2 and 3): synthetic Ln targets (chemotype nuisance, noise sd 0.11, signal 0.15 · u · tetrad_e3) shared by the three worlds of a replicate; synthetic An targets on the real An(III) record set. Counts k/n with exact 95 % intervals. Rules are an upper bound of the sealed ones (vs B1 only, no harm rule, no B2, one fold, no seed mean).

| world | crit1 S11 | crit1 S13 | S13 − S11 ≥ 0.005 | transfer claim | selectivity overall / given claim | gate-equivalent | S13 chose w_X > 0 | mean ΔR² S11 / S13 / S14 / S15 / S16 | paired S13 − S11 minus NULL's |
|---|---|---|---|---|---|---|---|---|---|
| NULL | 1/6 [0.00, 0.64] | 1/6 [0.00, 0.64] | 0/6 [0.00, 0.46] | 0/6 [0.00, 0.46] | 1/6 [0.00, 0.64] / 0/0 | 0/6 [0.00, 0.46] | 2/6 | +0.0189 / +0.0189 / +0.0189 / +0.0189 / +0.0189 | +0.0000 (min +0.0000, max +0.0000) |
| SMOOTH_NULL | 1/6 [0.00, 0.64] | 1/6 [0.00, 0.64] | 0/6 [0.00, 0.46] | 0/6 [0.00, 0.46] | 2/6 [0.04, 0.78] / 0/0 | 0/6 [0.00, 0.46] | 3/6 | +0.0189 / +0.0194 / +0.0195 / +0.0189 / +0.0189 | +0.0005 (min +0.0000, max +0.0016) |
| AN_SIGNAL | 1/6 [0.00, 0.64] | 1/6 [0.00, 0.64] | 0/6 [0.00, 0.46] | 0/6 [0.00, 0.46] | 3/6 [0.12, 0.88] / 0/0 | 0/6 [0.00, 0.46] | 3/6 | +0.0189 / +0.0196 / +0.0190 / +0.0189 / +0.0189 | +0.0007 (min +0.0000, max +0.0020) |

S11's chosen networks: {'w_Ln': 1.0, 'WD': 0.3, 'w_X': 0.0}, {'w_Ln': 0.0, 'WD': 0.3, 'w_X': 0.0}, zero, {'w_Ln': 0.3, 'WD': 0.3, 'w_X': 0.0}, {'w_Ln': 1.0, 'WD': 0.3, 'w_X': 0.0}, {'w_Ln': 0.0, 'WD': 0.3, 'w_X': 0.0}
