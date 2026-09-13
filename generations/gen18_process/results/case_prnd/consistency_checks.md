# Consistency checks against the cited numbers (DESIGN.md section 13.4) -- never validation

Regime: cohort=Pr/Nd case: literature placeholder entries PC88A and Cyanex 272 (no corpus rows), feed_status assumed (U1); holdout=none (computed regimes, consistency checks are not validation); averaging_unit=draw (parameter draw, seed 18); status_of_parameters=PLACEHOLDER_PARAMETERS (assumed with ranges, --allow-placeholders), feed_status assumed, spec_status assumed

Feeds, acidities and loadings of the sources are unknown (LITERATURE_NOTES.md section 2); none of (a)-(c) can validate the model.

## (a) Thakur 1993 (doi 10.1016/0304-386X(93)90084-Q): 97 % purity at > 85 % recovery, counter-current PC88A

Cell (0.97, 0.85), PC88A, subset all: **not reachable within the assumed window**; no draw reached the cell in the sweep.

## (b) EP2388344A1: PC-88A Nd/Pr circuit 72 extraction + 72 scrub + 8 strip stages at SF 1.4

Stage ladder (n_ext = n_scr = N, n_str = 8, 8 LHS rows per rung, SF 1.4, median draw otherwise): **at SF 1.4 the (0.99, 0.99) cell is not reached up to n_ext = n_scr = 40 (+ 8 strip; the solver budget ends there): the stage count the model needs is of the order of 70 + 70 rather than 7 + 7**.

Analytic Fenske-type minimum at total reflux for the (0.99, 0.99) cell with SF 1.4 and feed Nd:Pr 3.00: N_min = 24.0 theoretical stages (a bound from the constant-SF ideal, not the cascade); practical countercurrent circuits need a multiple of it, which is the order of the patent's 72 + 72.

| system_id | sf_nd_pr | n_ext | n_scr | n_str | n_lhs | n_converged | n_failed | cell_reached | n_reached | best_purity_mol | best_recovery_at_purity_ge_0.99 | max_purity_recovery_product |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sys_29976921e156a0a0 | 1.4 | 3 | 3 | 8 | 8 | 8 | 0 | False | 0 | 1 | 3.693e-07 | 3.693e-07 |
| sys_29976921e156a0a0 | 1.4 | 5 | 5 | 8 | 8 | 8 | 0 | False | 0 | 1 | 2.65e-11 | 2.65e-11 |
| sys_29976921e156a0a0 | 1.4 | 7 | 7 | 8 | 8 | 8 | 0 | False | 0 | 1 | 1.701e-15 | 1.701e-15 |
| sys_29976921e156a0a0 | 1.4 | 10 | 10 | 8 | 8 | 8 | 0 | False | 0 | 1 | 8.373e-22 | 8.373e-22 |
| sys_29976921e156a0a0 | 1.4 | 15 | 15 | 8 | 8 | 8 | 0 | False | 0 | 1 | 2.497e-32 | 2.497e-32 |
| sys_29976921e156a0a0 | 1.4 | 20 | 20 | 8 | 8 | 8 | 0 | False | 0 | 1 | 7.306e-43 | 7.306e-43 |
| sys_29976921e156a0a0 | 1.4 | 30 | 30 | 8 | 8 | 6 | 2 | False | 0 | 1 | 6.313e-64 | 6.313e-64 |
| sys_29976921e156a0a0 | 1.4 | 40 | 40 | 8 | 8 | 6 | 2 | False | 0 | 1 | 5.456e-85 | 5.456e-85 |


## (c) Banda 2014 (doi 10.1016/j.jiec.2014.03.002): maximum SF about 1.5

**enters only as the upper end of the SF placeholder range [1.3, 1.5]**; no number of Banda 2014 is reproduced or compared.
