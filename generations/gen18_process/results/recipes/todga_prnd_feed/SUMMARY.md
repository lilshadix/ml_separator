# Recipes for feed todga_prnd_feed.json (anion nitrate), spec prnd_spec.json

*regime: cohort=database systems with medium.anion == nitrate; holdout=none (computed regimes); averaging_unit=candidate; status_of_parameters=per system (column status_of_parameters); LHS n = 300, seed 18; on-spec = loosest cell {'purity_min': 0.95, 'recovery_min': 0.8}; knee = consistency cell (0.97, 0.85)*

Systems considered: 10; parameterised: 10; not_parameterised: 0. Pre-registered decision m1_adopted = False.

## Family diglycolamide

| system_id | name | status | params_source | status_of_parameters | n_converged | n_failed | n_on_spec_loosest | n_front_in_domain | reason |
|---|---|---|---|---|---|---|---|---|---|
| sys_07ee9637c98c1e20 | TODGA in other, nitrate medium, no additive | parameterised | nearest | nearest (origin corpus; ASSUMED_PLACEHOLDER values) | 300 | 0 | 0 | 11 |  |
| sys_1419f400c83e9ad8 | DMDODGA in aliphatic hydrocarbon, nitrate medium, no additive | parameterised | nearest | nearest (origin corpus; ASSUMED_PLACEHOLDER values) | 300 | 0 | 0 | 0 |  |
| sys_5cb78e5000d40860 | TODGA in aliphatic hydrocarbon, nitrate medium, no additive | parameterised | nearest | nearest (origin corpus; ASSUMED_PLACEHOLDER values) | 299 | 1 | 0 | 15 |  |
| sys_a2a472b5124b09c6 | TODGA in aromatic, nitrate medium, no additive | parameterised | nearest | nearest (origin corpus; ASSUMED_PLACEHOLDER values) | 300 | 0 | 1 | 24 |  |
| sys_f58e3a596f8d50f9 | TDdDGA in aliphatic hydrocarbon, nitrate medium, no additive | parameterised | nearest | nearest (origin corpus) | 300 | 0 | 1 | 9 |  |


Top three on-spec regimes per system (by consumption index):

| family | front | regime_status | n_ext | n_scr | n_str | oa_ext | s_over_a | w_over_a | scrub_acid_M | strip_acid_M | ligand_total_M | saponification_degree | purity_mol | recovery_from_feed | consumption_index | n_stages_total | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diglycolamide | 1 | IN_DOMAIN_WITH_CAVEATS | 9 | 6 | 6 | 3.812 | 1.333 | 0.7819 | 1.575 | 3.678 | 0.2068 | 0 | 0.9857 | 0.984 | 263.1 | 21 | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|PHASE_BEHAVIOUR_UNKNOWN |
| diglycolamide | 1 | OUT_OF_DOMAIN | 7 | 12 | 7 | 2.783 | 4.243 | 3.654 | 1.428 | 0.6059 | 0.2923 | 0 | 0.9714 | 0.9684 | 309.3 | 26 | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|OOD_HULL\|OOD_LIGAND\|OOD_METAL\|PHASE_BEHAVIOUR_UNKNOWN |


Pareto knee of the consistency cell:

| family | system_id | subset | purity_min | recovery_min | reachable | n_feasible | n_stages_total | consumption_index | regime_status | flags |
|---|---|---|---|---|---|---|---|---|---|---|
| diglycolamide | sys_07ee9637c98c1e20 | all | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |
| diglycolamide | sys_07ee9637c98c1e20 | in_domain_only | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |
| diglycolamide | sys_1419f400c83e9ad8 | all | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |
| diglycolamide | sys_1419f400c83e9ad8 | in_domain_only | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |
| diglycolamide | sys_5cb78e5000d40860 | all | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |
| diglycolamide | sys_5cb78e5000d40860 | in_domain_only | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |
| diglycolamide | sys_a2a472b5124b09c6 | all | 0.97 | 0.85 | True | 1 | 21 | 263.1 | IN_DOMAIN_WITH_CAVEATS | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|PHASE_BEHAVIOUR_UNKNOWN |
| diglycolamide | sys_a2a472b5124b09c6 | in_domain_only | 0.97 | 0.85 | True | 1 | 21 | 263.1 | IN_DOMAIN_WITH_CAVEATS | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|PHASE_BEHAVIOUR_UNKNOWN |
| diglycolamide | sys_f58e3a596f8d50f9 | all | 0.97 | 0.85 | True | 1 | 26 | 309.3 | OUT_OF_DOMAIN | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|OOD_HULL\|OOD_LIGAND\|OOD_METAL\|PHASE_BEHAVIOUR_UNKNOWN |
| diglycolamide | sys_f58e3a596f8d50f9 | in_domain_only | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |


## Family n_donor

| system_id | name | status | params_source | status_of_parameters | n_converged | n_failed | n_on_spec_loosest | n_front_in_domain | reason |
|---|---|---|---|---|---|---|---|---|---|
| sys_fbff75db47e9b852 | C5BTBP in aliphatic hydrocarbon, nitrate medium, no additive | parameterised | nearest | nearest (origin corpus) | 300 | 0 | 0 | 0 |  |


Top three on-spec regimes per system (by consumption index):

(none on spec)

Pareto knee of the consistency cell:

| family | system_id | subset | purity_min | recovery_min | reachable | n_feasible | n_stages_total | consumption_index | regime_status | flags |
|---|---|---|---|---|---|---|---|---|---|---|
| n_donor | sys_fbff75db47e9b852 | all | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |
| n_donor | sys_fbff75db47e9b852 | in_domain_only | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |


## Family other

| system_id | name | status | params_source | status_of_parameters | n_converged | n_failed | n_on_spec_loosest | n_front_in_domain | reason |
|---|---|---|---|---|---|---|---|---|---|
| sys_50088af14793f41f | DOODA (C12) in aliphatic hydrocarbon, nitrate medium, no additive | parameterised | nearest | nearest (origin corpus) | 300 | 0 | 0 | 0 |  |
| sys_a6ca4cd427ac7fcd | DMDO-HPyranDGA in aliphatic hydrocarbon, nitrate medium, no additive | parameterised | nearest | nearest (origin corpus) | 300 | 0 | 0 | 0 |  |
| sys_d93f24ac2fbe9b76 | 2-N-6-N-dimethyl-2-N-6-N-diphenylpyridine-2-6-dicarboxamide in chlorinated, nitrate medium, no additive | parameterised | nearest | nearest (origin corpus) | 300 | 0 | 1 | 0 |  |
| sys_edd93a49877e0b03 | DOODA (C8) in aliphatic hydrocarbon, nitrate medium, no additive | parameterised | nearest | nearest (origin corpus) | 300 | 0 | 0 | 0 |  |


Top three on-spec regimes per system (by consumption index):

| family | front | regime_status | n_ext | n_scr | n_str | oa_ext | s_over_a | w_over_a | scrub_acid_M | strip_acid_M | ligand_total_M | saponification_degree | purity_mol | recovery_from_feed | consumption_index | n_stages_total | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| other | 1 | OUT_OF_DOMAIN | 11 | 12 | 4 | 4.689 | 4.112 | 4.14 | 2.701 | 5.996 | 0.7357 | 0.1006 | 0.9807 | 0.823 | 980.2 | 27 | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|OOD_ACID\|OOD_HULL\|OOD_LIGAND\|OOD_LOADING\|OOD_METAL\|PHASE_BEHAVIOUR_UNKNOWN\|SAPONIFICATION_RANGE_UNKNOWN |


Pareto knee of the consistency cell:

| family | system_id | subset | purity_min | recovery_min | reachable | n_feasible | n_stages_total | consumption_index | regime_status | flags |
|---|---|---|---|---|---|---|---|---|---|---|
| other | sys_50088af14793f41f | all | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |
| other | sys_50088af14793f41f | in_domain_only | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |
| other | sys_a6ca4cd427ac7fcd | all | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |
| other | sys_a6ca4cd427ac7fcd | in_domain_only | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |
| other | sys_d93f24ac2fbe9b76 | all | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |
| other | sys_d93f24ac2fbe9b76 | in_domain_only | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |
| other | sys_edd93a49877e0b03 | all | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |
| other | sys_edd93a49877e0b03 | in_domain_only | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |

