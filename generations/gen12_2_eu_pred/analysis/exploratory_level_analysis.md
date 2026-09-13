# Exploratory analysis — coordination topology against the level target

Development fold: design B, seed 104729, fold 0. 153 training extractants, 77 training chemotypes. No held-out extractant and no model output contributed. The descriptor specification was frozen and digested before this ran, so nothing here can feed back into descriptor design.

**This section describes a representation. It does not claim chemistry: a Spearman correlation between a graph descriptor and an average log D over heterogeneous conditions is not a mechanism.**

## Strongest univariate associations (top 20 of 107 informative descriptors)

| descriptor                               | family   |   spearman |   abs_spearman |   partial_spearman_given_size |   n_extractants |
|:-----------------------------------------|:---------|-----------:|---------------:|------------------------------:|----------------:|
| coord__motif__n_dga_unit                 | motif    |      0.645 |          0.645 |                         0.613 |             153 |
| coord__donor__n_hard_O                   | donor    |      0.591 |          0.591 |                         0.526 |             153 |
| coord__donor__frac_hard_O                | donor    |      0.58  |          0.58  |                         0.516 |             153 |
| coord__donor__n_O_ether                  | donor    |      0.544 |          0.544 |                         0.522 |             153 |
| coord__arm__n_amide_branches             | arm      |      0.532 |          0.532 |                         0.549 |             153 |
| coord__donor__n_O_amide_carbonyl         | donor    |      0.532 |          0.532 |                         0.549 |             153 |
| coord__donor__frac_N                     | donor    |     -0.49  |          0.49  |                        -0.398 |             153 |
| coord__motif__max_repeated_named_unit    | motif    |      0.479 |          0.479 |                         0.412 |             153 |
| coord__donor__n_N_aromatic               | donor    |     -0.478 |          0.478 |                        -0.398 |             153 |
| coord__donor__n_N                        | donor    |     -0.476 |          0.476 |                        -0.39  |             153 |
| coord__arm__n_clusters_with_2plus_donors | arm      |      0.473 |          0.473 |                         0.313 |             153 |
| coord__motif__n_distinct_named_units     | motif    |      0.471 |          0.471 |                         0.431 |             153 |
| coord__motif__n_named_chelating_units    | motif    |      0.47  |          0.47  |                         0.413 |             153 |
| coord__arm__n_clusters_with_3plus_donors | arm      |      0.443 |          0.443 |                         0.351 |             153 |
| coord__arch__n_aromatic_rings            | arch     |     -0.423 |          0.423 |                        -0.376 |             153 |
| coord__arch__n_rings                     | arch     |     -0.401 |          0.401 |                        -0.348 |             153 |
| coord__arch__mean_heavy_degree           | arch     |     -0.384 |          0.384 |                        -0.387 |             153 |
| coord__motif__n_pyridine_like_ring       | motif    |     -0.375 |          0.375 |                        -0.27  |             153 |
| coord__donor__element_diversity          | donor    |     -0.351 |          0.351 |                        -0.235 |             153 |
| coord__arch__n_rotatable_bonds           | arch     |      0.339 |          0.339 |                         0.253 |             153 |

## The pre-registered headline descriptors

| descriptor                                 | family   |   spearman |   abs_spearman |   partial_spearman_given_size |   n_extractants |
|:-------------------------------------------|:---------|-----------:|---------------:|------------------------------:|----------------:|
| coord__motif__n_dga_unit                   | motif    |      0.645 |          0.645 |                         0.613 |             153 |
| coord__motif__n_named_chelating_units      | motif    |      0.47  |          0.47  |                         0.413 |             153 |
| coord__dist__min_intercluster_distance     | dist     |      0.258 |          0.258 |                         0.218 |             153 |
| coord__arm__repeated_arm_count             | arm      |      0.227 |          0.227 |                         0.133 |             153 |
| coord__arm__local_donor_cluster_count      | arm      |      0.215 |          0.215 |                         0.055 |             153 |
| coord__dist__donor_network_diameter        | dist     |      0.213 |          0.213 |                         0.232 |             153 |
| coord__arm__max_connected_donor_motif_size | arm      |     -0.15  |          0.15  |                        -0.077 |             153 |
| coord__arch__frac_atoms_in_largest_orbit   | arch     |     -0.147 |          0.147 |                        -0.055 |             153 |
| coord__donor__potential_donor_count        | donor    |      0.114 |          0.114 |                         0.059 |             153 |
| coord__arch__mol_weight_per_donor          | arch     |      0.063 |          0.063 |                        -0.056 |             153 |
| coord__arch__heavy_atoms_per_donor         | arch     |      0.062 |          0.062 |                        -0.059 |             153 |
| coord__arch__n_degree3_atoms               | arch     |     -0.037 |          0.037 |                        -0.137 |             153 |
| coord__arch__clogp_per_donor               | arch     |     -0.026 |          0.026 |                        -0.168 |             153 |
| coord__dist__frac_donor_pairs_within_3     | dist     |      0.015 |          0.015 |                         0.068 |             153 |

## Level against pocket count

|   coord__arm__local_donor_cluster_count |   n_extractants |   mean_level |   median_level |   sd_level |   mean_cells |
|----------------------------------------:|----------------:|-------------:|---------------:|-----------:|-------------:|
|                                       1 |             116 |       -0.48  |         -0.203 |      1.575 |        8.121 |
|                                       2 |              26 |        0.149 |          0.924 |      2.359 |        5     |
|                                       3 |               7 |        1.488 |          2.306 |      1.779 |        6.286 |
|                                       4 |               4 |       -0.413 |          0.096 |      1.896 |        8     |

### within the diglycolamide family only

|   coord__arm__local_donor_cluster_count |   n_extractants |   mean_level |
|----------------------------------------:|----------------:|-------------:|
|                                       1 |              44 |        0.729 |
|                                       2 |              18 |        0.793 |
|                                       3 |               6 |        2.136 |
|                                       4 |               3 |        0.449 |

## Family-stratified correlations

| descriptor                                 |   amide_other |   diglycolamide |   n_heterocyclic_polydentate |   phosphoryl |   sulfur_donor |
|:-------------------------------------------|--------------:|----------------:|-----------------------------:|-------------:|---------------:|
| coord__arch__clogp_per_donor               |        -0.218 |          -0.312 |                        0.401 |        0.072 |         -0.319 |
| coord__arch__frac_atoms_in_largest_orbit   |        -0.283 |          -0.129 |                       -0.031 |        0.847 |          0.116 |
| coord__arch__heavy_atoms_per_donor         |        -0.21  |          -0.343 |                        0.356 |        0.356 |         -0.841 |
| coord__arch__mol_weight_per_donor          |        -0.212 |          -0.344 |                        0.335 |        0.366 |         -0.725 |
| coord__arch__n_degree3_atoms               |        -0.094 |           0.093 |                        0.509 |        0.565 |         -0.845 |
| coord__arm__local_donor_cluster_count      |        -0.286 |           0.263 |                       -0.223 |      nan     |         -0.399 |
| coord__arm__max_connected_donor_motif_size |         0.247 |           0.331 |                       -0.002 |       -0.494 |          0.567 |
| coord__arm__repeated_arm_count             |        -0.48  |           0.248 |                      nan     |      nan     |         -0.399 |
| coord__dist__donor_network_diameter        |        -0.267 |           0.45  |                        0.177 |       -0.075 |          0.133 |
| coord__dist__frac_donor_pairs_within_3     |         0.561 |          -0.195 |                        0.112 |       -0.454 |         -0.03  |
| coord__dist__min_intercluster_distance     |        -0.254 |           0.343 |                       -0.229 |      nan     |         -0.399 |
| coord__donor__potential_donor_count        |         0.016 |           0.478 |                       -0.038 |       -0.494 |          0.546 |
| coord__motif__n_dga_unit                   |       nan     |           0.461 |                      nan     |      nan     |        nan     |
| coord__motif__n_named_chelating_units      |         0.244 |           0.461 |                        0.524 |        0.282 |          0.42  |
