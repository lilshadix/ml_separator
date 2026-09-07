| family           | reference          | candidate           | statistic   |   bootstrap_se |   power |    mde |   n_blocks |   n_units |
|:-----------------|:-------------------|:--------------------|:------------|---------------:|--------:|-------:|-----------:|----------:|
| zero_shot        | T1_EXTRATREES      | B0_GLOBAL_MEAN      | mae         |         0.1009 |     0.8 | 0.2828 |         97 |       183 |
| zero_shot        | T1_EXTRATREES      | B1_COND_ONLY        | mae         |         0.1038 |     0.8 | 0.2908 |         97 |       183 |
| zero_shot        | T1_EXTRATREES      | B2_NN_CHEMICAL      | mae         |         0.1451 |     0.8 | 0.4066 |         97 |       183 |
| zero_shot        | T1_EXTRATREES      | B3_NN_LEVEL_ONLY    | mae         |         0.1043 |     0.8 | 0.2923 |         97 |       183 |
| zero_shot        | T1_EXTRATREES      | T1_CATBOOST         | mae         |         0.0274 |     0.8 | 0.0769 |         97 |       183 |
| zero_shot        | T1_EXTRATREES      | T1_RANDOMFOREST     | mae         |         0.0174 |     0.8 | 0.0489 |         97 |       183 |
| zero_shot        | T1_EXTRATREES      | T1_XGBOOST          | mae         |         0.0187 |     0.8 | 0.0523 |         97 |       183 |
| zero_shot        | T1_EXTRATREES      | T2_MLP              | mae         |         0.05   |     0.8 | 0.14   |         97 |       183 |
| zero_shot        | T1_EXTRATREES      | T3_DMPNN_COND       | mae         |         0.0474 |     0.8 | 0.1328 |         97 |       183 |
| zero_shot        | T1_EXTRATREES      | T3_DMPNN_COND_DESC  | mae         |         0.0685 |     0.8 | 0.1918 |         97 |       183 |
| zero_shot        | T1_EXTRATREES      | T3_DMPNN_GRAPH_ONLY | mae         |         0.0816 |     0.8 | 0.2287 |         97 |       183 |
| multi_lanthanide | T4_MATCHED_EU_ONLY | T4_STRICT           | mae         |         0.0113 |     0.8 | 0.0317 |         97 |       183 |
| multi_lanthanide | T4_MATCHED_EU_ONLY | T4_PERMUTED_METAL   | mae         |         0.0131 |     0.8 | 0.0368 |         97 |       183 |
| multi_lanthanide | T4_MATCHED_EU_ONLY | T4_SHUFFLED_TARGET  | mae         |         0.0333 |     0.8 | 0.0932 |         97 |       183 |
| multi_lanthanide | T4_MATCHED_EU_ONLY | T4_LEAKY_NO_FILTER  | mae         |         0.0685 |     0.8 | 0.192  |         97 |       183 |
| multi_lanthanide | T4_MATCHED_EU_ONLY | EU_ONLY_ANCHOR      | mae         |         0.0098 |     0.8 | 0.0273 |         97 |       183 |
