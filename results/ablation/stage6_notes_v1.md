# Stage-6 Ablation

Inputs
- dataset: data/dataset_v1_frozen_scaffoldfix_v2.csv
- model_table: data/features/model_table_v1.csv.gz

Training pool
- role filter: NONE

Configuration
- SEED = 42
- random_splits = 5
- scaffold_splits = 5
- scaffold_group_col = Murcko_scaffold_ID_v2
- n_samples = 69
- n_scaffold_groups = 39

Feature sets
- n_physchem = 10
- n_ecfp = 2048
- n_hybrid = 2058

Outputs
- results/ablation/ablation_results.xlsx
- results/ablation/ablation_split_comparison_v1.csv
- results/ablation/ablation_feature_sets_v1.csv
- results/ablation/ablation_model_panel_v1.csv
