# Stage-5.3 — SHAP (FULL-X aligned to training features)

Inputs
- model_bundle: results/stage5_freeze_v1/final_xgb_model_v1.pkl
- model_table: data/features/model_table_v1.csv.gz
- dataset: data/dataset_v1_frozen_scaffoldfix_v2.csv

Bundle info
- model_bundle_type: <class 'dict'>
- estimator_type: <class 'xgboost.sklearn.XGBRegressor'>
- feature_list_found_in_bundle: True

Alignment
- aligned_to_bundle_features: True
- n_features_used: 2058
- missing_cols_filled_with_0: []
- extra_cols_ignored: ['logP_median', 'Tier', 'external_role', 'set_role', 'outer_fold', 'Murcko_scaffold_ID']

Outputs
- results/interpretability/shap_physchem_global_v1.csv
- results/interpretability/shap_physchem_direction_v1.csv
- results/interpretability/shap_physchem_per_compound_v1.csv.gz
- results/interpretability/shap_global_full_v1.csv
