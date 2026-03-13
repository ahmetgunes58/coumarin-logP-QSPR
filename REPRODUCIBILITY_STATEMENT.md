# Reproducibility Statement

This repository provides the complete computational workflow required to reproduce the results reported in the manuscript:

“Lipophilicity modelling of nitrogen-containing coumarin derivatives: a scaffold-aware interpretable QSPR study.”

All datasets, scripts, and modelling outputs used in the study are included in this repository to enable full reproducibility of the reported results.

---

## Dataset

The modelling dataset consists of **95 nitrogen-containing coumarin derivatives** curated from experimentally measured lipophilicity values reported in the primary literature.

The dataset contains **49 unique Bemis–Murcko scaffolds** and was partitioned as follows:

Training set (STRICT + GOLD tiers): 69 compounds  
External validation set A: 9 compounds  
External validation set B: 10 compounds  
Out-of-distribution (OOD) subset: 7 compounds  

The frozen dataset used for modelling is provided at:

data/dataset_v1_frozen_scaffoldfix_v2.csv

Additional dataset files:

data/exclusions_v1.csv  
data/scaffold_groups.csv  
data/splits_manifest.csv  
data/swissadme_raw_v1.csv  

A checksum file is included to ensure dataset integrity:

data/metadata/checksums_stage2.sha256

---

## Feature Generation

Molecular representations combine physicochemical descriptors and circular fingerprints.

Physicochemical descriptors (10):

MW  
TPSA  
HBD  
HBA  
RotB  
AromaticRingCount  
FractionCSP3  
MR  
RingCount  
HeavyAtomCount  

Circular fingerprints:

Morgan fingerprints (ECFP)

radius = 2  
length = 2048 bits  

Total modelling features per molecule: **2058**

Feature matrices:

data/features/features_physchem_v1.csv.gz  
data/features/features_ecfp2048_r2_v1.csv.gz  

---

## Modelling Workflow

The repository contains all scripts required to reproduce the modelling pipeline, including dataset preparation, scaffold assignment, feature generation, model training, validation, interpretability analysis, and figure generation.

The workflow can be reproduced by running the scripts in the following order.

### Data processing

scripts/data_processing/01_replicate_collapse_and_freeze.py  
scripts/data_processing/02_make_scaffolds.py  
scripts/data_processing/03_make_splits_manifest.py  
scripts/data_processing/04_build_features_v1.py  
scripts/data_processing/05_make_fold_scalers_v1.py  

### Model training

scripts/modeling/06_train_stage3_panel_v1.py  

### Model validation

scripts/modeling/09_stage4_y_scrambling_v1.py  
scripts/modeling/13_stage4_benchmark_swissadme_v1.py  

### Model freezing

scripts/modeling/16_stage5_freeze_final_model_v1.py  

### Interpretability analysis

scripts/modeling/21_stage5_3_shap_physchem_direction_v1.py  

### Feature ablation analysis

scripts/modeling/22_stage6_ablation_v1.py  

### Figure generation

scripts/modeling/23_stage7_generate_figures_v2_FINAL.py  

### Additional analysis utilities

scripts/analysis/external_metrics.py  
scripts/modeling/24_plot_residual_external.py  
scripts/modeling/25_extract_top_scaffolds.py  

Supplementary Figure S8 (TPSA–SHAP scatter plot used for interpretability analysis) is generated using:

`scripts/modeling/26_plot_tpsa_shap_scatter.py`

Output files:

- `figures/supplementary/Figure_S8_tpsa_shap_scatter.png`
- `figures/supplementary/Figure_S8_tpsa_shap_scatter.tiff`

Running the full pipeline starting from the frozen dataset reproduces the trained models, evaluation metrics, tables, and figures reported in the manuscript.

---

## Model and Validation

The final predictive model is a **hybrid XGBoost regression model** combining physicochemical descriptors and ECFP fingerprints.

Final model artifact:

results/stage5_freeze_v1/final_xgb_model_v1.pkl

Typical predictive performance on the independent external validation set (n = 19):

RMSE ≈ 0.433 log units  
MAE ≈ 0.365  
CCC ≈ 0.496  

Model robustness was evaluated using multiple validation procedures:

• scaffold-aware nested cross-validation  
• independent external validation  
• applicability-domain analysis  
• permutation-based Y-scrambling (200 permutations)

Validation results are available in:

results/external_validation  
results/y_scrambling  
results/benchmark_swissadme  
results/ablation  

---

## Figures

All manuscript and supplementary figures can be regenerated using:

scripts/modeling/23_stage7_generate_figures_v2_FINAL.py

Generated figures are stored in:

figures/main  
figures/supplementary  

---

## Software Environment

The modelling workflow was implemented in Python using:

Python  
RDKit  
scikit-learn  
XGBoost  
NumPy  
pandas  

RDKit version used in the study: **2025.03.6**

---

## Deterministic Reproducibility

The workflow ensures deterministic reproducibility through:

• a frozen dataset snapshot  
• fixed random seeds  
• scaffold-aware dataset splitting  
• fold-specific feature scaling  
• archived model artifacts  

Running the provided scripts reproduces the modelling results reported in the manuscript.