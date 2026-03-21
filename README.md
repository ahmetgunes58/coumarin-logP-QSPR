> **Reproducibility note:** The exact repository version corresponding to the manuscript submission is archived as the GitHub release **submission-v1**.

# Coumarin Lipophilicity QSPR — Reproducible Modelling Workflow

This repository contains the complete computational workflow accompanying the manuscript:

**“Lipophilicity modelling of nitrogen-containing coumarin derivatives: a scaffold-aware interpretable QSPR study.”**

The project implements a reproducible quantitative structure–property relationship (QSPR) modelling framework for predicting the octanol–water partition coefficient (logP) of nitrogen-containing coumarin derivatives using scaffold-aware machine learning and interpretable modelling.

All datasets, scripts, and modelling outputs required to reproduce the results reported in the manuscript are provided.

---

# Study Overview

Lipophilicity (logP) is a key physicochemical property influencing membrane permeability, solubility, and pharmacokinetic behaviour of drug-like molecules. Accurate prediction of lipophilicity remains challenging for heteroaromatic scaffolds such as coumarins, where heteroatom substitution and extended π-conjugation introduce non-additive structural effects.

This work develops a scaffold-aware QSPR modelling framework based on:

- curated experimental lipophilicity data
- hybrid molecular descriptors
- scaffold-aware validation
- interpretable machine learning

The repository provides the full modelling pipeline required to reproduce the results reported in the study.

---

# Repository Structure



data/
dataset_v1_frozen_scaffoldfix_v2.csv
exclusions_v1.csv
scaffold_groups.csv
splits_manifest.csv
swissadme_raw_v1.csv

features/
features_ecfp2048_r2_v1.csv.gz
features_physchem_v1.csv.gz
model_table_v1.csv.gz

metadata/
checksums_stage2.sha256
feature_metadata.json

scripts/

data_processing/
01_replicate_collapse_and_freeze.py
02_make_scaffolds.py
03_make_splits_manifest.py
04_build_features_v1.py
05_make_fold_scalers_v1.py

modeling/
06_train_stage3_panel_v1.py
09_stage4_y_scrambling_v1.py
13_stage4_benchmark_swissadme_v1.py
16_stage5_freeze_final_model_v1.py
21_stage5_3_shap_physchem_direction_v1.py
22_stage6_ablation_v1.py
23_stage7_generate_figures_v2_FINAL.py
24_plot_residual_external.py
25_extract_top_scaffolds.py

analysis/
external_metrics.py

results/
external_validation/
benchmark_swissadme/
interpretability/
ablation/
y_scrambling/
scaffold_analysis/
stage5_freeze_v1/

figures/
main/
supplementary/

environment/

README.md
REPRODUCIBILITY_STATEMENT.md


---

# Dataset

The curated dataset consists of:

- **95 nitrogen-containing coumarin derivatives**
- **49 Bemis–Murcko scaffolds**
- experimentally measured logP values from primary literature sources

Dataset partitions:

| Dataset | Compounds |
|--------|-----------|
| Training (STRICT + GOLD) | 69 |
| External validation A | 9 |
| External validation B | 10 |
| Out-of-distribution (OOD) | 7 |

Frozen dataset file:


data/dataset_v1_frozen_scaffoldfix_v2.csv


Dataset integrity verification:


data/metadata/checksums_stage2.sha256


---

# Feature Representation

Each molecule is encoded using a hybrid descriptor representation.

### Physicochemical descriptors (10)

- Molecular weight (MW)
- Topological polar surface area (TPSA)
- Hydrogen bond donors (HBD)
- Hydrogen bond acceptors (HBA)
- Rotatable bonds (RotB)
- Aromatic ring count
- Fraction of sp³ carbon atoms (FractionCSP3)
- Molar refractivity (MR)
- Ring count
- Heavy atom count

### Circular fingerprints

Morgan fingerprints (ECFP)


radius = 2
length = 2048 bits


Total modelling features per compound: **2058**

Feature matrices:


data/features/features_physchem_v1.csv.gz
data/features/features_ecfp2048_r2_v1.csv.gz


---

# Modelling Workflow

The modelling pipeline follows a staged workflow.

### Dataset preparation


scripts/data_processing/01_replicate_collapse_and_freeze.py


### Scaffold assignment


scripts/data_processing/02_make_scaffolds.py


### Dataset splitting


scripts/data_processing/03_make_splits_manifest.py


### Feature generation


scripts/data_processing/04_build_features_v1.py


### Fold-specific scaling


scripts/data_processing/05_make_fold_scalers_v1.py


### Model training


scripts/modeling/06_train_stage3_panel_v1.py


Model panel:

- Ridge regression
- Kernel Ridge Regression (Tanimoto kernel)
- XGBoost

---

# Validation

Multiple validation procedures were applied to ensure robust model evaluation.

### Scaffold-aware cross-validation

Murcko scaffold grouping prevents analogue leakage between folds.

### External validation

Independent external set size:


n = 19


Performance:


RMSE ≈ 0.433
MAE ≈ 0.365
CCC ≈ 0.496


Results stored in:


results/external_validation/


---

### Y-scrambling validation


scripts/modeling/09_stage4_y_scrambling_v1.py


Permutations:


200


Results stored in:


results/y_scrambling/


---

### Benchmark comparison

Fragment-based predictors from SwissADME were evaluated using:


scripts/modeling/13_stage4_benchmark_swissadme_v1.py


Results stored in:


results/benchmark_swissadme/


---

### Feature ablation analysis


scripts/modeling/22_stage6_ablation_v1.py


Results stored in:


results/ablation/


---

# Interpretability Analysis

Model interpretability was investigated using SHAP.


scripts/modeling/21_stage5_3_shap_physchem_direction_v1.py


Key descriptors influencing lipophilicity:

- TPSA
- Aromatic ring count
- FractionCSP3

Results stored in:


results/interpretability/


---

# Final Model

Final trained model:


results/stage5_freeze_v1/final_xgb_model_v1.pkl


---

# Reproducing the Figures

All manuscript and supplementary figures can be regenerated using:


scripts/modeling/23_stage7_generate_figures_v2_FINAL.py


Generated figures are stored in:


figures/main/
figures/supplementary/


---

# Software Environment

The modelling workflow was implemented in Python using:

- RDKit
- scikit-learn
- XGBoost
- NumPy
- pandas

RDKit version used in the study:


2025.03.6


---

# Reproducibility

A detailed reproducibility description is provided in:


REPRODUCIBILITY_STATEMENT.md


Running the modelling pipeline starting from the frozen dataset reproduces the trained models, evaluation metrics, and figures reported in the manuscript.

---

# Citation

If you use this dataset or modelling workflow, please cite:

**Ahmet Güneş**  
Lipophilicity modelling of nitrogen-containing coumarin derivatives: a scaffold-aware interpretable QSPR study.

---

# Contact

For questions regarding the dataset or modelling workflow, please contact the corresponding auth



