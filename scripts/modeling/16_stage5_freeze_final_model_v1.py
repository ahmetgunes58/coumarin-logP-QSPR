#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stage-5 FINAL MODEL FREEZE
Reproducible JMGM pipeline
"""

import pandas as pd
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
FEATURE_DIR = DATA_DIR / "features"

RESULT_DIR = BASE_DIR / "results"
OUT_DIR = RESULT_DIR / "stage5_freeze_v1"

OUT_DIR.mkdir(parents=True, exist_ok=True)


DATASET_PATH = DATA_DIR / "dataset_v1_frozen_scaffoldfix_v2.csv"
MODEL_TABLE_PATH = FEATURE_DIR / "model_table_v1.csv.gz"


# --------------------------------------------------
# MODEL CONFIG  (MUST MATCH STAGE-3)
# --------------------------------------------------

SEED = 42

XGB_PARAMS = dict(
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=10.0,
    gamma=0.0,
    objective="reg:squarederror",
    random_state=SEED,
    n_jobs=-1,
    verbosity=0
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

dataset = pd.read_csv(DATASET_PATH)
model_table = pd.read_csv(MODEL_TABLE_PATH)

train_ids = dataset.loc[
    dataset["external_role"] == "NONE",
    "Compound_ID"
].astype(str)

model_table["Compound_ID"] = model_table["Compound_ID"].astype(str)

train_df = model_table[
    model_table["Compound_ID"].isin(train_ids)
].copy()

y = train_df["logP_median"].astype(float)


# --------------------------------------------------
# FEATURE SELECTION
# --------------------------------------------------

ecfp_cols = [c for c in train_df.columns if c.startswith("ECFP_")]

phys_cols = [
    c for c in [
        "MW","TPSA","HBD","HBA","RotB",
        "AromaticRings","FractionCSP3",
        "MR","RingCount","HeavyAtomCount"
    ] if c in train_df.columns
]

feature_cols = ecfp_cols + phys_cols

X = train_df[feature_cols].copy()

scaler = StandardScaler()
X[phys_cols] = scaler.fit_transform(X[phys_cols])


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

model = XGBRegressor(**XGB_PARAMS)

model.fit(X.values, y.values)


# --------------------------------------------------
# SAVE MODEL BUNDLE
# --------------------------------------------------

bundle = {
    "model": model,
    "scaler": scaler,
    "feature_cols": feature_cols,
    "ecfp_cols": ecfp_cols,
    "phys_cols": phys_cols,
    "params": XGB_PARAMS,
    "seed": SEED
}

joblib.dump(bundle, OUT_DIR / "final_xgb_model_v1.pkl")


# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

booster = model.get_booster()

score = booster.get_score(importance_type="gain")

rows = []

for k, v in score.items():

    idx = int(k[1:])

    feature = feature_cols[idx]

    rows.append((feature, v))


importance_df = (
    pd.DataFrame(rows, columns=["feature","gain"])
    .sort_values("gain", ascending=False)
)

importance_df.to_csv(
    OUT_DIR / "feature_importance_gain_v1.csv",
    index=False
)


# --------------------------------------------------
# TRAINING PREDICTIONS
# --------------------------------------------------

pred = model.predict(X.values)

pred_df = pd.DataFrame({
    "Compound_ID": train_df["Compound_ID"],
    "y_true": y,
    "y_pred": pred,
    "residual": y - pred
})

pred_df.to_csv(
    OUT_DIR / "training_predictions_v1.csv",
    index=False
)


# --------------------------------------------------
# METADATA
# --------------------------------------------------

metadata = {
    "seed": SEED,
    "n_train": len(train_df),
    "n_features": len(feature_cols),
    "n_ecfp": len(ecfp_cols),
    "n_physchem": len(phys_cols),
    "dataset": str(DATASET_PATH),
    "model_table": str(MODEL_TABLE_PATH)
}

with open(OUT_DIR / "feature_metadata_v1.json", "w") as f:

    json.dump(metadata, f, indent=2)


print("Stage-5 FINAL MODEL FREEZE completed.")
print("Output:", OUT_DIR)