#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate z-score scalers for physicochemical descriptors
for each outer CV fold.

Scalers are fitted on the training portion of the NONE pool
for each fold.

Input
-----
data/features/model_table_v1.csv.gz

Output
------
data/metadata/scalers_v1/zscore_scaler_outer_fold_1.json
...
data/metadata/scalers_v1/zscore_scaler_outer_fold_5.json

data/metadata/scalers_v1/scalers_summary.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
FEATURE_DIR = DATA_DIR / "features"
META_DIR = DATA_DIR / "metadata"

MODEL_TABLE_PATH = FEATURE_DIR / "model_table_v1.csv.gz"

SCALER_DIR = META_DIR / "scalers_v1"
SCALER_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Configuration
# ============================================================

PHYS_COLS = [
    "MW",
    "TPSA",
    "HBD",
    "HBA",
    "RotB",
    "AromaticRings",
    "FractionCSP3",
    "MR",
    "RingCount",
    "HeavyAtomCount",
]

N_FOLDS = 5


# ============================================================
# Main
# ============================================================

def main():

    if not MODEL_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"Missing feature table: {MODEL_TABLE_PATH}\n"
            "Run 04_build_features_v1.py first."
        )

    df = pd.read_csv(MODEL_TABLE_PATH)

    required_cols = ["Compound_ID", "set_role", "outer_fold"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"model_table missing required column: {col}")

    for col in PHYS_COLS:
        if col not in df.columns:
            raise ValueError(f"model_table missing physicochemical descriptor: {col}")

    none_pool = df[df["set_role"] == "NONE"].copy()

    if none_pool.empty:
        raise ValueError("NONE pool is empty. Check split manifest.")

    summary = {}

    for fold in range(1, N_FOLDS + 1):

        train_df = none_pool[none_pool["outer_fold"] != fold].copy()
        test_df = none_pool[none_pool["outer_fold"] == fold].copy()

        if train_df.empty or test_df.empty:
            raise ValueError(
                f"Fold {fold}: train/test split empty. Check outer_fold assignments."
            )

        X = train_df[PHYS_COLS].astype(float)

        mean = X.mean(axis=0).to_dict()
        std = X.std(axis=0, ddof=0).replace(0, np.nan).to_dict()

        scaler = {
            "fold": fold,
            "fit_on": "set_role == NONE and outer_fold != fold",
            "physchem_columns": PHYS_COLS,
            "mean": mean,
            "std": std,
        }

        scaler_path = SCALER_DIR / f"zscore_scaler_outer_fold_{fold}.json"

        scaler_path.write_text(
            json.dumps(scaler, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        summary[fold] = {
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
        }

    summary_path = SCALER_DIR / "scalers_summary.json"

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nFold scalers generated successfully.")
    print("Output directory:", SCALER_DIR)

    for fold, stats in summary.items():
        print(f"Fold {fold}: train={stats['n_train']} test={stats['n_test']}")


if __name__ == "__main__":
    main()