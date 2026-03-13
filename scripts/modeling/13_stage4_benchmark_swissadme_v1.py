#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark comparison between the final Hybrid-XGB model and SwissADME logP predictors.

Compared methods
----------------
- SwissADME Consensus Log P
- SwissADME XLOGP3
- SwissADME iLOGP
- Final Hybrid-XGB model predictions

Inputs
------
- data/dataset_v1_frozen_scaffoldfix_v2.csv
- data/swissadme_raw_v1.csv
- results/external_validation/<latest_run>/predictions_external_final.csv

Outputs
-------
results/benchmark_swissadme/
  - baseline_benchmark_table_v1.csv
  - fig_baseline_benchmark_rmse_v1.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import inchi


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

EXTERNAL_VAL_DIR = RESULTS_DIR / "external_validation"
OUT_DIR = RESULTS_DIR / "benchmark_swissadme"

FROZEN_DATASET_PATH = DATA_DIR / "dataset_v1_frozen_scaffoldfix_v2.csv"
SWISSADME_PATH = DATA_DIR / "swissadme_raw_v1.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Configuration
# ============================================================

ID_COL = "Compound_ID"
SMILES_COL = "Canonical_SMILES"
Y_COL = "logP_median"

ROLE_COL_CANDIDATES = ["external_role", "set_role"]
TARGET_ROLES = ["EXT_A", "EXT_B"]

SWISS_SMILES_COL = "Canonical SMILES"
SWISS_CONSENSUS_COL = "Consensus Log P"
SWISS_XLOGP3_COL = "XLOGP3"
SWISS_ILOGP_COL = "iLOGP"

PRED_COL = "y_pred"
MODEL_COL = "model"

FINAL_MODEL_LABEL = "XGB_ECFP+PhysChem"
REPORT_MODEL_LABEL = "Hybrid-XGB (ECFP2048 r=2 + PhysChem)"


# ============================================================
# Metrics
# ============================================================

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def ccc(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mean_true = y_true.mean()
    mean_pred = y_pred.mean()

    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    var_true = np.mean((y_true - mean_true) ** 2)
    var_pred = np.mean((y_pred - mean_pred) ** 2)

    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    return float((2 * cov) / denom) if denom != 0 else np.nan


# ============================================================
# Utilities
# ============================================================

def smiles_to_inchikey(smiles: str):
    if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
        return None

    smiles = str(smiles).strip()
    if not smiles or smiles.lower() == "nan":
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        return inchi.MolToInchiKey(mol)
    except Exception:
        return None


def find_role_col(df: pd.DataFrame) -> str:
    for col in ROLE_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        f"Frozen dataset must contain one of role columns: {ROLE_COL_CANDIDATES}"
    )


def find_latest_predictions() -> Path:
    if not EXTERNAL_VAL_DIR.exists():
        raise FileNotFoundError(f"Missing directory: {EXTERNAL_VAL_DIR}")

    candidate_files = sorted(EXTERNAL_VAL_DIR.glob("*/predictions_external_final.csv"))
    if not candidate_files:
        raise FileNotFoundError(
            "No predictions_external_final.csv file found under "
            f"{EXTERNAL_VAL_DIR}"
        )

    return candidate_files[-1]


# ============================================================
# Main
# ============================================================

def main() -> None:
    if not FROZEN_DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing frozen dataset: {FROZEN_DATASET_PATH}")

    if not SWISSADME_PATH.exists():
        raise FileNotFoundError(f"Missing SwissADME file: {SWISSADME_PATH}")

    preds_path = find_latest_predictions()

    frozen = pd.read_csv(FROZEN_DATASET_PATH)
    swiss = pd.read_csv(SWISSADME_PATH)
    preds = pd.read_csv(preds_path)

    role_col = find_role_col(frozen)

    # Validate required columns
    for col in [ID_COL, SMILES_COL, Y_COL, role_col]:
        if col not in frozen.columns:
            raise ValueError(
                f"Frozen dataset is missing required column: {col}"
            )

    for col in [SWISS_SMILES_COL, SWISS_CONSENSUS_COL, SWISS_XLOGP3_COL, SWISS_ILOGP_COL]:
        if col not in swiss.columns:
            raise ValueError(
                f"SwissADME file is missing required column: {col}"
            )

    for col in [ID_COL, PRED_COL, MODEL_COL]:
        if col not in preds.columns:
            raise ValueError(
                f"Predictions file is missing required column: {col}"
            )

    # Restrict predictions to final model
    preds = preds[preds[MODEL_COL].astype(str) == FINAL_MODEL_LABEL].copy()
    if preds.empty:
        raise ValueError(
            f"No rows found for final model '{FINAL_MODEL_LABEL}' in {preds_path}"
        )

    # Build InChIKey mapping
    frozen_subset = frozen[[ID_COL, SMILES_COL, Y_COL, role_col]].copy()
    frozen_subset["_inchikey"] = frozen_subset[SMILES_COL].apply(smiles_to_inchikey)

    swiss_subset = swiss[
        [SWISS_SMILES_COL, SWISS_CONSENSUS_COL, SWISS_XLOGP3_COL, SWISS_ILOGP_COL]
    ].copy()
    swiss_subset["_inchikey"] = swiss_subset[SWISS_SMILES_COL].apply(smiles_to_inchikey)

    frozen_bad = int(frozen_subset["_inchikey"].isna().sum())
    swiss_bad = int(swiss_subset["_inchikey"].isna().sum())

    if frozen_bad > 0:
        print(f"Warning: frozen SMILES -> InChIKey failed for {frozen_bad} rows.")
    if swiss_bad > 0:
        print(f"Warning: SwissADME SMILES -> InChIKey failed for {swiss_bad} rows.")

    # Merge frozen dataset with SwissADME predictions
    merged = frozen_subset.merge(
        swiss_subset.drop(columns=[SWISS_SMILES_COL]),
        on="_inchikey",
        how="left",
    )

    missing_swiss = merged[merged[SWISS_CONSENSUS_COL].isna()][ID_COL].tolist()
    if missing_swiss:
        raise ValueError(
            "SwissADME merge failed for some compounds after InChIKey matching. "
            f"Missing entries: {missing_swiss[:20]}"
        )

    # Restrict to EXT_A + EXT_B only
    subset = merged[merged[role_col].astype(str).isin(TARGET_ROLES)].copy()
    if subset.empty:
        raise ValueError(
            f"No rows found for roles {TARGET_ROLES}. "
            f"Available roles: {sorted(merged[role_col].astype(str).unique())}"
        )

    # Merge final model predictions
    subset = subset.merge(preds[[ID_COL, PRED_COL]], on=ID_COL, how="left")

    missing_preds = subset[subset[PRED_COL].isna()][ID_COL].tolist()
    if missing_preds:
        raise ValueError(
            "Final model predictions are missing for some benchmark compounds. "
            f"Missing entries: {missing_preds[:20]}"
        )

    y_true = subset[Y_COL].astype(float).values

    methods = [
        ("SwissADME_Consensus", subset[SWISS_CONSENSUS_COL].astype(float).values),
        ("SwissADME_XLOGP3", subset[SWISS_XLOGP3_COL].astype(float).values),
        ("SwissADME_iLOGP", subset[SWISS_ILOGP_COL].astype(float).values),
        (REPORT_MODEL_LABEL, subset[PRED_COL].astype(float).values),
    ]

    rows = []
    for method_name, y_pred in methods:
        rows.append(
            {
                "set": "+".join(TARGET_ROLES),
                "method": method_name,
                "n": int(len(subset)),
                "RMSE": rmse(y_true, y_pred),
                "MAE": mae(y_true, y_pred),
                "CCC": ccc(y_true, y_pred),
            }
        )

    out_table = (
        pd.DataFrame(rows)
        .sort_values("RMSE", ascending=True)
        .reset_index(drop=True)
    )

    out_csv = OUT_DIR / "baseline_benchmark_table_v1.csv"
    out_table.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # RMSE bar plot
    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.add_subplot(111)

    ax.bar(out_table["method"], out_table["RMSE"])
    ax.set_ylabel("RMSE")
    ax.set_title(f"Benchmark comparison ({'+'.join(TARGET_ROLES)})")
    ax.tick_params(axis="x", rotation=25)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")

    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    out_fig = OUT_DIR / "fig_baseline_benchmark_rmse_v1.png"
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Benchmark comparison completed.")
    print(f"Roles: {TARGET_ROLES} (n={len(subset)})")
    print(f"Predictions file: {preds_path}")
    print(f"Output table: {out_csv}")
    print(f"Output figure: {out_fig}")
    print()
    print(out_table.to_string(index=False))


if __name__ == "__main__":
    main()