#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Y-scrambling for the final Hybrid-XGB model in the JMGM reproducibility pipeline.

Model
-----
Hybrid-XGB = ECFP2048 (radius=2) + z-scored PhysChem descriptors

Input
-----
- data/features/model_table_v1.parquet
  or
- data/features/model_table_v1.csv.gz

Output
------
results/y_scrambling/
  - y_scrambling_results_final_hybridxgb_v2.csv
  - y_scrambling_summary_final_hybridxgb_v2.json
  - fig_y_scrambling_final_hybridxgb_v2.png
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
FEATURE_DIR = DATA_DIR / "features"

RESULTS_DIR = BASE_DIR / "results"
OUT_DIR = RESULTS_DIR / "y_scrambling"


# ============================================================
# Configuration
# ============================================================

SEED = 42
N_PERM = 200

ROLE_COL = "set_role"
TRAIN_ROLE = "NONE"
FOLD_COL = "outer_fold"
Y_COL = "logP_median"

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

# Stage-3 aligned final Hybrid-XGB parameters
XGB_MAX_DEPTH = 4
XGB_N_ESTIMATORS = 3000
XGB_LEARNING_RATE = 0.03
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
XGB_REG_LAMBDA = 10.0
XGB_MIN_CHILD_WEIGHT = 5.0
XGB_GAMMA = 0.0

FINAL_MODEL_LABEL = "Hybrid-XGB (ECFP2048 r=2 + PhysChem)"


# ============================================================
# IO helpers
# ============================================================

def resolve_model_table() -> Path:
    parquet_path = FEATURE_DIR / "model_table_v1.parquet"
    csvgz_path = FEATURE_DIR / "model_table_v1.csv.gz"

    if parquet_path.exists():
        return parquet_path
    if csvgz_path.exists():
        return csvgz_path

    raise FileNotFoundError(
        "Model table not found. Expected one of:\n"
        f"- {parquet_path}\n"
        f"- {csvgz_path}"
    )


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.name.endswith(".csv.gz") or path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path}")


# ============================================================
# Metrics
# ============================================================

def rmse(y_true, y_pred) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def ccc(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) < 2:
        return float("nan")

    mean_x = float(np.mean(y_true))
    mean_y = float(np.mean(y_pred))
    var_x = float(np.var(y_true, ddof=1))
    var_y = float(np.var(y_pred, ddof=1))
    cov_xy = float(np.cov(y_true, y_pred, ddof=1)[0, 1])

    denom = var_x + var_y + (mean_x - mean_y) ** 2
    if denom == 0:
        return float("nan")

    return float((2.0 * cov_xy) / denom)


# ============================================================
# Feature helpers
# ============================================================

def get_ecfp_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("ECFP_")]
    if len(cols) != 2048:
        raise ValueError(f"Expected 2048 ECFP columns, found {len(cols)}")
    return sorted(cols)


def fit_zscore_scaler(df: pd.DataFrame, cols: list[str]) -> dict:
    mean = {}
    std = {}

    for col in cols:
        x = df[col].astype(float).values
        mean[col] = float(np.nanmean(x))
        std[col] = float(np.nanstd(x, ddof=0))

    return {
        "mean": mean,
        "std": std,
    }


def apply_zscore(df: pd.DataFrame, scaler: dict, cols: list[str]) -> np.ndarray:
    x = df[cols].astype(float).copy()
    mean = scaler["mean"]
    std = scaler["std"]

    for col in cols:
        mu = float(mean[col])
        sd = float(std[col]) if std[col] is not None else np.nan

        if np.isnan(sd) or sd == 0.0:
            x[col] = x[col] - mu
        else:
            x[col] = (x[col] - mu) / sd

    return x.values


def build_hybrid_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ecfp_cols: list[str],
    scaler: dict,
):
    x_train_fp = train_df[ecfp_cols].astype(np.float32).values
    x_test_fp = test_df[ecfp_cols].astype(np.float32).values

    x_train_phys = apply_zscore(train_df, scaler, PHYS_COLS).astype(np.float32)
    x_test_phys = apply_zscore(test_df, scaler, PHYS_COLS).astype(np.float32)

    x_train = np.hstack([x_train_fp, x_train_phys])
    x_test = np.hstack([x_test_fp, x_test_phys])

    return x_train, x_test


# ============================================================
# Core evaluation
# ============================================================

def evaluate_hybrid_xgb_outercv(df_none: pd.DataFrame, y_vec: np.ndarray) -> dict:
    ecfp_cols = get_ecfp_cols(df_none)
    folds = sorted(df_none[FOLD_COL].astype(int).unique().tolist())

    fold_rows = []

    for fold in folds:
        train_df = df_none[df_none[FOLD_COL].astype(int) != fold].copy()
        test_df = df_none[df_none[FOLD_COL].astype(int) == fold].copy()

        if train_df.empty or test_df.empty:
            raise ValueError(f"Fold {fold}: empty train/test split detected.")

        y_train = y_vec[train_df.index.to_numpy()]
        y_test = y_vec[test_df.index.to_numpy()]

        scaler = fit_zscore_scaler(train_df, PHYS_COLS)
        x_train, x_test = build_hybrid_matrices(train_df, test_df, ecfp_cols, scaler)

        model = xgb.XGBRegressor(
            n_estimators=XGB_N_ESTIMATORS,
            learning_rate=XGB_LEARNING_RATE,
            max_depth=XGB_MAX_DEPTH,
            subsample=XGB_SUBSAMPLE,
            colsample_bytree=XGB_COLSAMPLE_BYTREE,
            reg_lambda=XGB_REG_LAMBDA,
            min_child_weight=XGB_MIN_CHILD_WEIGHT,
            gamma=XGB_GAMMA,
            random_state=SEED,
            objective="reg:squarederror",
            n_jobs=0,
            verbosity=0,
        )

        model.fit(x_train, y_train, verbose=False)
        y_pred = model.predict(x_test)

        fold_rows.append(
            {
                "fold": int(fold),
                "RMSE": rmse(y_test, y_pred),
                "MAE": mae(y_test, y_pred),
                "CCC": ccc(y_test, y_pred),
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
            }
        )

    fold_df = pd.DataFrame(fold_rows)

    return {
        "RMSE": float(fold_df["RMSE"].mean()),
        "MAE": float(fold_df["MAE"].mean()),
        "CCC": float(fold_df["CCC"].mean()),
        "fold_metrics": fold_df,
    }


# ============================================================
# Plot
# ============================================================

def plot_rmse_histogram(result_df: pd.DataFrame, out_path: Path) -> None:
    real_rmse = float(result_df.loc[result_df["is_real"], "RMSE"].iloc[0])
    scrambled_rmse = result_df.loc[~result_df["is_real"], "RMSE"].to_numpy(dtype=float)
    empirical_p = (np.sum(scrambled_rmse <= real_rmse) + 1.0) / (len(scrambled_rmse) + 1.0)

    fig = plt.figure(figsize=(6.2, 4.2))
    ax = fig.add_subplot(111)

    ax.hist(scrambled_rmse, bins=18)
    ax.axvline(real_rmse, linestyle="--")
    ax.text(
        0.98,
        0.95,
        f"Real RMSE = {real_rmse:.3f}\nempirical p ≈ {empirical_p:.3f}",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=8,
    )

    ax.set_title("Y-scrambling RMSE distribution")
    ax.set_xlabel("Mean RMSE across scaffold-aware folds")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.25)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:
    model_table_path = resolve_model_table()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = read_table(model_table_path)

    required_cols = [ROLE_COL, FOLD_COL, Y_COL, "Compound_ID"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in model_table: {col}")

    for col in PHYS_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing physicochemical descriptor in model_table: {col}")

    _ = get_ecfp_cols(df)

    df_none = df[df[ROLE_COL] == TRAIN_ROLE].copy().reset_index(drop=True)
    if df_none.empty:
        raise ValueError(f"No rows found with {ROLE_COL} == '{TRAIN_ROLE}'")

    if df_none[FOLD_COL].isna().any():
        raise ValueError("outer_fold contains missing values inside the NONE pool.")

    y_true = df_none[Y_COL].to_numpy(dtype=float)

    rng = np.random.default_rng(SEED)
    rows = []

    # Real model
    real_metrics = evaluate_hybrid_xgb_outercv(df_none, y_true)
    rows.append(
        {
            "model": FINAL_MODEL_LABEL,
            "perm_i": 0,
            "is_real": True,
            "RMSE": real_metrics["RMSE"],
            "MAE": real_metrics["MAE"],
            "CCC": real_metrics["CCC"],
        }
    )

    # Permutations
    for i in range(1, N_PERM + 1):
        y_perm = y_true.copy()
        rng.shuffle(y_perm)

        perm_metrics = evaluate_hybrid_xgb_outercv(df_none, y_perm)
        rows.append(
            {
                "model": FINAL_MODEL_LABEL,
                "perm_i": i,
                "is_real": False,
                "RMSE": perm_metrics["RMSE"],
                "MAE": perm_metrics["MAE"],
                "CCC": perm_metrics["CCC"],
            }
        )

        if i % 25 == 0 or i == N_PERM:
            print(f"Completed permutation {i}/{N_PERM}")

    result_df = pd.DataFrame(rows)

    result_csv = OUT_DIR / "y_scrambling_results_final_hybridxgb_v2.csv"
    result_df.to_csv(result_csv, index=False, encoding="utf-8-sig")

    real_rmse = float(result_df.loc[result_df["is_real"], "RMSE"].iloc[0])
    scrambled_rmse = result_df.loc[~result_df["is_real"], "RMSE"].to_numpy(dtype=float)
    empirical_p = (np.sum(scrambled_rmse <= real_rmse) + 1.0) / (len(scrambled_rmse) + 1.0)

    summary = {
        "model": FINAL_MODEL_LABEL,
        "model_table": model_table_path.relative_to(BASE_DIR).as_posix(),
        "train_pool": TRAIN_ROLE,
        "train_n": int(len(df_none)),
        "n_perm": int(N_PERM),
        "seed": int(SEED),
        "real_RMSE": float(real_rmse),
        "real_MAE": float(result_df.loc[result_df["is_real"], "MAE"].iloc[0]),
        "real_CCC": float(result_df.loc[result_df["is_real"], "CCC"].iloc[0]),
        "scrambled_mean_RMSE": float(np.mean(scrambled_rmse)),
        "scrambled_std_RMSE": float(np.std(scrambled_rmse, ddof=1)),
        "scrambled_min_RMSE": float(np.min(scrambled_rmse)),
        "scrambled_max_RMSE": float(np.max(scrambled_rmse)),
        "empirical_p": float(empirical_p),
        "xgb_params": {
            "max_depth": XGB_MAX_DEPTH,
            "n_estimators": XGB_N_ESTIMATORS,
            "learning_rate": XGB_LEARNING_RATE,
            "subsample": XGB_SUBSAMPLE,
            "colsample_bytree": XGB_COLSAMPLE_BYTREE,
            "reg_lambda": XGB_REG_LAMBDA,
            "min_child_weight": XGB_MIN_CHILD_WEIGHT,
            "gamma": XGB_GAMMA,
            "random_state": SEED,
        },
        "outputs": {
            "results_csv": result_csv.name,
            "summary_json": "y_scrambling_summary_final_hybridxgb_v2.json",
            "figure_png": "fig_y_scrambling_final_hybridxgb_v2.png",
        },
    }

    summary_json = OUT_DIR / "y_scrambling_summary_final_hybridxgb_v2.json"
    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fig_path = OUT_DIR / "fig_y_scrambling_final_hybridxgb_v2.png"
    plot_rmse_histogram(result_df, fig_path)

    print("\nY-scrambling completed successfully.")
    print(f"Training pool: {TRAIN_ROLE} (n={len(df_none)})")
    print(f"Permutations: {N_PERM}")
    print(f"Real RMSE: {real_rmse:.4f}")
    print(f"Empirical p: {empirical_p:.4f}")
    print(f"Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()