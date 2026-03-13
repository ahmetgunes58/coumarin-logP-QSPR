#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model panel for the JMGM reproducibility pipeline.

Outer CV on NONE pool only:
- Ridge on z-scored PhysChem descriptors
- Kernel Ridge Regression with Tanimoto kernel on ECFP2048
- XGBoost on ECFP2048 + z-scored PhysChem descriptors

Final external evaluation:
- Train once on full NONE pool
- Predict EXT_A / EXT_B / OOD once
- Save final external predictions and metrics

Outputs
-------
results/external_validation/<timestamp>/
  - metrics_outercv.csv
  - predictions_outercv.csv
  - predictions_external_byfold.csv
  - predictions_external_final.csv
  - metrics_external_final.csv
  - configs.json
"""

import json
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

import xgboost as xgb


# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
FEATURE_DIR = DATA_DIR / "features"

RESULTS_DIR = BASE_DIR / "results"
OUT_ROOT = RESULTS_DIR / "external_validation"


# ---------------- CONFIG ----------------
N_OUTER_FOLDS = 5
SEED = 42

ALPHAS_KRR = [0.01, 0.1, 1, 10, 100]
XGB_DEPTHS = [3, 4]

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


# ---------------- IO HELPERS ----------------
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


# ---------------- METRICS ----------------
def rmse(y_true, y_pred) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def safe_r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return float("nan")
    denom = ((y_true - y_true.mean()) ** 2).sum()
    if denom == 0:
        return float("nan")
    return float(1.0 - (((y_true - y_pred) ** 2).sum() / denom))


def ccc(y_true, y_pred) -> float:
    """
    Lin's Concordance Correlation Coefficient.
    """
    x = np.asarray(y_true, dtype=float)
    y = np.asarray(y_pred, dtype=float)

    if len(x) < 2:
        return float("nan")

    mx = float(np.mean(x))
    my = float(np.mean(y))
    vx = float(np.var(x, ddof=0))
    vy = float(np.var(y, ddof=0))

    if (vx + vy) == 0 and (mx - my) == 0:
        return 1.0

    cov = float(np.mean((x - mx) * (y - my)))
    denom = vx + vy + (mx - my) ** 2
    if denom == 0:
        return float("nan")

    return float((2.0 * cov) / denom)


def metrics_dict(y_true, y_pred) -> dict:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": safe_r2(y_true, y_pred),
        "CCC": ccc(y_true, y_pred),
        "n": int(len(y_true)),
    }


def external_metrics_table(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, set_role), g in pred_df.groupby(["model", "set_role"], sort=False):
        y = g["y_true"].to_numpy(dtype=float)
        yhat = g["y_pred"].to_numpy(dtype=float)
        m = metrics_dict(y, yhat)
        rows.append(
            [model, set_role, m["n"], m["RMSE"], m["MAE"], m["R2"], m["CCC"]]
        )

    out = pd.DataFrame(
        rows,
        columns=["model", "set_role", "n", "RMSE", "MAE", "R2", "CCC"],
    )
    return out.sort_values(["set_role", "RMSE"], ascending=[True, True]).reset_index(drop=True)


# ---------------- SCALING ----------------
def fit_zscore_scaler(df: pd.DataFrame, cols) -> dict:
    mean = {}
    std = {}

    for c in cols:
        x = df[c].astype(float).values
        mu = float(np.nanmean(x))
        sd = float(np.nanstd(x, ddof=0))
        mean[c] = mu
        std[c] = sd

    return {
        "mean": mean,
        "std": std,
    }


def apply_zscore(df: pd.DataFrame, scaler: dict, cols) -> np.ndarray:
    mean = scaler["mean"]
    std = scaler["std"]
    X = df[cols].astype(float).copy()

    for c in cols:
        mu = float(mean[c])
        sd = float(std[c]) if std[c] is not None else np.nan
        if sd == 0 or np.isnan(sd):
            X[c] = X[c] - mu
        else:
            X[c] = (X[c] - mu) / sd

    return X.values


# ---------------- FEATURES ----------------
def get_ecfp_cols(df: pd.DataFrame):
    ecfp = [c for c in df.columns if c.startswith("ECFP_")]
    if len(ecfp) != 2048:
        raise ValueError(f"Expected 2048 ECFP columns, found {len(ecfp)}")
    return sorted(ecfp)


def tanimoto_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A.astype(np.uint8)
    B = B.astype(np.uint8)

    inter = A @ B.T
    a_sum = A.sum(axis=1).reshape(-1, 1)
    b_sum = B.sum(axis=1).reshape(1, -1)
    denom = a_sum + b_sum - inter

    K = np.where(denom > 0, inter / denom, 0.0)
    return K.astype(np.float64)


# ---------------- INNER CV MODEL SELECTION ----------------
def inner_cv_select_krr_alpha(train_df: pd.DataFrame, y: np.ndarray, ecfp_cols, groups: np.ndarray, strata: np.ndarray) -> float:
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    X = train_df[ecfp_cols].astype(np.uint8).values

    best_alpha = None
    best_score = float("inf")

    for alpha in ALPHAS_KRR:
        fold_rmses = []

        for tr_idx, va_idx in sgkf.split(train_df, y=strata, groups=groups):
            Xtr, Xva = X[tr_idx], X[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]

            Ktr = tanimoto_kernel(Xtr, Xtr)
            Kva = tanimoto_kernel(Xva, Xtr)

            model = KernelRidge(alpha=alpha, kernel="precomputed")
            model.fit(Ktr, ytr)
            pred = model.predict(Kva)
            fold_rmses.append(rmse(yva, pred))

        score = float(np.mean(fold_rmses))
        if score < best_score:
            best_score = score
            best_alpha = alpha

    return float(best_alpha)


def inner_cv_select_xgb_depth(train_X: np.ndarray, y: np.ndarray, groups: np.ndarray, strata: np.ndarray) -> int:
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)

    best_depth = None
    best_score = float("inf")

    for depth in XGB_DEPTHS:
        fold_rmses = []

        for tr_idx, va_idx in sgkf.split(train_X, y=strata, groups=groups):
            Xtr, Xva = train_X[tr_idx], train_X[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]

            model = xgb.XGBRegressor(
                n_estimators=2000,
                learning_rate=0.03,
                max_depth=depth,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=10.0,
                min_child_weight=5.0,
                gamma=0.0,
                random_state=SEED,
                objective="reg:squarederror",
                n_jobs=0,
            )
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            pred = model.predict(Xva)
            fold_rmses.append(rmse(yva, pred))

        score = float(np.mean(fold_rmses))
        if score < best_score:
            best_score = score
            best_depth = depth

    return int(best_depth)


# ---------------- MAIN ----------------
def main():
    model_table_path = resolve_model_table()
    df = read_table(model_table_path)

    required_cols = [
        "Compound_ID",
        "logP_median",
        "set_role",
        "outer_fold",
        "Murcko_scaffold_ID",
        "Tier",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"model_table missing required column: {c}")

    for c in PHYS_COLS:
        if c not in df.columns:
            raise ValueError(f"model_table missing physchem column: {c}")

    ecfp_cols = get_ecfp_cols(df)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = OUT_ROOT / ts
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "timestamp": ts,
        "inputs": {
            "model_table": model_table_path.relative_to(BASE_DIR).as_posix(),
        },
        "panel": {
            "ridge": {
                "features": "PhysChem(zscore)",
                "alpha": 1.0,
            },
            "krr": {
                "kernel": "Tanimoto",
                "features": "ECFP2048",
                "alphas": ALPHAS_KRR,
            },
            "xgb": {
                "features": "ECFP2048 + PhysChem(zscore)",
                "depths": XGB_DEPTHS,
                "n_estimators_inner_cv": 2000,
                "n_estimators_final": 3000,
                "learning_rate": 0.03,
                "reg_lambda": 10.0,
                "min_child_weight": 5.0,
                "gamma": 0.0,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": SEED,
            },
            "inner_cv": {
                "n_splits": 3,
                "method": "StratifiedGroupKFold",
                "strata": "Tier",
                "groups": "Murcko_scaffold_ID",
            },
            "outer_cv": {
                "n_splits": N_OUTER_FOLDS,
                "pool": "set_role == NONE",
            },
            "external_policy": "Train once on full NONE pool; predict EXT_A, EXT_B, OOD once",
            "metrics": ["RMSE", "MAE", "R2", "CCC"],
        },
        "outputs": {
            "metrics_outercv": "metrics_outercv.csv",
            "predictions_outercv": "predictions_outercv.csv",
            "predictions_external_byfold": "predictions_external_byfold.csv",
            "predictions_external_final": "predictions_external_final.csv",
            "metrics_external_final": "metrics_external_final.csv",
        },
    }
    (outdir / "configs.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    none_pool = df[df["set_role"] == "NONE"].copy()
    ext_a = df[df["set_role"] == "EXT_A"].copy()
    ext_b = df[df["set_role"] == "EXT_B"].copy()
    ood = df[df["set_role"] == "OOD"].copy()

    if len(none_pool) == 0:
        raise ValueError("No NONE pool rows found in model_table.")
    if none_pool["outer_fold"].isna().any():
        raise ValueError("NONE pool contains missing outer_fold values.")

    outer_metrics = []
    outer_preds = []
    ext_byfold = []

    # ---------------- OUTER CV ----------------
    for fold in range(1, N_OUTER_FOLDS + 1):
        train = none_pool[none_pool["outer_fold"] != fold].copy()
        test = none_pool[none_pool["outer_fold"] == fold].copy()

        if len(train) == 0 or len(test) == 0:
            raise ValueError(f"Fold {fold} is empty. Check outer_fold assignments.")

        y_train = train["logP_median"].astype(float).values
        y_test = test["logP_median"].astype(float).values

        groups_train = train["Murcko_scaffold_ID"].astype(str).values
        strata_train = train["Tier"].astype(str).values

        scaler = fit_zscore_scaler(train, PHYS_COLS)

        # Ridge
        Xtr_phys = apply_zscore(train, scaler, PHYS_COLS)
        Xte_phys = apply_zscore(test, scaler, PHYS_COLS)

        ridge = Ridge(alpha=1.0, random_state=SEED)
        ridge.fit(Xtr_phys, y_train)
        pred_test = ridge.predict(Xte_phys)

        outer_metrics.append(
            {"model": "Ridge_PhysChem", "outer_fold": fold, **metrics_dict(y_test, pred_test)}
        )
        outer_preds.append(
            pd.DataFrame({
                "Compound_ID": test["Compound_ID"].values,
                "set_role": "NONE",
                "outer_fold": fold,
                "y_true": y_test,
                "y_pred": pred_test,
                "model": "Ridge_PhysChem",
            })
        )

        # KRR
        best_alpha = inner_cv_select_krr_alpha(
            train_df=train,
            y=y_train,
            ecfp_cols=ecfp_cols,
            groups=groups_train,
            strata=strata_train,
        )

        Xtr_fp = train[ecfp_cols].astype(np.uint8).values
        Xte_fp = test[ecfp_cols].astype(np.uint8).values

        Ktr = tanimoto_kernel(Xtr_fp, Xtr_fp)
        Kte = tanimoto_kernel(Xte_fp, Xtr_fp)

        krr_model = KernelRidge(alpha=best_alpha, kernel="precomputed")
        krr_model.fit(Ktr, y_train)
        pred_test = krr_model.predict(Kte)

        outer_metrics.append(
            {
                "model": "KRR_Tanimoto_ECFP",
                "outer_fold": fold,
                "alpha": best_alpha,
                **metrics_dict(y_test, pred_test),
            }
        )
        outer_preds.append(
            pd.DataFrame({
                "Compound_ID": test["Compound_ID"].values,
                "set_role": "NONE",
                "outer_fold": fold,
                "y_true": y_test,
                "y_pred": pred_test,
                "model": "KRR_Tanimoto_ECFP",
                "alpha": best_alpha,
            })
        )

        # XGB
        Xtr_comb = np.hstack([Xtr_fp.astype(np.float32), Xtr_phys.astype(np.float32)])
        Xte_comb = np.hstack([Xte_fp.astype(np.float32), Xte_phys.astype(np.float32)])

        best_depth = inner_cv_select_xgb_depth(
            train_X=Xtr_comb,
            y=y_train,
            groups=groups_train,
            strata=strata_train,
        )

        xgb_model = xgb.XGBRegressor(
            n_estimators=3000,
            learning_rate=0.03,
            max_depth=best_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=10.0,
            min_child_weight=5.0,
            gamma=0.0,
            random_state=SEED,
            objective="reg:squarederror",
            n_jobs=0,
        )
        xgb_model.fit(Xtr_comb, y_train, verbose=False)
        pred_test = xgb_model.predict(Xte_comb)

        outer_metrics.append(
            {
                "model": "XGB_ECFP+PhysChem",
                "outer_fold": fold,
                "max_depth": best_depth,
                **metrics_dict(y_test, pred_test),
            }
        )
        outer_preds.append(
            pd.DataFrame({
                "Compound_ID": test["Compound_ID"].values,
                "set_role": "NONE",
                "outer_fold": fold,
                "y_true": y_test,
                "y_pred": pred_test,
                "model": "XGB_ECFP+PhysChem",
                "max_depth": best_depth,
            })
        )

        # Diagnostic external predictions by fold
        for tag, extdf in [("EXT_A", ext_a), ("EXT_B", ext_b), ("OOD", ood)]:
            if len(extdf) == 0:
                continue

            yext = extdf["logP_median"].astype(float).values
            Xext_phys = apply_zscore(extdf, scaler, PHYS_COLS)
            Xext_fp = extdf[ecfp_cols].astype(np.uint8).values
            Xext_comb = np.hstack([Xext_fp.astype(np.float32), Xext_phys.astype(np.float32)])

            ext_byfold.append(
                pd.DataFrame({
                    "Compound_ID": extdf["Compound_ID"].values,
                    "set_role": tag,
                    "outer_fold": fold,
                    "y_true": yext,
                    "y_pred": ridge.predict(Xext_phys),
                    "model": "Ridge_PhysChem",
                })
            )

            Kext = tanimoto_kernel(Xext_fp, Xtr_fp)
            ext_byfold.append(
                pd.DataFrame({
                    "Compound_ID": extdf["Compound_ID"].values,
                    "set_role": tag,
                    "outer_fold": fold,
                    "y_true": yext,
                    "y_pred": krr_model.predict(Kext),
                    "model": "KRR_Tanimoto_ECFP",
                    "alpha": best_alpha,
                })
            )

            ext_byfold.append(
                pd.DataFrame({
                    "Compound_ID": extdf["Compound_ID"].values,
                    "set_role": tag,
                    "outer_fold": fold,
                    "y_true": yext,
                    "y_pred": xgb_model.predict(Xext_comb),
                    "model": "XGB_ECFP+PhysChem",
                    "max_depth": best_depth,
                })
            )

        print(f"Fold {fold} OK | KRR alpha={best_alpha} | XGB depth={best_depth}")

    metrics_df = pd.DataFrame(outer_metrics)
    metrics_df.to_csv(outdir / "metrics_outercv.csv", index=False, encoding="utf-8-sig")

    preds_df = pd.concat(outer_preds, ignore_index=True)
    preds_df.to_csv(outdir / "predictions_outercv.csv", index=False, encoding="utf-8-sig")

    ext_byfold_df = pd.concat(ext_byfold, ignore_index=True) if ext_byfold else pd.DataFrame()
    ext_byfold_df.to_csv(outdir / "predictions_external_byfold.csv", index=False, encoding="utf-8-sig")

    # ---------------- FINAL EXTERNAL EVALUATION ----------------
    full_train = none_pool.copy()
    y_full = full_train["logP_median"].astype(float).values

    groups_full = full_train["Murcko_scaffold_ID"].astype(str).values
    strata_full = full_train["Tier"].astype(str).values

    full_scaler = fit_zscore_scaler(full_train, PHYS_COLS)

    X_full_phys = apply_zscore(full_train, full_scaler, PHYS_COLS)
    X_full_fp = full_train[ecfp_cols].astype(np.uint8).values
    X_full_comb = np.hstack([X_full_fp.astype(np.float32), X_full_phys.astype(np.float32)])

    # Final Ridge
    ridge_final = Ridge(alpha=1.0, random_state=SEED)
    ridge_final.fit(X_full_phys, y_full)

    # Final KRR
    best_alpha_full = inner_cv_select_krr_alpha(
        train_df=full_train,
        y=y_full,
        ecfp_cols=ecfp_cols,
        groups=groups_full,
        strata=strata_full,
    )
    K_full = tanimoto_kernel(X_full_fp, X_full_fp)
    krr_final = KernelRidge(alpha=best_alpha_full, kernel="precomputed")
    krr_final.fit(K_full, y_full)

    # Final XGB
    best_depth_full = inner_cv_select_xgb_depth(
        train_X=X_full_comb,
        y=y_full,
        groups=groups_full,
        strata=strata_full,
    )
    xgb_final = xgb.XGBRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=best_depth_full,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=10.0,
        min_child_weight=5.0,
        gamma=0.0,
        random_state=SEED,
        objective="reg:squarederror",
        n_jobs=0,
    )
    xgb_final.fit(X_full_comb, y_full, verbose=False)

    final_preds = []
    for tag, extdf in [("EXT_A", ext_a), ("EXT_B", ext_b), ("OOD", ood)]:
        if len(extdf) == 0:
            continue

        yext = extdf["logP_median"].astype(float).values
        Xext_phys = apply_zscore(extdf, full_scaler, PHYS_COLS)
        Xext_fp = extdf[ecfp_cols].astype(np.uint8).values
        Xext_comb = np.hstack([Xext_fp.astype(np.float32), Xext_phys.astype(np.float32)])

        final_preds.append(
            pd.DataFrame({
                "Compound_ID": extdf["Compound_ID"].values,
                "set_role": tag,
                "y_true": yext,
                "y_pred": ridge_final.predict(Xext_phys),
                "model": "Ridge_PhysChem",
            })
        )

        Kext = tanimoto_kernel(Xext_fp, X_full_fp)
        final_preds.append(
            pd.DataFrame({
                "Compound_ID": extdf["Compound_ID"].values,
                "set_role": tag,
                "y_true": yext,
                "y_pred": krr_final.predict(Kext),
                "model": "KRR_Tanimoto_ECFP",
                "alpha": best_alpha_full,
            })
        )

        final_preds.append(
            pd.DataFrame({
                "Compound_ID": extdf["Compound_ID"].values,
                "set_role": tag,
                "y_true": yext,
                "y_pred": xgb_final.predict(Xext_comb),
                "model": "XGB_ECFP+PhysChem",
                "max_depth": best_depth_full,
            })
        )

    final_pred_df = pd.concat(final_preds, ignore_index=True) if final_preds else pd.DataFrame()
    final_pred_df.to_csv(outdir / "predictions_external_final.csv", index=False, encoding="utf-8-sig")

    final_metrics_df = external_metrics_table(final_pred_df) if not final_pred_df.empty else pd.DataFrame()
    final_metrics_df.to_csv(outdir / "metrics_external_final.csv", index=False, encoding="utf-8-sig")

    print("\nOK - Model panel completed")
    print(f"Run folder: {outdir}")
    print("Saved: metrics_outercv.csv")
    print("Saved: predictions_outercv.csv")
    print("Saved: predictions_external_byfold.csv")
    print("Saved: predictions_external_final.csv")
    print("Saved: metrics_external_final.csv")
    print(f"Final params: KRR alpha={best_alpha_full} | XGB depth={best_depth_full}")


if __name__ == "__main__":
    main()