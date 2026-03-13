#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage-6 ablation study for the JMGM reproducibility pipeline.

Compares:
- Random KFold vs Scaffold GroupKFold
- PhysChem vs ECFP vs Hybrid feature sets
- Ridge vs KRR (Tanimoto) vs XGBoost models

Inputs
------
- data/dataset_v1_frozen_scaffoldfix_v2.csv
- data/features/model_table_v1.parquet
  or
- data/features/model_table_v1.csv.gz

Outputs
-------
results/ablation/
  - ablation_results.xlsx
  - ablation_split_comparison_v1.csv
  - ablation_feature_sets_v1.csv
  - ablation_model_panel_v1.csv
  - stage6_notes_v1.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.kernel_ridge import KernelRidge

import xgboost as xgb


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
FEATURE_DIR = DATA_DIR / "features"

RESULTS_DIR = BASE_DIR / "results"
OUT_DIR = RESULTS_DIR / "ablation"

P_DS = DATA_DIR / "dataset_v1_frozen_scaffoldfix_v2.csv"

OUT_XLSX = OUT_DIR / "ablation_results.xlsx"
OUT_SPLIT = OUT_DIR / "ablation_split_comparison_v1.csv"
OUT_FEAT = OUT_DIR / "ablation_feature_sets_v1.csv"
OUT_MODEL = OUT_DIR / "ablation_model_panel_v1.csv"
OUT_NOTES = OUT_DIR / "stage6_notes_v1.md"


# ============================================================
# Config
# ============================================================

SEED = 42
N_SPLITS = 5

PHYS_CANDIDATES = [
    "MW",
    "TPSA",
    "MR",
    "AromaticRings",
    "FractionCSP3",
    "FractionCsp3",
    "HBD",
    "HBA",
    "RotB",
    "RingCount",
    "HeavyAtomCount",
]

META_DROP_ALWAYS = {
    "Compound_ID",
    "external_role",
    "set_role",
    "outer_fold",
    "Tier",
    "Murcko_scaffold_ID",
    "Murcko_scaffold_ID_v2",
    "Murcko_scaffold_SMILES_v2",
    "logP_median",
    "logP",
    "LogP",
    "exp_logP",
    "experimental_logP",
    "y",
    "target",
}


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
    raise ValueError(f"Unsupported table format: {path}")


def relative_or_absolute(path: Path) -> str:
    try:
        return path.relative_to(BASE_DIR).as_posix()
    except Exception:
        return str(path)


# ============================================================
# Metrics
# ============================================================

def ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mt = np.mean(y_true)
    mp = np.mean(y_pred)
    vt = np.var(y_true)
    vp = np.var(y_pred)
    cov = np.mean((y_true - mt) * (y_pred - mp))

    denom = vt + vp + (mt - mp) ** 2
    if denom == 0:
        return np.nan

    return float((2.0 * cov) / denom)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ============================================================
# Utilities
# ============================================================

def find_y_col(df: pd.DataFrame) -> str:
    for c in ["logP_median", "logP", "LogP", "exp_logP", "experimental_logP"]:
        if c in df.columns:
            return c

    for c in df.columns:
        if "logp" in c.lower() and pd.api.types.is_numeric_dtype(df[c]):
            return c

    raise ValueError("Could not locate target logP column in dataset.")


def find_role_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["external_role", "set_role"]:
        if c in df.columns:
            return c
    return None


def find_scaffold_group_col(ds: pd.DataFrame, mt: pd.DataFrame) -> str:
    for c in ["Murcko_scaffold_ID_v2", "Murcko_scaffold_ID"]:
        if c in ds.columns:
            return c
    for c in ["Murcko_scaffold_ID_v2", "Murcko_scaffold_ID"]:
        if c in mt.columns:
            return c
    raise ValueError("Could not locate scaffold group column in dataset or model_table.")


def numericize(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.fillna(x.median(numeric_only=True))
    x = x.fillna(0.0)
    return x


def canonical_phys_cols(columns: List[str]) -> List[str]:
    cols = []
    seen = set()

    for c in columns:
        if c == "FractionCsp3":
            key = "FractionCSP3"
        else:
            key = c

        if key not in seen:
            cols.append(c)
            seen.add(key)

    return cols


def infer_phys_cols(feature_cols: List[str]) -> List[str]:
    hits = [c for c in feature_cols if c in PHYS_CANDIDATES]
    hits = canonical_phys_cols(hits)
    return hits


def infer_ecfp_cols(feature_cols: List[str], phys_cols: List[str]) -> List[str]:
    ecfp = [c for c in feature_cols if c.startswith("ECFP_")]
    if len(ecfp) >= 512:
        return sorted(ecfp)

    non_phys = [c for c in feature_cols if c not in set(phys_cols)]
    if len(non_phys) >= 512:
        return non_phys

    raise ValueError(
        f"Could not infer ECFP columns. Detected {len(ecfp)} explicit ECFP columns."
    )


def flatten_columns(cols) -> List[str]:
    if not isinstance(cols, pd.MultiIndex):
        return [str(c) for c in cols]

    out = []
    for tup in cols.to_list():
        parts = [str(x) for x in tup if x is not None and str(x) != ""]
        out.append("_".join(parts))
    return out


def tanimoto_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = (A > 0).astype(np.uint8)
    B = (B > 0).astype(np.uint8)

    inter = A @ B.T
    sumA = A.sum(axis=1, keepdims=True)
    sumB = B.sum(axis=1, keepdims=True).T
    union = sumA + sumB - inter

    sim = inter / np.clip(union, 1, None)
    return sim.astype(np.float64)


def xgb_params() -> dict:
    return dict(
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=10.0,
        min_child_weight=5.0,
        gamma=0.0,
        random_state=SEED,
        n_jobs=0,
        objective="reg:squarederror",
        verbosity=0,
    )


# ============================================================
# CV + models
# ============================================================

@dataclass
class EvalResult:
    split_strategy: str
    feature_set: str
    model_name: str
    n: int
    rmse: float
    mae: float
    r2: float
    ccc: float


def cv_evaluate(
    X: pd.DataFrame,
    y: np.ndarray,
    split_name: str,
    splitter,
    groups: Optional[np.ndarray],
    model_name: str,
    feature_set: str,
) -> EvalResult:
    preds_all = np.full(shape=len(y), fill_value=np.nan, dtype=float)

    for train_idx, test_idx in splitter.split(X, y, groups):
        Xtr = X.iloc[train_idx].to_numpy()
        Xte = X.iloc[test_idx].to_numpy()
        ytr = y[train_idx]

        if model_name == "Ridge":
            pipe = Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("ridge", Ridge(alpha=1.0))
            ])
            pipe.fit(Xtr, ytr)
            preds = pipe.predict(Xte)

        elif model_name == "KRR_Tanimoto":
            K_train = tanimoto_similarity(Xtr, Xtr)
            K_test = tanimoto_similarity(Xte, Xtr)
            krr = KernelRidge(alpha=0.01, kernel="precomputed")
            krr.fit(K_train, ytr)
            preds = krr.predict(K_test)

        elif model_name == "XGB":
            reg = xgb.XGBRegressor(**xgb_params())
            reg.fit(Xtr, ytr)
            preds = reg.predict(Xte)

        else:
            raise ValueError(f"Unknown model: {model_name}")

        preds_all[test_idx] = preds

    mask = ~np.isnan(preds_all)
    yy = y[mask]
    pp = preds_all[mask]

    return EvalResult(
        split_strategy=split_name,
        feature_set=feature_set,
        model_name=model_name,
        n=int(mask.sum()),
        rmse=rmse(yy, pp),
        mae=float(mean_absolute_error(yy, pp)),
        r2=float(r2_score(yy, pp)),
        ccc=float(ccc(yy, pp)),
    )


# ============================================================
# Main
# ============================================================

def main():
    model_table_path = resolve_model_table()

    if not P_DS.exists():
        raise FileNotFoundError(f"Missing dataset: {P_DS}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = pd.read_csv(P_DS)
    mt = read_table(model_table_path)

    if "Compound_ID" not in ds.columns or "Compound_ID" not in mt.columns:
        raise ValueError("Compound_ID must exist in both dataset and model_table.")

    ds["Compound_ID"] = ds["Compound_ID"].astype(str)
    mt["Compound_ID"] = mt["Compound_ID"].astype(str)

    ycol = find_y_col(ds)
    role_col = find_role_col(ds)
    scaffold_col = find_scaffold_group_col(ds, mt)

    # Use NONE training pool only
    if role_col is not None:
        ds_use = ds[ds[role_col].astype(str) == "NONE"].copy()
    else:
        ds_use = ds.copy()

    if ds_use.empty:
        raise ValueError("Training pool is empty.")

    # Align model table to dataset order
    mt_use = (
        mt.merge(ds_use[["Compound_ID"]], on="Compound_ID", how="inner")
          .set_index("Compound_ID")
          .loc[ds_use["Compound_ID"].tolist()]
          .reset_index()
    )

    if len(mt_use) != len(ds_use):
        raise ValueError("Dataset and model_table alignment mismatch after Compound_ID join.")

    # Scaffold groups
    if scaffold_col in ds_use.columns:
        groups_scaffold = ds_use[scaffold_col].astype(str).to_numpy()
    else:
        groups_scaffold = mt_use[scaffold_col].astype(str).to_numpy()

    n_groups = pd.Series(groups_scaffold).nunique()
    if n_groups < 2:
        raise ValueError(f"Need at least 2 scaffold groups for GroupKFold, found {n_groups}.")

    n_splits_scaffold = min(N_SPLITS, int(n_groups))
    print(f"[INFO] Training compounds: {len(ds_use)} | Scaffold groups: {n_groups} | Scaffold folds: {n_splits_scaffold}")

    # Build X
    feat_cols_all = [c for c in mt_use.columns if c not in META_DROP_ALWAYS]
    X_all = numericize(mt_use[feat_cols_all].copy())

    phys_cols = infer_phys_cols(list(X_all.columns))
    if len(phys_cols) < 3:
        raise ValueError(f"Too few PhysChem columns found. Found: {phys_cols}")

    ecfp_cols = infer_ecfp_cols(list(X_all.columns), phys_cols)

    feature_sets = {
        "PhysChem": phys_cols,
        "ECFP": ecfp_cols,
        "Hybrid": list(dict.fromkeys(phys_cols + ecfp_cols)),
    }

    y = pd.to_numeric(ds_use[ycol], errors="coerce").to_numpy()
    if np.isnan(y).any():
        raise ValueError(f"Target column '{ycol}' contains NaN values.")

    splitters: List[Tuple[str, object, Optional[np.ndarray]]] = [
        ("RandomKFold", KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED), None),
        ("ScaffoldGroupKFold", GroupKFold(n_splits=n_splits_scaffold), groups_scaffold),
    ]

    model_panel = ["Ridge", "KRR_Tanimoto", "XGB"]
    results: List[EvalResult] = []

    for split_name, splitter, groups in splitters:
        for fs_name, cols in feature_sets.items():
            X_fs = X_all[cols].copy()

            for model_name in model_panel:
                if model_name == "KRR_Tanimoto" and fs_name != "ECFP":
                    continue
                if model_name == "Ridge" and fs_name == "ECFP":
                    continue

                res = cv_evaluate(
                    X=X_fs,
                    y=y,
                    split_name=split_name,
                    splitter=splitter,
                    groups=groups,
                    model_name=model_name,
                    feature_set=fs_name,
                )
                results.append(res)

                print(
                    f"{split_name} | {fs_name} | {model_name} | "
                    f"RMSE={res.rmse:.6f} | CCC={res.ccc:.6f}"
                )

    res_df = pd.DataFrame([r.__dict__ for r in results])

    split_comp = res_df.pivot_table(
        index=["feature_set", "model_name"],
        columns=["split_strategy"],
        values=["rmse", "ccc", "mae", "r2"],
        aggfunc="first",
    )
    split_comp = split_comp.reset_index()
    split_comp.columns = flatten_columns(split_comp.columns)

    feat_comp = res_df.sort_values(["split_strategy", "model_name", "rmse"]).copy()
    model_comp = res_df.sort_values(["split_strategy", "feature_set", "rmse"]).copy()

    res_df.to_csv(OUT_MODEL, index=False, encoding="utf-8-sig")
    split_comp.to_csv(OUT_SPLIT, index=False, encoding="utf-8-sig")
    feat_comp.to_csv(OUT_FEAT, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        res_df.to_excel(writer, sheet_name="all_results", index=False)
        split_comp.to_excel(writer, sheet_name="split_comparison", index=False)
        feat_comp.to_excel(writer, sheet_name="feature_sets", index=False)
        model_comp.to_excel(writer, sheet_name="model_panel", index=False)

        cfg = pd.DataFrame([
            {"key": "SEED", "value": SEED},
            {"key": "N_SPLITS_random", "value": N_SPLITS},
            {"key": "N_SPLITS_scaffold", "value": n_splits_scaffold},
            {"key": "target_column", "value": ycol},
            {"key": "n_samples_used", "value": len(y)},
            {"key": "scaffold_group_col", "value": scaffold_col},
            {"key": "n_groups_scaffold", "value": int(n_groups)},
            {"key": "n_physchem", "value": len(phys_cols)},
            {"key": "n_ecfp", "value": len(ecfp_cols)},
            {"key": "n_hybrid", "value": len(feature_sets["Hybrid"])},
        ])
        cfg.to_excel(writer, sheet_name="config", index=False)

    notes = f"""# Stage-6 Ablation

Inputs
- dataset: {relative_or_absolute(P_DS)}
- model_table: {relative_or_absolute(model_table_path)}

Training pool
- role filter: {"NONE" if role_col else "not applied (role column missing)"}

Configuration
- SEED = {SEED}
- random_splits = {N_SPLITS}
- scaffold_splits = {n_splits_scaffold}
- scaffold_group_col = {scaffold_col}
- n_samples = {len(y)}
- n_scaffold_groups = {int(n_groups)}

Feature sets
- n_physchem = {len(phys_cols)}
- n_ecfp = {len(ecfp_cols)}
- n_hybrid = {len(feature_sets["Hybrid"])}

Outputs
- {relative_or_absolute(OUT_XLSX)}
- {relative_or_absolute(OUT_SPLIT)}
- {relative_or_absolute(OUT_FEAT)}
- {relative_or_absolute(OUT_MODEL)}
"""
    OUT_NOTES.write_text(notes, encoding="utf-8")

    print("\nOK - Stage-6 ablation exported.")
    print(f"Excel: {OUT_XLSX}")
    print(f"Split comparison: {OUT_SPLIT}")
    print(f"Feature sets: {OUT_FEAT}")
    print(f"Model panel: {OUT_MODEL}")
    print(f"Notes: {OUT_NOTES}")


if __name__ == "__main__":
    main()