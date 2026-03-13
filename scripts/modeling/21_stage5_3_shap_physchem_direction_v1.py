#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage-5.3 SHAP analysis for the final XGBoost model.

Workflow
--------
- Load trained XGBoost model (or model bundle)
- Recover exact training feature list if stored in the bundle
- Build feature matrix from model_table for the training pool only
- Align columns exactly to the stored training feature list when available
- Compute SHAP values on the aligned matrix
- Export PhysChem subset reports:
    * global importance
    * directionality
    * per-compound SHAP values
- Export full global SHAP summary

Outputs
-------
results/interpretability/
  - shap_physchem_global_v1.csv
  - shap_physchem_direction_v1.csv
  - shap_physchem_per_compound_v1.csv.gz
  - shap_global_full_v1.csv
  - stage5_3_notes_shap_v1.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
FEATURE_DIR = DATA_DIR / "features"
RESULTS_DIR = BASE_DIR / "results"

OUT_DIR = RESULTS_DIR / "interpretability"
OUT_PHYS_GLOBAL = OUT_DIR / "shap_physchem_global_v1.csv"
OUT_PHYS_DIR = OUT_DIR / "shap_physchem_direction_v1.csv"
OUT_PHYS_PER = OUT_DIR / "shap_physchem_per_compound_v1.csv.gz"
OUT_FULL_GLOBAL = OUT_DIR / "shap_global_full_v1.csv"
OUT_NOTES = OUT_DIR / "stage5_3_notes_shap_v1.md"


# ============================================================
# Input candidates
# ============================================================

MODEL_CANDIDATES = [
    RESULTS_DIR / "stage5_freeze_v1" / "final_xgb_model_v1.pkl",
    BASE_DIR / "04_results" / "stage5_freeze_v1" / "final_xgb_model_v1.pkl",
]

MODEL_TABLE_CANDIDATES = [
    FEATURE_DIR / "model_table_v1.parquet",
    FEATURE_DIR / "model_table_v1.csv.gz",
    DATA_DIR / "model_table_v1.csv.gz",
    BASE_DIR / "01_data" / "_LOCKED_FREEZE_v1" / "model_table_v1.csv.gz",
]

DATASET_CANDIDATES = [
    DATA_DIR / "dataset_v1_frozen_scaffoldfix_v2.csv",
    DATA_DIR / "dataset_v1_frozen.csv",
    BASE_DIR / "01_data" / "_LOCKED_FREEZE_v1" / "dataset_v1_frozen_scaffoldfix_v2.csv",
    BASE_DIR / "01_data" / "_LOCKED_FREEZE_v1" / "dataset_v1_frozen.csv",
]


# ============================================================
# Configuration
# ============================================================

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


# ============================================================
# Utilities
# ============================================================

def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


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


def extract_estimator(obj):
    if not isinstance(obj, dict):
        return obj

    for key in ["model", "estimator", "xgb_model", "xgb", "regressor"]:
        if key in obj:
            return obj[key]

    for _, value in obj.items():
        if hasattr(value, "predict"):
            return value

    raise TypeError(f"Model bundle is dict but no estimator found. Keys: {list(obj.keys())}")


def extract_feature_list(bundle) -> Optional[List[str]]:
    """
    Try to recover the exact feature column list used at training time.
    """
    if not isinstance(bundle, dict):
        return None

    candidate_keys = ["feature_names", "feature_cols", "columns", "X_cols", "feature_list", "x_cols"]

    for key in candidate_keys:
        if key in bundle:
            value = bundle[key]
            if isinstance(value, (list, tuple)) and len(value) > 0:
                return [str(x) for x in value]

    for _, value in bundle.items():
        if isinstance(value, dict):
            for nested_key in candidate_keys:
                if nested_key in value:
                    nested_val = value[nested_key]
                    if isinstance(nested_val, (list, tuple)) and len(nested_val) > 0:
                        return [str(x) for x in nested_val]

    return None


def numericize(df: pd.DataFrame) -> pd.DataFrame:
    x = df.apply(lambda s: pd.to_numeric(s, errors="coerce"))
    x = x.fillna(x.median(numeric_only=True))
    x = x.fillna(0.0)
    return x


def canonical_phys_cols(cols: List[str]) -> List[str]:
    out = []
    seen = set()
    for c in cols:
        key = "FractionCSP3" if c == "FractionCsp3" else c
        if key not in seen:
            out.append(c)
            seen.add(key)
    return out


def find_role_col(df: pd.DataFrame) -> str:
    for candidate in ["external_role", "set_role", "role", "split_role"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Dataset must contain one of the following role columns: "
        "external_role, set_role, role, split_role"
    )


# ============================================================
# Main
# ============================================================

def main():
    model_path = first_existing(MODEL_CANDIDATES)
    model_table_path = first_existing(MODEL_TABLE_CANDIDATES)
    dataset_path = first_existing(DATASET_CANDIDATES)

    if model_path is None:
        raise FileNotFoundError(
            "Missing model pickle. Checked:\n- "
            + "\n- ".join(str(p) for p in MODEL_CANDIDATES)
        )

    if model_table_path is None:
        raise FileNotFoundError(
            "Missing model_table. Checked:\n- "
            + "\n- ".join(str(p) for p in MODEL_TABLE_CANDIDATES)
        )

    if dataset_path is None:
        raise FileNotFoundError(
            "Missing frozen dataset. Checked:\n- "
            + "\n- ".join(str(p) for p in DATASET_CANDIDATES)
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(model_path)
    model = extract_estimator(bundle)
    feature_list = extract_feature_list(bundle)  # may be None

    # dataset -> training IDs
    ds = pd.read_csv(dataset_path)

    if "Compound_ID" not in ds.columns:
        raise ValueError(f"{dataset_path.name} missing column: Compound_ID")

    role_col = find_role_col(ds)

    ds["Compound_ID"] = ds["Compound_ID"].astype(str)
    train_ids = set(
        ds.loc[ds[role_col].astype(str) == "NONE", "Compound_ID"].astype(str).tolist()
    )

    if len(train_ids) == 0:
        raise ValueError(f"No training IDs found in dataset ({role_col} == NONE).")

    # model_table
    mt = read_table(model_table_path)

    if "Compound_ID" not in mt.columns:
        raise ValueError(f"{model_table_path.name} missing column: Compound_ID")

    mt["Compound_ID"] = mt["Compound_ID"].astype(str)

    train = mt.loc[mt["Compound_ID"].isin(train_ids)].copy()
    if len(train) == 0:
        raise ValueError("No matching training IDs found in model_table.")

    # Build raw feature DF
    raw_feat = train.drop(columns=["Compound_ID"]).copy()

    aligned = False
    missing_cols: List[str] = []
    extra_cols: List[str] = []

    if feature_list is not None:
        have = set(raw_feat.columns.astype(str))
        need = list(map(str, feature_list))

        missing_cols = [c for c in need if c not in have]
        extra_cols = [c for c in raw_feat.columns.astype(str) if c not in set(need)]

        x_aligned = raw_feat.reindex(columns=need)
        x_aligned = numericize(x_aligned)

        x_full = x_aligned
        feat_cols = need
        aligned = True
    else:
        x_full = numericize(raw_feat)
        feat_cols = list(x_full.columns)

    # SHAP on FULL-X
    import shap

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(x_full)

    shap_full = pd.DataFrame(shap_vals, columns=[f"SHAP_{c}" for c in feat_cols])

    # Full global
    full_global = []
    for col in feat_cols:
        s = shap_full[f"SHAP_{col}"].to_numpy()
        full_global.append(
            {
                "feature": col,
                "mean_abs_shap": float(np.mean(np.abs(s))),
                "mean_shap": float(np.mean(s)),
            }
        )

    full_global_df = (
        pd.DataFrame(full_global)
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    full_global_df.to_csv(OUT_FULL_GLOBAL, index=False, encoding="utf-8-sig")

    # PhysChem subset
    phys_cols = canonical_phys_cols([c for c in PHYS_CANDIDATES if c in feat_cols])

    if len(phys_cols) < 3:
        raise ValueError(f"Too few PhysChem columns found among model features. Found: {phys_cols}")

    out_phys_per = pd.concat(
        [
            train[["Compound_ID"]].reset_index(drop=True),
            x_full[phys_cols].reset_index(drop=True),
            shap_full[[f"SHAP_{c}" for c in phys_cols]].reset_index(drop=True),
        ],
        axis=1,
    )
    out_phys_per.to_csv(OUT_PHYS_PER, index=False, compression="gzip", encoding="utf-8-sig")

    phys_global = []
    for col in phys_cols:
        s = out_phys_per[f"SHAP_{col}"].to_numpy()
        phys_global.append(
            {
                "feature": col,
                "mean_abs_shap": float(np.mean(np.abs(s))),
                "mean_shap": float(np.mean(s)),
            }
        )

    phys_global_df = (
        pd.DataFrame(phys_global)
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    phys_global_df.to_csv(OUT_PHYS_GLOBAL, index=False, encoding="utf-8-sig")

    # Directionality
    dir_rows = []
    for col in phys_cols:
        x = out_phys_per[col].to_numpy()
        s = out_phys_per[f"SHAP_{col}"].to_numpy()

        corr = float(np.corrcoef(x, s)[0, 1]) if np.std(x) > 0 and np.std(s) > 0 else 0.0
        q20 = np.quantile(x, 0.20)
        q80 = np.quantile(x, 0.80)

        low_mask = x <= q20
        high_mask = x >= q80

        mean_low = float(np.mean(s[low_mask])) if np.any(low_mask) else float("nan")
        mean_high = float(np.mean(s[high_mask])) if np.any(high_mask) else float("nan")

        if np.isfinite(mean_low) and np.isfinite(mean_high):
            if mean_high > mean_low:
                direction = "positive"
            elif mean_high < mean_low:
                direction = "negative"
            else:
                direction = "mixed"
        else:
            direction = "mixed"

        dir_rows.append(
            {
                "feature": col,
                "corr_feature_shap": corr,
                "q20": float(q20),
                "q80": float(q80),
                "mean_shap_low_q20": mean_low,
                "mean_shap_high_q80": mean_high,
                "direction_tag": direction,
            }
        )

    dir_df = (
        pd.DataFrame(dir_rows)
        .merge(phys_global_df[["feature", "mean_abs_shap"]], on="feature", how="left")
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    dir_df.to_csv(OUT_PHYS_DIR, index=False, encoding="utf-8-sig")

    notes = f"""# Stage-5.3 — SHAP (FULL-X aligned to training features)

Inputs
- model_bundle: {relative_or_absolute(model_path)}
- model_table: {relative_or_absolute(model_table_path)}
- dataset: {relative_or_absolute(dataset_path)}

Bundle info
- model_bundle_type: {type(bundle)}
- estimator_type: {type(model)}
- feature_list_found_in_bundle: {feature_list is not None}

Alignment
- aligned_to_bundle_features: {aligned}
- n_features_used: {len(feat_cols)}
- missing_cols_filled_with_0: {missing_cols}
- extra_cols_ignored: {extra_cols}

Outputs
- {relative_or_absolute(OUT_PHYS_GLOBAL)}
- {relative_or_absolute(OUT_PHYS_DIR)}
- {relative_or_absolute(OUT_PHYS_PER)}
- {relative_or_absolute(OUT_FULL_GLOBAL)}
"""
    OUT_NOTES.write_text(notes, encoding="utf-8")

    print("OK - Stage-5.3 SHAP exported.")
    print(f"Feature list found in bundle: {feature_list is not None}")
    print(f"Aligned to bundle features: {aligned}")
    print(f"n_features_used: {len(feat_cols)}")
    if feature_list is not None:
        print(f"missing filled with 0 (n={len(missing_cols)}): {missing_cols[:10]}")
        print(f"extra ignored (n={len(extra_cols)}): {extra_cols[:10]}")
    print(f"Output: {OUT_PHYS_GLOBAL}")
    print(f"Output: {OUT_PHYS_DIR}")
    print(f"Output: {OUT_PHYS_PER}")
    print(f"Output: {OUT_FULL_GLOBAL}")
    print(f"Output: {OUT_NOTES}")


if __name__ == "__main__":
    main()