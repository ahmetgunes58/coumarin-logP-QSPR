#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

FROZEN = "01_data/frozen/dataset_v1_frozen.csv"
SCAFF  = "01_data/processed/scaffold_groups.csv"
OUT    = "01_data/processed/splits_manifest.csv"

N_SPLITS = 5
SEED = 42


def main():
    df = pd.read_csv(FROZEN)
    sc = pd.read_csv(SCAFF)

    # --- basic checks ---
    required_frozen = ["Compound_ID", "Tier"]
    missing = [c for c in required_frozen if c not in df.columns]
    if missing:
        raise ValueError(f"Frozen dataset missing columns: {missing}")

    required_sc = ["Compound_ID", "Murcko_scaffold_ID"]
    missing_sc = [c for c in required_sc if c not in sc.columns]
    if missing_sc:
        raise ValueError(f"scaffold_groups missing columns: {missing_sc}")

    # If external_role column is absent, create it as NONE (defensive)
    if "external_role" not in df.columns:
        df["external_role"] = "NONE"

    # --- merge scaffold id (handle collision with placeholder) ---
    # If frozen already has Murcko_scaffold_ID placeholder, merge will create suffix cols.
    merged = df.merge(sc, on="Compound_ID", how="left", suffixes=("_frozen", "_scaf"))

    # Resolve scaffold column deterministically
    if "Murcko_scaffold_ID_scaf" in merged.columns:
        merged["Murcko_scaffold_ID"] = merged["Murcko_scaffold_ID_scaf"]
    elif "Murcko_scaffold_ID" in merged.columns:
        # no collision case
        pass
    else:
        # very unlikely, but keep explicit error
        raise ValueError("Murcko_scaffold_ID not found after merge.")

    # Drop any leftover scaffold columns from collision
    for c in ["Murcko_scaffold_ID_frozen", "Murcko_scaffold_ID_scaf"]:
        if c in merged.columns:
            merged = merged.drop(columns=[c])

    # Check for missing scaffolds
    if merged["Murcko_scaffold_ID"].isna().any():
        missing_ids = merged.loc[merged["Murcko_scaffold_ID"].isna(), "Compound_ID"].tolist()
        raise ValueError(f"Missing Murcko_scaffold_ID for {len(missing_ids)} compounds. Example: {missing_ids[:10]}")

    # --- role assignment ---
    merged["set_role"] = merged["external_role"].astype(str).fillna("NONE")

    # --- outer CV only on NONE pool ---
    pool = merged[merged["set_role"] == "NONE"].copy()
    y = pool["Tier"].astype(str).values
    groups = pool["Murcko_scaffold_ID"].astype(str).values

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    pool["outer_fold"] = -1
    for fold, (_, test_idx) in enumerate(sgkf.split(pool, y=y, groups=groups), start=1):
        pool.iloc[test_idx, pool.columns.get_loc("outer_fold")] = fold

    if (pool["outer_fold"] == -1).any():
        raise RuntimeError("Some NONE-pool rows were not assigned to any fold. Check StratifiedGroupKFold inputs.")

    # Merge fold assignment back (external/OOD get outer_fold=0)
    merged = merged.merge(pool[["Compound_ID", "outer_fold"]], on="Compound_ID", how="left")
    merged["outer_fold"] = merged["outer_fold"].fillna(0).astype(int)

    # --- output manifest ---
    manifest = merged[[
        "Compound_ID",
        "Tier",
        "external_role",
        "set_role",
        "Murcko_scaffold_ID",
        "outer_fold"
    ]].sort_values(["set_role", "outer_fold", "Compound_ID"])

    Path("01_data/processed").mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUT, index=False)

    print("OK ✅ splits_manifest generated")
    print("Saved:", OUT)

    print("\nCounts by set_role:")
    print(manifest["set_role"].value_counts(dropna=False).to_string())

    print("\nOuter CV fold sizes (set_role=NONE only):")
    print(manifest[manifest["set_role"] == "NONE"]["outer_fold"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()