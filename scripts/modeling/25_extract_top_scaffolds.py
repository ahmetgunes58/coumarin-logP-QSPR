#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
25_extract_top_scaffolds.py

Purpose
-------
Extract and summarize the top Bemis–Murcko scaffolds from the frozen coumarin dataset.

Outputs
-------
- results/scaffold_analysis/scaffold_counts_all_v1.csv
- results/scaffold_analysis/scaffold_top10_v1.csv
- results/scaffold_analysis/scaffold_top10_mapping_v1.csv

What this script does
---------------------
1) Loads the frozen dataset
2) Computes Bemis–Murcko scaffold SMILES from Canonical_SMILES
3) Counts scaffold frequencies
4) Assigns S1–S10 labels to the ten most frequent scaffolds
5) Exports CSV files for Figure S7 verification

Notes
-----
- This script is intended for scaffold verification and SI support.
- It does not draw the scaffolds; it extracts the exact scaffold identities and counts.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
OUT_DIR = RESULTS_DIR / "scaffold_analysis"

OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_CANDIDATES = [
    DATA_DIR / "dataset_v1_frozen_scaffoldfix_v2.csv",
    DATA_DIR / "dataset_v1_frozen.csv",
    BASE_DIR / "01_data" / "_LOCKED_FREEZE_v1" / "dataset_v1_frozen_scaffoldfix_v2.csv",
    BASE_DIR / "01_data" / "_LOCKED_FREEZE_v1" / "dataset_v1_frozen.csv",
]

OUT_ALL = OUT_DIR / "scaffold_counts_all_v1.csv"
OUT_TOP10 = OUT_DIR / "scaffold_top10_v1.csv"
OUT_MAP = OUT_DIR / "scaffold_top10_mapping_v1.csv"


# ============================================================
# Helpers
# ============================================================

def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Frozen dataset not found. Checked:\n" + "\n".join(str(p) for p in paths)
    )


def mol_from_smiles(smiles: str):
    if pd.isna(smiles):
        return None
    s = str(smiles).strip()
    if not s:
        return None
    return Chem.MolFromSmiles(s)


def murcko_scaffold_smiles(smiles: str) -> str:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return "INVALID"
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None:
        return "INVALID"
    return Chem.MolToSmiles(scaffold)


# ============================================================
# Main
# ============================================================

def main() -> None:
    dataset_path = first_existing(DATASET_CANDIDATES)
    print(f"[INFO] Using dataset: {dataset_path}")

    df = pd.read_csv(dataset_path)

    required = ["Compound_ID", "Canonical_SMILES"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["Compound_ID"] = df["Compound_ID"].astype(str)

    # Compute scaffold SMILES
    df["scaffold_smiles"] = df["Canonical_SMILES"].apply(murcko_scaffold_smiles)

    # Count frequencies
    counts = (
        df.groupby("scaffold_smiles", dropna=False)
        .agg(
            Count=("Compound_ID", "size"),
            Representative_Compound_ID=("Compound_ID", "first"),
            Representative_SMILES=("Canonical_SMILES", "first"),
        )
        .reset_index()
        .sort_values(["Count", "scaffold_smiles"], ascending=[False, True])
        .reset_index(drop=True)
    )

    # Add global rank
    counts["GlobalRank"] = range(1, len(counts) + 1)

    # Export all scaffold counts
    counts.to_csv(OUT_ALL, index=False, encoding="utf-8-sig")

    # Top-10 scaffolds
    top10 = counts.head(10).copy().reset_index(drop=True)
    top10["ScaffoldCode"] = [f"S{i}" for i in range(1, len(top10) + 1)]

    # Export top-10 summary
    top10 = top10[
        [
            "ScaffoldCode",
            "GlobalRank",
            "scaffold_smiles",
            "Count",
            "Representative_Compound_ID",
            "Representative_SMILES",
        ]
    ]
    top10.to_csv(OUT_TOP10, index=False, encoding="utf-8-sig")

    # Compound-level mapping for top-10 scaffolds
    top10_map = {row["scaffold_smiles"]: row["ScaffoldCode"] for _, row in top10.iterrows()}

    mapping = df[df["scaffold_smiles"].isin(top10_map.keys())].copy()
    mapping["ScaffoldCode"] = mapping["scaffold_smiles"].map(top10_map)

    cols = ["ScaffoldCode", "Compound_ID", "Canonical_SMILES", "scaffold_smiles"]
    extra_cols = [c for c in ["logP_median", "TPSA", "external_role", "Tier"] if c in mapping.columns]
    mapping = mapping[cols + extra_cols].sort_values(["ScaffoldCode", "Compound_ID"])

    mapping.to_csv(OUT_MAP, index=False, encoding="utf-8-sig")

    # Console summary
    print("\n[OK] Scaffold extraction completed.")
    print(f"Total compounds: {len(df)}")
    print(f"Unique scaffolds: {counts.shape[0]}")
    print(f"Top-10 total compounds: {int(top10['Count'].sum())}")
    print(f"Top-10 fraction: {top10['Count'].sum() / len(df):.3f}")

    print("\nTop-10 scaffolds:")
    print(top10[["ScaffoldCode", "Count", "Representative_Compound_ID", "scaffold_smiles"]].to_string(index=False))

    print("\nOutputs:")
    print(f"- {OUT_ALL}")
    print(f"- {OUT_TOP10}")
    print(f"- {OUT_MAP}")


if __name__ == "__main__":
    main()