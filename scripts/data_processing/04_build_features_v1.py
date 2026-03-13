#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature engineering for the JMGM reproducibility pipeline.

Outputs
-------
- data/features/model_table_v1.(parquet|csv.gz)
- data/features/features_physchem_v1.(parquet|csv.gz)
- data/features/features_ecfp2048_r2_v1.(parquet|csv.gz)
- data/reports/feature_metadata.json
- data/reports/checksums_stage2.sha256
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem, Descriptors, Lipinski, Crippen, rdMolDescriptors


# ----------------- paths -----------------
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
FEATURE_DIR = DATA_DIR / "features"
REPORT_DIR = DATA_DIR / "reports"

FROZEN = DATA_DIR / "dataset_v1_frozen_scaffoldfix_v2.csv"
MANIFEST = DATA_DIR / "splits_manifest.csv"


# ----------------- config -----------------
ECFP_RADIUS = 2
ECFP_NBITS = 2048

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

MODEL_TABLE_BASENAME = "model_table_v1"


# ----------------- helpers -----------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def try_write_table(df: pd.DataFrame, out_base: Path) -> Path:
    """
    Prefer parquet if available; otherwise write csv.gz.
    Returns the written file path.
    """
    out_parquet = out_base.with_suffix(".parquet")
    out_csvgz = out_base.with_suffix(".csv.gz")

    try:
        df.to_parquet(out_parquet, index=False)
        return out_parquet
    except Exception:
        df.to_csv(out_csvgz, index=False, compression="gzip", encoding="utf-8-sig")
        return out_csvgz


def mol_from_smiles(smiles: str):
    if pd.isna(smiles):
        return None
    s = str(smiles).strip()
    if not s:
        return None
    return Chem.MolFromSmiles(s)


def compute_physchem(mol) -> dict:
    return {
        "MW": float(Descriptors.MolWt(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "HBD": float(Lipinski.NumHDonors(mol)),
        "HBA": float(Lipinski.NumHAcceptors(mol)),
        "RotB": float(Lipinski.NumRotatableBonds(mol)),
        "AromaticRings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "FractionCSP3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "MR": float(Crippen.MolMR(mol)),
        "RingCount": float(rdMolDescriptors.CalcNumRings(mol)),
        "HeavyAtomCount": float(mol.GetNumHeavyAtoms()),
    }


def compute_ecfp_bits(mol, radius: int = 2, nbits: int = 2048) -> np.ndarray:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.uint8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def main():
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().isoformat(timespec="seconds")

    if not FROZEN.exists():
        raise FileNotFoundError(f"Frozen dataset not found: {FROZEN}")
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    frozen = pd.read_csv(FROZEN)
    manifest = pd.read_csv(MANIFEST)

    # ---- basic checks ----
    for c in ["Compound_ID", "Canonical_SMILES", "logP_median", "Tier", "external_role"]:
        if c not in frozen.columns:
            raise ValueError(f"Frozen dataset missing required column: {c}")

    for c in ["Compound_ID", "set_role", "outer_fold", "Murcko_scaffold_ID"]:
        if c not in manifest.columns:
            raise ValueError(f"Manifest missing required column: {c}")

    # ---- merge (handle Murcko_scaffold_ID collision) ----
    df = frozen.merge(
        manifest[["Compound_ID", "set_role", "outer_fold", "Murcko_scaffold_ID"]],
        on="Compound_ID",
        how="left",
        validate="one_to_one",
        suffixes=("_frozen", "_manifest"),
    )

    if "Murcko_scaffold_ID" not in df.columns:
        if "Murcko_scaffold_ID_manifest" in df.columns:
            df["Murcko_scaffold_ID"] = df["Murcko_scaffold_ID_manifest"]
        elif "Murcko_scaffold_ID_frozen" in df.columns:
            df["Murcko_scaffold_ID"] = df["Murcko_scaffold_ID_frozen"]

    for c in ["Murcko_scaffold_ID_frozen", "Murcko_scaffold_ID_manifest"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    if df["set_role"].isna().any():
        miss = df.loc[df["set_role"].isna(), ["Compound_ID"]].head(10)
        raise ValueError(f"Some Compound_ID values are missing in manifest. Examples:\n{miss}")

    if df["Murcko_scaffold_ID"].isna().any():
        miss = df.loc[df["Murcko_scaffold_ID"].isna(), ["Compound_ID"]].head(10)
        raise ValueError(f"Some Murcko_scaffold_ID values are missing after merge. Examples:\n{miss}")

    # ---- compute features ----
    phys_rows = []
    ecfp_mat = np.zeros((len(df), ECFP_NBITS), dtype=np.uint8)

    invalid = []
    for i, (cid, smi) in enumerate(zip(df["Compound_ID"], df["Canonical_SMILES"])):
        mol = mol_from_smiles(smi)
        if mol is None:
            invalid.append(str(cid))
            phys_rows.append({k: np.nan for k in PHYS_COLS})
            ecfp_mat[i, :] = 0
            continue

        phys_rows.append(compute_physchem(mol))
        ecfp_mat[i, :] = compute_ecfp_bits(mol, radius=ECFP_RADIUS, nbits=ECFP_NBITS)

    if invalid:
        raise ValueError(f"Invalid SMILES found for {len(invalid)} compounds. Examples: {invalid[:10]}")

    phys = pd.DataFrame(phys_rows, columns=PHYS_COLS)
    ecfp_cols = [f"ECFP_{j:04d}" for j in range(ECFP_NBITS)]
    ecfp = pd.DataFrame(ecfp_mat, columns=ecfp_cols)

    # ---- assemble model table ----
    core_cols = [
        "Compound_ID",
        "logP_median",
        "Tier",
        "external_role",
        "set_role",
        "outer_fold",
        "Murcko_scaffold_ID",
    ]
    model = pd.concat([df[core_cols], phys, ecfp], axis=1)

    # ---- write outputs ----
    model_path = try_write_table(model, FEATURE_DIR / MODEL_TABLE_BASENAME)
    phys_path = try_write_table(
        pd.concat([df[["Compound_ID"]], phys], axis=1),
        FEATURE_DIR / "features_physchem_v1",
    )
    ecfp_path = try_write_table(
        pd.concat([df[["Compound_ID"]], ecfp], axis=1),
        FEATURE_DIR / "features_ecfp2048_r2_v1",
    )

    # ---- metadata ----
    meta = {
        "timestamp": ts,
        "inputs": {
            "frozen": FROZEN.name,
            "manifest": MANIFEST.name,
        },
        "rdkit": {
            "version": rdBase.rdkitVersion,
        },
        "features": {
            "ecfp": {
                "type": "Morgan",
                "radius": ECFP_RADIUS,
                "nBits": ECFP_NBITS,
                "binary": True,
            },
            "physchem": {
                "columns": PHYS_COLS,
                "scaling": "z-score scaling fitted on training folds only",
            },
        },
        "outputs": {
            "model_table": model_path.relative_to(BASE_DIR).as_posix(),
            "features_physchem": phys_path.relative_to(BASE_DIR).as_posix(),
            "features_ecfp": ecfp_path.relative_to(BASE_DIR).as_posix(),
        },
        "schema": {
            "target": "logP_median",
            "roles": "external_role / set_role",
            "cv": "outer_fold",
            "group": "Murcko_scaffold_ID",
        },
    }

    meta_path = REPORT_DIR / "feature_metadata.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- checksums ----
    checksums_path = REPORT_DIR / "checksums_stage2.sha256"
    lines = [
        f"{sha256_file(FROZEN)}  {FROZEN.relative_to(BASE_DIR).as_posix()}",
        f"{sha256_file(MANIFEST)}  {MANIFEST.relative_to(BASE_DIR).as_posix()}",
        f"{sha256_file(model_path)}  {model_path.relative_to(BASE_DIR).as_posix()}",
        f"{sha256_file(phys_path)}  {phys_path.relative_to(BASE_DIR).as_posix()}",
        f"{sha256_file(ecfp_path)}  {ecfp_path.relative_to(BASE_DIR).as_posix()}",
        f"{sha256_file(meta_path)}  {meta_path.relative_to(BASE_DIR).as_posix()}",
    ]
    checksums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("OK - Feature tables built successfully")
    print(f"Model table: {model_path}")
    print(f"PhysChem:    {phys_path}")
    print(f"ECFP:        {ecfp_path}")
    print(f"Metadata:    {meta_path}")
    print(f"Checksums:   {checksums_path}")
    print(f"Rows: {len(model)} | ECFP bits: {ECFP_NBITS} | PhysChem: {len(PHYS_COLS)}")


if __name__ == "__main__":
    main()