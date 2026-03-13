#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coumarin logP Project — JCIM (LOCKED v1.1 aligned)

Script: 01_replicate_collapse_and_freeze.py

Purpose
-------
AŞAMA-0 / DATASET FREEZE (Step-1)
1) Load measurement-level raw curated dataset (raw_dataset_original.xlsx)
2) Collapse replicate measurements to compound-level (grouped by Compound_ID)
3) Compute replicate statistics (median/min/max/n_measurements + uncertainty per LOCKED rules)
4) Apply OOD rule: is_OOD = (TPSA > 150) and external_role override to "OOD"
5) Emit frozen master dataset + exclusions log + checksum + audit + replicate report

LOCKED rules implemented
------------------------
- logP_median: median of Experimental_logP
- logP_uncertainty:
    n >= 3 -> MAD
    n == 2 -> |x1-x2| / 2
    n == 1 -> NA
- is_OOD: TPSA > 150  (user decision)
- external_role mapping (raw -> frozen):
    Training        -> NONE
    External_Test_A -> EXT_A
    External_Test_B -> EXT_B
  If is_OOD True, external_role is forced to OOD.

Outputs (stable names; overwrite to ensure single "frozen" reference)
-------------------------------------------------------------------
<outdir>/dataset_v1_frozen.csv
<outdir>/exclusions_v1.csv
<outdir>/checksums.sha256
<reports_dir>/replicate_report_v1.csv
<reports_dir>/audit_faz0_freeze.json

Usage (Windows)
---------------
python 02_scripts\\data_processing\\01_replicate_collapse_and_freeze.py ^
  --in 01_data\\raw\\raw_dataset_original.xlsx ^
  --outdir 01_data\\frozen ^
  --reportsdir 01_data\\reports
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    med = np.median(values)
    return float(np.median(np.abs(values - med)))


def compute_uncertainty(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float)
    n = vals.size
    if n >= 3:
        return mad(vals)
    if n == 2:
        return float(abs(vals[0] - vals[1]) / 2.0)
    return np.nan


def norm_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def uniq_nonempty(series: pd.Series) -> list[str]:
    s = series.dropna().map(norm_str)
    s = s[s != ""]
    return list(pd.unique(s))


def role_normalize(raw_role: str) -> str:
    m = {
        "Training": "NONE",
        "NONE": "NONE",
        "External_Test_A": "EXT_A",
        "EXT_A": "EXT_A",
        "External-A": "EXT_A",
        "External_Test_B": "EXT_B",
        "EXT_B": "EXT_B",
        "External-B": "EXT_B",
        "OOD": "OOD",
    }
    rr = norm_str(raw_role)
    return m.get(rr, rr if rr else "NONE")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to raw_dataset_original.xlsx")
    ap.add_argument("--outdir", required=True, help="Output directory for frozen files")
    ap.add_argument("--reportsdir", required=True, help="Output directory for reports/audit")
    ap.add_argument("--ood_tpsa_threshold", type=float, default=150.0, help="OOD rule: TPSA > threshold")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    inp = Path(args.inp)
    outdir = Path(args.outdir)
    reportsdir = Path(args.reportsdir)

    outdir.mkdir(parents=True, exist_ok=True)
    reportsdir.mkdir(parents=True, exist_ok=True)

    ts_iso = datetime.now().isoformat(timespec="seconds")

    df = pd.read_excel(inp)

    required = [
        "Compound_ID",
        "Canonical_SMILES",
        "InChIKey",
        "Experimental_logP",
        "TPSA",
        "Molecular_Weight",
        "Data_Tier",
        "External_Role",
        "Coumarin_Type",
        "Replicate_ID",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik kolonlar var: {missing}")

    df["Experimental_logP"] = pd.to_numeric(df["Experimental_logP"], errors="coerce")
    df["TPSA"] = pd.to_numeric(df["TPSA"], errors="coerce")
    df["Molecular_Weight"] = pd.to_numeric(df["Molecular_Weight"], errors="coerce")

    df["Compound_ID"] = df["Compound_ID"].map(norm_str)
    bad_id_rows = df["Compound_ID"] == ""
    if bad_id_rows.any():
        df.loc[bad_id_rows, "Compound_ID"] = df.index[bad_id_rows].map(lambda i: f"__MISSING_ID_ROW_{i}__")

    exclusions: list[dict] = []
    frozen_rows: list[dict] = []
    rep_report_rows: list[dict] = []

    consistency_fields = [
        "Canonical_SMILES",
        "InChIKey",
        "TPSA",
        "Molecular_Weight",
        "Data_Tier",
        "External_Role",
        "Coumarin_Type",
    ]

    for compound_id, g in df.groupby("Compound_ID", dropna=False):
        g = g.copy()

        smiles_vals = uniq_nonempty(g["Canonical_SMILES"])
        inchikey_vals = uniq_nonempty(g["InChIKey"])

        if (len(smiles_vals) == 0) and (len(inchikey_vals) == 0):
            exclusions.append(
                {
                    "Compound_ID": compound_id,
                    "Reason_for_exclusion": "Missing both InChIKey and Canonical_SMILES",
                    "Source": "raw_dataset_original.xlsx",
                }
            )
            continue

        vals = g["Experimental_logP"].dropna().to_numpy(dtype=float)
        n_measurements = int(vals.size)

        if n_measurements == 0:
            exclusions.append(
                {
                    "Compound_ID": compound_id,
                    "Reason_for_exclusion": "Missing Experimental_logP (n_measurements=0)",
                    "Source": "raw_dataset_original.xlsx",
                }
            )
            continue

        logP_median = float(np.median(vals))
        logP_min = float(np.min(vals))
        logP_max = float(np.max(vals))
        logP_uncertainty = compute_uncertainty(vals)

        rep = g[g["Experimental_logP"].notna()].iloc[0] if g["Experimental_logP"].notna().any() else g.iloc[0]

        consistency = {}
        inconsistent_any = False
        for f in consistency_fields:
            u = uniq_nonempty(g[f])
            consistency[f"{f}_nunique"] = len(u)
            if len(u) > 1:
                inconsistent_any = True

        rep_ids = uniq_nonempty(g["Replicate_ID"])
        rep_ids_join = "; ".join(rep_ids[:50]) + (f"; ...(+{len(rep_ids)-50})" if len(rep_ids) > 50 else "")

        external_role = role_normalize(rep.get("External_Role", ""))

        tpsa_val = rep.get("TPSA", np.nan)
        is_ood = bool(pd.notna(tpsa_val) and float(tpsa_val) > float(args.ood_tpsa_threshold))
        if is_ood:
            external_role = "OOD"

        frozen_rows.append(
            {
                "Compound_ID": compound_id,
                "Canonical_SMILES": rep.get("Canonical_SMILES", ""),
                "InChIKey": rep.get("InChIKey", ""),
                "logP_median": logP_median,
                "logP_uncertainty": (np.nan if pd.isna(logP_uncertainty) else float(logP_uncertainty)),
                "n_measurements": n_measurements,
                "logP_min": logP_min,
                "logP_max": logP_max,
                "Tier": rep.get("Data_Tier", ""),
                "TPSA": (np.nan if pd.isna(rep.get("TPSA", np.nan)) else float(rep.get("TPSA"))),
                "MW": (np.nan if pd.isna(rep.get("Molecular_Weight", np.nan)) else float(rep.get("Molecular_Weight"))),
                "Coumarin_Type": rep.get("Coumarin_Type", ""),
                "Cluster_ID": "",
                "Murcko_scaffold_ID": "",
                "is_OOD": is_ood,
                "external_role": external_role,
            }
        )

        rep_report_rows.append(
            {
                "Compound_ID": compound_id,
                "n_rows": int(len(g)),
                "n_measurements": n_measurements,
                "logP_median": logP_median,
                "logP_min": logP_min,
                "logP_max": logP_max,
                "logP_uncertainty": (np.nan if pd.isna(logP_uncertainty) else float(logP_uncertainty)),
                "Replicate_IDs": rep_ids_join,
                "replicate_inconsistent_any": bool(inconsistent_any),
                "is_OOD": is_ood,
                **{k: v for k, v in consistency.items() if k.endswith("_nunique")},
            }
        )

    frozen = pd.DataFrame(frozen_rows)
    frozen = frozen.sort_values(["external_role", "Tier", "Compound_ID"], kind="mergesort").reset_index(drop=True)

    replicate_report = pd.DataFrame(rep_report_rows)
    if not replicate_report.empty:
        replicate_report = replicate_report.sort_values(
            ["replicate_inconsistent_any", "n_measurements", "Compound_ID"],
            ascending=[False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)

    exclusions_df = pd.DataFrame(exclusions)
    if not exclusions_df.empty:
        exclusions_df = exclusions_df.sort_values(["Reason_for_exclusion", "Compound_ID"], kind="mergesort").reset_index(
            drop=True
        )

    frozen_csv = outdir / "dataset_v1_frozen.csv"
    exclusions_csv = outdir / "exclusions_v1.csv"
    checksums_file = outdir / "checksums.sha256"
    rep_report_csv = reportsdir / "replicate_report_v1.csv"
    audit_json = reportsdir / "audit_faz0_freeze.json"

    frozen.to_csv(frozen_csv, index=False, encoding="utf-8-sig")
    exclusions_df.to_csv(exclusions_csv, index=False, encoding="utf-8-sig")
    replicate_report.to_csv(rep_report_csv, index=False, encoding="utf-8-sig")

    audit = {
        "timestamp": ts_iso,
        "input": str(inp),
        "outputs": {
            "frozen_csv": str(frozen_csv),
            "exclusions_csv": str(exclusions_csv),
            "checksums_sha256": str(checksums_file),
            "replicate_report_csv": str(rep_report_csv),
            "audit_json": str(audit_json),
        },
        "policies": {
            "groupby_key": "Compound_ID",
            "collapsed_logp_policy": "logP_median = median(Experimental_logP)",
            "uncertainty_policy": "n>=3 MAD; n==2 |x1-x2|/2; n==1 NA",
            "ood_policy": f"is_OOD = (TPSA > {args.ood_tpsa_threshold}) ; external_role forced to OOD",
            "external_role_mapping": {"Training": "NONE", "External_Test_A": "EXT_A", "External_Test_B": "EXT_B"},
        },
        "counts": {
            "n_raw_rows": int(len(df)),
            "n_unique_compound_ids_raw": int(df["Compound_ID"].nunique()),
            "n_frozen_rows": int(len(frozen)),
            "n_excluded_compounds": int(len(exclusions_df)),
            "n_missing_logp_rows": int(df["Experimental_logP"].isna().sum()),
            "n_ood": int(frozen["is_OOD"].sum()) if not frozen.empty else 0,
            "roles": frozen["external_role"].value_counts().to_dict() if not frozen.empty else {},
            "tiers": frozen["Tier"].value_counts().to_dict() if not frozen.empty else {},
            "n_inconsistent_compounds": int(replicate_report["replicate_inconsistent_any"].sum())
            if not replicate_report.empty
            else 0,
        },
        "notes": {"Cluster_ID": "placeholder (filled later)", "Murcko_scaffold_ID": "placeholder (filled later)"},
    }
    audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    checksums_lines = [
        f"{sha256_file(frozen_csv)}  {frozen_csv.name}",
        f"{sha256_file(exclusions_csv)}  {exclusions_csv.name}",
        f"{sha256_file(rep_report_csv)}  {rep_report_csv.name}",
        f"{sha256_file(audit_json)}  {audit_json.name}",
        f"{sha256_file(inp)}  {inp.name}",
    ]
    checksums_file.write_text("\n".join(checksums_lines) + "\n", encoding="utf-8")

    print("OK ✅ AŞAMA-0 / Step-1 completed")
    print("Frozen:", frozen_csv)
    print("Exclusions:", exclusions_csv)
    print("Checksums:", checksums_file)
    print("Replicate report:", rep_report_csv)
    print("Audit:", audit_json)
    if not frozen.empty:
        print("Frozen rows:", len(frozen), "| OOD:", int(frozen["is_OOD"].sum()), "| Excluded:", int(len(exclusions_df)))


if __name__ == "__main__":
    main()