#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage-7 figure generation for the JMGM reproducibility pipeline.

Design goals
------------
- Single script; each figure is an isolated function (no cross-figure side effects).
- Minimal, safe patches to improve correctness and publication readability.
- Robust to missing inputs: placeholder figure + manifest logging.
- Figure locking: optional per-figure toggles + copy accepted outputs to a stable folder.

Primary inputs
--------------
- data/dataset_v1_frozen_scaffoldfix_v2.csv
- data/features/model_table_v1.csv.gz
  (fallback: data/features/features_physchem_v1.csv.gz)

Optional inputs
---------------
- results/interpretability/shap_physchem_global_v1.csv
- results/interpretability/shap_physchem_direction_v1.csv
- results/ablation/ablation_model_panel_v1.csv
- results/external_validation/.../predictions_external_final.csv
- results/y_scrambling/y_scrambling_results_final_hybridxgb_v2.csv
- results/benchmark_swissadme/baseline_benchmark_table_v1.csv

Figures
-------
- Fig01_dataset_curation
- Fig02_chemical_space_pca
- Fig03_tpsa_vs_mw_roles
- Fig04_pred_vs_obs
- Fig05_applicability_domain
- Fig06_y_scrambling
- Fig07_shap_physchem_importance
- Fig08_shap_physchem_direction
- Fig09_baseline_benchmark
- Fig10_ablation_summary
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ============================================================
# Config
# ============================================================

SEED = 42

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
FEATURE_DIR = DATA_DIR / "features"
RESULT_DIR = BASE_DIR / "results"
FIGURE_DIR = BASE_DIR / "figures"

P_DS = DATA_DIR / "dataset_v1_frozen_scaffoldfix_v2.csv"
P_MT = FEATURE_DIR / "model_table_v1.csv.gz"
if not P_MT.exists():
    P_MT = FEATURE_DIR / "features_physchem_v1.csv.gz"

P_SHAP_GLOBAL = RESULT_DIR / "interpretability" / "shap_physchem_global_v1.csv"
P_SHAP_DIR = RESULT_DIR / "interpretability" / "shap_physchem_direction_v1.csv"
P_ABL_MODEL = RESULT_DIR / "ablation" / "ablation_model_panel_v1.csv"

OUT_ROOT = FIGURE_DIR / "main" / "stage7"

FINAL_MODEL_LABEL = "Hybrid-XGB (ECFP2048 r=2 + PhysChem)"

FIGURE_TOGGLES = {
    "Fig01_dataset_curation": True,
    "Fig02_chemical_space_pca": True,
    "Fig03_tpsa_vs_mw_roles": True,
    "Fig04_pred_vs_obs": True,
    "Fig05_applicability_domain": True,
    "Fig06_y_scrambling": True,
    "Fig07_shap_physchem_importance": True,
    "Fig08_shap_physchem_direction": True,
    "Fig09_baseline_benchmark": True,
    "Fig10_ablation_summary": True,
}

COPY_TO_FINAL_LOCKED = False
FINAL_LOCKED_DIR = FIGURE_DIR / "_FINAL_LOCKED_STAGE7"

ROLE_COL_CANDIDATES = ["external_role", "set_role", "role", "split_role"]
Y_COL_CANDIDATES = ["logP_median", "logP", "LogP", "exp_logP", "experimental_logP"]

PHYS_CANDIDATES = [
    "MW", "TPSA", "MR", "AromaticRings", "FractionCSP3", "FractionCsp3",
    "HBD", "HBA", "RotB", "RingCount", "HeavyAtomCount"
]


# ============================================================
# Matplotlib publication style
# ============================================================

def set_pub_style():
    mpl.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "lines.linewidth": 1.0,
        "grid.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_fig(fig: plt.Figure, out_base: Path, manifest: Dict, key: str):
    out_base = Path(out_base)
    if out_base.suffix:
        out_base = out_base.with_suffix("")

    png = out_base.with_suffix(".png")
    pdf = out_base.with_suffix(".pdf")

    try:
        fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.18)
    except Exception:
        pass

    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    manifest["figures"][key]["outputs"] = {"png": str(png), "pdf": str(pdf)}

    if COPY_TO_FINAL_LOCKED:
        try:
            FINAL_LOCKED_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(png, FINAL_LOCKED_DIR / f"{key}.png")
            shutil.copy2(pdf, FINAL_LOCKED_DIR / f"{key}.pdf")
        except Exception:
            pass


def placeholder_figure(title: str, lines: List[str]) -> plt.Figure:
    fig = plt.figure(figsize=(6.5, 3.6))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.02, 0.90, title, fontsize=12, weight="bold", transform=ax.transAxes)
    ax.text(0.02, 0.72, "PLACEHOLDER", fontsize=11, weight="bold", transform=ax.transAxes)
    ax.text(0.02, 0.62, "Reason / debug:", fontsize=9, transform=ax.transAxes)
    y = 0.56
    for m in lines[:12]:
        ax.text(0.04, y, f"- {m}", fontsize=8, transform=ax.transAxes)
        y -= 0.05
    if len(lines) > 12:
        ax.text(0.04, y, f"... +{len(lines)-12} more", fontsize=8, transform=ax.transAxes)
    return fig


def find_first_existing(globs: List[str]) -> Optional[Path]:
    for g in globs:
        hits = sorted(BASE_DIR.glob(g))
        if hits:
            return hits[0]
    return None


def find_y_col(df: pd.DataFrame) -> str:
    for c in Y_COL_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns:
        if "logp" in c.lower() and pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError("Could not locate experimental logP column in dataset.")


def find_role_col(df: pd.DataFrame) -> Optional[str]:
    for c in ROLE_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _normalize_role_values(arr_like) -> pd.Series:
    s = pd.Series(arr_like).astype(str)
    s = s.replace({
        "NONE": "Training", "None": "Training", "none": "Training",
        "TRAIN": "Training", "train": "Training",
        "TRAINING": "Training", "training": "Training",
    })
    return s


def read_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not P_DS.exists():
        raise FileNotFoundError(f"Missing dataset: {P_DS}")
    if not P_MT.exists():
        raise FileNotFoundError(f"Missing model_table/features table: {P_MT}")

    ds = pd.read_csv(P_DS)
    mt = pd.read_csv(P_MT, compression="gzip") if P_MT.name.endswith(".gz") else pd.read_csv(P_MT)

    if "Compound_ID" not in ds.columns or "Compound_ID" not in mt.columns:
        raise ValueError("Compound_ID must exist in both dataset and model_table/features table.")

    ds["Compound_ID"] = ds["Compound_ID"].astype(str)
    mt["Compound_ID"] = mt["Compound_ID"].astype(str)
    return ds, mt


def add_identity_line(ax, x, y):
    mn = np.nanmin([np.nanmin(x), np.nanmin(y)])
    mx = np.nanmax([np.nanmax(x), np.nanmax(y)])
    ax.plot([mn, mx], [mn, mx], linestyle="--")
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)


def _parse_sim_bin_to_midpoint(val) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan

    if isinstance(val, (int, float, np.integer, np.floating)):
        x = float(val)
        return x if 0.0 <= x <= 1.0 else np.nan

    s = str(val).strip()

    if s.startswith("<"):
        try:
            x = float(s[1:])
            return 0.5 * x
        except Exception:
            return np.nan

    if s.startswith(">"):
        try:
            x = float(s[1:])
            return 0.5 * (x + 1.0)
        except Exception:
            return np.nan

    s2 = s.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    s2 = s2.replace(",", "-")
    s2 = re.sub(r"\s+", "", s2)

    m = re.match(r"^([-+]?\d*\.?\d+)[-–—]([-+]?\d*\.?\d+)$", s2)
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        mid = 0.5 * (a + b)
        return mid if 0.0 <= mid <= 1.0 else np.nan

    m = re.match(r"^([-+]?\d*\.?\d+)$", s2)
    if m:
        x = float(m.group(1))
        return x if 0.0 <= x <= 1.0 else np.nan

    return np.nan


# ============================================================
# Figure builders
# ============================================================

def fig01_dataset_curation(ds: pd.DataFrame, manifest: Dict, outdir: Path):
    key = "Fig01_dataset_curation"
    manifest["figures"][key]["inputs"] = [str(P_DS)]

    ycol = find_y_col(ds)
    role_col = find_role_col(ds)

    scaffold_col = None
    for c in ["Murcko_scaffold_ID_v2", "Murcko_scaffold_ID", "scaffold_id", "Scaffold_ID"]:
        if c in ds.columns:
            scaffold_col = c
            break

    fig = plt.figure(figsize=(9.6, 6.0))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    if role_col:
        roles = _normalize_role_values(ds[role_col])
        order = ["Training", "EXT_A", "EXT_B", "OOD"]
        counts = roles.value_counts()
        labels = [r for r in order if r in counts.index] + [r for r in counts.index if r not in order]
        vals = [int(counts[r]) for r in labels]
        ax1.bar(labels, vals)
        ax1.set_title("A) Dataset split")
        ax1.set_ylabel("Count")
        ax1.tick_params(axis="x", rotation=30)
        for t in ax1.get_xticklabels():
            t.set_ha("right")
    else:
        ax1.text(0.5, 0.5, "role column not found", ha="center", va="center")
        ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    y = pd.to_numeric(ds[ycol], errors="coerce").dropna()
    ax2.hist(y.values, bins=14)
    ax2.set_title("B) Experimental logP distribution")
    ax2.set_xlabel("Experimental logP")
    ax2.set_ylabel("Count")

    ax3 = fig.add_subplot(gs[1, 0])
    if "TPSA" in ds.columns:
        tpsa = pd.to_numeric(ds["TPSA"], errors="coerce").dropna()
        ax3.hist(tpsa.values, bins=14)
        ax3.axvline(150.0, linestyle="--")
        ax3.set_title("C) TPSA distribution (OOD threshold)")
        ax3.set_xlabel("TPSA")
        ax3.set_ylabel("Count")
    else:
        ax3.text(0.5, 0.5, "TPSA not found", ha="center", va="center")
        ax3.axis("off")

    ax4 = fig.add_subplot(gs[1, 1])
    if scaffold_col:
        sc = ds[scaffold_col].astype(str)
        sc_counts = sc.value_counts()
        top = sc_counts.head(10)
        codes = [f"S{i+1}" for i in range(len(top))]
        ax4.bar(codes, top.values.astype(int))
        ax4.set_title(f"D) Top-10 scaffolds (n_scaffolds={sc_counts.shape[0]})")
        ax4.set_xlabel("Scaffold code (see SI for scaffold structures.)")
        ax4.set_ylabel("Count")

        rep_name_col = "Common_Name" if "Common_Name" in ds.columns else None
        rows = []
        for i, (sid, cnt) in enumerate(top.items()):
            code = f"S{i+1}"
            rep = None
            if rep_name_col:
                sub = ds.loc[ds[scaffold_col].astype(str) == sid, rep_name_col].astype(str)
                rep = sub.value_counts().index[0] if len(sub) else None
            rows.append({
                "ScaffoldCode": code,
                "Murcko_scaffold_ID_v2": sid,
                "Count": int(cnt),
                "RepresentativeName": rep
            })
        mapping_df = pd.DataFrame(rows)
        mapping_path = outdir / "Fig01_panelD_scaffold_mapping.csv"
        mapping_df.to_csv(mapping_path, index=False)
        manifest["figures"][key]["extras"] = {"panelD_mapping_csv": str(mapping_path)}
    else:
        ax4.text(0.5, 0.5, "Murcko scaffold column not found", ha="center", va="center")
        ax4.axis("off")

    fig.suptitle("Figure 01 — Dataset curation", y=0.98)
    save_fig(fig, outdir / key, manifest, key)


def fig02_chemical_space_pca(ds: pd.DataFrame, mt: pd.DataFrame, manifest: Dict, outdir: Path):
    key = "Fig02_chemical_space_pca"
    manifest["figures"][key]["inputs"] = [str(P_DS), str(P_MT)]

    role_col = find_role_col(ds)

    use_cols = [c for c in PHYS_CANDIDATES if c in ds.columns]
    if len(use_cols) < 3:
        use_cols = [c for c in PHYS_CANDIDATES if c in mt.columns]

    if len(use_cols) < 3:
        fig = placeholder_figure("Figure 02 — Chemical space (PCA)", ["Need >=3 PhysChem columns in dataset or model_table"])
        save_fig(fig, outdir / key, manifest, key)
        return

    base = ds[["Compound_ID"]].copy()
    if all(c in ds.columns for c in use_cols):
        X = ds[use_cols].copy()
    else:
        mt_use = mt.merge(base, on="Compound_ID", how="inner").set_index("Compound_ID").loc[base["Compound_ID"]].reset_index()
        X = mt_use[use_cols].copy()

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    Xs = StandardScaler().fit_transform(X.values)
    pca = PCA(n_components=2, random_state=SEED)
    Z = pca.fit_transform(Xs)

    fig = plt.figure(figsize=(6.5, 4.2))
    ax = fig.add_subplot(111)

    if role_col:
        roles = _normalize_role_values(ds[role_col]).values
        for r in sorted(pd.Series(roles).unique()):
            m = roles == r
            ax.scatter(Z[m, 0], Z[m, 1], s=18, alpha=0.85, label=r)
        ax.legend(frameon=False, ncols=2)
    else:
        ax.scatter(Z[:, 0], Z[:, 1], s=18, alpha=0.85)

    ax.set_title("Figure 02 — Chemical space (PCA on PhysChem)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.grid(True, alpha=0.25)

    save_fig(fig, outdir / key, manifest, key)


def fig03_tpsa_vs_mw_roles(ds: pd.DataFrame, manifest: Dict, outdir: Path):
    key = "Fig03_tpsa_vs_mw_roles"
    manifest["figures"][key]["inputs"] = [str(P_DS)]

    if "TPSA" not in ds.columns or "MW" not in ds.columns:
        fig = placeholder_figure("Figure 03 — TPSA vs MW", ["dataset must contain TPSA and MW"])
        save_fig(fig, outdir / key, manifest, key)
        return

    role_col = find_role_col(ds)

    tpsa = pd.to_numeric(ds["TPSA"], errors="coerce").values
    mw = pd.to_numeric(ds["MW"], errors="coerce").values

    fig = plt.figure(figsize=(6.5, 4.2))
    ax = fig.add_subplot(111)

    if role_col:
        roles = _normalize_role_values(ds[role_col]).values
        for r in sorted(pd.Series(roles).unique()):
            m = roles == r
            ax.scatter(mw[m], tpsa[m], s=20, alpha=0.85, label=r)
        ax.legend(frameon=False, ncols=2)
    else:
        ax.scatter(mw, tpsa, s=20, alpha=0.85)

    ax.axhline(150, linestyle="--")
    ax.set_title("Figure 03 — TPSA vs MW (OOD: TPSA>150)")
    ax.set_xlabel("MW")
    ax.set_ylabel("TPSA")
    ax.grid(True, alpha=0.25)

    save_fig(fig, outdir / key, manifest, key)


def fig04_pred_vs_obs_auto(ds: pd.DataFrame, manifest: Dict, outdir: Path):
    key = "Fig04_pred_vs_obs"
    cand = [
        "results/external_validation/*/predictions_external_final*.csv",
        "results/**/predictions_external*.csv",
        "results/**/predictions_oof*.csv",
        "results/**/oof_predictions*.csv",
        "results/**/training_predictions*.csv",
        "results/**/predictions*.csv",
    ]
    p = find_first_existing(cand)
    if p is None:
        fig = placeholder_figure("Figure 04 — Predicted vs Observed", [f"None of these globs matched: {cand}"])
        manifest["figures"][key]["inputs"] = []
        save_fig(fig, outdir / key, manifest, key)
        return

    manifest["figures"][key]["inputs"] = [str(p)]
    df = pd.read_csv(p)

    y_true_cols = [c for c in df.columns if c.lower() in ["y_true", "y", "logp_median", "logp_exp", "exp_logp", "observed"]]
    y_pred_cols = [c for c in df.columns if c.lower() in ["y_pred", "pred", "prediction", "predicted"]]

    if not y_true_cols or not y_pred_cols:
        fig = placeholder_figure(
            "Figure 04 — Predicted vs Observed",
            [f"Found file but could not infer y_true/y_pred columns: {p}", f"Columns: {list(df.columns)}"]
        )
        save_fig(fig, outdir / key, manifest, key)
        return

    set_label = "External" if "external" in p.name.lower() else ("OOF" if "oof" in p.name.lower() else "Predictions")

    role_col = find_role_col(df)
    df_use = df.copy()
    if role_col:
        r = _normalize_role_values(df_use[role_col])
        df_use["_role_norm"] = r
        if set_label == "External":
            df_use = df_use[df_use["_role_norm"] != "Training"].copy()

    y_true = pd.to_numeric(df_use[y_true_cols[0]], errors="coerce").values
    y_pred = pd.to_numeric(df_use[y_pred_cols[0]], errors="coerce").values
    m = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]

    if len(y_true) < 5:
        fig = placeholder_figure("Figure 04 — Predicted vs Observed", [f"Too few valid points after filtering: n={len(y_true)}", str(p)])
        save_fig(fig, outdir / key, manifest, key)
        return

    fig = plt.figure(figsize=(6.2, 4.2))
    ax = fig.add_subplot(111)
    ax.scatter(y_true, y_pred, s=22, alpha=0.85)
    add_identity_line(ax, y_true, y_pred)
    ax.set_title(f"Figure 04 — Predicted vs Observed ({set_label}, {FINAL_MODEL_LABEL})")
    ax.set_xlabel("Observed logP")
    ax.set_ylabel("Predicted logP")
    ax.grid(True, alpha=0.25)

    save_fig(fig, outdir / key, manifest, key)


def fig05_applicability_domain_auto(manifest: Dict, outdir: Path):
    key = "Fig05_applicability_domain"
    cand = [
        "results/**/similarity_error_bins.csv",
        "results/**/knn*_ad*_summary*.csv",
        "results/**/applicability*domain*.csv",
        "results/**/similarity*error*.csv",
    ]
    p = find_first_existing(cand)
    if p is None:
        fig = placeholder_figure("Figure 05 — Applicability Domain", [f"No AD file found via: {cand}"])
        manifest["figures"][key]["inputs"] = []
        save_fig(fig, outdir / key, manifest, key)
        return

    manifest["figures"][key]["inputs"] = [str(p)]
    df = pd.read_csv(p)

    def _find_col(preds: List[str]) -> Optional[str]:
        for pr in preds:
            for c in df.columns:
                if pr in c.lower():
                    return c
        return None

    if "sim_bin" in df.columns:
        ycol = None
        for c in ["median", "mean", "max"]:
            if c in df.columns:
                ycol = c
                break
        if ycol is None:
            fig = placeholder_figure(
                "Figure 05 — Applicability Domain",
                [f"sim_bin present but no stat column among median/mean/max: {p}", f"Columns: {list(df.columns)}"]
            )
            save_fig(fig, outdir / key, manifest, key)
            return

        x = df["sim_bin"].map(_parse_sim_bin_to_midpoint)
        y = pd.to_numeric(df[ycol], errors="coerce")
        m = (~pd.isna(x)) & (~y.isna())
        x = np.asarray(x)[m.values]
        y = y[m].values

        if len(x) == 0:
            fig = placeholder_figure(
                "Figure 05 — Applicability Domain",
                [f"sim_bin present but could not parse midpoint into [0,1]: {p}",
                 f"Unique sim_bin: {sorted(df['sim_bin'].astype(str).unique())[:8]} ..."]
            )
            save_fig(fig, outdir / key, manifest, key)
            return

        if np.nanmin(x) < 0.0 or np.nanmax(x) > 1.0:
            fig = placeholder_figure(
                "Figure 05 — Applicability Domain",
                [f"Invalid similarity range detected (expected 0..1).",
                 f"x_min={np.nanmin(x):.3f}, x_max={np.nanmax(x):.3f}",
                 f"File: {p}"]
            )
            save_fig(fig, outdir / key, manifest, key)
            return

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        fig = plt.figure(figsize=(6.4, 4.2))
        ax = fig.add_subplot(111)
        ax.plot(x, y, marker="o")
        ax.set_title("Figure 05 — Applicability Domain (similarity-bin summary)")
        ax.set_xlabel("Nearest-neighbor Tanimoto similarity (bin midpoint)")
        ax.set_ylabel(f"{ycol} (error summary)")
        ax.grid(True, alpha=0.25)

        save_fig(fig, outdir / key, manifest, key)
        return

    bin_low_col = _find_col(["bin_low"])
    bin_high_col = _find_col(["bin_high"])
    if bin_low_col and bin_high_col:
        err_col = _find_col(["rmse"]) or _find_col(["mae"]) or _find_col(["abs_error", "mean_abs", "mean_error", "error_mean"])
        if err_col is None:
            fig = placeholder_figure(
                "Figure 05 — Applicability Domain",
                [f"Found bins file but couldn't infer error column: {p}", f"Columns: {list(df.columns)}"]
            )
            save_fig(fig, outdir / key, manifest, key)
            return

        bin_low = pd.to_numeric(df[bin_low_col], errors="coerce")
        bin_high = pd.to_numeric(df[bin_high_col], errors="coerce")
        x = (bin_low + bin_high) / 2.0
        y = pd.to_numeric(df[err_col], errors="coerce")
        m = (~x.isna()) & (~y.isna())
        x, y = x[m].values, y[m].values

        if len(x) == 0:
            fig = placeholder_figure("Figure 05 — Applicability Domain", [f"No valid numeric bin/error rows: {p}"])
            save_fig(fig, outdir / key, manifest, key)
            return

        if np.nanmin(x) < 0.0 or np.nanmax(x) > 1.0:
            fig = placeholder_figure(
                "Figure 05 — Applicability Domain",
                [f"Invalid similarity range detected (expected 0..1).",
                 f"x_min={np.nanmin(x):.3f}, x_max={np.nanmax(x):.3f}",
                 f"File: {p}"]
            )
            save_fig(fig, outdir / key, manifest, key)
            return

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        fig = plt.figure(figsize=(6.4, 4.2))
        ax = fig.add_subplot(111)
        ax.plot(x, y, marker="o")
        ax.set_title("Figure 05 — Applicability Domain (binned similarity–error)")
        ax.set_xlabel("Nearest-neighbor Tanimoto similarity (bin center)")
        ax.set_ylabel(err_col)
        ax.grid(True, alpha=0.25)

        save_fig(fig, outdir / key, manifest, key)
        return

    sim_col = _find_col(["tanimoto", "similarity", "nn_sim", "knn_sim", "sim"])
    err_col = _find_col(["abs_error", "absolute_error", "abs_err", "error", "rmse", "mae"])

    if sim_col is None or err_col is None:
        fig = placeholder_figure(
            "Figure 05 — Applicability Domain",
            [f"Found file but couldn't infer similarity/error cols: {p}", f"Columns: {list(df.columns)}"]
        )
        save_fig(fig, outdir / key, manifest, key)
        return

    x = pd.to_numeric(df[sim_col], errors="coerce").values
    y = pd.to_numeric(df[err_col], errors="coerce").values
    m = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[m], y[m]

    if len(x) == 0:
        fig = placeholder_figure("Figure 05 — Applicability Domain", [f"No valid numeric similarity/error rows: {p}"])
        save_fig(fig, outdir / key, manifest, key)
        return

    if np.nanmin(x) < 0.0 or np.nanmax(x) > 1.0:
        fig = placeholder_figure(
            "Figure 05 — Applicability Domain",
            [f"Invalid similarity range detected in column '{sim_col}' (expected 0..1).",
             f"x_min={np.nanmin(x):.3f}, x_max={np.nanmax(x):.3f}",
             f"File: {p}"]
        )
        save_fig(fig, outdir / key, manifest, key)
        return

    fig = plt.figure(figsize=(6.4, 4.2))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=20, alpha=0.85)
    ax.set_title("Figure 05 — Applicability Domain (similarity vs error)")
    ax.set_xlabel(sim_col)
    ax.set_ylabel(err_col)
    ax.grid(True, alpha=0.25)

    save_fig(fig, outdir / key, manifest, key)


def fig06_y_scrambling_auto(manifest: Dict, outdir: Path):
    key = "Fig06_y_scrambling"
    cand = [
        "results/y_scrambling/y_scrambling_results_final_hybridxgb*.csv",
        "results/**/y_scrambling_results*.csv",
        "results/**/y_scram*summary*.csv",
        "results/**/yscram*summary*.csv",
        "results/**/y_scram*.csv",
    ]
    p = find_first_existing(cand)
    if p is None:
        fig = placeholder_figure("Figure 06 — Y-scrambling", [f"No y-scrambling file found via: {cand}"])
        manifest["figures"][key]["inputs"] = []
        save_fig(fig, outdir / key, manifest, key)
        return

    manifest["figures"][key]["inputs"] = [str(p)]
    df = pd.read_csv(p)

    if "RMSE" not in df.columns:
        rmse_col = None
        for c in df.columns:
            if "rmse" in c.lower():
                rmse_col = c
                break
        if rmse_col is None:
            fig = placeholder_figure("Figure 06 — Y-scrambling", [f"Could not find RMSE column in: {p}", f"Columns: {list(df.columns)}"])
            save_fig(fig, outdir / key, manifest, key)
            return
        df = df.rename(columns={rmse_col: "RMSE"})

    if "is_real" in df.columns:
        real = df[df["is_real"].astype(bool) == True]
        scr = df[df["is_real"].astype(bool) == False]
        if len(real) == 0 or len(scr) == 0:
            fig = placeholder_figure("Figure 06 — Y-scrambling", [f"is_real present but cannot split real vs scrambled in: {p}"])
            save_fig(fig, outdir / key, manifest, key)
            return
        real_rmse = float(pd.to_numeric(real["RMSE"], errors="coerce").dropna().iloc[0])
        scr_rmse = pd.to_numeric(scr["RMSE"], errors="coerce").dropna().values
    else:
        real_rmse = None
        scr_rmse = pd.to_numeric(df["RMSE"], errors="coerce").dropna().values

    if len(scr_rmse) == 0:
        fig = placeholder_figure("Figure 06 — Y-scrambling", [f"No valid scrambled RMSE values in: {p}"])
        save_fig(fig, outdir / key, manifest, key)
        return

    fig = plt.figure(figsize=(6.2, 4.2))
    ax = fig.add_subplot(111)
    ax.hist(scr_rmse, bins=18)
    if real_rmse is not None and np.isfinite(real_rmse):
        ax.axvline(real_rmse, linestyle="--")
        p_emp = (np.sum(scr_rmse <= real_rmse) + 1.0) / (len(scr_rmse) + 1.0)
        ax.text(0.98, 0.95, f"Real RMSE={real_rmse:.3f}\nempirical p≈{p_emp:.3f}",
                ha="right", va="top", transform=ax.transAxes, fontsize=8)
        title = f"Figure 06 — Y-scrambling ({FINAL_MODEL_LABEL})"
    else:
        title = "Figure 06 — Y-scrambling (scrambled RMSE distribution)"

    ax.set_title(title)
    ax.set_xlabel("Scrambled RMSE")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.25)

    save_fig(fig, outdir / key, manifest, key)


def fig07_shap_importance(manifest: Dict, outdir: Path):
    key = "Fig07_shap_physchem_importance"
    if not P_SHAP_GLOBAL.exists():
        fig = placeholder_figure("Figure 07 — SHAP global (PhysChem)", [str(P_SHAP_GLOBAL)])
        manifest["figures"][key]["inputs"] = []
        save_fig(fig, outdir / key, manifest, key)
        return

    manifest["figures"][key]["inputs"] = [str(P_SHAP_GLOBAL)]
    df = pd.read_csv(P_SHAP_GLOBAL)

    for c in ["feature", "mean_abs_shap"]:
        if c not in df.columns:
            fig = placeholder_figure("Figure 07 — SHAP global (PhysChem)", [f"Missing column {c} in {P_SHAP_GLOBAL}"])
            save_fig(fig, outdir / key, manifest, key)
            return

    top = df.sort_values("mean_abs_shap", ascending=False).head(10).copy().iloc[::-1]

    fig = plt.figure(figsize=(6.8, 4.2))
    ax = fig.add_subplot(111)
    ax.barh(top["feature"].astype(str).tolist(), top["mean_abs_shap"].values)
    ax.set_title("Figure 07 — Global importance (PhysChem, mean|SHAP|)")
    ax.set_xlabel("mean(|SHAP|)")
    ax.grid(True, axis="x", alpha=0.25)

    save_fig(fig, outdir / key, manifest, key)


def fig08_shap_direction(manifest: Dict, outdir: Path):
    key = "Fig08_shap_physchem_direction"
    if not P_SHAP_DIR.exists():
        fig = placeholder_figure("Figure 08 — SHAP directionality (PhysChem)", [str(P_SHAP_DIR)])
        manifest["figures"][key]["inputs"] = []
        save_fig(fig, outdir / key, manifest, key)
        return

    manifest["figures"][key]["inputs"] = [str(P_SHAP_DIR)]
    df = pd.read_csv(P_SHAP_DIR)

    need = ["feature", "corr_feature_shap", "mean_abs_shap"]
    for c in need:
        if c not in df.columns:
            fig = placeholder_figure("Figure 08 — SHAP directionality (PhysChem)", [f"Missing column {c} in {P_SHAP_DIR}"])
            save_fig(fig, outdir / key, manifest, key)
            return

    top = df.sort_values("mean_abs_shap", ascending=False).head(10).copy().iloc[::-1]

    fig = plt.figure(figsize=(6.8, 4.2))
    ax = fig.add_subplot(111)
    ax.barh(top["feature"].astype(str).tolist(), top["corr_feature_shap"].values)
    ax.axvline(0, linestyle="--")
    ax.set_title("Figure 08 — Directionality (corr(feature, SHAP))")
    ax.set_xlabel("corr(feature value, SHAP contribution)")
    ax.grid(True, axis="x", alpha=0.25)

    save_fig(fig, outdir / key, manifest, key)


def fig09_baseline_benchmark_auto(manifest: Dict, outdir: Path):
    key = "Fig09_baseline_benchmark"
    cand = [
        "results/benchmark_swissadme/baseline_benchmark_table*.csv",
        "results/**/baseline_benchmark_table*.csv",
        "results/**/baseline*benchmark*.csv",
        "results/**/benchmark*baseline*.csv",
        "results/**/swissadme*benchmark*.csv",
    ]
    p = find_first_existing(cand)
    if p is None:
        fig = placeholder_figure("Figure 09 — Baseline benchmark", [f"No baseline benchmark file found via: {cand}"])
        manifest["figures"][key]["inputs"] = []
        save_fig(fig, outdir / key, manifest, key)
        return

    manifest["figures"][key]["inputs"] = [str(p)]
    df = pd.read_csv(p)

    method_col = None
    rmse_col = None
    for c in df.columns:
        cl = c.lower()
        if method_col is None and ("method" in cl or "model" in cl or "predictor" in cl):
            method_col = c
        if rmse_col is None and "rmse" in cl:
            rmse_col = c

    if method_col is None or rmse_col is None:
        fig = placeholder_figure("Figure 09 — Baseline benchmark", [f"Couldn't infer method/rmse cols: {p}", f"Columns: {list(df.columns)}"])
        save_fig(fig, outdir / key, manifest, key)
        return

    sub = df[[method_col, rmse_col]].dropna().copy()
    sub[method_col] = sub[method_col].replace({"OurModel": FINAL_MODEL_LABEL})
    sub[rmse_col] = pd.to_numeric(sub[rmse_col], errors="coerce")
    sub = sub.dropna().sort_values(rmse_col, ascending=True).head(10)

    fig = plt.figure(figsize=(7.2, 4.2))
    ax = fig.add_subplot(111)
    ax.bar(sub[method_col].astype(str).tolist(), sub[rmse_col].values)
    ax.set_title("Figure 09 — Baseline benchmark (lower is better)")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=30)
    for t in ax.get_xticklabels():
        t.set_ha("right")
    ax.grid(True, axis="y", alpha=0.25)

    fig.subplots_adjust(bottom=0.35)
    save_fig(fig, outdir / key, manifest, key)


def fig10_ablation_summary(manifest: Dict, outdir: Path):
    key = "Fig10_ablation_summary"
    if not P_ABL_MODEL.exists():
        fig = placeholder_figure("Figure 10 — Ablation summary", [str(P_ABL_MODEL)])
        manifest["figures"][key]["inputs"] = []
        save_fig(fig, outdir / key, manifest, key)
        return

    manifest["figures"][key]["inputs"] = [str(P_ABL_MODEL)]
    df = pd.read_csv(P_ABL_MODEL)

    need = ["split_strategy", "feature_set", "model_name", "rmse", "ccc"]
    for c in need:
        if c not in df.columns:
            fig = placeholder_figure("Figure 10 — Ablation summary", [f"Missing column {c} in {P_ABL_MODEL}"])
            save_fig(fig, outdir / key, manifest, key)
            return

    wanted = [
        ("PhysChem", "Ridge"),
        ("ECFP", "KRR_Tanimoto"),
        ("Hybrid", "Ridge"),
        ("Hybrid", "XGB"),
        ("ECFP", "XGB"),
        ("PhysChem", "XGB"),
    ]
    sub = df.copy()
    sub["key"] = sub["feature_set"].astype(str) + "+" + sub["model_name"].astype(str)
    wanted_keys = [a + "+" + b for a, b in wanted]
    sub = sub[sub["key"].isin(wanted_keys)].copy()

    sub["key"] = pd.Categorical(sub["key"], categories=wanted_keys, ordered=True)
    sub = sub.sort_values(["split_strategy", "key"])

    fig = plt.figure(figsize=(7.6, 4.6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    for split in ["RandomKFold", "ScaffoldGroupKFold"]:
        s = sub[sub["split_strategy"] == split]
        if len(s) == 0:
            continue
        ax1.plot(s["key"].astype(str).tolist(), s["rmse"].values, marker="o", label=split)
    ax1.set_title("A) Ablation — RMSE")
    ax1.set_ylabel("RMSE")
    ax1.tick_params(axis="x", rotation=30)
    for t in ax1.get_xticklabels():
        t.set_ha("right")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(frameon=False)

    ax2 = fig.add_subplot(gs[0, 1])
    for split in ["RandomKFold", "ScaffoldGroupKFold"]:
        s = sub[sub["split_strategy"] == split]
        if len(s) == 0:
            continue
        ax2.plot(s["key"].astype(str).tolist(), s["ccc"].values, marker="o", label=split)
    ax2.set_title("B) Ablation — CCC")
    ax2.set_ylabel("CCC")
    ax2.tick_params(axis="x", rotation=30)
    for t in ax2.get_xticklabels():
        t.set_ha("right")
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(frameon=False)

    fig.subplots_adjust(bottom=0.38)
    save_fig(fig, outdir / key, manifest, key)


# ============================================================
# Main
# ============================================================

def main():
    set_pub_style()

    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = OUT_ROOT / ts
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "stage": "Stage-7",
        "level": "publication-ready stable",
        "timestamp": ts,
        "outdir": str(outdir),
        "final_model_label": FINAL_MODEL_LABEL,
        "figures": {
            "Fig01_dataset_curation": {"inputs": [], "outputs": {}, "extras": {}},
            "Fig02_chemical_space_pca": {"inputs": [], "outputs": {}, "extras": {}},
            "Fig03_tpsa_vs_mw_roles": {"inputs": [], "outputs": {}, "extras": {}},
            "Fig04_pred_vs_obs": {"inputs": [], "outputs": {}, "extras": {}},
            "Fig05_applicability_domain": {"inputs": [], "outputs": {}, "extras": {}},
            "Fig06_y_scrambling": {"inputs": [], "outputs": {}, "extras": {}},
            "Fig07_shap_physchem_importance": {"inputs": [], "outputs": {}, "extras": {}},
            "Fig08_shap_physchem_direction": {"inputs": [], "outputs": {}, "extras": {}},
            "Fig09_baseline_benchmark": {"inputs": [], "outputs": {}, "extras": {}},
            "Fig10_ablation_summary": {"inputs": [], "outputs": {}, "extras": {}},
        }
    }

    ds, mt = read_inputs()

    if FIGURE_TOGGLES["Fig01_dataset_curation"]:
        fig01_dataset_curation(ds, manifest, outdir)

    if FIGURE_TOGGLES["Fig02_chemical_space_pca"]:
        fig02_chemical_space_pca(ds, mt, manifest, outdir)

    if FIGURE_TOGGLES["Fig03_tpsa_vs_mw_roles"]:
        fig03_tpsa_vs_mw_roles(ds, manifest, outdir)

    if FIGURE_TOGGLES["Fig04_pred_vs_obs"]:
        fig04_pred_vs_obs_auto(ds, manifest, outdir)

    if FIGURE_TOGGLES["Fig05_applicability_domain"]:
        fig05_applicability_domain_auto(manifest, outdir)

    if FIGURE_TOGGLES["Fig06_y_scrambling"]:
        fig06_y_scrambling_auto(manifest, outdir)

    if FIGURE_TOGGLES["Fig07_shap_physchem_importance"]:
        fig07_shap_importance(manifest, outdir)

    if FIGURE_TOGGLES["Fig08_shap_physchem_direction"]:
        fig08_shap_direction(manifest, outdir)

    if FIGURE_TOGGLES["Fig09_baseline_benchmark"]:
        fig09_baseline_benchmark_auto(manifest, outdir)

    if FIGURE_TOGGLES["Fig10_ablation_summary"]:
        fig10_ablation_summary(manifest, outdir)

    man_path = outdir / "stage7_manifest_v2.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("OK - Stage-7 figures exported.")
    print(f"outdir: {outdir}")
    print(f"manifest: {man_path}")


if __name__ == "__main__":
    main()