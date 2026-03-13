#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
26_plot_tpsa_shap_scatter.py

Purpose
-------
Generate a supplementary scatter plot illustrating the relationship between
TPSA descriptor values and their corresponding SHAP contributions for the
training pool of the final Hybrid-XGB model.

Colour-coding by Bemis-Murcko scaffold class (top-10 scaffolds) reveals
that the positive TPSA-SHAP correlation is driven primarily by
nitrogen-embedded aromatic scaffold classes rather than compounds with
isolated polar substituents.

Inputs
------
- results/interpretability/shap_physchem_per_compound_v1.csv.gz
    Per-compound TPSA values and SHAP_TPSA contributions (training pool only).

- results/scaffold_analysis/scaffold_top10_mapping_v1.csv
    Compound-to-scaffold mapping for the ten most frequent Bemis-Murcko
    scaffolds (S1-S10).

    Note: S2, S3, S9, S10 are external/OOD compounds and therefore absent
    from the SHAP training-pool file. Only S1, S4, S5, S6, S7, S8 appear
    in both files and receive explicit scaffold labels. Remaining training
    compounds are labelled "Other (training)".

Outputs
-------
figures/supplementary/
  - Figure_SX_tpsa_shap_scatter.png   (600 dpi)
  - Figure_SX_tpsa_shap_scatter.tiff  (600 dpi)

Notes
-----
- Run after 21_stage5_3_shap_physchem_direction_v1.py and
  25_extract_top_scaffolds.py have completed successfully.
- The Pearson correlation coefficient shown on the figure corresponds to
  the full training pool (r ≈ +0.62 as reported in the manuscript).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]  # C:\JMGM\scripts\modeling\ -> C:\JMGM

RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures" / "supplementary"

SHAP_PER_COMPOUND_CANDIDATES = [
    RESULTS_DIR / "interpretability" / "shap_physchem_per_compound_v1.csv.gz",
    BASE_DIR / "04_results" / "interpretability" / "shap_physchem_per_compound_v1.csv.gz",
    BASE_DIR / "04_results" / "interpretability" / "shap_physchem_per_compound_v1.csv.gz",
]

SCAFFOLD_MAP_CANDIDATES = [
    RESULTS_DIR / "scaffold_analysis" / "scaffold_top10_mapping_v1.csv",
    BASE_DIR / "04_results" / "scaffold_analysis" / "scaffold_top10_mapping_v1.csv",
    BASE_DIR / "results" / "scaffold_analysis" / "scaffold_top10_mapping_v1.csv",
]

OUT_PNG  = FIGURES_DIR / "Figure_SX_tpsa_shap_scatter.png"
OUT_TIFF = FIGURES_DIR / "Figure_SX_tpsa_shap_scatter.tiff"


# ============================================================
# Configuration
# ============================================================

# Scaffolds present in the training pool (S2/S3/S9/S10 are external/OOD)
TRAINING_SCAFFOLDS = ["S1", "S4", "S5", "S6", "S7", "S8"]

# Colour palette — distinct, print-safe colours for each scaffold group
SCAFFOLD_COLORS: dict[str, str] = {
    "S1": "#E41A1C",   # red
    "S4": "#FF7F00",   # orange
    "S5": "#4DAF4A",   # green
    "S6": "#984EA3",   # purple
    "S7": "#377EB8",   # blue
    "S8": "#A65628",   # brown
    "Other (training)": "#AAAAAA",   # grey
}

OTHER_LABEL = "Other (training)"

# Figure dimensions (in inches) — consistent with pipeline figure style
FIG_WIDTH  = 6.4
FIG_HEIGHT = 5.2
DPI_SAVE   = 600


# ============================================================
# Helpers
# ============================================================

def first_existing(paths: list[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Required input file not found. Checked:\n"
        + "\n".join(f"  - {p}" for p in paths)
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load inputs ----
    shap_path    = first_existing(SHAP_PER_COMPOUND_CANDIDATES)
    scaffold_path = first_existing(SCAFFOLD_MAP_CANDIDATES)

    shap_df = pd.read_csv(shap_path)
    scaff_df = pd.read_csv(scaffold_path)

    # Validate required columns
    for col in ["Compound_ID", "TPSA", "SHAP_TPSA"]:
        if col not in shap_df.columns:
            raise ValueError(f"SHAP file missing required column: {col}")

    for col in ["Compound_ID", "ScaffoldCode"]:
        if col not in scaff_df.columns:
            raise ValueError(f"Scaffold mapping file missing required column: {col}")

    # ---- Merge scaffold labels ----
    # Keep only training-pool scaffolds for the legend
    training_scaff = scaff_df[
        scaff_df["ScaffoldCode"].isin(TRAINING_SCAFFOLDS)
    ][["Compound_ID", "ScaffoldCode"]].drop_duplicates("Compound_ID")

    df = shap_df[["Compound_ID", "TPSA", "SHAP_TPSA"]].merge(
        training_scaff, on="Compound_ID", how="left"
    )
    df["ScaffoldCode"] = df["ScaffoldCode"].fillna(OTHER_LABEL)

    # ---- Pearson r for full training pool ----
    r_val, p_val = stats.pearsonr(df["TPSA"], df["SHAP_TPSA"])

    # ---- Plot ----
    plt.rcParams.update({
        "font.family":      "Arial",
        "font.size":        11,
        "axes.labelsize":   12,
        "axes.titlesize":   12,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "legend.fontsize":   9,
    })

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # Draw "Other" first so named scaffolds sit on top
    plot_order = [OTHER_LABEL] + TRAINING_SCAFFOLDS

    for group in plot_order:
        sub = df[df["ScaffoldCode"] == group]
        if sub.empty:
            continue

        color  = SCAFFOLD_COLORS.get(group, "#AAAAAA")
        marker = "o"
        zorder = 2 if group == OTHER_LABEL else 3
        alpha  = 0.55 if group == OTHER_LABEL else 0.90
        size   = 55  if group == OTHER_LABEL else 75
        lw     = 0.3 if group == OTHER_LABEL else 0.6

        ax.scatter(
            sub["TPSA"],
            sub["SHAP_TPSA"],
            c=color,
            marker=marker,
            s=size,
            alpha=alpha,
            edgecolors="black",
            linewidths=lw,
            zorder=zorder,
            label=group,
        )

    # OLS trend line (full dataset)
    x_sorted = np.sort(df["TPSA"].values)
    slope, intercept, *_ = stats.linregress(df["TPSA"], df["SHAP_TPSA"])
    ax.plot(
        x_sorted,
        slope * x_sorted + intercept,
        color="black",
        linewidth=1.2,
        linestyle="--",
        zorder=1,
        label=None,
    )

    # Annotation: Pearson r
    p_str = "< 0.001" if p_val < 0.001 else f"= {p_val:.3f}"
    ax.text(
        0.05, 0.95,
        f"$r$ = {r_val:.2f},  $p$ {p_str}\n$n$ = {len(df)}",
        transform=ax.transAxes,
        fontsize=9,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.85),
    )

    # Axes
    ax.set_xlabel("TPSA (Å²)")
    ax.set_ylabel("SHAP contribution (TPSA)")
    ax.axhline(0, color="0.6", linewidth=0.8, linestyle=":")

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.45)

    # Legend — named scaffolds first, then Other
    handles, labels = ax.get_legend_handles_labels()
    label_order = TRAINING_SCAFFOLDS + [OTHER_LABEL]
    order_map = {lbl: i for i, lbl in enumerate(label_order)}
    paired = sorted(zip(labels, handles), key=lambda x: order_map.get(x[0], 99))
    sorted_labels, sorted_handles = zip(*paired) if paired else ([], [])

    ax.legend(
        sorted_handles,
        sorted_labels,
        title="Scaffold class",
        title_fontsize=9,
        loc="lower right",
        framealpha=0.9,
        edgecolor="0.7",
    )

    fig.tight_layout()

    # ---- Save ----
    fig.savefig(OUT_PNG,  dpi=DPI_SAVE, bbox_inches="tight")
    fig.savefig(OUT_TIFF, dpi=DPI_SAVE, bbox_inches="tight")
    plt.close(fig)

    print("Figure saved successfully.")
    print(f"  PNG:  {OUT_PNG}")
    print(f"  TIFF: {OUT_TIFF}")
    print(f"  n compounds: {len(df)}")
    print(f"  Pearson r (TPSA vs SHAP_TPSA): {r_val:.3f}  p {p_str}")
    print(f"  Scaffold groups in plot: {df['ScaffoldCode'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
