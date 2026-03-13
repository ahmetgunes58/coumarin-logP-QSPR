#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# -------------------------------------------------
# PATHS
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures" / "supplementary"

EXTERNAL_VAL_DIR = RESULTS_DIR / "external_validation"


def find_latest_predictions_file() -> Path:
    if not EXTERNAL_VAL_DIR.exists():
        raise FileNotFoundError(f"Directory not found:\n{EXTERNAL_VAL_DIR}")

    candidates = sorted(EXTERNAL_VAL_DIR.glob("*/predictions_external_final.csv"))
    if not candidates:
        raise FileNotFoundError(
            "No predictions_external_final.csv found under:\n"
            f"{EXTERNAL_VAL_DIR}"
        )

    return candidates[-1]


file_path = find_latest_predictions_file()

if not file_path.exists():
    raise FileNotFoundError(f"CSV not found:\n{file_path}")


# -------------------------------------------------
# READ DATA
# -------------------------------------------------
df = pd.read_csv(file_path)

print("Columns:", df.columns.tolist())
print("Total rows:", len(df))


# -------------------------------------------------
# ONLY EXT_A + EXT_B
# -------------------------------------------------
df = df[df["set_role"].isin(["EXT_A", "EXT_B"])].copy()

print("\nModel names:")
print(df["model"].unique())


# -------------------------------------------------
# FINAL MODEL FILTER
# -------------------------------------------------
final_model_name = "XGB_ECFP+PhysChem"
df = df[df["model"] == final_model_name].copy()

print("\nRows after filter:", len(df))

if df.empty:
    raise ValueError(
        "Filtered dataframe is empty. Check model name. "
        f"Available models: {pd.read_csv(file_path)['model'].unique()}"
    )


# -------------------------------------------------
# RESIDUAL CALCULATION
# -------------------------------------------------
df["residual"] = df["y_pred"] - df["y_true"]

print("\nResidual summary:")
print(df["residual"].describe())


# -------------------------------------------------
# VISUAL SETTINGS
# -------------------------------------------------
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

fig, ax = plt.subplots(figsize=(6.4, 4.8))

# Scatter
ax.scatter(
    df["y_true"],
    df["residual"],
    s=75,
    alpha=0.90,
    edgecolors="black",
    linewidths=0.5
)

# Zero line
ax.axhline(y=0, linestyle="--", linewidth=1.2)

# Optional visual guides
ax.axhline(y=0.5, linestyle=":", linewidth=0.8)
ax.axhline(y=-0.5, linestyle=":", linewidth=0.8)

# Labels
ax.set_xlabel("Experimental logP")
ax.set_ylabel("Residual (Predicted − Experimental logP)")

# Limits
x_min = df["y_true"].min()
x_max = df["y_true"].max()
y_min = df["residual"].min()
y_max = df["residual"].max()

x_range = x_max - x_min
y_range = y_max - y_min

x_pad = 0.10 * x_range if x_range > 0 else 0.2
y_pad = 0.15 * y_range if y_range > 0 else 0.2

ax.set_xlim(x_min - x_pad, x_max + x_pad)
ax.set_ylim(y_min - y_pad, y_max + y_pad)

# Style cleanup
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

plt.tight_layout()


# -------------------------------------------------
# SAVE
# -------------------------------------------------
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

out_png = FIGURES_DIR / "Figure_S6_residual_plot.png"
out_tiff = FIGURES_DIR / "Figure_S6_residual_plot.tiff"

fig.savefig(out_png, dpi=600, bbox_inches="tight")
fig.savefig(out_tiff, dpi=600, bbox_inches="tight")

print("\nSaved:")
print(out_png)
print(out_tiff)

plt.show()