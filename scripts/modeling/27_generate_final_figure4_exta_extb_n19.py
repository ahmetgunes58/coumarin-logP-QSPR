from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "results" / "external_validation" / "final_EXTA_EXTB_n19" / "figure4_external_hybridxgb_exta_extb_n19_data.csv"
METRICS_PATH = ROOT / "results" / "external_validation" / "final_EXTA_EXTB_n19" / "figure4_external_hybridxgb_exta_extb_n19_metrics.csv"

MAIN_OUT = ROOT / "figures" / "main" / "stage7" / "20260309_102934"
EVIDENCE_OUT = ROOT / "results" / "external_validation" / "final_EXTA_EXTB_n19"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def standardize_ext_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = (
                out[col]
                .astype(str)
                .str.replace("EXT_A+EXT_B", "EXT-A + EXT-B", regex=False)
                .str.replace("EXT_A + EXT_B", "EXT-A + EXT-B", regex=False)
                .str.replace("EXT_A", "EXT-A", regex=False)
                .str.replace("EXT_B", "EXT-B", regex=False)
            )
    return out


def infer_column(columns, include_terms, exclude_terms=()):
    cols = list(columns)
    for col in cols:
        low = col.lower()
        if all(term in low for term in include_terms) and not any(term in low for term in exclude_terms):
            return col
    return None


def infer_observed_predicted_columns(df: pd.DataFrame):
    columns = df.columns

    observed_candidates = [
        "Experimental_logP",
        "experimental_logP",
        "experimental_logp",
        "Observed_logP",
        "observed_logP",
        "observed_logp",
        "y_true",
        "y_exp",
        "exp_logP",
        "exp_logp",
    ]

    predicted_candidates = [
        "Predicted_logP",
        "predicted_logP",
        "predicted_logp",
        "Pred_logP",
        "pred_logP",
        "pred_logp",
        "y_pred",
        "prediction",
    ]

    obs = next((c for c in observed_candidates if c in columns), None)
    pred = next((c for c in predicted_candidates if c in columns), None)

    if obs is None:
        obs = infer_column(columns, ["experimental"])
    if obs is None:
        obs = infer_column(columns, ["observed"])
    if obs is None:
        obs = infer_column(columns, ["exp"], exclude_terms=("pred", "resid"))

    if pred is None:
        pred = infer_column(columns, ["predicted"])
    if pred is None:
        pred = infer_column(columns, ["pred"], exclude_terms=("minus", "bias", "resid"))

    if obs is None or pred is None:
        raise RuntimeError(
            "Could not infer observed/predicted columns.\n"
            f"Columns found: {list(columns)}\n"
            "Please rename the observed column to Experimental_logP and the predicted column to Predicted_logP."
        )

    return obs, pred


def add_identity_line(ax, y_obs, y_pred):
    values = np.concatenate([np.asarray(y_obs, dtype=float), np.asarray(y_pred, dtype=float)])
    low = np.nanmin(values)
    high = np.nanmax(values)
    pad = (high - low) * 0.08 if high > low else 0.2
    lims = [low - pad, high + pad]
    ax.plot(lims, lims, linestyle="--", linewidth=1.2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)


def main():
    df = standardize_ext_labels(read_csv(DATA_PATH))
    metrics = standardize_ext_labels(read_csv(METRICS_PATH))

    DATA_PATH.write_text(df.to_csv(index=False), encoding="utf-8")
    METRICS_PATH.write_text(metrics.to_csv(index=False), encoding="utf-8")

    obs_col, pred_col = infer_observed_predicted_columns(df)

    y_obs = pd.to_numeric(df[obs_col], errors="coerce")
    y_pred = pd.to_numeric(df[pred_col], errors="coerce")
    valid = y_obs.notna() & y_pred.notna()

    df_plot = df.loc[valid].copy()
    y_obs = y_obs.loc[valid]
    y_pred = y_pred.loc[valid]

    if len(df_plot) != 19:
        raise RuntimeError(f"Expected n=19 for EXT-A + EXT-B, but found n={len(df_plot)}.")

    row = metrics.iloc[0]
    rmse = float(row["RMSE"])
    mae = float(row["MAE"])
    ccc = float(row["CCC"])
    bias = float(row["Bias_pred_minus_exp"])

    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
    })

    fig, ax = plt.subplots(figsize=(3.6, 3.4))

    if "subset" in df_plot.columns:
        for subset, sub in df_plot.groupby("subset"):
            label = str(subset).replace("EXT_A", "EXT-A").replace("EXT_B", "EXT-B")
            ax.scatter(sub[obs_col], sub[pred_col], s=24, alpha=0.82, label=label)
        ax.legend(frameon=False, loc="best")
    else:
        ax.scatter(y_obs, y_pred, s=24, alpha=0.82)

    add_identity_line(ax, y_obs, y_pred)

    ax.set_xlabel("Experimental logP")
    ax.set_ylabel("Predicted logP")
    ax.grid(True, alpha=0.22, linewidth=0.6)

    text = (
        "EXT-A + EXT-B, n = 19\n"
        f"RMSE = {rmse:.3f}\n"
        f"MAE = {mae:.3f}\n"
        f"CCC = {ccc:.3f}\n"
        f"Bias = {bias:+.3f}"
    )

    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.7", alpha=0.92),
    )

    MAIN_OUT.mkdir(parents=True, exist_ok=True)
    EVIDENCE_OUT.mkdir(parents=True, exist_ok=True)

    main_base = MAIN_OUT / "Fig04_pred_vs_obs"
    evidence_base = EVIDENCE_OUT / "Figure_04_pred_vs_observed_external_HybridXGB_EXTA_EXTB_n19"

    for base in [main_base, evidence_base]:
        fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")

    plt.close(fig)

    print("Final Figure 4 generated successfully.")
    print(f"Observed column: {obs_col}")
    print(f"Predicted column: {pred_col}")
    print(f"n = {len(df_plot)}")
    print(f"RMSE = {rmse:.3f}, MAE = {mae:.3f}, CCC = {ccc:.3f}, Bias = {bias:+.3f}")


if __name__ == "__main__":
    main()