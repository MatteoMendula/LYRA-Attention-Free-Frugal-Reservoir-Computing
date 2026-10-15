#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

VARIABILITY_CSV = Path(
    "csv/ett_variability_results/"
    "local_variability_dataset_summary.csv"
)

SPECTRAL_RADIUS_CSV = Path(
    "csv/ETT-DeepRCvsMoment.csv"
)

OUTPUT_PDF = Path(
    "plots/ett_variability_and_spectral_radius.pdf"
)

OUTPUT_PNG = Path(
    "plots/ett_variability_and_spectral_radius.png"
)


DATASET_ORDER = [
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
]

WINDOW_ORDER = [24, 168]

METRICS = {
    "local_std": "Local STD",
    "rms_increment": "RMS Increment",
}

WINDOW_COLORS = {
    24: "#4C78A8",
    168: "#D5AC82",
}

DATASET_COLORS = {
    "ETTh1": "#54A24B",
    "ETTh2": "#54A24B",
    "ETTm1": "#54A24B",
    "ETTm2": "#54A24B",
}


# ---------------------------------------------------------------------
# Optional filtering of spectral-radius results
# ---------------------------------------------------------------------
#
# Ideally, the boxplot should contain the final spectral-radius values
# selected or accepted by the HP search, rather than every candidate
# explored during optimization.
#
# If the CSV contains a column identifying the selected configurations,
# set the following variables accordingly.
#
# Example:
#
# SR_SELECTION_COLUMN = "status"
# SR_SELECTION_VALUE = "selected"
#
# If no filtering is needed, leave both as None.
# ---------------------------------------------------------------------

SR_SELECTION_COLUMN = None
SR_SELECTION_VALUE = None


# ---------------------------------------------------------------------
# Unit spectral-radius reference
# ---------------------------------------------------------------------

UNIT_SR_REFERENCE = 1.0


# ---------------------------------------------------------------------
# IEEE-oriented style
# ---------------------------------------------------------------------

sns.set_theme(style="whitegrid")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
})


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def normalize_dataset_names(series):
    """
    Normalize ETT dataset labels.

    Examples
    --------
    ETTh1.csv -> ETTh1
    ./data/ETTm2.csv -> ETTm2
    ETTm1 -> ETTm1
    """

    normalized = (
        series
        .astype(str)
        .str.strip()
        .str.replace("\\", "/", regex=False)
        .str.split("/")
        .str[-1]
        .str.replace(".csv", "", regex=False)
    )

    return normalized


def validate_columns(df, required_columns, file_name):
    """
    Verify that all required columns are available.
    """

    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(
            f"{file_name} is missing the following columns:\n"
            + "\n".join(sorted(missing))
        )


# ---------------------------------------------------------------------
# Load local-variability results
# ---------------------------------------------------------------------

variability_df = pd.read_csv(VARIABILITY_CSV)

required_variability_columns = {
    "dataset",
    "resolution",
    "window_hours",
    "feature",
    "metric",
    "median",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
}

validate_columns(
    variability_df,
    required_variability_columns,
    VARIABILITY_CSV,
)

variability_df["dataset"] = normalize_dataset_names(
    variability_df["dataset"]
)

plot_df = variability_df.loc[
    (variability_df["resolution"] == "hourly_aligned")
    & (variability_df["feature"] == "ALL_MEDIAN")
    & (variability_df["window_hours"].isin(WINDOW_ORDER))
    & (variability_df["metric"].isin(METRICS))
].copy()

if plot_df.empty:
    raise ValueError(
        "No local-variability rows are available for plotting. "
        "Expected resolution='hourly_aligned', "
        "feature='ALL_MEDIAN', windows 24/168 hours, "
        "and metrics local_std/rms_increment."
    )


# ---------------------------------------------------------------------
# Validate all variability combinations
# ---------------------------------------------------------------------

expected = pd.MultiIndex.from_product(
    [
        METRICS.keys(),
        DATASET_ORDER,
        WINDOW_ORDER,
    ],
    names=[
        "metric",
        "dataset",
        "window_hours",
    ],
)

available = pd.MultiIndex.from_frame(
    plot_df[
        [
            "metric",
            "dataset",
            "window_hours",
        ]
    ]
)

missing_combinations = expected.difference(available)

if len(missing_combinations) > 0:
    raise ValueError(
        "Missing variability combinations:\n"
        + "\n".join(map(str, missing_combinations))
    )


# ---------------------------------------------------------------------
# Load spectral-radius results
# ---------------------------------------------------------------------

spectral_df = pd.read_csv(SPECTRAL_RADIUS_CSV)

required_spectral_columns = {
    "dataset",
    "spectral_radius",
}

validate_columns(
    spectral_df,
    required_spectral_columns,
    SPECTRAL_RADIUS_CSV,
)

spectral_df["dataset"] = normalize_dataset_names(
    spectral_df["dataset"]
)

spectral_df["spectral_radius"] = pd.to_numeric(
    spectral_df["spectral_radius"],
    errors="coerce",
)

spectral_df = spectral_df.dropna(
    subset=[
        "dataset",
        "spectral_radius",
    ]
).copy()


# Keep only the four ETT domains.
spectral_df = spectral_df.loc[
    spectral_df["dataset"].isin(DATASET_ORDER)
].copy()


# Optional filtering of selected HP configurations.
if SR_SELECTION_COLUMN is not None:

    if SR_SELECTION_COLUMN not in spectral_df.columns:
        raise ValueError(
            f"Selection column '{SR_SELECTION_COLUMN}' "
            f"does not exist in {SPECTRAL_RADIUS_CSV}."
        )

    spectral_df = spectral_df.loc[
        spectral_df[SR_SELECTION_COLUMN]
        == SR_SELECTION_VALUE
    ].copy()


if spectral_df.empty:
    raise ValueError(
        "No valid spectral-radius observations are available "
        "after preprocessing and optional filtering."
    )


# Use an ordered categorical variable to guarantee the same order
# across all three panels.
spectral_df["dataset"] = pd.Categorical(
    spectral_df["dataset"],
    categories=DATASET_ORDER,
    ordered=True,
)

spectral_df = spectral_df.sort_values("dataset")


# ---------------------------------------------------------------------
# Print sample size for reproducibility
# ---------------------------------------------------------------------

sr_counts = (
    spectral_df
    .groupby("dataset", observed=False)
    .size()
    .reindex(DATASET_ORDER)
)

print("\nSpectral-radius observations included in the boxplot:")

for dataset, count in sr_counts.items():
    print(f"  {dataset}: n = {count}")


# ---------------------------------------------------------------------
# Create output directory
# ---------------------------------------------------------------------

OUTPUT_PDF.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PNG.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------

fig, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(10.5, 3.25),
    constrained_layout=True,
    gridspec_kw={
        "width_ratios": [1, 1, 1.1],
    },
)

x = np.arange(len(DATASET_ORDER))

bar_width = 0.34


# ---------------------------------------------------------------------
# Panels (a) and (b): local variability
# ---------------------------------------------------------------------

for panel_idx, (metric, title) in enumerate(METRICS.items()):

    ax = axes[panel_idx]

    panel_max = 0.0

    for window_idx, window_hours in enumerate(WINDOW_ORDER):

        subset = (
            plot_df.loc[
                (plot_df["metric"] == metric)
                & (
                    plot_df["window_hours"]
                    == window_hours
                )
            ]
            .set_index("dataset")
            .reindex(DATASET_ORDER)
        )

        medians = subset["median"].to_numpy(
            dtype=float
        )

        ci_low = subset[
            "bootstrap_ci_low"
        ].to_numpy(dtype=float)

        ci_high = subset[
            "bootstrap_ci_high"
        ].to_numpy(dtype=float)

        lower_errors = np.maximum(
            medians - ci_low,
            0,
        )

        upper_errors = np.maximum(
            ci_high - medians,
            0,
        )

        yerr = np.vstack([
            lower_errors,
            upper_errors,
        ])

        panel_max = max(
            panel_max,
            float(np.nanmax(ci_high)),
        )

        positions = (
            x
            + (
                window_idx
                - (len(WINDOW_ORDER) - 1) / 2
            )
            * bar_width
        )

        ax.bar(
            positions,
            medians,
            width=bar_width,
            color=WINDOW_COLORS[window_hours],
            edgecolor="0.30",
            linewidth=0.5,
            label=f"{window_hours} h",
            zorder=2,
        )

        ax.errorbar(
            positions,
            medians,
            yerr=yerr,
            fmt="none",
            color="black",
            ecolor="black",
            capsize=2.0,
            elinewidth=0.8,
            capthick=0.8,
            zorder=3,
        )

        # Explicit marker for the median.
        ax.scatter(
            positions,
            medians,
            color="black",
            edgecolor="black",
            linewidth=0.2,
            s=10,
            zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_ORDER)

    ax.set_ylabel(
        "Robust-scaled value"
    )

    ax.set_title(
        f"({chr(97 + panel_idx)}) {title}"
    )

    ax.grid(
        axis="y",
        linestyle=":",
        linewidth=0.6,
        alpha=0.5,
        zorder=0,
    )

    ax.grid(
        axis="x",
        visible=False,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylim(
        0,
        panel_max * 1.16,
    )


# Add the window legend only once.
axes[1].legend(
    loc="upper right",
    frameon=True,
    edgecolor="0.70",
    title="Window",
    title_fontsize=8,
)


# ---------------------------------------------------------------------
# Panel (c): selected spectral-radius distributions
# ---------------------------------------------------------------------

ax_sr = axes[2]


# Boxplot
sns.boxplot(
    data=spectral_df,
    x="dataset",
    y="spectral_radius",
    order=DATASET_ORDER,
    palette=DATASET_COLORS,
    width=0.48,
    showfliers=False,
    saturation=0.75,
    linewidth=0.8,
    boxprops={
        "edgecolor": "0.20",
        "alpha": 0.80,
    },
    medianprops={
        "color": "black",
        "linewidth": 1.2,
    },
    whiskerprops={
        "color": "0.25",
        "linewidth": 0.8,
    },
    capprops={
        "color": "0.25",
        "linewidth": 0.8,
    },
    ax=ax_sr,
)


# Individual selected values
sns.stripplot(
    data=spectral_df,
    x="dataset",
    y="spectral_radius",
    order=DATASET_ORDER,
    color="black",
    alpha=0.48,
    jitter=0.16,
    size=2.8,
    linewidth=0,
    ax=ax_sr,
    zorder=3,
)


# Reference at rho = 1.
ax_sr.axhline(
    y=UNIT_SR_REFERENCE,
    color="#605858",
    linestyle="--",
    linewidth=1.0,
    zorder=1,
)


# Determine a suitable position for the annotation.
sr_min = float(
    spectral_df["spectral_radius"].min()
)

sr_max = float(
    spectral_df["spectral_radius"].max()
)

sr_range = max(
    sr_max - sr_min,
    0.10,
)


ax_sr.annotate(
    "unit spectral-radius reference",
    xy=(
        len(DATASET_ORDER) - 0.55,
        UNIT_SR_REFERENCE,
    ),
    xytext=(0, 4),
    textcoords="offset points",
    ha="right",
    va="bottom",
    color="#605858",
    fontsize=6.8,
)


# Spectral-radius panel styling
ax_sr.set_title(
    "(c) Selected Spectral Radius"
)

ax_sr.set_xlabel("")

ax_sr.set_ylabel(
    "Effective spectral radius"
)

ax_sr.set_xticklabels(DATASET_ORDER)

ax_sr.grid(
    axis="y",
    linestyle=":",
    linewidth=0.6,
    alpha=0.5,
    zorder=0,
)

ax_sr.grid(
    axis="x",
    visible=False,
)

ax_sr.spines["top"].set_visible(False)
ax_sr.spines["right"].set_visible(False)


# Make sure the unit reference is visible even if all selected
# spectral-radius values lie entirely above or below one.
lower_limit = min(
    sr_min - 0.10 * sr_range,
    UNIT_SR_REFERENCE - 0.12 * sr_range,
)

upper_limit = max(
    sr_max + 0.12 * sr_range,
    UNIT_SR_REFERENCE + 0.16 * sr_range,
)

ax_sr.set_ylim(
    lower_limit,
    upper_limit,
)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

fig.savefig(
    OUTPUT_PDF,
    bbox_inches="tight",
    pad_inches=0.02,
)

fig.savefig(
    OUTPUT_PNG,
    bbox_inches="tight",
    pad_inches=0.02,
    dpi=300,
)

plt.close(fig)

print(f"\nSaved PDF: {OUTPUT_PDF.resolve()}")
print(f"Saved PNG: {OUTPUT_PNG.resolve()}")