#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT_CSV = Path("csv/ett_variability_results/local_variability_dataset_summary.csv")

OUTPUT_PDF = Path("plots/ett_multiscale_local_variability.pdf")
OUTPUT_PNG = Path("plots/ett_multiscale_local_variability.png")

DATASET_ORDER = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
WINDOW_ORDER = [24, 168]

METRICS = {
    "local_std": "Local STD",
    "rms_increment": "RMS Increment",
}


COLORS = {
    24: "#4C78A8",
    168: "#D5AC82",
}




# ---------------------------------------------------------------------
# IEEE style
# ---------------------------------------------------------------------

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
# Load data
# ---------------------------------------------------------------------

df = pd.read_csv(INPUT_CSV)

required_columns = {
    "dataset",
    "resolution",
    "window_hours",
    "feature",
    "metric",
    "median",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
}

missing = required_columns.difference(df.columns)

if missing:
    raise ValueError(
        "Missing columns:\n" + "\n".join(sorted(missing))
    )


plot_df = df.loc[
    (df["resolution"] == "hourly_aligned")
    & (df["feature"] == "ALL_MEDIAN")
    & (df["window_hours"].isin(WINDOW_ORDER))
    & (df["metric"].isin(METRICS))
].copy()

if plot_df.empty:
    raise ValueError("No rows available for plotting.")


expected = pd.MultiIndex.from_product(
    [
        METRICS.keys(),
        DATASET_ORDER,
        WINDOW_ORDER,
    ],
    names=["metric", "dataset", "window_hours"],
)

available = pd.MultiIndex.from_frame(
    plot_df[["metric", "dataset", "window_hours"]]
)

missing_combinations = expected.difference(available)

if len(missing_combinations) > 0:
    raise ValueError(
        "Missing combinations:\n"
        + "\n".join(map(str, missing_combinations))
    )


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(7.16, 3.2),
    constrained_layout=True,
)

x = np.arange(len(DATASET_ORDER))
bar_width = 0.34


for panel_idx, (metric, title) in enumerate(METRICS.items()):

    ax = axes[panel_idx]

    panel_max = 0

    cached = {}

    for window_idx, window_hours in enumerate(WINDOW_ORDER):

        subset = (
            plot_df.loc[
                (plot_df["metric"] == metric)
                & (plot_df["window_hours"] == window_hours)
            ]
            .set_index("dataset")
            .reindex(DATASET_ORDER)
        )

        medians = subset["median"].to_numpy(dtype=float)

        ci_low = subset["bootstrap_ci_low"].to_numpy(dtype=float)
        ci_high = subset["bootstrap_ci_high"].to_numpy(dtype=float)

        cached[window_hours] = medians

        lower = np.maximum(medians - ci_low, 0)
        upper = np.maximum(ci_high - medians, 0)

        yerr = np.vstack([lower, upper])

        panel_max = max(
            panel_max,
            np.max(ci_high),
        )

        positions = (
            x
            + (window_idx - 0.5)
            * bar_width
        )

        ax.bar(
            positions,
            medians,
            width=bar_width,
            color=COLORS[window_hours],
            edgecolor="0.3",
            linewidth=0.4,
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

        # Median marker
        ax.scatter(
            positions,
            medians,
            color="black",
            s=10,
            zorder=4,
        )

    # ---------------------------------------------------------
    # Delta annotations
    # ---------------------------------------------------------

    med_24 = cached[24]
    med_168 = cached[168]

    for i in range(len(DATASET_ORDER)):

        delta = med_168[i] - med_24[i]

        y_text = max(
            med_24[i],
            med_168[i],
        )

        # ax.text(
        #     x[i],
        #     y_text + 0.03 * panel_max,
        #     f"{delta:+.2f}",
        #     ha="center",
        #     va="bottom",
        #     fontsize=7,
        #     color="0.25",
        # )

    # ---------------------------------------------------------
    # Axes styling
    # ---------------------------------------------------------

    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_ORDER)

    ax.set_ylabel("Robust-scaled value")

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

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylim(
        0,
        panel_max * 1.18,
    )

    if panel_idx == 1:
        ax.legend(
            loc="upper right",
            frameon=True,
            edgecolor="gray",
            title="Window",
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

print(f"Saved PDF : {OUTPUT_PDF.resolve()}")
print(f"Saved PNG : {OUTPUT_PNG.resolve()}")