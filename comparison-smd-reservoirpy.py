import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ============================================================================
# IEEE DOUBLE-COLUMN PUBLICATION STYLE
# ============================================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600
})

# ============================================================================
# 0. Define input/output directories
# ============================================================================

csv_location = "./csv/"
result_plot_location = "./plots/"

# ============================================================================
# 1. Map CSV files to model labels
# ============================================================================

file_mapping = {
    "SMD-DeepRC-no-reservoirpy.csv": "ReservoirPy",
    "SMD-DeepRC-ok.csv": "LYRA"
}

dfs = []

for file, label in file_mapping.items():
    try:
        df = pd.read_csv(csv_location + file)
        df["Model_Label"] = label
        dfs.append(df)

    except FileNotFoundError:
        print(f"Warning: file '{file}' not found. Skipping.")

if not dfs:
    print("No CSV files found. Execution terminated.")
    exit()

df_all = pd.concat(dfs, ignore_index=True)

# ============================================================================
# 2. Filter data for the selected experiment
# ============================================================================

wp_name = "WP1_HighDiv_6_Machines"

df_wp1 = df_all[
    df_all["Name"].str.contains(
        wp_name,
        na=False,
        case=False
    )
].copy()

if df_wp1.empty:
    print(f"No data found for experiment '{wp_name}'.")
    exit()

# ============================================================================
# 3. Replace NaN values with zero
# ============================================================================

cols_to_check = [
    "accuracy_regime",
    "rmse_forecast_denorm",
    "rmse_stability",
    "duration_sec"
]

for col in cols_to_check:
    if col in df_wp1.columns:
        df_wp1[col] = df_wp1[col].fillna(0.0)

# ============================================================================
# 4. Handle standard deviation columns
# ============================================================================

std_columns = [
    "accuracy_regime_std",
    "rmse_forecast_std",
    "rmse_stability_std",
    "duration_sec_std"
]

for col in std_columns:
    if col not in df_wp1.columns:
        df_wp1[col] = 0.0
    else:
        df_wp1[col] = df_wp1[col].fillna(0.0)

# ============================================================================
# 5. Aggregate results by model
# ============================================================================

df_grouped = (
    df_wp1
    .groupby("Model_Label")
    .mean(numeric_only=True)
    .reset_index()
)

# ============================================================================
# 6. Metrics configuration
# (metric column, std column, log scale)
# ============================================================================

metrics = {
    "Accuracy (↑)\nRegime Classification":
        ("accuracy_regime", "accuracy_regime_std", False),

    "RMSE (↓)\nForecasting":
        ("rmse_forecast_denorm", "rmse_forecast_std", False),

    "RMSE (↓)\nReconstruction":
        ("rmse_stability", "rmse_stability_std", False),

    "Duration (↓)\nHP Search [s]":
        ("duration_sec", "duration_sec_std", True)
}

# ============================================================================
# 7. Visual configuration
# ============================================================================

models = ["ReservoirPy", "LYRA"]

# Publication-friendly colors
colors = ["#ECECFF", "#55A868"]

# Hatches for grayscale printing
hatches = ["...", "\\\\\\"]

x_pos = np.arange(len(models))
bar_width = 0.55

# IEEE double-column width (~7.16 inches)
fig, axes = plt.subplots(
    1,
    4,
    figsize=(5.5, 2.2)
)

# ============================================================================
# 8. Generate subplots
# ============================================================================

for i, (title, (val_col, std_col, use_log)) in enumerate(metrics.items()):

    ax = axes[i]

    means = []
    stds = []

    for model in models:

        subset = df_grouped[df_grouped["Model_Label"] == model]

        if not subset.empty:

            means.append(subset[val_col].values[0])
            stds.append(subset[std_col].values[0])

        else:

            means.append(0.0)
            stds.append(0.0)

    # ------------------------------------------------------------------------
    # Draw bars
    # ------------------------------------------------------------------------

    bars = ax.bar(
        x_pos,
        means,
        bar_width,
        yerr=stds,
        color=colors,
        edgecolor="black",
        linewidth=0.9,
        capsize=4,
        error_kw={
            "elinewidth": 1.0,
            "capthick": 1.0
        }
    )

    # Apply hatches
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    # ------------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------------

    ax.set_title(
        title,
        fontweight="bold",
        pad=10
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(models)

    if use_log:
        ax.set_yscale("log")

    # Light publication-style grid
    ax.grid(
        axis="y",
        linestyle=":",
        linewidth=0.5,
        alpha=0.5
    )

    ax.set_axisbelow(True)

    # ------------------------------------------------------------------------
    # Add value annotations
    # ------------------------------------------------------------------------

    for idx, mean_val in enumerate(means):

        if mean_val <= 0:
            continue

        if use_log:
            label = f"{mean_val:.0f}"
        else:
            label = f"{mean_val:.3f}"

        ax.annotate(
            label,
            xy=(x_pos[idx], mean_val),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold"
        )

    # ------------------------------------------------------------------------
    # Adjust limits
    # ------------------------------------------------------------------------

    if not use_log:

        ymax = max(np.array(means) + np.array(stds))

        if ymax > 0:
            ax.set_ylim(0, ymax * 1.18)

    else:
        ymin = min(np.array(means) - np.array(stds))
        ymax = max(np.array(means) + np.array(stds))

        if ymin <= 0:
            ymin = 1e-3

        ax.set_ylim(ymin * 0.8, ymax * 2)

    # Clean upper/right borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if i == 0:
        ax.set_ylabel("Metric Value")

# ============================================================================
# 9. Global legend
# ============================================================================

legend_elements = [
    Patch(
        facecolor=colors[0],
        edgecolor="black",
        hatch=hatches[0],
        label="ReservoirPy"
    ),
    Patch(
        facecolor=colors[1],
        edgecolor="black",
        hatch=hatches[1],
        label="LYRA"
    )
]

fig.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.10),
    ncol=2,
    frameon=False
)

# ============================================================================
# 10. Final layout and export
# ============================================================================

plt.subplots_adjust(
    top=0.78,
    wspace=0.45
)

output_filename = (
    result_plot_location +
    "WP1_HighDiv_Head2Head_IEEE.pdf"
)

plt.savefig(
    output_filename,
    format="pdf",
    bbox_inches="tight"
)

plt.close()

print(
    f"✅ IEEE-ready vector figure successfully saved:\n"
    f"{output_filename}"
)