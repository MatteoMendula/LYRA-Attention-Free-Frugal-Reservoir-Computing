#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IEEE-style Class Distribution Visualization for
SMD, ETT, and ECG5000 datasets.

Features
--------
- Percentage distributions instead of counts
- Original dataset-inspired colors
- Hatches for grayscale printing
- IEEE-style publication formatting
- High-quality PNG/PDF/EPS export
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ==========================================================
# Configuration
# ==========================================================

SMD_DIR = '../datasets/SMD/OmniAnomaly-master/ServerMachineDataset'
ETT_DIR = '../datasets/etth'
ECG_DIR = '../datasets/ECG5000'

ETT_N_CLUSTERS = 4

# Dataset colors (close to original palettes)
SMD_COLOR = "#440154"   # viridis
ETT_COLOR = "#B12A90"   # magma
ECG_COLOR = "#2A788E"   # crest

HATCHES = [
    '///',
    '\\\\',
    'xxx',
    '---',
    '+++',
    'ooo',
    '***'
]

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ==========================================================
# Utilities
# ==========================================================

def compute_distribution(labels):
    """
    Compute counts and percentages.
    """

    series = pd.Series(labels)

    counts = series.value_counts().sort_index()
    percentages = counts / counts.sum() * 100

    return {
        "counts": {
            str(int(k)): int(v)
            for k, v in counts.items()
        },
        "percentages": {
            str(int(k)): round(float(v), 4)
            for k, v in percentages.items()
        }
    }


def plot_percentage_distribution(
        ax,
        labels,
        title,
        xlabel,
        color):
    """
    IEEE-style percentage distribution.
    """

    labels = np.asarray(labels)

    unique_labels, counts = np.unique(
        labels,
        return_counts=True
    )

    percentages = counts / counts.sum() * 100

    bars = ax.bar(
        [str(x) for x in unique_labels],
        percentages,
        color=color,
        width=0.55,
        edgecolor='black',
        linewidth=1.1
    )

    for idx, (bar, pct) in enumerate(
            zip(bars, percentages)):

        bar.set_hatch(
            HATCHES[idx % len(HATCHES)]
        )

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{pct:.1f}%",
            ha='center',
            va='bottom',
            fontsize=8
        )

    ax.set_title(
        title,
        fontweight='bold'
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Percentage (%)")

    ax.grid(
        axis='y',
        linestyle='--',
        linewidth=0.6,
        alpha=0.5
    )

    ax.set_axisbelow(True)

    ymax = max(percentages) * 1.18

    ax.set_ylim(
        0,
        min(ymax, 105)
    )


# ==========================================================
# Main Function
# ==========================================================

def plot_class_distributions():

    class_distributions = {}

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(5, 7),
        constrained_layout=True
    )

    # ======================================================
    # SMD
    # ======================================================

    smd_labels_dir = os.path.join(
        SMD_DIR,
        "regime_labels"
    )

    smd_train_labels = []
    smd_test_labels = []

    if os.path.exists(smd_labels_dir):

        for file_name in os.listdir(
                smd_labels_dir):

            if not file_name.endswith(".npy"):
                continue

            labels = np.load(
                os.path.join(
                    smd_labels_dir,
                    file_name
                )
            )

            if file_name.startswith("train_"):
                smd_train_labels.extend(
                    labels.tolist()
                )

            elif file_name.startswith("test_"):
                smd_test_labels.extend(
                    labels.tolist()
                )

        class_distributions["SMD"] = {}

        if len(smd_train_labels) > 0:

            class_distributions["SMD"]["train"] = \
                compute_distribution(
                    smd_train_labels
                )

            plot_percentage_distribution(
                axes[0, 0],
                smd_train_labels,
                "SMD Train",
                "Regime Label",
                SMD_COLOR
            )

        if len(smd_test_labels) > 0:

            class_distributions["SMD"]["test"] = \
                compute_distribution(
                    smd_test_labels
                )

            plot_percentage_distribution(
                axes[0, 1],
                smd_test_labels,
                "SMD Test",
                "Regime Label",
                SMD_COLOR
            )

    # ======================================================
    # ETT
    # ======================================================

    ett_file = os.path.join(
        ETT_DIR,
        "ETTh1.csv"
    )

    if os.path.exists(ett_file):

        df_ett = pd.read_csv(
            ett_file
        )

        if "date" in df_ett.columns:
            df_ett = df_ett.drop(
                columns=["date"]
            )

        raw_data = df_ett.values.astype(
            np.float32
        )

        train_end_idx = int(
            len(raw_data) * 0.60
        )

        val_end_idx = int(
            len(raw_data) * 0.70
        )

        train_data = raw_data[:train_end_idx]
        test_data = raw_data[val_end_idx:]

        scaler = StandardScaler()

        train_scaled = scaler.fit_transform(
            train_data
        )

        test_scaled = scaler.transform(
            test_data
        )

        kmeans = KMeans(
            n_clusters=ETT_N_CLUSTERS,
            random_state=42,
            n_init=10
        )

        ett_train_labels = \
            kmeans.fit_predict(train_scaled)

        ett_test_labels = \
            kmeans.predict(test_scaled)

        class_distributions["ETT"] = {
            "train": compute_distribution(
                ett_train_labels
            ),
            "test": compute_distribution(
                ett_test_labels
            )
        }

        plot_percentage_distribution(
            axes[1, 0],
            ett_train_labels,
            "ETTh1 Train",
            f"Cluster Label (k={ETT_N_CLUSTERS})",
            ETT_COLOR
        )

        plot_percentage_distribution(
            axes[1, 1],
            ett_test_labels,
            "ETTh1 Test",
            f"Cluster Label (k={ETT_N_CLUSTERS})",
            ETT_COLOR
        )

    # ======================================================
    # ECG5000
    # ======================================================

    ecg_train_file = os.path.join(
        ECG_DIR,
        "ECG5000_TRAIN.tsv"
    )

    ecg_test_file = os.path.join(
        ECG_DIR,
        "ECG5000_TEST.tsv"
    )

    class_distributions["ECG5000"] = {}

    if os.path.exists(ecg_train_file):

        ecg_train_data = np.loadtxt(
            ecg_train_file,
            delimiter="\t"
        )

        ecg_train_labels = (
            ecg_train_data[:, 0]
            .astype(int) - 1
        )

        class_distributions[
            "ECG5000"
        ]["train"] = compute_distribution(
            ecg_train_labels
        )

        plot_percentage_distribution(
            axes[2, 0],
            ecg_train_labels,
            "ECG5000 Train",
            "Class Label",
            ECG_COLOR
        )

    if os.path.exists(ecg_test_file):

        ecg_test_data = np.loadtxt(
            ecg_test_file,
            delimiter="\t"
        )

        ecg_test_labels = (
            ecg_test_data[:, 0]
            .astype(int) - 1
        )

        class_distributions[
            "ECG5000"
        ]["test"] = compute_distribution(
            ecg_test_labels
        )

        plot_percentage_distribution(
            axes[2, 1],
            ecg_test_labels,
            "ECG5000 Test",
            "Class Label",
            ECG_COLOR
        )

    # ======================================================
    # Save JSON
    # ======================================================

    json_output = \
        "dataset_class_distributions.json"

    with open(json_output, "w") as f:
        json.dump(
            class_distributions,
            f,
            indent=4
        )

    print(
        f"Saved distributions to "
        f"'{json_output}'"
    )

    # ======================================================
    # Save Figures
    # ======================================================

    os.makedirs(
        "./plots",
        exist_ok=True
    )

    plt.savefig(
        "./plots/dataset_class_distributions_train_test.png",
        dpi=600,
        bbox_inches="tight"
    )

    plt.savefig(
        "./plots/dataset_class_distributions_train_test.pdf",
        bbox_inches="tight"
    )

    plt.savefig(
        "./plots/dataset_class_distributions_train_test.eps",
        bbox_inches="tight"
    )

    print("Figures saved successfully.")



if __name__ == "__main__":
    plot_class_distributions()