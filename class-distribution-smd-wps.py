#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experiment-Specific Class Distribution Visualization for SMD.
Generates a grid of subplots (N rows x 2 columns) to compare the 
regime distributions of the Training Corpus vs. the Target Machine.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import Counter

# --- Configuration & Paths ---
# Path to the pre-computed regime labels
SMD_LABELS_DIR = '../datasets/SMD/OmniAnomaly-master/ServerMachineDataset/regime_labels'

# The experimental setups
EXPERIMENTS = [
    {"name": "WP1_LowDiv_2_Machines",
     "corpus": ['machine-1-1.txt', 'machine-1-2.txt'],
     "val_machine": 'machine-1-2.txt',
     "target": 'machine-3-11.txt'},
    {"name": "WP1_MedDiv_4_Machines",
     "corpus": ['machine-1-1.txt', 'machine-1-2.txt', 'machine-2-1.txt', 'machine-2-2.txt'],
     "val_machine": 'machine-1-2.txt',
     "target": 'machine-3-11.txt'},
    {"name": "WP1_HighDiv_6_Machines",
     "corpus": ['machine-1-1.txt', 'machine-1-2.txt', 'machine-2-1.txt',
                'machine-2-2.txt', 'machine-3-1.txt', 'machine-3-2.txt'],
     "val_machine": 'machine-1-2.txt',
     "target": 'machine-3-11.txt'},
    {"name": "WP2_LODO_Groups_1_and_2",
     "corpus": ['machine-1-1.txt', 'machine-1-2.txt', 'machine-1-3.txt',
                'machine-2-1.txt', 'machine-2-2.txt', 'machine-2-3.txt'],
     "val_machine": 'machine-2-3.txt',
     "target": 'machine-3-11.txt'},
    {"name": "WP3_Random_Mix_1",
     "corpus": ['machine-1-3.txt', 'machine-1-5.txt', 'machine-2-4.txt',
                'machine-2-7.txt', 'machine-3-4.txt', 'machine-3-8.txt'],
     "val_machine": 'machine-2-7.txt',
     "target": 'machine-3-11.txt'},
    {"name": "WP3_Random_Mix_2",
     "corpus": ['machine-1-4.txt', 'machine-1-7.txt', 'machine-2-3.txt',
                'machine-2-5.txt', 'machine-3-5.txt', 'machine-3-9.txt'],
     "val_machine": 'machine-2-5.txt',
     "target": 'machine-3-11.txt'},
    {"name": "WP4_Target_1-8",
     "corpus": ['machine-1-1.txt', 'machine-1-2.txt', 'machine-2-1.txt',
                'machine-2-2.txt', 'machine-3-1.txt', 'machine-3-2.txt'],
     "val_machine": 'machine-2-2.txt',
     "target": 'machine-1-8.txt'},
    {"name": "WP4_Target_2-9",
     "corpus": ['machine-1-1.txt', 'machine-1-2.txt', 'machine-2-1.txt',
                'machine-2-2.txt', 'machine-3-1.txt', 'machine-3-2.txt'],
     "val_machine": 'machine-2-2.txt',
     "target": 'machine-2-9.txt'},
]

def load_labels_for_machines(machine_list):
    """
    Loads and concatenates both 'train_' and 'test_' .npy label files 
    for a given list of machine text file names.
    """
    all_labels = []
    for machine in machine_list:
        base_name = machine.replace('.txt', '.npy')
        
        # Check for both train and test partitions to get the full machine profile
        for prefix in ['train_', 'test_']:
            file_path = os.path.join(SMD_LABELS_DIR, prefix + base_name)
            if os.path.exists(file_path):
                labels = np.load(file_path)
                all_labels.extend(labels)
            else:
                print(f"Warning: Could not find {file_path}")
                
    return all_labels

def plot_experiment_distributions():
    """
    Generates a figure with len(EXPERIMENTS) rows and 2 columns.
    Col 0: Training Corpus distribution
    Col 1: Target Machine distribution

    Also saves the plotted counts to a JSON file.
    """
    n_exp = len(EXPERIMENTS)
    fig, axes = plt.subplots(nrows=n_exp, ncols=2, figsize=(14, 4 * n_exp))
    sns.set_theme(style="whitegrid")

    class_order = [0, 1, 2, 3]

    # Dictionary that will be dumped to JSON
    plot_results = {}

    for i, exp in enumerate(EXPERIMENTS):
        exp_name = exp['name']

        # Load data
        corpus_labels = load_labels_for_machines(exp['corpus'])
        target_labels = load_labels_for_machines([exp['target']])

        # Compute counts
        corpus_counts = {str(c): 0 for c in class_order}
        target_counts = {str(c): 0 for c in class_order}

        corpus_counter = Counter(corpus_labels)
        target_counter = Counter(target_labels)

        for c in class_order:
            corpus_counts[str(c)] = int(corpus_counter.get(c, 0))
            target_counts[str(c)] = int(target_counter.get(c, 0))

        # Save counts for JSON
        corpus_total = len(corpus_labels)
        target_total = len(target_labels)

        corpus_percentages = {
            str(c): round(
                100.0 * corpus_counts[str(c)] / corpus_total,
                4
            ) if corpus_total > 0 else 0.0
            for c in class_order
        }

        target_percentages = {
            str(c): round(
                100.0 * target_counts[str(c)] / target_total,
                4
            ) if target_total > 0 else 0.0
            for c in class_order
        }

        plot_results[exp_name] = {
            "corpus_machines": exp["corpus"],
            "target_machine": exp["target"],

            "corpus_total_samples": int(corpus_total),
            "target_total_samples": int(target_total),

            "corpus_counts": corpus_counts,
            "target_counts": target_counts,

            "corpus_percentages": corpus_percentages,
            "target_percentages": target_percentages
        }

        # Plot Corpus
        ax_corpus = axes[i, 0]
        if corpus_labels:
            sns.countplot(
                x=corpus_labels,
                ax=ax_corpus,
                order=class_order,
                palette="Blues_d"
            )
            ax_corpus.set_title(
                f"{exp_name}\nCorpus ({len(exp['corpus'])} machines)",
                fontweight='bold'
            )
            ax_corpus.set_xlabel('Regime / Class Label')
            ax_corpus.set_ylabel('Count')
        else:
            ax_corpus.text(0.5, 0.5, 'Data missing', ha='center')

        # Plot Target
        ax_target = axes[i, 1]
        if target_labels:
            sns.countplot(
                x=target_labels,
                ax=ax_target,
                order=class_order,
                palette="Reds_d"
            )
            ax_target.set_title(
                f"{exp_name}\nTarget ({exp['target']})",
                fontweight='bold'
            )
            ax_target.set_xlabel('Regime / Class Label')
            ax_target.set_ylabel('Count')
        else:
            ax_target.text(0.5, 0.5, 'Data missing', ha='center')

    plt.tight_layout(pad=2.0)

    # Save plot
    output_plot = './plots/smd_experiments_covariate_shift.png'
    plt.savefig(output_plot, dpi=300)

    # Save JSON
    output_json = './plots/smd_experiments_covariate_shift.json'
    with open(output_json, "w") as f:
        json.dump(plot_results, f, indent=4)

    print(f"\nPlot saved to: {output_plot}")
    print(f"JSON saved to: {output_json}")

    plt.show()

if __name__ == "__main__":
    if not os.path.exists(SMD_LABELS_DIR):
        print(f"Error: Directory not found at {SMD_LABELS_DIR}")
        print("Please ensure you have run 'generate_regime_labels.py' first.")
    else:
        plot_experiment_distributions()