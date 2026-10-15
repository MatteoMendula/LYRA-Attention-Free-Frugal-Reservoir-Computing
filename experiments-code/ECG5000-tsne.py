#!/usr/bin/env python3

import argparse
import gc
import gzip
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

# Custom modules
from psi_DeepRC_IP import DeepReservoir
from momentfm import MOMENTPipeline


# =====================================================================
# Configuration
# =====================================================================

BASE_DIR = Path("/home/mmendula/last-rush/datasets/ECG5000")
TEST_PATH = BASE_DIR / "ECG5000_TEST.tsv"

RESULTS_DIR = Path("../csv")
CACHE_PATH = RESULTS_DIR / "ecg5000_embedding_results.json.gz"

OUTPUT_PDF = RESULTS_DIR / "ecg5000_tsne_comparison_readable.pdf"
OUTPUT_PNG = RESULTS_DIR / "ecg5000_tsne_comparison_readable.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GLOBAL_SEED = 42

SOFT_FLOOR_C = 0.02
CLIP_VAL = 4.0

RC_PARAMS = {
    "tot_units": 100,
    "n_layers": 4,
    "spectral_radius": 0.7091296583705226,
    "leaky": 1.0,
    "connectivity_recurrent": 20,
    "input_scaling": 0.11046872015680255,
}

N_TAPS = 4
BIAS_SCALING = 0.1
CONCAT_UNITS = True
LLE_TUNING = True

MOMENT_SIZE = "small"

DEFAULT_TSNE_PARAMS = {
    "n_components": 2,
    "perplexity": 30.0,
    "max_iter": 1000,
    "init": "pca",
    "learning_rate": "auto",
    "random_state": GLOBAL_SEED,
    "n_jobs": -1,
}

CLASS_NAMES = [
    "Normal",
    "R-on-T PVC",
    "PVC",
    "SP",
    "Ub",
]

# Color-blind-friendly palette
CLASS_COLORS = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermilion
    "#CC79A7",  # purple
]

CLASS_MARKERS = [
    "o",
    "s",
    "^",
    "D",
    "P",
]


# =====================================================================
# Reproducibility
# =====================================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# Data loading and preprocessing
# =====================================================================

def process_ecg_sequences(filepath):
    raw_data = np.loadtxt(filepath, delimiter="\t")

    labels = raw_data[:, 0].astype(np.int64) - 1
    series = raw_data[:, 1:].astype(np.float32)

    mu = series.mean(axis=1, keepdims=True)
    sd = series.std(axis=1, keepdims=True)
    sd = np.sqrt(sd ** 2 + SOFT_FLOOR_C ** 2)

    series_loc = (series - mu) / sd
    np.clip(series_loc, -CLIP_VAL, CLIP_VAL, out=series_loc)

    # [B, T, 1]
    series_loc = series_loc[..., np.newaxis]

    return series_loc, labels


# =====================================================================
# DeepRC embeddings
# =====================================================================

def build_reservoir(input_dim, params, device, seed):
    seed_everything(seed)

    model = DeepReservoir(
        input_size=input_dim,
        tot_units=params["tot_units"],
        n_layers=params["n_layers"],
        concat=CONCAT_UNITS,
        input_scaling=params["input_scaling"],
        inter_scaling=1.0,
        spectral_radius=params["spectral_radius"],
        leaky=params["leaky"],
        connectivity_recurrent=params["connectivity_recurrent"],
        connectivity_input=params["connectivity_recurrent"],
        connectivity_inter=params["connectivity_recurrent"],
        bias_bool=True,
        bias_scaling=BIAS_SCALING,
        effective_rescaling=True,
        nonlin="tanh",
        lle_tuning=LLE_TUNING,
        n_taps=N_TAPS,
        verbose=False,
    ).to(device)

    return model


def get_rc_embeddings(X_numpy, params):
    model = build_reservoir(
        input_dim=1,
        params=params,
        device=DEVICE,
        seed=GLOBAL_SEED,
    )

    model.eval()

    l1_states = []
    ldeep_states = []

    batch_size = 2048

    with torch.no_grad():
        for start in range(0, len(X_numpy), batch_size):
            stop = start + batch_size

            batch = torch.from_numpy(
                X_numpy[start:stop]
            ).float().to(DEVICE)

            _, states_last = model(
                batch,
                return_all_layers=True,
            )

            l1_states.append(
                states_last[0].cpu().numpy()
            )

            ldeep_states.append(
                states_last[-1].cpu().numpy()
            )

    r_l1 = np.concatenate(l1_states, axis=0)
    r_ld = np.concatenate(ldeep_states, axis=0)

    embeddings = np.hstack([r_l1, r_ld])

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()

    return embeddings


# =====================================================================
# MOMENT embeddings
# =====================================================================

def get_moment_embeddings(X_numpy, model_size="small"):
    model_name = f"AutonLab/MOMENT-1-{model_size}"

    model = MOMENTPipeline.from_pretrained(
        model_name,
        model_kwargs={
            "task_name": "embedding",
        },
    )

    model.init()
    model = model.to(DEVICE)
    model.eval()

    x_tensor = (
        torch.from_numpy(X_numpy)
        .float()
        .permute(0, 2, 1)
    )

    sequence_length = x_tensor.shape[2]
    pad_length = (8 - sequence_length % 8) % 8

    states = []
    batch_size = 128

    with torch.no_grad():
        for start in range(0, len(x_tensor), batch_size):
            stop = start + batch_size
            batch = x_tensor[start:stop].to(DEVICE)

            if pad_length > 0:
                batch = F.pad(
                    batch,
                    (0, pad_length),
                    mode="constant",
                    value=0,
                )

                mask = torch.ones(
                    (batch.shape[0], batch.shape[2]),
                    dtype=torch.long,
                    device=DEVICE,
                )

                mask[:, -pad_length:] = 0
            else:
                mask = None

            outputs = model(
                x_enc=batch,
                input_mask=mask,
            )

            embeddings_flat = (
                outputs.embeddings
                .reshape(batch.shape[0], -1)
                .cpu()
                .numpy()
            )

            states.append(embeddings_flat)

    embeddings = np.concatenate(states, axis=0)

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()

    return embeddings


# =====================================================================
# t-SNE
# =====================================================================

def compute_tsne(embeddings, params, model_name):
    print(
        f"Calcolo t-SNE per {model_name}: "
        f"shape={embeddings.shape}, "
        f"perplexity={params['perplexity']}"
    )

    reducer = TSNE(**params)
    coordinates = reducer.fit_transform(embeddings)

    return coordinates.astype(np.float32)


# =====================================================================
# JSON cache
# =====================================================================

def array_to_list(array, dtype=np.float32):
    return np.asarray(array, dtype=dtype).tolist()


def save_results_json(
    output_path,
    labels,
    rc_embeddings,
    moment_embeddings,
    rc_tsne,
    moment_tsne,
    tsne_params,
):
    """
    Save raw embeddings and current t-SNE coordinates.

    Raw embeddings allow t-SNE to be recomputed without rerunning the models.
    The JSON is gzip-compressed because embedding arrays can be large.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    class_counts = np.bincount(
        labels,
        minlength=len(CLASS_NAMES),
    )

    payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": {
            "name": "ECG5000",
            "split": "TEST",
            "source_path": str(TEST_PATH),
            "n_samples": int(len(labels)),
            "sequence_length": 140,
            "n_classes": len(CLASS_NAMES),
            "class_names": CLASS_NAMES,
            "class_counts": class_counts.tolist(),
        },
        "preprocessing": {
            "soft_floor_c": SOFT_FLOOR_C,
            "clip_value": CLIP_VAL,
            "description": (
                "Per-sequence local normalization with "
                "soft variance flooring and defensive clipping."
            ),
        },
        "models": {
            "DeepRC": {
                "rc_params": RC_PARAMS,
                "n_taps": N_TAPS,
                "bias_scaling": BIAS_SCALING,
                "concat_units": CONCAT_UNITS,
                "lle_tuning": LLE_TUNING,
                "embedding_shape": list(rc_embeddings.shape),
            },
            "MOMENT": {
                "model_size": MOMENT_SIZE,
                "model_name": (
                    f"AutonLab/MOMENT-1-{MOMENT_SIZE}"
                ),
                "embedding_shape": list(moment_embeddings.shape),
            },
        },
        "tsne": {
            "parameters": tsne_params,
            "DeepRC": array_to_list(rc_tsne),
            "MOMENT": array_to_list(moment_tsne),
        },
        "labels": np.asarray(
            labels,
            dtype=np.int64,
        ).tolist(),
        "embeddings": {
            "DeepRC": array_to_list(rc_embeddings),
            "MOMENT": array_to_list(moment_embeddings),
        },
    }

    with gzip.open(
        output_path,
        mode="wt",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            separators=(",", ":"),
            allow_nan=False,
        )

    size_mb = output_path.stat().st_size / (1024 ** 2)

    print(
        f"Cache JSON salvata in: {output_path.resolve()}"
    )
    print(f"Dimensione compressa: {size_mb:.2f} MB")


def load_results_json(input_path):
    if not input_path.exists():
        raise FileNotFoundError(
            f"Cache non trovata: {input_path}"
        )

    with gzip.open(
        input_path,
        mode="rt",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    labels = np.asarray(
        payload["labels"],
        dtype=np.int64,
    )

    rc_embeddings = np.asarray(
        payload["embeddings"]["DeepRC"],
        dtype=np.float32,
    )

    moment_embeddings = np.asarray(
        payload["embeddings"]["MOMENT"],
        dtype=np.float32,
    )

    rc_tsne = np.asarray(
        payload["tsne"]["DeepRC"],
        dtype=np.float32,
    )

    moment_tsne = np.asarray(
        payload["tsne"]["MOMENT"],
        dtype=np.float32,
    )

    return {
        "payload": payload,
        "labels": labels,
        "rc_embeddings": rc_embeddings,
        "moment_embeddings": moment_embeddings,
        "rc_tsne": rc_tsne,
        "moment_tsne": moment_tsne,
    }


def update_tsne_in_json(
    cache_path,
    rc_tsne,
    moment_tsne,
    tsne_params,
):
    """
    Update only the t-SNE coordinates and parameters while preserving
    the expensive model embeddings.
    """

    with gzip.open(
        cache_path,
        mode="rt",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    payload["tsne"] = {
        "parameters": tsne_params,
        "DeepRC": array_to_list(rc_tsne),
        "MOMENT": array_to_list(moment_tsne),
    }

    payload["tsne_updated_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()

    with gzip.open(
        cache_path,
        mode="wt",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            separators=(",", ":"),
            allow_nan=False,
        )

    print(
        "Coordinate t-SNE aggiornate nella cache: "
        f"{cache_path.resolve()}"
    )


# =====================================================================
# Visualization
# =====================================================================

def robust_display_limits(coordinates, padding=0.06):
    """
    Compute readable limits without hard clipping.

    Entire coordinate range is preserved; padding is added to prevent
    points from touching the axes.
    """

    x_min = float(np.min(coordinates[:, 0]))
    x_max = float(np.max(coordinates[:, 0]))
    y_min = float(np.min(coordinates[:, 1]))
    y_max = float(np.max(coordinates[:, 1]))

    x_range = max(x_max - x_min, 1e-8)
    y_range = max(y_max - y_min, 1e-8)

    return (
        x_min - padding * x_range,
        x_max + padding * x_range,
        y_min - padding * y_range,
        y_max + padding * y_range,
    )


def plot_embedding_panel(
    ax,
    coordinates,
    labels,
    title,
    panel_letter,
):
    class_counts = np.bincount(
        labels,
        minlength=len(CLASS_NAMES),
    )

    # Largest class first, rare classes last.
    # Rare classes therefore remain visible on top.
    drawing_order = np.argsort(class_counts)[::-1]

    for class_id in drawing_order:
        mask = labels == class_id

        # Majority class is made lighter to reduce visual dominance.
        if class_counts[class_id] == class_counts.max():
            alpha = 0.32
            point_size = 8
        else:
            alpha = 0.78
            point_size = 15

        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            c=CLASS_COLORS[class_id],
            marker=CLASS_MARKERS[class_id],
            s=point_size,
            alpha=alpha,
            linewidths=0,
            edgecolors="none",
            rasterized=True,
            zorder=2 + class_id,
        )

    x_min, x_max, y_min, y_max = robust_display_limits(
        coordinates
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_title(
        f"({panel_letter}) {title}",
        fontsize=9,
        pad=6,
    )

    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.5,
        alpha=0.45,
        zorder=0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        direction="out",
        length=3,
        width=0.6,
    )


def create_tsne_plot(
    labels,
    rc_coordinates,
    moment_coordinates,
    output_pdf,
    output_png,
):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
    })

    output_pdf.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    class_counts = np.bincount(
        labels,
        minlength=len(CLASS_NAMES),
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(7.16, 3.35),
        constrained_layout=False,
    )

    plot_embedding_panel(
        ax=axes[0],
        coordinates=rc_coordinates,
        labels=labels,
        title="LYRA multi-scale representation",
        panel_letter="a",
    )

    plot_embedding_panel(
        ax=axes[1],
        coordinates=moment_coordinates,
        labels=labels,
        title="MOMENT-1-small representation",
        panel_letter="b",
    )

    legend_handles = []

    for class_id, class_name in enumerate(CLASS_NAMES):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=CLASS_MARKERS[class_id],
                linestyle="none",
                markerfacecolor=CLASS_COLORS[class_id],
                markeredgecolor="none",
                markersize=5.5,
                label=(
                    f"{class_name} "
                    f"(n={class_counts[class_id]})"
                ),
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=5,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.05,
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        top=0.93,
        bottom=0.23,
        wspace=0.24,
    )

    fig.savefig(
        output_pdf,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    fig.savefig(
        output_png,
        bbox_inches="tight",
        pad_inches=0.02,
        dpi=300,
    )

    plt.close(fig)

    print(f"PDF salvato: {output_pdf.resolve()}")
    print(f"PNG salvato: {output_png.resolve()}")


# =====================================================================
# Pipeline modes
# =====================================================================

def run_full_pipeline(tsne_params):
    seed_everything(GLOBAL_SEED)

    print("1. Caricamento ECG5000 TEST...")
    x_test, labels = process_ecg_sequences(TEST_PATH)

    print(
        f"Dataset caricato: {x_test.shape[0]} campioni; "
        f"shape={x_test.shape}"
    )

    print("\n2. Estrazione embedding DeepRC...")
    rc_embeddings = get_rc_embeddings(
        x_test,
        RC_PARAMS,
    )

    print(
        f"Embedding DeepRC: {rc_embeddings.shape}"
    )

    print("\n3. Estrazione embedding MOMENT...")
    moment_embeddings = get_moment_embeddings(
        x_test,
        model_size=MOMENT_SIZE,
    )

    print(
        f"Embedding MOMENT: {moment_embeddings.shape}"
    )

    print("\n4. Calcolo t-SNE...")
    rc_tsne = compute_tsne(
        rc_embeddings,
        tsne_params,
        "DeepRC",
    )

    moment_tsne = compute_tsne(
        moment_embeddings,
        tsne_params,
        "MOMENT",
    )

    print("\n5. Salvataggio cache...")
    save_results_json(
        output_path=CACHE_PATH,
        labels=labels,
        rc_embeddings=rc_embeddings,
        moment_embeddings=moment_embeddings,
        rc_tsne=rc_tsne,
        moment_tsne=moment_tsne,
        tsne_params=tsne_params,
    )

    print("\n6. Generazione figura...")
    create_tsne_plot(
        labels=labels,
        rc_coordinates=rc_tsne,
        moment_coordinates=moment_tsne,
        output_pdf=OUTPUT_PDF,
        output_png=OUTPUT_PNG,
    )


def run_plot_only():
    print(f"Lettura cache: {CACHE_PATH}")

    results = load_results_json(CACHE_PATH)

    create_tsne_plot(
        labels=results["labels"],
        rc_coordinates=results["rc_tsne"],
        moment_coordinates=results["moment_tsne"],
        output_pdf=OUTPUT_PDF,
        output_png=OUTPUT_PNG,
    )


def run_tsne_only(tsne_params):
    print(f"Lettura embedding dalla cache: {CACHE_PATH}")

    results = load_results_json(CACHE_PATH)

    rc_tsne = compute_tsne(
        results["rc_embeddings"],
        tsne_params,
        "DeepRC",
    )

    moment_tsne = compute_tsne(
        results["moment_embeddings"],
        tsne_params,
        "MOMENT",
    )

    update_tsne_in_json(
        cache_path=CACHE_PATH,
        rc_tsne=rc_tsne,
        moment_tsne=moment_tsne,
        tsne_params=tsne_params,
    )

    create_tsne_plot(
        labels=results["labels"],
        rc_coordinates=rc_tsne,
        moment_coordinates=moment_tsne,
        output_pdf=OUTPUT_PDF,
        output_png=OUTPUT_PNG,
    )


# =====================================================================
# Command line
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract, cache, recompute, and visualize "
            "ECG5000 embeddings."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "compute",
            "plot",
            "tsne",
        ],
        default="compute",
        help=(
            "'compute': run models and save everything; "
            "'plot': regenerate the figure from cached coordinates; "
            "'tsne': recompute t-SNE from cached embeddings."
        ),
    )

    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=GLOBAL_SEED,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tsne_params = {
        "n_components": 2,
        "perplexity": args.perplexity,
        "max_iter": args.iterations,
        "init": "pca",
        "learning_rate": "auto",
        "random_state": args.seed,
        "n_jobs": -1,
    }

    if args.mode == "compute":
        run_full_pipeline(tsne_params)

    elif args.mode == "plot":
        run_plot_only()

    elif args.mode == "tsne":
        run_tsne_only(tsne_params)