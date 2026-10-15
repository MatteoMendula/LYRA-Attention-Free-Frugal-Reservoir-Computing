import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir

from psi_DeepRC_IP import DeepReservoir

rpy.verbosity(0)

# Reproducibility and numerical experiment size
SEED = 42
UNITS = 1000

# Publication-oriented Matplotlib settings for an IEEE two-column figure.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.0,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    rpy.set_seed(seed)


def compute_spectra(spectral_radius, leaky, connectivity, seed=SEED):
    """Return effective-transition eigenvalues for ReservoirPy and DeepRC."""
    set_seed(seed)
    identity = np.eye(UNITS)

    # Keep the ReservoirPy baseline consistent with the original scripts.
    # rc_connectivity=1.0 creates a dense recurrent matrix.
    node_rpy = Reservoir(
        units=UNITS,
        sr=spectral_radius,
        lr=leaky,
        rc_connectivity=1.0,
    )
    node_rpy(np.zeros((1, 1)))  # force parameter initialization
    W_rpy_raw = node_rpy.W
    W_rpy = (
        W_rpy_raw.toarray()
        if hasattr(W_rpy_raw, "toarray")
        else np.asarray(W_rpy_raw)
    )
    W_eff_rpy = (1.0 - leaky) * identity + leaky * W_rpy

    deep_rc = DeepReservoir(
        input_size=1,
        tot_units=UNITS,
        n_layers=1,
        spectral_radius=spectral_radius,
        leaky=leaky,
        connectivity_recurrent=connectivity,
        lle_tuning=False,
        effective_rescaling=True,
        seed=seed,
    )
    W_deeprc = (
        deep_rc.reservoir[0]
        .cell.recurrent_kernel
        .detach()
        .cpu()
        .numpy()
    )
    W_eff_deeprc = (1.0 - leaky) * identity + leaky * W_deeprc

    return np.linalg.eigvals(W_eff_rpy), np.linalg.eigvals(W_eff_deeprc)


def add_reference_circle(ax, center, radius, label=None):
    theta = np.linspace(0.0, 2.0 * np.pi, 800)
    ax.plot(
        center[0] + radius * np.cos(theta),
        center[1] + radius * np.sin(theta),
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.75,
        label=label,
        zorder=3,
    )


def draw_panel(ax, eigenvalues, panel_label, method, color, leaky, target_sr):
    x = eigenvalues.real
    y = eigenvalues.imag
    ax.scatter(
        x,
        y,
        s=8,
        c=color,
        alpha=0.55,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )

    # The declared target is imposed on the effective transition matrix itself.
    # Therefore, the reference boundary is centered at the origin and has
    # radius target_sr, independently of the leakage value.
    add_reference_circle(
        ax,
        center=(0.0, 0.0),
        radius=target_sr,
        label=rf"target $\rho={target_sr:g}$",
    )
    ax.axhline(0.0, color="0.75", linewidth=0.45, zorder=0)
    ax.axvline(0.0, color="0.75", linewidth=0.45, zorder=0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.90", linewidth=0.4)
    ax.set_title(f"({panel_label}) {method}", fontweight="bold", pad=3)
    ax.set_xlabel(r"Re$(\lambda)$")
    ax.set_ylabel(r"Im$(\lambda)$")


def main():
    # Order chosen to make the visual argument read from left to right:
    # first the alpha=1 control case, where the methods coincide; then the
    # alpha=0.5 leaky case, where effective rescaling preserves the target.
    experiments = [
        {
            "name": r"Control: $\alpha=1.0$, $\rho=0.99$",
            "sr": 0.99,
            "leaky": 1.0,
            "connectivity": 10,
        },
        {
            "name": r"Leaky regime: $\alpha=0.5$, $\rho=1.2$",
            "sr": 1.2,
            "leaky": 0.5,
            "connectivity": 1000,
        },
    ]

    spectra = []
    for exp in experiments:
        spectra_file = f"spectra/spectra_{experiments.index(exp)}.npz"
        if not os.path.exists(spectra_file):
            print(f"Computing {exp['name']} ...")
            s = compute_spectra(
                spectral_radius=exp["sr"],
                leaky=exp["leaky"],
                connectivity=exp["connectivity"],
            )
        else:
            print(f"Loading precomputed spectra from {spectra_file} ...")
            data = np.load(spectra_file)
            s = (data["eig_rpy"], data["eig_deeprc"])
        spectra.append(s)

    # try to save the spectra for later use, but don't fail if the directory doesn't exist
    try:
        os.makedirs("spectra", exist_ok=True)
        for i, exp in enumerate(experiments):

            # if it exists, do not overwrite, to avoid accidental loss of data
            if os.path.exists(f"spectra/spectra_{i}.npz"):
                print(f"Warning: spectra/spectra_{i}.npz already exists, skipping save.")
                continue 
            np.savez(
                f"spectra/spectra_{i}.npz",
                eig_rpy=spectra[i][0],
                eig_deeprc=spectra[i][1],
                sr=exp["sr"],
                leaky=exp["leaky"],
                connectivity=exp["connectivity"],
            )
    except Exception as e:
        print(f"Warning: could not save spectra due to {e}")


    # Full IEEE two-column width. Four compact square panels on one line.
    fig, axes = plt.subplots(2, 2, figsize=(5, 5), constrained_layout=True)
    colors = {"standard": "#3366AA", "effective": "#228855"}
    panel_labels = ("a", "b", "c", "d")
    methods = ("Standard", "Effective (LYRA)", "Standard", "Effective (LYRA)")

    panel = 0
    for group, (exp, (eig_rpy, eig_deeprc)) in enumerate(zip(experiments, spectra)):
        values = np.concatenate((eig_rpy, eig_deeprc))
        x_min, x_max = values.real.min(), values.real.max()
        y_abs = np.max(np.abs(values.imag))
        # Include the target circle |lambda|=rho in the common limits.
        radius = exp["sr"]
        x_min = min(x_min, -radius)
        x_max = max(x_max, radius)
        y_abs = max(y_abs, radius)
        pad = 0.06 * max(x_max - x_min, 2.0 * y_abs)

        for eig, color_key in ((eig_rpy, "standard"), (eig_deeprc, "effective")):
            ax = axes[panel]
            draw_panel(
                ax,
                eig,
                panel_labels[panel],
                methods[panel],
                colors[color_key],
                exp["leaky"],
                exp["sr"],
            )
            ax.set_xlim(x_min - pad, x_max + pad)
            ax.set_ylim(-y_abs - pad, y_abs + pad)
            ax.set_xlabel(r"Re$(\lambda)$", labelpad=1)
            ax.set_ylabel(r"Im$(\lambda)$" if panel in (0, 2) else "", labelpad=1)
            ax.tick_params(axis="both", pad=1, length=2)
            panel += 1

    # Group labels above panels (a,b) and (c,d).
    fig.canvas.draw()
    for first, last, label in (
        (0, 1, experiments[0]["name"]),
        (2, 3, experiments[1]["name"]),
    ):
        left = axes[first].get_position().x0
        right = axes[last].get_position().x1
        top = max(axes[first].get_position().y1, axes[last].get_position().y1)
        fig.text(
            (left + right) / 2.0,
            top + 0.035,
            label,
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
        )

    # Avoid a separate legend consuming vertical space; the dashed boundary is
    # explained in the paper caption.
    out_dir = "./imgs"
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "LYRA_effective_rescaling_eigenvalues.pdf")
    png_path = os.path.join(out_dir, "LYRA_effective_rescaling_eigenvalues.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()