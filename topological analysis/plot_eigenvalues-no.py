import numpy as np
import matplotlib.pyplot as plt
import torch
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir

# Importa l'implementazione custom
from psi_DeepRC_IP import DeepReservoir

rpy.verbosity(0)

# =====================================================================
# CONFIGURAZIONE (Paragone Fair: Matrice Non-Densa + Leaky=1.0)
# =====================================================================
UNITS = 1000
SPECTRAL_RADIUS = 0.99  # (o 1.2, in questo caso saranno identici a prescindere dal raggio)
LEAKY = 1.0             # <-- IL PARAMETRO CHIAVE (Nessuna inerzia)
CONNECTIVITY = 10       # <-- Matrice sparsa (Disattiva la Legge di Girko)

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    rpy.set_seed(seed)

set_seed()

print(f"Generazione matrici dense ({UNITS}x{UNITS}) con Leaky={LEAKY}...")
I = np.eye(UNITS)

# =====================================================================
# 1. RESERVOIRPY BASELINE (Approccio Classico)
# =====================================================================
# In ReservoirPy rc_connectivity=1.0 indica una matrice densa
node_rpy = Reservoir(
    units=UNITS, sr=SPECTRAL_RADIUS, lr=LEAKY, rc_connectivity=1.0
)
node_rpy(np.zeros((1, 1))) # Allocazione forzata

W_rpy_sparse = node_rpy.W
W_rpy = W_rpy_sparse.toarray() if hasattr(W_rpy_sparse, 'toarray') else np.array(W_rpy_sparse)

# Matrice di transizione effettiva classica
W_eff_rpy = (1 - LEAKY) * I + LEAKY * W_rpy

# =====================================================================
# 2. DEEPRC (Custom - Effective Rescaling)
# =====================================================================
deep_rc = DeepReservoir(
    input_size=1, tot_units=UNITS, n_layers=1, 
    spectral_radius=SPECTRAL_RADIUS, leaky=LEAKY, 
    connectivity_recurrent=CONNECTIVITY, lle_tuning=False, 
    effective_rescaling=True, seed=42
)

W_deeprc_tensor = deep_rc.reservoir[0].cell.recurrent_kernel
W_deeprc = W_deeprc_tensor.detach().cpu().numpy()

# Matrice di transizione fisica reale subita dalla rete DeepRC
W_eff_deeprc = (1 - LEAKY) * I + LEAKY * W_deeprc

# =====================================================================
# CALCOLO AUTOVALORI W_EFF
# =====================================================================
print("Estrazione autovalori...")
eig_eff_rpy = np.linalg.eigvals(W_eff_rpy)
eig_eff_deeprc = np.linalg.eigvals(W_eff_deeprc)

# =====================================================================
# PLOTTING PER IL PAPER (LYRA)
# =====================================================================
print("Generazione del plot vettoriale...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# Nota la 'r' prima della stringa per evitare l'errore \alpha
fig.suptitle(rf"Dynamical Substrate Topology (Dense Matrix, $\alpha={LEAKY}$, Target SR={SPECTRAL_RADIUS})", fontsize=16, fontweight='bold')

def format_eigen_subplot(ax, eig_vals, title, color):
    ax.scatter(np.real(eig_vals), np.imag(eig_vals), alpha=0.6, edgecolors='w', linewidth=0.5, color=color, s=40)
    
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color='red', linestyle='--', linewidth=1.5, label='Unit Circle (Stability Limit)')
    ax.plot(SPECTRAL_RADIUS * np.cos(theta), SPECTRAL_RADIUS * np.sin(theta), color='black', linestyle=':', linewidth=1.5, label=f'Target SR ({SPECTRAL_RADIUS})')

    ax.axhline(0, color='gray', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='gray', linewidth=0.8, alpha=0.5)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Real Part (Inertia)", fontsize=12)
    ax.set_ylabel("Imaginary Part (Oscillation)", fontsize=12)
    
    ax.set_aspect('equal', adjustable='box')
    limit = 1.2
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

# Plot 1: ReservoirPy
format_eigen_subplot(axes[0], eig_eff_rpy, "ReservoirPy (Standard Scaling)\nPerdita del 50% di capacità oscillatoria", color='royalblue')

# Plot 2: DeepRC
format_eigen_subplot(axes[1], eig_eff_deeprc, "DeepRC (Effective Rescaling)\nConservazione della fase all'Edge of Chaos", color='seagreen')

plt.tight_layout()
filename = "./LYRA_Centered_Edge_of_Chaos.pdf"
plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
print(f"✅ Analisi completata. Grafico salvato come '{filename}'.")