import matplotlib.pyplot as plt
import numpy as np

# Definizione dei modelli e delle metriche
models = [
    "DeepRC_IP",
    "MOMENT-1-small",
    "DeepRC_IP - PCA 140D",
    "MOMENT-1-small - PCA 15D"
]

metrics = {
    "Silhouette Score (↑)": [-0.0222, -0.0246, -0.0236, -0.0270],
    "Davies-Bouldin (↓)": [23.7578, 9.7984, 23.3023, 9.6837],
    "Trustworthiness (↑)": [0.9936, 0.9105, 0.9921, 0.8875]
}

# Configurazione della figura con 3 subplot affiancati
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Colori distinti per i modelli

for i, (metric_name, values) in enumerate(metrics.items()):
    ax = axes[i]
    
    # Creazione delle barre per ogni modello
    bars = ax.bar(models, values, color=colors, edgecolor='black', alpha=0.85, width=0.6)
    
    # Titolo e formattazione dell'asse
    ax.set_title(metric_name, fontsize=13, fontweight='bold', pad=15)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Rendi la linea dello zero più evidente per lo Silhouette Score (valori negativi)
    if min(values) < 0:
        ax.axhline(0, color='black', linewidth=0.8, linestyle='-')

    # Aggiunta dei valori numerici sopra o sotto le barre
    for bar in bars:
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 3 if height >= 0 else -12
        
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, offset),
                    textcoords="offset points",
                    ha='center', va=va, fontsize=9, fontweight='bold')

# Ottimizzazione degli spazi per evitare sovrapposizioni delle etichette
plt.tight_layout()

# Salvataggio del grafico
plt.savefig('confronto_metriche.png', dpi=300)