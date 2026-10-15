import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# 1. Definizione del file
file_path = './csv/ETT-DeepRCvsMoment-inference-scaling.csv'

# 2. Caricamento e pulizia
df = pd.read_csv(file_path)
cols_to_numeric = ['seq_len', 'batch_size', 'inference_latency_ms', 'peak_vram_mb', 'throughput_seq_per_sec']
for col in cols_to_numeric:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. Mettere in scala usando il Batch Size
df['latency_per_seq'] = df['inference_latency_ms'] / df['batch_size']
df['vram_per_seq'] = df['peak_vram_mb'] / df['batch_size']
df['throughput'] = df['throughput_seq_per_sec']

# 4. Creazione Etichetta Unica (Modello + Batch Size)
df['Config_Label'] = df['model'] + ' (BS=' + df['batch_size'].fillna(0).astype(int).astype(str) + ')'

# 5. Raggruppamento
df_grouped = df.groupby(['Config_Label', 'model', 'seq_len'])[['latency_per_seq', 'vram_per_seq', 'throughput']].mean().reset_index()

# --- MODIFICA: Ordinamento personalizzato (MOMENT prima, poi DeepRC, ordinati per Batch Size) ---
def sort_configs(cfg):
    bs_match = re.search(r'BS=(\d+)', cfg)
    bs = int(bs_match.group(1)) if bs_match else 0
    # Diamo priorità a MOMENT (0) rispetto a DeepRC (1)
    model_prio = 0 if 'MOMENT' in cfg else 1
    return (model_prio, bs)

configs = sorted(df_grouped['Config_Label'].unique(), key=sort_configs)
seq_lengths = sorted(df_grouped['seq_len'].unique())

# 6. Definizione Stili Dinamici (Colori, Hatch, Linestyle)
style_map = {}

# Colori allineati al primo plot: Blu per MOMENT, Verde per DeepRC (sfumature basate su tab10)
moment_colors = ['#99ccff', '#66b3ff', '#3399ff', '#0059b3'] # Da chiaro a scuro
deeprc_colors = ['#b3e6b3', '#66cc66', '#2ca02c', '#006600'] # Da chiaro a scuro

d_idx, m_idx = 0, 0
for cfg in configs:
    model_name = df_grouped[df_grouped['Config_Label'] == cfg]['model'].iloc[0]
    
    if 'MOMENT' in str(model_name):
        style_map[cfg] = {
            'facecolor': moment_colors[m_idx % len(moment_colors)],
            'edgecolor': '#001a33', 
            'hatch': '/',   # Tratteggio coerente col primo plot
            'linestyle': '-'
        }
        m_idx += 1
    else:
        style_map[cfg] = {
            'facecolor': deeprc_colors[d_idx % len(deeprc_colors)],
            'edgecolor': '#003300', 
            'hatch': '\\',  # Tratteggio coerente col primo plot
            'linestyle': '-'
        }
        d_idx += 1

# 7. Configurazione Metriche
metrics = {
    '(a) Latency per Single Sequence (↓)': ('latency_per_seq', 'Latency [ms / sequence]'),
    '(b) Amortized VRAM per Sequence (↓)': ('vram_per_seq', 'VRAM [MB / sequence]'),
    '(c) Global Throughput (↑)': ('throughput', 'Processed Sequences per Second')
}

# 8. Creazione Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))

x = np.arange(len(seq_lengths))
width = 0.8 / len(configs)

for i, (title, (col, ylabel)) in enumerate(metrics.items()):
    ax = axes[i]
    
    for j, cfg in enumerate(configs):
        subset = df_grouped[df_grouped['Config_Label'] == cfg]
        
        y_values = []
        for sl in seq_lengths:
            val = subset[subset['seq_len'] == sl][col]
            if not val.empty:
                y_values.append(val.values[0])
            else:
                y_values.append(0.0)
        
        offset = x - (0.8 / 2) + (j + 0.5) * width
        
        ax.bar(offset, y_values, width, label=cfg, 
               color=style_map[cfg]['facecolor'],
               edgecolor=style_map[cfg]['edgecolor'],
               hatch=style_map[cfg]['hatch'],
               linestyle=style_map[cfg]['linestyle'],
               linewidth=1.2, alpha=0.9)

        print(f"Plotted {title} for config {cfg}: values={y_values}")
        
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(seq_lengths)
    # Uniformiamo anche la griglia al primo grafico
    ax.grid(axis='y', linestyle='--', alpha=0.7) 

# --- MODIFICA: Legenda in alto, pulita e unificata ---
handles, labels = axes[0].get_legend_handles_labels()
# Posizioniamo la legenda in alto, senza cornice (frameon=False)
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.90), 
           ncol=min(4, len(configs)), fontsize=12, frameon=False)

plt.tight_layout()
# Regoliamo i margini: lasciamo spazio in alto per la legenda e stringiamo il fondo
plt.subplots_adjust(top=0.76, bottom=0.1) 
plt.savefig('plots/ett_scalability_plots.pdf', dpi=300, bbox_inches='tight')