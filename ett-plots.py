import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# 1. Definizione del file
files = ['./csv/ETT-DeepRCvsMoment.csv']

# 2. Caricamento del dataset
dfs = []
for file_path in files:
    df = pd.read_csv(file_path)
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# 3. Pulizia e Unificazione dei Nomi degli Esperimenti
def unify_experiment_name(name):
    prefixes_to_remove = ['DeepRC_IP_', 'DeepRC_', 'MOMENT_Aligned_', 'MOMENT_']
    name_str = str(name)
    for prefix in prefixes_to_remove:
        if name_str.startswith(prefix):
            return name_str[len(prefix):]
    return name_str

df_all['Unified_Name'] = df_all['Name'].apply(unify_experiment_name)

# 4. Raggruppamento sui nomi unificati e per architettura
df_grouped = df_all.groupby(['Unified_Name', 'architecture']).mean(numeric_only=True).reset_index()

# 5. Funzione per l'ordinamento "Naturale"
def natural_keys(text):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]

experiments = sorted(df_grouped['Unified_Name'].unique(), key=natural_keys)
architectures = df_grouped['architecture'].unique()
architectures = list(architectures)
architectures.reverse()

print(architectures)

architecture_labels = {
    'DeepRC_Ridge_MultiTap_EffRescale_LLE': 'LYRA',
    'MOMENT-small': 'MOMENT',
}

# --- NUOVA LOGICA PER CLUSTERIZZARE I DATASET ---
x_coords = []
current_x = 0
current_dataset = None
GAP_TRA_DATASET = 0.6  # Regola questo valore se vuoi i dataset più vicini (es. 0.3) o lontani (es. 0.8)

for exp in experiments:
    # Estrae la porzione del dataset (es. 'ETTh1' da 'ETTh1-H96' o 'ETTh1_H96')
    dataset_name = exp.split('_')[0] if '_' in exp else exp.split('-')[0]
    
    if current_dataset is None:
        current_dataset = dataset_name
    elif current_dataset != dataset_name:
        current_x += GAP_TRA_DATASET  # Inserisce lo stacco
        current_dataset = dataset_name
        
    x_coords.append(current_x)
    current_x += 1

x = np.array(x_coords)
# ------------------------------------------------

# 6. Configurazione delle metriche da plottare
metrics = {
    '(a) Accuracy [%] (↑)': 'accuracy_regime',
    '(b) Forecasting [RMSE] (↓)': 'rmse_forecast_denorm',
    '(c) Reconstruction [RMSE] (↓)': 'rmse_stability' 
}

width = 0.6 / len(architectures)

colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
hatches = ['/', '\\', 'x', '-']

# 7. Creazione dei plot (Larghezza aumentata leggermente a 10 per accogliere i gap)
fig, axes = plt.subplots(3, 1, figsize=(8, 10))

for i, (metric_title, val_col) in enumerate(metrics.items()):
    ax = axes[i]
    
    for j, arch in enumerate(architectures):
        arch_means = []
        
        for exp in experiments:
            subset = df_grouped[(df_grouped['Unified_Name'] == exp) & (df_grouped['architecture'] == arch)]
            if not subset.empty and val_col in subset.columns:
                arch_means.append(subset[val_col].values[0])
            else:
                arch_means.append(0)
                
        # L'offset ora calcola le posizioni usando il nuovo array x con i gap
        offset = x - (0.8 / 2) + (j + 0.5) * width
        
        ax.bar(offset, arch_means, width, label=architecture_labels.get(arch, arch) if i == 0 else "", 
               color=colors[j % len(colors)], hatch=hatches[j % len(hatches)], 
               edgecolor='black', alpha=0.9)
        
    ax.set_title(metric_title, fontsize=15)
    ax.set_xticks(x)
    
    if i == 2:
        ax.set_xticklabels([e.replace('_', '-') for e in experiments], rotation=30, ha='right', fontsize=12)
    else:
        ax.set_xticklabels([])
        
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(bottom=0)

fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.99), 
           ncol=len(architectures), fontsize=15, frameon=False)

plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.25)

plt.savefig('./plots/ett_plots.pdf', bbox_inches='tight', dpi=300)