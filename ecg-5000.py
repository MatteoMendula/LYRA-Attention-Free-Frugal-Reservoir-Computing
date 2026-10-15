import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# 1. Definizione del file
files = ['./csv/ECG5000-DeepRCvsMoment.csv'] 

# 2. Caricamento del dataset
dfs = []
for file_path in files:
    df = pd.read_csv(file_path)
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# Gestione delle deviazioni standard mancanti
# NOTA: I nomi delle colonne qui sono 'accuracy' e 'accuracy_std'
std_cols = ['accuracy_std', 'rmse_stability_std']
for col in std_cols:
    if col not in df_all.columns:
        df_all[col] = 0.0
    else:
        df_all[col] = df_all[col].fillna(0.0)

# 3. Pulizia e Unificazione dei Nomi degli Esperimenti
def unify_experiment_name(name):
    prefixes_to_remove = ['DeepRC_IP_', 'DeepRC_', 'MOMENT_Aligned_', 'MOMENT_']
    name_str = str(name)
    for prefix in prefixes_to_remove:
        if name_str.startswith(prefix):
            return name_str[len(prefix):]
    return name_str

# Creiamo la colonna unificata
df_all['Unified_Name'] = df_all['Name'].apply(unify_experiment_name)

# 4. Raggruppamento sui nomi unificati e per architettura
df_grouped = df_all.groupby(['Unified_Name', 'architecture']).mean(numeric_only=True).reset_index()

# 5. Funzione per l'ordinamento "Naturale"
def natural_keys(text):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]

experiments = sorted(df_grouped['Unified_Name'].unique(), key=natural_keys)
architectures = df_grouped['architecture'].unique()

# 6. Configurazione delle metriche da plottare (Solo Accuracy e Stability)
metrics = {
    'Accuracy': ('accuracy', 'accuracy_std'),
    'Stability (RMSE)': ('rmse_stability', 'rmse_stability_std')
}

# Impostazioni per il clustered bar chart
x = np.arange(len(experiments))
width = 0.8 / len(architectures)

# 7. Creazione dei plot
# Utilizziamo 2 righe e 1 colonna dato che manca il forecasting
fig, axes = plt.subplots(2, 1, figsize=(16, 14))
fig.suptitle('Confronto Architetture per Task ECG5000', fontsize=18, y=0.96)

cmap = plt.get_cmap('tab10') 

for i, (metric_title, (val_col, std_col)) in enumerate(metrics.items()):
    ax = axes[i]
    
    for j, arch in enumerate(architectures):
        arch_means = []
        arch_stds = []
        
        for exp in experiments:
            subset = df_grouped[(df_grouped['Unified_Name'] == exp) & (df_grouped['architecture'] == arch)]
            if not subset.empty:
                arch_means.append(subset[val_col].values[0])
                arch_stds.append(subset[std_col].values[0])
            else:
                arch_means.append(0)
                arch_stds.append(0)
                
        offset = x - (0.8 / 2) + (j + 0.5) * width
        
        ax.bar(offset, arch_means, width, yerr=arch_stds, label=arch, 
               color=cmap(j % 10), edgecolor='black', capsize=4, alpha=0.9)

        print(f"Plottato {metric_title} per architettura {arch} con {len(arch_means)} barre.")
        # print values
        print(f"Valori medi: {arch_means}")
        
    ax.set_title(metric_title, fontsize=15)
    ax.set_xticks(x)
    # Etichette basate sul nome unificato e ruotate per leggibilità
    ax.set_xticklabels(experiments, rotation=20, ha='right', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylabel('Valore della Metrica', fontsize=13)

# Unica legenda per l'intera figura in alto
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.93), 
           ncol=min(5, len(architectures)), fontsize=12, title='Architetture')

plt.tight_layout()
plt.subplots_adjust(top=0.86, hspace=0.4)

# save
plt.savefig('./plots/ecg5000_plots.pdf', dpi=300, bbox_inches='tight')
print("Plots salvati in './plots/ecg5000_plots.pdf'")