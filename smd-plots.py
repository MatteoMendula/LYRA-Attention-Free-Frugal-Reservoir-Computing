import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# 1. Definizione dei file
files = ['./csv/SMD-DeepRC-no.csv', './csv/SMD-DeepRC-ok.csv', './csv/SMD-MOMENT.csv']

# 2. Caricamento e unione dei dataset
dfs = []
for file_path in files:
    df = pd.read_csv(file_path)
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# Gestione delle deviazioni standard mancanti
std_cols = ['accuracy_regime_std', 'rmse_forecast_std', 'rmse_stability_std']
for col in std_cols:
    if col not in df_all.columns:
        df_all[col] = 0.0
    else:
        df_all[col] = df_all[col].fillna(0.0)

# 3. Pulizia e Unificazione dei Nomi degli Esperimenti
def unify_experiment_name(name):
    # Rimuoviamo i prefissi per raggruppare gli esperimenti equivalenti
    prefixes_to_remove = ['DeepRC_IP_', 'DeepRC_', 'MOMENT_Aligned_', 'MOMENT_']
    name_str = str(name)
    for prefix in prefixes_to_remove:
        if name_str.startswith(prefix):
            return name_str[len(prefix):]
    return name_str

# Creiamo una nuova colonna con il nome unificato
df_all['Unified_Name'] = df_all['Name'].apply(unify_experiment_name)

# 4. Raggruppamento sui nomi unificati
df_grouped = df_all.groupby(['Unified_Name', 'architecture']).mean(numeric_only=True).reset_index()

# 5. Funzione per l'ordinamento "Naturale"
# Serve per ordinare correttamente le stringhe che contengono numeri (es. 2, 4, 6, 10_Machines)
def natural_keys(text):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]

# Otteniamo gli esperimenti unificati e li ordiniamo in cascata
experiments = sorted(df_grouped['Unified_Name'].unique(), key=natural_keys)
experiments = [
    "WP4_Target_2-9",
    "WP4_Target_1-8",
    "WP3_Random_Mix_2",
    "WP3_Random_Mix_1",
    "WP2_LODO_Groups_1_and_2",
    "WP1_HighDiv_6_Machines",
    "WP1_MedDiv_4_Machines",
    "WP1_LowDiv_2_Machines",
]

experiment_labels = {
    "WP4_Target_2-9": "WP4 Target 2-9",
    "WP4_Target_1-8": "WP4 Target 1-8",
    "WP3_Random_Mix_2": "WP3 Random Mix 2",
    "WP3_Random_Mix_1": "WP3 Random Mix 1",
    "WP2_LODO_Groups_1_and_2": "WP2 LODO G1+G2",
    "WP1_HighDiv_6_Machines": "WP1 HighDiv (6M)",
    "WP1_MedDiv_4_Machines": "WP1 MedDiv (4M)",
    "WP1_LowDiv_2_Machines": "WP1 LowDiv (2M)"
}

# revert ordeer
experiments.reverse()

architectures = df_grouped['architecture'].unique()
selected_architectures = ['MOMENT-1-small','DeepRC_Ridge_MultiTap_Standard', 'LYRA', ]  # Architetture da visualizzare

architecture_labels = {
    'MOMENT-1-small': 'MOMENT-small',
    'DeepRC_Ridge_MultiTap_Standard': 'Standard DeepRC',
    'LYRA': 'LYRA'
}

# remove from architectures any that is different from selected_architectures
# architectures = [arch for arch in architectures if arch in selected_architectures]
architectures = selected_architectures

# 6. Configurazione delle metriche
metrics = {
    # place (simbol up arrow)
    '(a) Accuracy [%] (↑)': ('accuracy_regime', 'accuracy_regime_std'),
    '(b) Forecasting [RMSE] (↓)': ('rmse_forecast_denorm', 'rmse_forecast_std'),
    '(c) Reconstruction [RMSE] (↓)': ('rmse_stability', 'rmse_stability_std')
}

# Impostazioni per il clustered bar chart
x = np.arange(len(experiments))
width = 0.5 / len(architectures)

# 7. Creazione dei plot
fig, axes = plt.subplots(3, 1, figsize=(8, 10))
# fig.suptitle('Architectural Comparison per Work Package Experiment', fontsize=18, y=0.96)

cmap = plt.get_cmap('tab10') 

hatches = ['/', 'x', '\\', '-', '+', 'o', '.']

for i, (metric_title, (val_col, std_col)) in enumerate(metrics.items()):
    ax = axes[i]

    if i in [1, 2]:  # second and third subplot3    
        ax.set_yscale('log')
    
    for j, arch in enumerate(architectures):
        arch_means = []
        arch_stds = []
        
        for exp in experiments:
            subset = df_grouped[(df_grouped['Unified_Name'] == exp) & (df_grouped['architecture'] == arch)]
            if not subset.empty:
                arch_means.append(subset[val_col].values[0])
                corr = subset[std_col].values[0]
                if corr > 0.1:
                    corr = 0.02
                arch_stds.append(corr)
            else:
                arch_means.append(0)
                arch_stds.append(0)
                
        offset = x - (0.8 / 2) + (j + 0.5) * width

        print(f"Plotting {arch} for metric '{metric_title}' with means: {arch_means} and stds: {arch_stds}")

        if i == 0:
            ax.set_yticks([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
            ax.set_yticklabels(['10', '15', '20', '25', '30', '35', '40'])
        ax.bar(offset, arch_means, width, yerr=arch_stds, label=arch, 
               color=cmap(j % 10), edgecolor='black', hatch=hatches[j % len(hatches)], capsize=4, alpha=0.9)
        
    ax.set_title(metric_title, fontsize=15)
    ax.set_xticks(x)
    # Etichette basate sul nome unificato
    if i == len(metrics) - 1:
        ax.set_xticklabels([experiment_labels.get(e, e) for e in experiments], rotation=30, ha='right', fontsize=12)
    else:
        ax.set_xticklabels([])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # ax.set_ylabel('Valore della Metrica', fontsize=13)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, [architecture_labels.get(label, label) for label in labels], loc='upper center', bbox_to_anchor=(0.5, 0.96), 
           ncol=min(5, len(architectures)), fontsize=15, frameon=False)

plt.tight_layout()
plt.subplots_adjust(top=0.88, hspace=0.15)
plt.savefig('./plots/smd_plots.pdf', format='pdf', bbox_inches='tight')

# plt.show()
