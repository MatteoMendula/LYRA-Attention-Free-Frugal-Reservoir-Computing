import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# ==========================================
# 1. CARICAMENTO E PREPARAZIONE DATI
# ==========================================
files = ['./csv/SMD-DeepRC-no.csv', './csv/SMD-DeepRC-ok.csv', './csv/SMD-MOMENT.csv']

hatches = ['/', 'x', '\\', '-', '+', 'o', '.']

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

dfs = []
for file_path in files:
    try:
        df = pd.read_csv(file_path)
        dfs.append(df)
    except FileNotFoundError:
        pass

if not dfs:
    print("Errore: Impossibile trovare i file CSV nella directory corrente.")
    exit()

df_all = pd.concat(dfs, ignore_index=True)

def unify_experiment_name(name):
    prefixes = ['DeepRC_IP_', 'DeepRC_', 'MOMENT_Aligned_', 'MOMENT_']
    name_str = str(name)
    for p in prefixes:
        if name_str.startswith(p):
            return name_str[len(p):]
    return name_str

df_all['Unified_Name'] = df_all['Name'].apply(unify_experiment_name)

experiments = [
    "WP4_Target_2-9", "WP4_Target_1-8", "WP3_Random_Mix_2", "WP3_Random_Mix_1",
    "WP2_LODO_Groups_1_and_2", "WP1_HighDiv_6_Machines", "WP1_MedDiv_4_Machines", "WP1_LowDiv_2_Machines"
]
experiments.reverse()

architecture_labels = {
    'MOMENT-1-small': 'MOMENT-small',
    'DeepRC_Ridge_MultiTap_Standard': 'Standard DeepRC',
    'LYRA': 'LYRA'
}

selected_architectures = ['MOMENT-1-small', 'DeepRC_Ridge_MultiTap_Standard', 'LYRA']

df_all = df_all[df_all['architecture'].isin(selected_architectures)]
df_all['duration_sec'] = df_all['duration_sec'].fillna(0)

# ==========================================
# 2. CALCOLO DEI PARAMETRI (TOTALI E TRAINABILI)
# ==========================================
N_FEATURES = 38
N_CLASSES = 3   
H_FORECAST = 12 
F_DIM = 38      

def calculate_params(row):
    arch = row['architecture']
    if 'MOMENT' in arch:
        total_params = 14_500_000
        trainable = ( (768 + 38) * (H_FORECAST * F_DIM) ) + (768 * N_CLASSES) + (768 * N_FEATURES)
    else:
        n_units = row.get('tot_units', 500) if not pd.isna(row.get('tot_units', 500)) else 500
        n_layers = row.get('n_layers', 1) if not pd.isna(row.get('n_layers', 1)) else 1
        
        w_in = N_FEATURES * n_units
        w_res = (n_units * n_units) * n_layers
        
        readout_in = (n_units * n_layers) + N_FEATURES
        trainable = (readout_in * (H_FORECAST * F_DIM)) + (readout_in * N_CLASSES) + (readout_in * N_FEATURES)
        
        if 'IP' in arch:
            trainable += (n_units * 2 * n_layers) 
            
        total_params = w_in + w_res + trainable

    return pd.Series([total_params, trainable])

df_all[['Total_Params', 'Trainable_Params']] = df_all.apply(calculate_params, axis=1)

# ==========================================
# 3. CREAZIONE DELLA FIGURA E DEI SUBPLOT
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
# fig.suptitle('Efficiency Profiling: HPs Search Duration vs Parameter Count', fontsize=18, fontweight='bold', y=0.95)
cmap = plt.get_cmap('tab10')
colors = [cmap(0), cmap(1), cmap(2)]

# --- PLOT 1: HPs Search Duration (Bar Plot con Y Log) ---
df_grouped_dur = df_all.groupby(['Unified_Name', 'architecture'])['duration_sec'].agg(['mean', 'std']).reset_index()
df_grouped_dur['std'] = df_grouped_dur['std'].fillna(0)

x = np.arange(len(experiments))
width = 0.25

for j, arch in enumerate(selected_architectures):
    means, stds = [], []
    for exp in experiments:
        subset = df_grouped_dur[(df_grouped_dur['Unified_Name'] == exp) & (df_grouped_dur['architecture'] == arch)]
        if not subset.empty:
            means.append(subset['mean'].values[0])
            stds.append(subset['std'].values[0])
        else:
            means.append(0)
            stds.append(0)
            
    offset = x - width + (j * width)
    ax1.bar(offset, means, width, yerr=stds, label=architecture_labels.get(arch, arch), color=colors[j], edgecolor='black', hatch=hatches[j], capsize=4, alpha=0.9)

    print(f"Plotting {arch} for HP Search Duration with means: {means} and stds: {stds}")

ax1.set_title('(a) Hyperparameter Search Duration (200 Iterations) [s]', fontsize=15)
ax1.set_yscale('log') 
ax1.set_xticks(x)
ax1.set_xticklabels([experiment_labels.get(exp, exp) for exp in experiments], rotation=20, ha='right', fontsize=11)
ax1.set_ylabel('Duration [s]', fontsize=13)
ax1.grid(axis='y', linestyle='--', alpha=0.7, which='both')
ax1.legend(loc='upper left', fontsize=11)

# --- PLOT 2: Parameter Count (Clustered Bar Chart) ---
df_params = df_all.groupby('architecture')[['Trainable_Params', 'Total_Params']].mean().reset_index()
df_params['architecture'] = pd.Categorical(df_params['architecture'], categories=selected_architectures, ordered=True)
df_params = df_params.sort_values('architecture')

arch_labels = [architecture_labels.get(arch, arch) for arch in df_params['architecture'].values]
trainable_vals = df_params['Trainable_Params'].values
total_vals = df_params['Total_Params'].values
frozen_vals = total_vals - trainable_vals # I parametri NON allenabili

x_pos = np.arange(len(arch_labels))
bar_width = 0.35 # Larghezza ridotta per farle stare affiancate

# Barra 1: Parametri Allenabili (Sinistra)
p1 = ax2.bar(x_pos - bar_width/2, trainable_vals, bar_width, label='Trainable Parameters (Readout Layer)', 
             color='#FF6B6B', edgecolor='black', alpha=0.9)

# Barra 2: Parametri Congelati (Destra)
p2 = ax2.bar(x_pos + bar_width/2, frozen_vals, bar_width, 
             label='Frozen Parameters (Encoder / Reservoir)', 
             color='lightgray', hatch='//', edgecolor='black', alpha=0.9)

print(f"Plotting Parameter Counts with Trainable: {trainable_vals} and Frozen: {frozen_vals}")

# Aggiunta dei valori numerici in cima alle barre
for i in range(len(arch_labels)):
    # Valore Allenabili
    ax2.annotate(f'{int(trainable_vals[i]):,}',
                 xy=(x_pos[i] - bar_width/2, trainable_vals[i]), xytext=(0, 5),
                 textcoords="offset points", ha='center', va='bottom', 
                 fontsize=10, fontweight='bold', color='black')
    
    # Valore Congelati
    ax2.annotate(f'{int(frozen_vals[i]):,}',
                 xy=(x_pos[i] + bar_width/2, frozen_vals[i]), xytext=(0, 5),
                 textcoords="offset points", ha='center', va='bottom', 
                 fontsize=10, fontweight='bold', color='black')

ax2.set_title('(b) Frozen vs Trainable Parameters (Avg. across WPs)', fontsize=15)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(arch_labels, fontsize=12)
ax2.set_yscale('log')
# Forziamo il limite inferiore per dare respiro al grafico
ax2.set_ylim(bottom=10**4, top=max(total_vals)*2) 
ax2.set_ylabel('N. Params', fontsize=13)
ax2.grid(axis='y', linestyle='--', alpha=0.7, which='both')
ax2.legend(loc='upper right', fontsize=11)

plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.5)
plt.savefig('./plots/smd_efficiency.pdf', format='pdf', bbox_inches='tight')
print("Grafico salvato in formato vettoriale: smd_efficiency.pdf")
