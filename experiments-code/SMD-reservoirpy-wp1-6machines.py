import os
import time
import random
import numpy as np
import pandas as pd
import torch
import mlflow
import gc
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import time

# --- IMPORT RESERVOIRPY ---
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir
rpy.verbosity(0) # Disabilita i log prolissi durante la ricerca

# Moduli custom (assumendo siano disponibili nel tuo environment)
from utils import load_machine_data, robust_torch_ridge_fit

# =====================================================================
# CONFIGURAZIONE (Parità assoluta con DeepRC e MOMENT)
# =====================================================================
BASE_DIR = '/home/mmendula/last-rush/datasets/ServerMachineDataset'
WINDOW_SIZE = 60
FORECAST_HORIZON = 5
N_ITERATIONS = 100

SEEDS_SEARCH = [11, 22, 33]
SEEDS_TEST   = [11, 22, 33, 44, 55]
GLOBAL_SEED  = 42

SOFT_FLOOR_C = 0.02
CLIP_VAL     = 4.0      

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-8

# Spazio di ricerca adattato per ReservoirPy
HP_SPACE = {
    "tot_units": [50, 100, 250, 500, 1000],
    "spectral_radius": {"min": np.log(0.5), "max": np.log(1.2)},
    "leaky": [0.1, 0.3, 0.5, 1.0],
    "input_scaling": {"min": np.log(0.02), "max": np.log(0.5)},
    "ridge_alpha": {"min": np.log(1e-3), "max": np.log(1e4)}
}

# =====================================================================
# UTILITIES & RC-REVIN V5
# =====================================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    rpy.set_seed(seed)
    torch.manual_seed(seed)

def create_sequences_rcrevin(data_global, window_size, horizon):
    n = len(data_global) - window_size - horizon + 1
    X_glob = np.stack([data_global[i:i + window_size] for i in range(n)]).astype(np.float32)
    y_glob = np.stack([data_global[i + window_size:i + window_size + horizon] for i in range(n)]).astype(np.float32)

    mu = X_glob.mean(axis=1, keepdims=True)
    sd = X_glob.std(axis=1, keepdims=True)
    sd = np.sqrt(sd ** 2 + SOFT_FLOOR_C ** 2)

    X_loc = (X_glob - mu) / sd
    y_loc = (y_glob - mu) / sd

    np.clip(X_loc, -CLIP_VAL, CLIP_VAL, out=X_loc)

    x_last_loc = X_loc[:, -1, :].copy()
    x_last_glob = X_glob[:, -1, :].copy()
    return X_loc, y_loc, y_glob, x_last_loc, x_last_glob, mu.squeeze(1), sd.squeeze(1)

def load_and_process_machine(raw_data, machine_name, split_type, scaler):
    p_label = os.path.join(BASE_DIR, 'regime_labels', f"{split_type}_{machine_name.replace('.txt', '.npy')}")
    labels_raw = np.load(p_label)
    scaled_data = scaler.transform(raw_data)
    X_loc, y_loc, y_glob, x_last_loc, x_last_glob, mu, sd = create_sequences_rcrevin(scaled_data, WINDOW_SIZE, FORECAST_HORIZON)
    labels = labels_raw[WINDOW_SIZE - 1 : WINDOW_SIZE - 1 + len(X_loc)]
    return X_loc, y_loc, y_glob, x_last_loc, x_last_glob, mu, sd, labels

def add_bias(R):
    return np.hstack([R, np.ones((R.shape[0], 1), dtype=R.dtype)])

def naive_forecast_rmse(x_last_glob, y_glob):
    naive = np.repeat(x_last_glob[:, np.newaxis, :], y_glob.shape[1], axis=1)
    return np.sqrt(mean_squared_error(
        y_glob.reshape(len(y_glob), -1), naive.reshape(len(naive), -1)))

# =====================================================================
# RESERVOIRPY WRAPPER & EVALUATION
# =====================================================================
def get_rpy_states_in_batches(node, X_numpy, batch_size=2048, noise_std=0.0):
    states = []
    
    # Lavoriamo direttamente sul tensore numpy originario per iniettare il rumore
    X_data = X_numpy.copy()
    if noise_std > 0.0:
        X_data += np.random.normal(0, noise_std, X_data.shape)
        
    # Risoluzione del bug: Inizializziamo esplicitamente il nodo usando la prima sequenza.
    # Questo bypassa la logica fallata di self.initialize() su liste di array in reservoirpy.
    if not node.is_initialized:
        node.initialize(X_data[0])
        
    for i in range(0, len(X_data), batch_size):
        batch = X_data[i:i+batch_size]
        batch_states = []
        
        # Iteriamo esplicitamente sulla sequenza per garantire il reset=True indipendente
        for seq in batch:
            out = node.run(seq, reset=True)
            batch_states.append(out[-1]) # Estraiamo solo l'ultimo stato h(T)
            
        states.append(np.stack(batch_states))
        
    return np.concatenate(states, axis=0)

def evaluate_rpy_pipeline(params, seed, 
                          X_loc_fit, y_loc_fit, y_glob_fit, x_last_loc_fit, mu_fit, sd_fit, reg_fit,
                          X_loc_ev, y_loc_ev, y_glob_ev, x_last_loc_ev, mu_ev, sd_ev, reg_ev, naive_rmse, classes):
    
    seed_everything(seed)
    
    # Istanziamo un nodo Reservoir base di ReservoirPy
    units = int(round(params["tot_units"]))
    node = Reservoir(
        units=units,
        sr=params["spectral_radius"],
        lr=params["leaky"],
        input_scaling=params["input_scaling"],
        rc_connectivity=10/units, # Approssimazione alla connectivity di DeepRC (10)
        input_connectivity=10/X_loc_fit.shape[2]
    )

    noise_std = 0.05
    ridge_alpha = params.get("ridge_alpha", 1e-3)

    R_fit = get_rpy_states_in_batches(node, X_loc_fit)
    R_ev = get_rpy_states_in_batches(node, X_loc_ev)
    R_fit_n = get_rpy_states_in_batches(node, X_loc_fit, noise_std=noise_std)
    R_ev_n = get_rpy_states_in_batches(node, X_loc_ev, noise_std=noise_std)

    # ---- Head 1: Forecasting (Delta Prediction) ----
    F_fit = add_bias(np.hstack([R_fit, x_last_loc_fit]))
    F_ev  = add_bias(np.hstack([R_ev, x_last_loc_ev]))

    H, F_dim = y_loc_fit.shape[1], y_loc_fit.shape[2]
    baseline_fit = np.repeat(x_last_loc_fit[:, None, :], H, axis=1)
    baseline_ev  = np.repeat(x_last_loc_ev[:, None, :], H, axis=1)

    target_delta_fit_flat = (y_loc_fit - baseline_fit).reshape(len(y_loc_fit), -1)
    
    W_fore = robust_torch_ridge_fit(F_fit, target_delta_fit_flat, ridge_alpha, DEVICE)
    pred_delta_flat = (torch.from_numpy(F_ev).float().to(DEVICE) @ W_fore).cpu().numpy()

    pred_loc = baseline_ev + pred_delta_flat.reshape(len(y_loc_ev), H, F_dim)
    pred_glob = pred_loc * sd_ev[:, None, :] + mu_ev[:, None, :]

    rmse_fore = np.sqrt(mean_squared_error(
        y_glob_ev.reshape(len(y_glob_ev), -1), pred_glob.reshape(len(pred_glob), -1)
    ))

    # ---- Head 2: Regime Classification ----
    C_fit = add_bias(R_fit)
    C_ev = add_bias(R_ev)
    alpha_reg = ridge_alpha * 100.0 
    
    y_reg_oh = (reg_fit[:len(C_fit)][:, None] == classes[None, :]).astype(np.float32)
    W_reg = robust_torch_ridge_fit(C_fit, y_reg_oh, alpha_reg, DEVICE)
    
    pred_idx = np.argmax((torch.from_numpy(C_ev).float().to(DEVICE) @ W_reg).cpu().numpy(), axis=1)
    acc_reg = accuracy_score(reg_ev[:len(C_ev)], classes[pred_idx])

    # ---- Head 3: Reconstruction ----
    S_fit = add_bias(R_fit_n)
    S_ev = add_bias(R_ev_n)
    alpha_rec = ridge_alpha * 10.0 
    
    W_recon = robust_torch_ridge_fit(S_fit, x_last_loc_fit, alpha_rec, DEVICE)
    pred_recon = (torch.from_numpy(S_ev).float().to(DEVICE) @ W_recon).cpu().numpy()
    rmse_recon = np.sqrt(mean_squared_error(x_last_loc_ev, pred_recon))

    # ---- Scoring Composite (Standard LYRA Paper) ----
    s_fore = 1.0 / (1.0 + rmse_fore / (naive_rmse + EPS))
    score = 0.3 * s_fore + 0.4 * acc_reg + 0.3 * (1.0 / (1.0 + rmse_recon))

    return score, rmse_fore, acc_reg, rmse_recon, pred_glob

# =====================================================================
# MAIN EXECUTION (Solo WP1_HighDiv_6_Machines)
# =====================================================================
if __name__ == "__main__":
    seed_everything(GLOBAL_SEED)
    
    # Target dell'esperimento isolato
    exp = {
        "name": "WP1_HighDiv_6_Machines",
        "corpus": ['machine-1-1.txt', 'machine-1-2.txt', 'machine-2-1.txt', 'machine-2-2.txt', 'machine-3-1.txt', 'machine-3-2.txt'],
        "val_machine": 'machine-1-2.txt',
        "target": 'machine-3-11.txt'
    }

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("IEEE_COMPARISON_MOMENT_BENCHMARK-v13-noIP")

    print(f"\n[START] RESERVOIRPY BENCHMARK FAIR COMPARISON: {exp['name']}")

    val_machine = exp["val_machine"]
    train_machines = [m for m in exp["corpus"] if m != val_machine]

    # Caricamento e Fitting dello Scaler GLOBALE
    raw_tr_dict = {m: load_machine_data(os.path.join(BASE_DIR, 'train', m)) for m in train_machines}
    scaler = MinMaxScaler().fit(np.vstack(list(raw_tr_dict.values())))

    # Process Train
    tr_Xl, tr_yl, tr_yg, tr_xll, tr_m, tr_s, tr_reg = [], [], [], [], [], [], []
    for m in train_machines:
        Xl, yl, yg, xll, _, mu, sd, lab = load_and_process_machine(raw_tr_dict[m], m, 'train', scaler)
        tr_Xl.append(Xl); tr_yl.append(yl); tr_yg.append(yg); tr_xll.append(xll)
        tr_m.append(mu); tr_s.append(sd); tr_reg.append(lab)

    X_loc_train, y_loc_train, y_glob_train = np.concatenate(tr_Xl, axis=0), np.concatenate(tr_yl, axis=0), np.concatenate(tr_yg, axis=0)
    x_last_loc_train = np.concatenate(tr_xll, axis=0)
    mu_train, sd_train, regime_train = np.concatenate(tr_m, axis=0), np.concatenate(tr_s, axis=0), np.concatenate(tr_reg, axis=0)

    # Process Val & Test
    raw_val = load_machine_data(os.path.join(BASE_DIR, 'train', val_machine))
    raw_test = load_machine_data(os.path.join(BASE_DIR, 'test', exp['target']))
    
    X_loc_val, y_loc_val, y_glob_val, x_last_loc_val, x_last_glob_val, mu_val, sd_val, regime_val = load_and_process_machine(raw_val, val_machine, 'train', scaler)
    X_loc_test, y_loc_test, y_glob_test, x_last_loc_test, x_last_glob_test, mu_test, sd_test, regime_test = load_and_process_machine(raw_test, exp['target'], 'test', scaler)

    # Process Full Train (Train + Val)
    X_loc_full = np.concatenate([X_loc_train, X_loc_val], axis=0)
    y_loc_full = np.concatenate([y_loc_train, y_loc_val], axis=0)
    y_glob_full = np.concatenate([y_glob_train, y_glob_val], axis=0)
    x_last_loc_full = np.concatenate([x_last_loc_train, x_last_loc_val], axis=0)
    mu_full, sd_full = np.concatenate([mu_train, mu_val], axis=0), np.concatenate([sd_train, sd_val], axis=0)
    regime_full = np.concatenate([regime_train, regime_val], axis=0)

    classes = np.unique(regime_full)
    naive_rmse_val = naive_forecast_rmse(x_last_glob_val, y_glob_val)
    naive_rmse_test = naive_forecast_rmse(x_last_glob_test, y_glob_test)

    # ==================== RANDOM SEARCH ====================
    best_score = -np.inf
    best_params = {}
    best_params["architecture"] = "reservoirpy"

    start_time = time.time() # <--- INIZIO TIMER
    print(f"Esecuzione Search su {N_ITERATIONS} iterazioni...")
    for i in range(N_ITERATIONS):
        # Sampling HPs
        print(f"\n[Iter {i+1}/{N_ITERATIONS}] Sampling Hyperparameters...")
        params = {
            "tot_units": random.choice(HP_SPACE["tot_units"]),
            "spectral_radius": np.exp(random.uniform(HP_SPACE["spectral_radius"]["min"], HP_SPACE["spectral_radius"]["max"])),
            "leaky": random.choice(HP_SPACE["leaky"]),
            "input_scaling": np.exp(random.uniform(HP_SPACE["input_scaling"]["min"], HP_SPACE["input_scaling"]["max"])),
            "ridge_alpha": np.exp(random.uniform(HP_SPACE["ridge_alpha"]["min"], HP_SPACE["ridge_alpha"]["max"]))
        }

        # Valutazione multi-seed su validation set
        val_scores = []
        for seed in SEEDS_SEARCH:
            score, _, _, _, _ = evaluate_rpy_pipeline(
                params, seed, 
                X_loc_train, y_loc_train, y_glob_train, x_last_loc_train, mu_train, sd_train, regime_train,
                X_loc_val, y_loc_val, y_glob_val, x_last_loc_val, mu_val, sd_val, regime_val, naive_rmse_val, classes
            )
            val_scores.append(score)
            
        mean_score = np.mean(val_scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params.copy()
            print(f"Iter {i+1}: Nuovo Best Score = {best_score:.4f} | Param: {params}")

    duration = time.time() - start_time # <--- FINE TIMER
    print(f"\n[SEARCH COMPLETED] Best Score: {best_score:.4f} | Best Params: {best_params} | Duration: {duration:.2f}s")
    # ==================== TEST FINAL EVALUATION ====================
    print("\nEsecuzione Test Evaluation sui parametri migliori...")
    test_results = []
    rmse_denorm_list = []
    H, F_dim = y_glob_test.shape[1], y_glob_test.shape[2]

    with mlflow.start_run(run_name=f"RPY_Baseline_{exp['name']}"):
        mlflow.log_params(best_params)
        mlflow.log_metric("duration_sec", duration)
        
        for seed in SEEDS_TEST:
            score, r_fore, acc_reg, r_recon, pred_glob = evaluate_rpy_pipeline(
                best_params, seed, 
                X_loc_full, y_loc_full, y_glob_full, x_last_loc_full, mu_full, sd_full, regime_full,
                X_loc_test, y_loc_test, y_glob_test, x_last_loc_test, mu_test, sd_test, regime_test, naive_rmse_test, classes
            )
            
            test_results.append({"score": score, "rmse_forecast": r_fore, "acc_regime": acc_reg, "rmse_recon": r_recon})
            
            pred_physical = scaler.inverse_transform(pred_glob.reshape(-1, F_dim)).reshape(len(pred_glob), H, F_dim)
            y_physical = scaler.inverse_transform(y_glob_test.reshape(-1, F_dim)).reshape(len(y_glob_test), H, F_dim)
            rmse_denorm_list.append(np.sqrt(mean_squared_error(y_physical.reshape(len(y_physical), -1), pred_physical.reshape(len(pred_physical), -1))))
            gc.collect()

        # Log MLFlow
        mlflow.log_metric("val_composite_score", best_score)
        for k in ["score", "rmse_forecast", "acc_regime", "rmse_recon"]:
            vals = [m[k] for m in test_results]
            mlflow.log_metric(f"composite_{k}" if k == "score" else k, float(np.mean(vals)))
            mlflow.log_metric(f"composite_{k}_std" if k == "score" else f"{k}_std", float(np.std(vals)))
            
        rmse_denorm_mean = float(np.mean(rmse_denorm_list))
        mlflow.log_metric("rmse_forecast_denorm", rmse_denorm_mean)
        mlflow.log_metric("rmse_forecast_naive", naive_rmse_test)

        print(f"\n[RISULTATI FINALI TEST] {exp['name']}")
        print(f" - Acc Regime: {np.mean([m['acc_regime'] for m in test_results]):.2%}")
        print(f" - RMSE (Glob): {np.mean([m['rmse_forecast'] for m in test_results]):.4f}")
        print(f" - Naive (Glob): {naive_rmse_test:.4f}")
        print(f" - RMSE (Physical): {rmse_denorm_mean:.4f}")