import os
import time
import random
import numpy as np
import pandas as pd
import torch
import mlflow
import gc
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import torch.nn.functional as F

# --- IMPORT MOMENT ---
from momentfm import MOMENTPipeline

from utils import load_machine_data

# =====================================================================
# CONFIGURATION
# =====================================================================
BASE_DIR = '/home/mmendula/last-rush/datasets/ServerMachineDataset'

WINDOW_SIZE = 60
FORECAST_HORIZON = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_ITERATIONS = 200 # Aligned to DeepRC search budget

SEEDS_TEST = [11, 22, 33, 44, 55]
GLOBAL_SEED = 42

# --- RC-RevIN v5: soft floor + defensive clipping (Aligned) ---
SOFT_FLOOR_C = 0.02     
CLIP_VAL     = 4.0      

EPS = 1e-8

# =====================================================================
# UTILITIES (Structurally Aligned with DeepRC)
# =====================================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_sequences_rcrevin(data_global, window_size, horizon):
    """Exact match to DeepRC's RC-RevIN v5 Normalization logic"""
    n = len(data_global) - window_size - horizon + 1
    if n <= 0:
        raise ValueError("Series too short for requested window+horizon.")

    X_glob = np.stack([data_global[i:i + window_size] for i in range(n)]).astype(np.float32)
    y_glob = np.stack([data_global[i + window_size:i + window_size + horizon] for i in range(n)]).astype(np.float32)

    mu = X_glob.mean(axis=1, keepdims=True)
    sd = X_glob.std(axis=1, keepdims=True)

    # SOFT FLOOR
    sd = np.sqrt(sd ** 2 + SOFT_FLOOR_C ** 2)

    X_loc = (X_glob - mu) / sd
    y_loc = (y_glob - mu) / sd

    # Defensive Clipping
    np.clip(X_loc, -CLIP_VAL, CLIP_VAL, out=X_loc)

    return X_loc, y_loc, y_glob, mu.squeeze(1), sd.squeeze(1)

def naive_forecast_rmse(X_loc, y_glob, mu, sd):
    """Calculated on the global/denormalized scale to match DeepRC"""
    last_loc = X_loc[:, -1:, :]
    # Denormalize the last local known value back to global scale
    last_glob = last_loc * sd[:, np.newaxis, :] + mu[:, np.newaxis, :]
    naive = np.repeat(last_glob, y_glob.shape[1], axis=1)       
    return np.sqrt(mean_squared_error(
        y_glob.reshape(len(y_glob), -1), naive.reshape(len(naive), -1)))

def load_and_process_machine(raw_data, machine_name, split_type, scaler):
    p_label = os.path.join(BASE_DIR, 'regime_labels', f"{split_type}_{machine_name.replace('.txt', '.npy')}")
    labels_raw = np.load(p_label)
    
    scaled_data = scaler.transform(raw_data)
    X_loc, y_loc, y_glob, mu, sd = create_sequences_rcrevin(scaled_data, WINDOW_SIZE, FORECAST_HORIZON)
    labels = labels_raw[WINDOW_SIZE - 1 : WINDOW_SIZE - 1 + len(X_loc)]
    
    return X_loc, y_loc, y_glob, mu, sd, labels

# =====================================================================
# MOMENT SPECIFIC FUNCTIONS
# =====================================================================
def get_moment_embeddings_in_batches(model, X_tensor, batch_size=128, noise_std=0.0):
    states = []
    model.eval()
    
    if noise_std > 0.0:
        X_tensor = X_tensor + torch.randn_like(X_tensor) * noise_std
        
    X_tensor = X_tensor.permute(0, 2, 1)
    seq_len = X_tensor.shape[2]
    pad_len = (8 - (seq_len % 8)) % 8

    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size].to(DEVICE)
            
            if pad_len > 0:
                batch = F.pad(batch, (0, pad_len), "constant", 0)
                mask = torch.ones((batch.shape[0], batch.shape[2]), dtype=torch.long, device=DEVICE)
                mask[:, -pad_len:] = 0
            else:
                mask = None
            
            outputs = model(x_enc=batch, input_mask=mask)
            embeddings_flat = outputs.embeddings.reshape(batch.shape[0], -1).cpu().numpy()
            states.append(embeddings_flat)
            
            del batch, outputs, mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    return np.concatenate(states, axis=0)

def evaluate_moment_pipeline(X_train, y_train_loc, y_train_glob, regime_train,
                             X_val, y_val_loc, y_val_glob, regime_val, val_mu, val_sd,
                             X_train_full, y_train_full_loc, y_train_full_glob, regime_train_full,
                             X_test, y_test_loc, y_test_glob, regime_test, test_mu, test_sd, scaler,
                             model_size="small"):
    
    model_name = f"AutonLab/MOMENT-1-{model_size}"
    print(f"Loading MOMENT model: {model_name}...")
    
    moment_model = MOMENTPipeline.from_pretrained(
        model_name, 
        model_kwargs={"task_name": "embedding"} 
    )
    moment_model.init()
    moment_model = moment_model.to(DEVICE)

    noise_std = 0.05
    X_tr_t = torch.from_numpy(X_train).float()
    X_val_t = torch.from_numpy(X_val).float()
    X_tr_full_t = torch.from_numpy(X_train_full).float()
    X_test_t = torch.from_numpy(X_test).float()

    print("Extracting MOMENT embeddings...")
    # Clean embeddings for Head 1 & 2
    R_tr = get_moment_embeddings_in_batches(moment_model, X_tr_t)
    R_val = get_moment_embeddings_in_batches(moment_model, X_val_t)
    R_tr_full = get_moment_embeddings_in_batches(moment_model, X_tr_full_t)
    
    # Noisy embeddings for Head 3 (Extracted once to save compute during 200 iteration search)
    R_tr_n = get_moment_embeddings_in_batches(moment_model, X_tr_t, noise_std=noise_std)
    R_val_n = get_moment_embeddings_in_batches(moment_model, X_val_t, noise_std=noise_std)
    
    # Naive baselines for composite score calculation
    naive_rmse_val = naive_forecast_rmse(X_val, y_val_glob, val_mu, val_sd)
    naive_rmse_test = naive_forecast_rmse(X_test, y_test_glob, test_mu, test_sd)

    # Flatten targets
    y_tr_loc_flat = y_train_loc.reshape(len(y_train_loc), -1)
    y_tr_full_loc_flat = y_train_full_loc.reshape(len(y_train_full_loc), -1)
    
    # Target for Reconstruction (Head 3)
    tgt_tr = X_train[:, -1, :]
    tgt_val = X_val[:, -1, :]
    tgt_tr_full = X_train_full[:, -1, :]

    # Aligned Features (Adding last state to Head 1 like DeepRC)
    F_tr = np.hstack([R_tr, X_train[:, -1, :]])
    F_val = np.hstack([R_val, X_val[:, -1, :]])
    F_tr_full = np.hstack([R_tr_full, X_train_full[:, -1, :]])

    print(f"Tuning Readouts ({N_ITERATIONS} Iteration Random Search)...")
    best_val_score = -np.inf
    best_alpha = None
    
    np.random.seed(GLOBAL_SEED)
    
    for i in range(N_ITERATIONS):

        print(f"Iteration {i+1}/{N_ITERATIONS}...", end="\r")

        # Draw from log-uniform space matching DeepRC's 1e-6 to 1e-2 range
        alpha = np.exp(np.random.uniform(np.log(1e-6), np.log(1e-2)))
        
        # Head 1 (Forecast) -> Fit on local, evaluate on global (like DeepRC)
        clf_f = Ridge(alpha=alpha, solver='lsqr').fit(F_tr, y_tr_loc_flat)
        pred_loc_flat = clf_f.predict(F_val)
        
        H, F_dim = y_val_glob.shape[1], y_val_glob.shape[2]
        pred_loc = pred_loc_flat.reshape(len(pred_loc_flat), H, F_dim)
        pred_glob = pred_loc * val_sd[:, None, :] + val_mu[:, None, :]
        
        rmse_f = np.sqrt(mean_squared_error(
            y_val_glob.reshape(len(y_val_glob), -1),
            pred_glob.reshape(len(pred_glob), -1)
        ))
        
        # Head 2 (Regime)
        clf_reg = RidgeClassifier(alpha=alpha, solver='lsqr').fit(R_tr, regime_train)
        acc_reg = accuracy_score(regime_val, clf_reg.predict(R_val))
        
        # Head 3 (Reconstruction)
        clf_rec = Ridge(alpha=alpha, solver='lsqr').fit(R_tr_n, tgt_tr)
        rmse_rec = np.sqrt(mean_squared_error(tgt_val, clf_rec.predict(R_val_n)))
        
        # Aligned Composite Score
        s_fore = 1.0 / (1.0 + rmse_f / (naive_rmse_val + EPS))
        s_reg = acc_reg
        s_stab = 1.0 / (1.0 + rmse_rec)
        val_score = 0.3 * s_fore + 0.4 * s_reg + 0.3 * s_stab
        
        if val_score > best_val_score:
            best_val_score = val_score
            best_alpha = alpha

    print(f"Best Alpha selected: {best_alpha:.6f} (Val Score: {best_val_score:.4f})")

    # Fit final models on Full Train Set with Best Alpha
    R_tr_full_n = get_moment_embeddings_in_batches(moment_model, X_tr_full_t, noise_std=noise_std)
    
    clf_fore_final = Ridge(alpha=best_alpha, solver='lsqr').fit(F_tr_full, y_tr_full_loc_flat)
    clf_reg_final = RidgeClassifier(alpha=best_alpha, solver='lsqr').fit(R_tr_full, regime_train_full)
    clf_rec_final = Ridge(alpha=best_alpha, solver='lsqr').fit(R_tr_full_n, tgt_tr_full)
    
    # -------------------------------------------------------------
    # Multi-Seed Evaluation for Test
    # -------------------------------------------------------------
    print(f"Running Multi-Seed Test Evaluation over {len(SEEDS_TEST)} seeds...")
    test_results = []
    rmse_denorm_list = []
    
    # Clean embeddings and predictions are deterministic for the frozen model
    R_test = get_moment_embeddings_in_batches(moment_model, X_test_t)
    F_test = np.hstack([R_test, X_test[:, -1, :]])
    
    pred_loc_flat = clf_fore_final.predict(F_test)
    pred_reg = clf_reg_final.predict(R_test)
    
    H, F_dim = y_test_glob.shape[1], y_test_glob.shape[2]
    pred_loc = pred_loc_flat.reshape(len(pred_loc_flat), H, F_dim)
    pred_glob = pred_loc * test_sd[:, None, :] + test_mu[:, None, :]
    
    rmse_fore = np.sqrt(mean_squared_error(
        y_test_glob.reshape(len(y_test_glob), -1), 
        pred_glob.reshape(len(pred_glob), -1)
    ))
    acc_reg = accuracy_score(regime_test, pred_reg)
    
    # Fully Denormalized Forecast (Physical scale)
    pred_physical = scaler.inverse_transform(pred_glob.reshape(-1, F_dim)).reshape(len(pred_glob), H, F_dim)
    y_physical = scaler.inverse_transform(y_test_glob.reshape(-1, F_dim)).reshape(len(y_test_glob), H, F_dim)
    
    rmse_den = np.sqrt(mean_squared_error(
        y_physical.reshape(len(y_physical), -1), 
        pred_physical.reshape(len(pred_physical), -1)
    ))
    
    tgt_test = X_test[:, -1, :]

    for seed in SEEDS_TEST:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Head 3 injects noise, so embeddings change per seed
        R_test_n = get_moment_embeddings_in_batches(moment_model, X_test_t, noise_std=noise_std)
        pred_recon = clf_rec_final.predict(R_test_n)
        rmse_rec = np.sqrt(mean_squared_error(tgt_test, pred_recon))
        
        s_fore = 1.0 / (1.0 + rmse_fore / (naive_rmse_test + EPS))
        s_reg = acc_reg
        s_stab = 1.0 / (1.0 + rmse_rec)
        score = 0.3 * s_fore + 0.4 * s_reg + 0.3 * s_stab
        
        test_results.append({
            "score": score,
            "rmse_recon": rmse_rec
        })
        
    agg = {
        "val_composite_score": best_val_score,
        "best_alpha": best_alpha,
        "rmse_forecast": rmse_fore,
        "rmse_forecast_std": 0.0, # Deterministic
        "rmse_forecast_naive": naive_rmse_test,
        "rmse_forecast_denorm": rmse_den,
        "acc_regime": acc_reg,
        "acc_regime_std": 0.0,    # Deterministic
        "rmse_recon": float(np.mean([m["rmse_recon"] for m in test_results])),
        "rmse_recon_std": float(np.std([m["rmse_recon"] for m in test_results])),
        "composite_score": float(np.mean([m["score"] for m in test_results])),
        "composite_score_std": float(np.std([m["score"] for m in test_results]))
    }

    del moment_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return agg

# =====================================================================
# MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    
    experiments = [
        {"name": "WP1_LowDiv_2_Machines", "corpus": ['machine-1-1.txt', 'machine-1-2.txt'], "val_machine": 'machine-1-2.txt', "target": 'machine-3-11.txt'},
        {"name": "WP1_MedDiv_4_Machines", "corpus": ['machine-1-1.txt', 'machine-1-2.txt', 'machine-2-1.txt', 'machine-2-2.txt'], "val_machine": 'machine-1-2.txt', "target": 'machine-3-11.txt'},
        {"name": "WP1_HighDiv_6_Machines", "corpus": ['machine-1-1.txt', 'machine-1-2.txt', 'machine-2-1.txt', 'machine-2-2.txt', 'machine-3-1.txt', 'machine-3-2.txt'], "val_machine": 'machine-1-2.txt', "target": 'machine-3-11.txt'},
        {"name": "WP2_LODO_Groups_1_and_2", "corpus": ['machine-1-1.txt', 'machine-1-2.txt', 'machine-1-3.txt', 'machine-2-1.txt', 'machine-2-2.txt', 'machine-2-3.txt'], "val_machine": 'machine-2-3.txt', "target": 'machine-3-11.txt'},
        {"name": "WP3_Random_Mix_1", "corpus": ['machine-1-3.txt', 'machine-1-5.txt', 'machine-2-4.txt', 'machine-2-7.txt', 'machine-3-4.txt', 'machine-3-8.txt'], "val_machine": 'machine-2-7.txt', "target": 'machine-3-11.txt'},
        {"name": "WP3_Random_Mix_2", "corpus": ['machine-1-4.txt', 'machine-1-7.txt', 'machine-2-3.txt', 'machine-2-5.txt', 'machine-3-5.txt', 'machine-3-9.txt'], "val_machine": 'machine-2-5.txt', "target": 'machine-3-11.txt'},
        {"name": "WP4_Target_1-8", "corpus": ['machine-1-1.txt', 'machine-1-2.txt', 'machine-2-1.txt', 'machine-2-2.txt', 'machine-3-1.txt', 'machine-3-2.txt'], "val_machine": 'machine-2-2.txt', "target": 'machine-1-8.txt'},
        {"name": "WP4_Target_2-9", "corpus": ['machine-1-1.txt', 'machine-1-2.txt', 'machine-2-1.txt', 'machine-2-2.txt', 'machine-3-1.txt', 'machine-3-2.txt'], "val_machine": 'machine-2-2.txt', "target": 'machine-2-9.txt'}
    ]

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("IEEE_COMPARISON_MOMENT_BENCHMARK-v3-Aligned")

    print(f"--- STARTING ALIGNED MOMENT BENCHMARK ({len(experiments)} Runs) ---")
    print("NOTE: Global MinMaxScaler + RC-RevIN v5 and 200 Iteration Ridge Search Active.")

    for exp in experiments:
        seed_everything(GLOBAL_SEED)

        print("\n" + "="*60)
        print(f"RUNNING EXPERIMENT: {exp['name']}")
        print("="*60)

        with mlflow.start_run(run_name=f"MOMENT_Aligned_{exp['name']}"):
            model_size = "small"
            
            mlflow.log_param("architecture", f"MOMENT-1-{model_size}")
            mlflow.log_param("target_machine", exp['target'])
            mlflow.log_param("val_machine", exp['val_machine'])
            mlflow.log_param("corpus_size", len(exp['corpus']))
            mlflow.log_param("normalization", "Dual_Stream_RC_RevIN_v5_SoftFloor")
            mlflow.log_param("scaler_fit_scope", "train_machines_only")
            mlflow.log_param("soft_floor_c", SOFT_FLOOR_C)
            mlflow.log_param("clip_val", CLIP_VAL)
            mlflow.log_param("n_iterations", N_ITERATIONS)
            
            val_machine = exp["val_machine"]
            train_machines = [m for m in exp["corpus"] if m != val_machine]

            raw_tr_dict = {}
            for m in train_machines:
                raw_tr_dict[m] = load_machine_data(os.path.join(BASE_DIR, 'train', m))

            raw_val = load_machine_data(os.path.join(BASE_DIR, 'train', val_machine))
            raw_test = load_machine_data(os.path.join(BASE_DIR, 'test', exp['target']))

            # 1. FIT MINMAX SCALER GLOBALLY (Matched to DeepRC)
            scaler = MinMaxScaler()
            scaler.fit(np.vstack(list(raw_tr_dict.values())))

            # 2. PROCESS TRAIN DATA
            tr_Xl, tr_yl, tr_yg, tr_m, tr_s, tr_reg = [], [], [], [], [], []
            for m in train_machines:
                Xl, yl, yg, mu, sd, lab = load_and_process_machine(raw_tr_dict[m], m, 'train', scaler)
                tr_Xl.append(Xl); tr_yl.append(yl); tr_yg.append(yg)
                tr_m.append(mu); tr_s.append(sd); tr_reg.append(lab)

            X_loc_train = np.concatenate(tr_Xl, axis=0)
            y_loc_train = np.concatenate(tr_yl, axis=0)
            y_glob_train = np.concatenate(tr_yg, axis=0)
            mu_train = np.concatenate(tr_m, axis=0)
            sd_train = np.concatenate(tr_s, axis=0)
            regime_train = np.concatenate(tr_reg, axis=0)
            del tr_Xl, tr_yl, tr_yg, tr_m, tr_s, tr_reg

            # 3. PROCESS VAL DATA
            X_loc_val, y_loc_val, y_glob_val, mu_val, sd_val, regime_val = load_and_process_machine(raw_val, val_machine, 'train', scaler)

            X_loc_full = np.concatenate([X_loc_train, X_loc_val], axis=0)
            y_loc_full = np.concatenate([y_loc_train, y_loc_val], axis=0)
            y_glob_full = np.concatenate([y_glob_train, y_glob_val], axis=0)
            regime_full = np.concatenate([regime_train, regime_val], axis=0)

            # 4. PROCESS TEST DATA
            X_loc_test, y_loc_test, y_glob_test, mu_test, sd_test, regime_test = load_and_process_machine(raw_test, exp['target'], 'test', scaler)

            # 5. RUN EVALUATION
            start = time.time()
            results = evaluate_moment_pipeline(
                X_train=X_loc_train, y_train_loc=y_loc_train, y_train_glob=y_glob_train, regime_train=regime_train,
                X_val=X_loc_val, y_val_loc=y_loc_val, y_val_glob=y_glob_val, regime_val=regime_val, val_mu=mu_val, val_sd=sd_val,
                X_train_full=X_loc_full, y_train_full_loc=y_loc_full, y_train_full_glob=y_glob_full, regime_train_full=regime_full,
                X_test=X_loc_test, y_test_loc=y_loc_test, y_test_glob=y_glob_test, regime_test=regime_test, test_mu=mu_test, test_sd=sd_test,
                scaler=scaler, model_size=model_size
            )
            duration = time.time() - start

            # 6. LOG RESULTS TO MLFLOW
            mlflow.log_metric("val_composite_score", results['val_composite_score'])
            mlflow.log_metric("composite_score", results['composite_score'])
            mlflow.log_metric("composite_score_std", results['composite_score_std'])
            mlflow.log_metric("accuracy_regime", results['acc_regime'])
            mlflow.log_metric("accuracy_regime_std", results['acc_regime_std'])
            mlflow.log_metric("rmse_forecast", results['rmse_forecast'])
            mlflow.log_metric("rmse_forecast_std", results['rmse_forecast_std'])
            mlflow.log_metric("rmse_forecast_naive", results['rmse_forecast_naive'])
            mlflow.log_metric("rmse_forecast_denorm", results['rmse_forecast_denorm'])
            mlflow.log_metric("rmse_stability", results['rmse_recon'])
            mlflow.log_metric("rmse_stability_std", results['rmse_recon_std'])
            mlflow.log_metric("duration_sec", duration)
            mlflow.log_param("best_alpha", results['best_alpha'])

            print(f"\n[MLFLOW LOGGED] {exp['name']}"
                  f" - Acc: {results['acc_regime']:.2%} (+/- {results['acc_regime_std']:.2%})"
                  f" - RMSE (Glob): {results['rmse_forecast']:.4f} (+/- {results['rmse_forecast_std']:.4f})"
                  f" - Naive (Glob): {results['rmse_forecast_naive']:.4f}"
                  f" - RMSE (Physical): {results['rmse_forecast_denorm']:.4f}")

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED.")