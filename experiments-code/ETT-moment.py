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
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error, accuracy_score
import torch.nn.functional as F

from momentfm import MOMENTPipeline

# =====================================================================
# CONFIGURAZIONE & DRY RUN TOGGLE
# =====================================================================
BASE_DIR = '/home/mmendula/last-rush/datasets/ett'

DRY_RUN = False

WINDOW_SIZE = 96
HORIZONS = [96] if DRY_RUN else [96, 192, 336, 720]
# Allineato ai 100 iter di DeepRC per garantire fair comparison del budget di ricerca,
# alleggerendo al contempo il carico computazionale.
N_ITERATIONS = 5 if DRY_RUN else 100  

SEEDS_TEST   = [42] if DRY_RUN else [11, 22, 33, 44, 55]
GLOBAL_SEED  = 42

# --------- RC-RevIN v5: soft floor + clipping difensivo ---------
SOFT_FLOOR_C = 0.02     
CLIP_VAL     = 4.0      

EPS = 1e-8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================================
# UTILITY: SEED & PREPROCESSING ETT / RevIN
# =====================================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_ett_csv(filepath):
    df = pd.read_csv(filepath)
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    return df.values.astype(np.float32)

def create_sequences_rcrevin(data_global, window_size, horizon):
    n = len(data_global) - window_size - horizon + 1
    if n <= 0:
        raise ValueError("Serie troppo corta per window+horizon richiesti.")

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
    del X_glob
    return X_loc, y_loc, y_glob, x_last_loc, x_last_glob, mu.squeeze(1), sd.squeeze(1)

def naive_forecast_rmse(x_last_glob, y_glob):
    naive = np.repeat(x_last_glob[:, np.newaxis, :], y_glob.shape[1], axis=1)
    return np.sqrt(mean_squared_error(
        y_glob.reshape(len(y_glob), -1), naive.reshape(len(naive), -1)))

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

def evaluate_moment_pipeline(
    X_loc_train, y_loc_train, y_glob_train, x_last_loc_train, x_last_glob_train, mu_train, sd_train, regime_train,
    X_loc_val, y_loc_val, y_glob_val, x_last_loc_val, x_last_glob_val, mu_val, sd_val, regime_val,
    X_loc_full, y_loc_full, y_glob_full, x_last_loc_full, x_last_glob_full, mu_full, sd_full, regime_full,
    X_loc_test, y_loc_test, y_glob_test, x_last_loc_test, x_last_glob_test, mu_test, sd_test, regime_test,
    scaler, model_size="small"
):
    model_name = f"AutonLab/MOMENT-1-{model_size}"
    print(f"Loading MOMENT model: {model_name}...")
    
    moment_model = MOMENTPipeline.from_pretrained(
        model_name, 
        model_kwargs={"task_name": "embedding"} 
    )
    moment_model.init()
    moment_model = moment_model.to(DEVICE)

    noise_std = 0.05
    X_tr_t = torch.from_numpy(X_loc_train).float()
    X_val_t = torch.from_numpy(X_loc_val).float()
    X_tr_full_t = torch.from_numpy(X_loc_full).float()
    X_test_t = torch.from_numpy(X_loc_test).float()

    print("Extracting MOMENT embeddings...")
    R_tr = get_moment_embeddings_in_batches(moment_model, X_tr_t)
    R_val = get_moment_embeddings_in_batches(moment_model, X_val_t)
    R_tr_full = get_moment_embeddings_in_batches(moment_model, X_tr_full_t)
    
    R_tr_n = get_moment_embeddings_in_batches(moment_model, X_tr_t, noise_std=noise_std)
    R_val_n = get_moment_embeddings_in_batches(moment_model, X_val_t, noise_std=noise_std)
    
    naive_rmse_val = naive_forecast_rmse(x_last_glob_val, y_glob_val)
    naive_rmse_test = naive_forecast_rmse(x_last_glob_test, y_glob_test)

    H, F_dim = y_glob_val.shape[1], y_glob_val.shape[2]

    # --- Costruzione Features: Allineamento con logica Residual/Delta di DeepRC ---
    F_tr = np.hstack([R_tr, x_last_loc_train])
    F_val = np.hstack([R_val, x_last_loc_val])
    F_tr_full = np.hstack([R_tr_full, x_last_loc_full])

    baseline_train = np.repeat(x_last_loc_train[:, None, :], H, axis=1)
    target_delta_train = y_loc_train - baseline_train
    target_delta_train_flat = target_delta_train.reshape(len(target_delta_train), -1)

    print(f"Tuning Readouts ({N_ITERATIONS} Iterations Log-Uniform Search)...")
    best_val_score = -np.inf
    best_alpha = None
    
    np.random.seed(GLOBAL_SEED)
    
    for i in range(N_ITERATIONS):
        # Range corrispondente a {"min": np.log(1e-3), "max": np.log(1e4)} usato in DeepRC
        alpha = np.exp(np.random.uniform(np.log(1e-3), np.log(1e4)))
        
        # Head 1 (Forecast Residuals come in DeepRC)
        clf_f = Ridge(alpha=alpha).fit(F_tr, target_delta_train_flat)
        
        baseline_val = np.repeat(x_last_loc_val[:, None, :], H, axis=1)
        pred_delta_val_flat = clf_f.predict(F_val)
        pred_delta_val = pred_delta_val_flat.reshape(len(y_loc_val), H, F_dim)
        
        pred_loc_val = baseline_val + pred_delta_val
        pred_glob_val = pred_loc_val * sd_val[:, None, :] + mu_val[:, None, :]
        
        rmse_f = np.sqrt(mean_squared_error(
            y_glob_val.reshape(len(y_glob_val), -1),
            pred_glob_val.reshape(len(pred_glob_val), -1)
        ))
        
        # Head 2 (Regime) - DeepRC usa alpha * 100
        clf_reg = RidgeClassifier(alpha=alpha * 100.0).fit(R_tr, regime_train)
        acc_reg = accuracy_score(regime_val, clf_reg.predict(R_val))
        
        # Head 3 (Reconstruction) - DeepRC usa alpha * 10
        clf_rec = Ridge(alpha=alpha * 10.0).fit(R_tr_n, x_last_loc_train)
        rmse_rec = np.sqrt(mean_squared_error(x_last_loc_val, clf_rec.predict(R_val_n)))
        
        s_fore = 1.0 / (1.0 + rmse_f / (naive_rmse_val + EPS))
        s_reg = acc_reg
        s_stab = 1.0 / (1.0 + rmse_rec)
        val_score = 0.3 * s_fore + 0.4 * s_reg + 0.3 * s_stab
        
        if val_score > best_val_score:
            best_val_score = val_score
            best_alpha = alpha

    print(f"Best Alpha selected: {best_alpha:.6f} (Val Score: {best_val_score:.4f})")

    # Fit final models on Full Train Set
    R_tr_full_n = get_moment_embeddings_in_batches(moment_model, X_tr_full_t, noise_std=noise_std)
    
    baseline_full = np.repeat(x_last_loc_full[:, None, :], y_glob_full.shape[1], axis=1)
    target_delta_full = y_loc_full - baseline_full
    
    clf_fore_final = Ridge(alpha=best_alpha).fit(F_tr_full, target_delta_full.reshape(len(target_delta_full), -1))
    clf_reg_final = RidgeClassifier(alpha=best_alpha * 100.0).fit(R_tr_full, regime_full)
    clf_rec_final = Ridge(alpha=best_alpha * 10.0).fit(R_tr_full_n, x_last_loc_full)
    
    # --- Multi-Seed Test Evaluation ---
    print(f"Running Multi-Seed Test Evaluation over {len(SEEDS_TEST)} seeds...")
    test_results = []
    
    R_test = get_moment_embeddings_in_batches(moment_model, X_test_t)
    F_test = np.hstack([R_test, x_last_loc_test])
    
    H_test = y_glob_test.shape[1]
    baseline_test = np.repeat(x_last_loc_test[:, None, :], H_test, axis=1)
    
    pred_delta_test_flat = clf_fore_final.predict(F_test)
    pred_delta_test = pred_delta_test_flat.reshape(len(y_loc_test), H_test, F_dim)
    
    pred_loc_test = baseline_test + pred_delta_test
    pred_glob_test = pred_loc_test * sd_test[:, None, :] + mu_test[:, None, :]
    
    rmse_fore = np.sqrt(mean_squared_error(
        y_glob_test.reshape(len(y_glob_test), -1), 
        pred_glob_test.reshape(len(pred_glob_test), -1)
    ))
    
    pred_reg_test = clf_reg_final.predict(R_test)
    acc_reg = accuracy_score(regime_test, pred_reg_test)
    
    pred_physical = scaler.inverse_transform(pred_glob_test.reshape(-1, F_dim)).reshape(len(pred_glob_test), H_test, F_dim)
    y_physical = scaler.inverse_transform(y_glob_test.reshape(-1, F_dim)).reshape(len(y_glob_test), H_test, F_dim)
    
    rmse_den = np.sqrt(mean_squared_error(
        y_physical.reshape(len(y_physical), -1), 
        pred_physical.reshape(len(pred_physical), -1)
    ))

    for seed in SEEDS_TEST:
        seed_everything(seed)
        R_test_n = get_moment_embeddings_in_batches(moment_model, X_test_t, noise_std=noise_std)
        pred_recon = clf_rec_final.predict(R_test_n)
        rmse_rec = np.sqrt(mean_squared_error(x_last_loc_test, pred_recon))
        
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
        "rmse_forecast_std": 0.0,
        "rmse_forecast_naive": naive_rmse_test,
        "rmse_forecast_denorm": rmse_den,
        "acc_regime": acc_reg,
        "acc_regime_std": 0.0,
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
# MAIN ETT EXECUTION CON CICLO SUGLI ORIZZONTI
# =====================================================================
if __name__ == "__main__":
    experiments = [
        {"name": "ETTh1", "file": "ETTh1.csv"},
        {"name": "ETTh2", "file": "ETTh2.csv"},
        {"name": "ETTm1", "file": "ETTm1.csv"},
        {"name": "ETTm2", "file": "ETTm2.csv"}
    ]

    print(f"--- STARTING MOMENT ETT ABLATION SUITE (DRY_RUN={DRY_RUN}) ---")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("IEEE_ETT_Horizon_Depth_Ablation")

    for exp in experiments:
        for current_h in HORIZONS:
            seed_everything(GLOBAL_SEED)
            
            run_name = f"MOMENT_{exp['name']}_H{current_h}"
            print("\n" + "=" * 60)
            print(f"RUNNING EXPERIMENT: {run_name}")
            print("=" * 60)

            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("architecture", "MOMENT-1-small")
                mlflow.log_param("dataset", exp['file'])
                mlflow.log_param("forecast_horizon", current_h)
                mlflow.log_param("window_size", WINDOW_SIZE)
                mlflow.log_param("n_iterations", N_ITERATIONS)
                
                # 1. LOAD RAW DATA
                p_data = os.path.join(BASE_DIR, exp['file'])
                raw_data = load_ett_csv(p_data)
                train_end_idx = int(len(raw_data) * 0.6)

                # 2. FIT SCALER (Strictly on 60%)
                scaler = MinMaxScaler()
                scaler.fit(raw_data[:train_end_idx])
                scaled_data = scaler.transform(raw_data)

                # 3. GMM + BIC CLUSTERING
                print(f"[{exp['name']}] Ricerca del numero ottimale di regimi operativi (GMM + BIC)...")
                MIN_REGIMES, MAX_REGIMES = 2, 10
                lowest_bic, best_k, best_gmm = np.inf, MIN_REGIMES, None
                train_data_for_clustering = scaled_data[:train_end_idx]

                for k in range(MIN_REGIMES, MAX_REGIMES + 1):
                    gmm_temp = GaussianMixture(n_components=k, covariance_type='full', random_state=GLOBAL_SEED)
                    gmm_temp.fit(train_data_for_clustering)
                    bic_score = gmm_temp.bic(train_data_for_clustering)
                    if bic_score < lowest_bic:
                        lowest_bic = bic_score
                        best_gmm = gmm_temp
                        best_k = k

                print(f"-> Regimi identificati: {best_k} (BIC: {lowest_bic:.0f})")
                regimes_raw = best_gmm.predict(scaled_data)
                
                mlflow.log_param("n_regimes_adaptive", best_k)
                mlflow.log_metric("regime_bic_score", lowest_bic)

                # 4. SEQUENZE E RC-RevIN
                X_loc, y_loc, y_glob, x_last_loc, x_last_glob, mu, sd = create_sequences_rcrevin(
                    scaled_data, WINDOW_SIZE, current_h
                )
                regimes_aligned = regimes_raw[WINDOW_SIZE - 1 : WINDOW_SIZE - 1 + len(X_loc)]

                # 5. CHRONOLOGICAL SPLITS (60/10/30)
                seq_train_end = int(len(X_loc) * 0.6)
                seq_val_end = int(len(X_loc) * 0.7)
                
                X_loc_train = X_loc[:seq_train_end]
                y_loc_train = y_loc[:seq_train_end]
                y_glob_train = y_glob[:seq_train_end]
                x_last_loc_train = x_last_loc[:seq_train_end]
                x_last_glob_train = x_last_glob[:seq_train_end]
                mu_train = mu[:seq_train_end]
                sd_train = sd[:seq_train_end]
                regime_train = regimes_aligned[:seq_train_end]

                X_loc_val = X_loc[seq_train_end:seq_val_end]
                y_loc_val = y_loc[seq_train_end:seq_val_end]
                y_glob_val = y_glob[seq_train_end:seq_val_end]
                x_last_loc_val = x_last_loc[seq_train_end:seq_val_end]
                x_last_glob_val = x_last_glob[seq_train_end:seq_val_end]
                mu_val = mu[seq_train_end:seq_val_end]
                sd_val = sd[seq_train_end:seq_val_end]
                regime_val = regimes_aligned[seq_train_end:seq_val_end]

                X_loc_full = np.concatenate([X_loc_train, X_loc_val], axis=0)
                y_loc_full = np.concatenate([y_loc_train, y_loc_val], axis=0)
                y_glob_full = np.concatenate([y_glob_train, y_glob_val], axis=0)
                x_last_loc_full = np.concatenate([x_last_loc_train, x_last_loc_val], axis=0)
                x_last_glob_full = np.concatenate([x_last_glob_train, x_last_glob_val], axis=0)
                mu_full = np.concatenate([mu_train, mu_val], axis=0)
                sd_full = np.concatenate([sd_train, sd_val], axis=0)
                regime_full = np.concatenate([regime_train, regime_val], axis=0)

                X_loc_test = X_loc[seq_val_end:]
                y_loc_test = y_loc[seq_val_end:]
                y_glob_test = y_glob[seq_val_end:]
                x_last_loc_test = x_last_loc[seq_val_end:]
                x_last_glob_test = x_last_glob[seq_val_end:]
                mu_test = mu[seq_val_end:]
                sd_test = sd[seq_val_end:]
                regime_test = regimes_aligned[seq_val_end:]

                # 6. RUN EVALUATION
                start = time.time()
                test_res = evaluate_moment_pipeline(
                    X_loc_train=X_loc_train, y_loc_train=y_loc_train, y_glob_train=y_glob_train, 
                    x_last_loc_train=x_last_loc_train, x_last_glob_train=x_last_glob_train, 
                    mu_train=mu_train, sd_train=sd_train, regime_train=regime_train,
                    
                    X_loc_val=X_loc_val, y_loc_val=y_loc_val, y_glob_val=y_glob_val, 
                    x_last_loc_val=x_last_loc_val, x_last_glob_val=x_last_glob_val, 
                    mu_val=mu_val, sd_val=sd_val, regime_val=regime_val,
                    
                    X_loc_full=X_loc_full, y_loc_full=y_loc_full, y_glob_full=y_glob_full, 
                    x_last_loc_full=x_last_loc_full, x_last_glob_full=x_last_glob_full, 
                    mu_full=mu_full, sd_full=sd_full, regime_full=regime_full,
                    
                    X_loc_test=X_loc_test, y_loc_test=y_loc_test, y_glob_test=y_glob_test, 
                    x_last_loc_test=x_last_loc_test, x_last_glob_test=x_last_glob_test, 
                    mu_test=mu_test, sd_test=sd_test, regime_test=regime_test,
                    
                    scaler=scaler, model_size="small"
                )
                duration = time.time() - start

                # 7. LOG RESULTS
                mlflow.log_param("best_alpha", test_res['best_alpha'])
                mlflow.log_metric("val_composite_score", test_res['val_composite_score'])
                mlflow.log_metric("composite_score", test_res['composite_score'])
                mlflow.log_metric("accuracy_regime", test_res['acc_regime'])
                mlflow.log_metric("rmse_forecast", test_res['rmse_forecast'])
                mlflow.log_metric("rmse_forecast_naive", test_res['rmse_forecast_naive'])
                mlflow.log_metric("rmse_forecast_denorm", test_res['rmse_forecast_denorm'])
                mlflow.log_metric("rmse_stability", test_res['rmse_recon'])
                mlflow.log_metric("duration_sec", duration)

                # Cleanup
                del X_loc_train, X_loc_val, X_loc_full, X_loc_test
                gc.collect()

                print(f"[{run_name}] COMPLETED. Acc: {test_res['acc_regime']:.2%} | RMSE (Denorm): {test_res['rmse_forecast_denorm']:.4f}")

    print("\n" + "="*60)
    print("ALL ETT EXPERIMENTS COMPLETED.")
