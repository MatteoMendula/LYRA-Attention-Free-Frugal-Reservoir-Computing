import os
import time
import random
import numpy as np
import pandas as pd
import torch
import mlflow
import gc
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

# --- IMPORT MOMENT ---
from momentfm import MOMENTPipeline

# =====================================================================
# CONFIGURATION
# =====================================================================
BASE_DIR = '/home/mmendula/last-rush/datasets/ECG5000' # Path ECG5000

WINDOW_SIZE = 120
FORECAST_HORIZON = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_ITERATIONS = 100 # Allineato al budget di DeepRC per ECG5000

SEEDS_TEST = [11, 22, 33, 44, 55]
GLOBAL_SEED = 42

# --- RC-RevIN v5: soft floor + defensive clipping (Allineato) ---
SOFT_FLOOR_C = 0.02     
CLIP_VAL     = 4.0      

EPS = 1e-8

# =====================================================================
# UTILITIES (Strutturalmente allineate con DeepRC)
# =====================================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def process_ecg_sequences(filepath):
    """
    Carica i dati TSV ECG5000, estrae i label ed esegue il
    RevIN localmente su ogni singolo battito indipendente.
    """
    raw_data = np.loadtxt(filepath, delimiter='\t')
    labels = raw_data[:, 0].astype(np.int64) - 1  # Allinea i label da 0 a 4
    series = raw_data[:, 1:].astype(np.float32)

    # RevIN Locale
    mu = series.mean(axis=1, keepdims=True)
    sd = series.std(axis=1, keepdims=True)
    sd = np.sqrt(sd ** 2 + SOFT_FLOOR_C ** 2)

    series_loc = (series - mu) / sd
    np.clip(series_loc, -CLIP_VAL, CLIP_VAL, out=series_loc)

    # Aggiungi la dimensione delle feature (1)
    series_loc = series_loc[..., np.newaxis]
    
    # Prendi l'ultimo step per la reconstruction
    x_last_loc = series_loc[:, -1, :].copy()

    return series_loc, x_last_loc, labels

# =====================================================================
# MOMENT SPECIFIC FUNCTIONS
# =====================================================================
def get_moment_embeddings_in_batches(model, X_tensor, batch_size=128, noise_std=0.0):
    states = []
    model.eval()
    
    if noise_std > 0.0:
        X_tensor = X_tensor + torch.randn_like(X_tensor) * noise_std
        
    # MOMENT si aspetta [batch_size, n_channels, seq_len]
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
            # Appiattiamo l'output per usarlo nella regressione lineare
            embeddings_flat = outputs.embeddings.reshape(batch.shape[0], -1).cpu().numpy()
            states.append(embeddings_flat)
            
            del batch, outputs, mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    return np.concatenate(states, axis=0)

def evaluate_moment_pipeline_class(X_train, x_last_train, reg_train,
                                   X_val, x_last_val, reg_val,
                                   X_train_full, x_last_train_full, reg_train_full,
                                   X_test, x_last_test, reg_test, 
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
    # Clean embeddings per Classificazione
    R_tr = get_moment_embeddings_in_batches(moment_model, X_tr_t)
    R_val = get_moment_embeddings_in_batches(moment_model, X_val_t)
    R_tr_full = get_moment_embeddings_in_batches(moment_model, X_tr_full_t)
    
    # Noisy embeddings per Ricostruzione
    R_tr_n = get_moment_embeddings_in_batches(moment_model, X_tr_t, noise_std=noise_std)
    R_val_n = get_moment_embeddings_in_batches(moment_model, X_val_t, noise_std=noise_std)

    print(f"Tuning Readouts ({N_ITERATIONS} Iteration Random Search)...")
    best_val_score = -np.inf
    best_alpha = None
    
    np.random.seed(GLOBAL_SEED)
    
    for i in range(N_ITERATIONS):
        # Spazio di ricerca log-uniforme per Alpha allineato a DeepRC (1e-3 -> 1e4)
        alpha = np.exp(np.random.uniform(np.log(1e-3), np.log(1e4)))
        
        # Head 1 (Classificazione)
        clf_reg = RidgeClassifier(alpha=alpha, solver='lsqr').fit(R_tr, reg_train)
        acc_class = accuracy_score(reg_val, clf_reg.predict(R_val))
        
        # Head 2 (Ricostruzione/Stabilità) - alpha scalato come in DeepRC
        alpha_rec = alpha * 10.0
        clf_rec = Ridge(alpha=alpha_rec, solver='lsqr').fit(R_tr_n, x_last_train)
        rmse_rec = np.sqrt(mean_squared_error(x_last_val, clf_rec.predict(R_val_n)))
        
        # Scoring Allineato (Dominato dall'accuratezza)
        s_reg = acc_class
        s_stab = 1.0 / (1.0 + rmse_rec)
        val_score = 0.8 * s_reg + 0.2 * s_stab
        
        if val_score > best_val_score:
            best_val_score = val_score
            best_alpha = alpha

    print(f"Best Alpha selected: {best_alpha:.6f} (Val Score: {best_val_score:.4f})")

    # Fit final models su Full Train Set (500 samples)
    R_tr_full_n = get_moment_embeddings_in_batches(moment_model, X_tr_full_t, noise_std=noise_std)
    
    clf_reg_final = RidgeClassifier(alpha=best_alpha, solver='lsqr').fit(R_tr_full, reg_train_full)
    clf_rec_final = Ridge(alpha=best_alpha * 10.0, solver='lsqr').fit(R_tr_full_n, x_last_train_full)
    
    # -------------------------------------------------------------
    # Multi-Seed Evaluation for Test
    # -------------------------------------------------------------
    print(f"Running Multi-Seed Test Evaluation over {len(SEEDS_TEST)} seeds...")
    test_results = []
    
    # Le predizioni pulite (Classificazione) sono deterministiche per il modello congelato
    R_test = get_moment_embeddings_in_batches(moment_model, X_test_t)
    pred_reg = clf_reg_final.predict(R_test)
    acc_class_test = accuracy_score(reg_test, pred_reg)

    for seed in SEEDS_TEST:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # L'Head di Ricostruzione inietta rumore, quindi gli embedding cambiano per seed
        R_test_n = get_moment_embeddings_in_batches(moment_model, X_test_t, noise_std=noise_std)
        pred_recon = clf_rec_final.predict(R_test_n)
        rmse_rec_test = np.sqrt(mean_squared_error(x_last_test, pred_recon))
        
        s_reg = acc_class_test
        s_stab = 1.0 / (1.0 + rmse_rec_test)
        score = 0.8 * s_reg + 0.2 * s_stab
        
        test_results.append({
            "score": score,
            "rmse_recon": rmse_rec_test
        })
        
    agg = {
        "val_composite_score": best_val_score,
        "best_alpha": best_alpha,
        "accuracy": acc_class_test,
        "accuracy_std": 0.0,    # Deterministico (nessun rumore in classificazione)
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
    seed_everything(GLOBAL_SEED)

    print("--- STARTING ALIGNED MOMENT BENCHMARK (ECG5000 Classification) ---")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("IEEE_ECG5000_Classification")

    with mlflow.start_run(run_name="MOMENT_Aligned_ECG5000_Classification"):
        model_size = "small"
        
        mlflow.log_param("architecture", f"MOMENT-1-{model_size}")
        mlflow.log_param("dataset", "ECG5000")
        mlflow.log_param("window_size", WINDOW_SIZE)
        mlflow.log_param("normalization", "RevIN_v5_SoftFloor_PerSequence")
        mlflow.log_param("soft_floor_c", SOFT_FLOOR_C)
        mlflow.log_param("clip_val", CLIP_VAL)
        mlflow.log_param("n_iterations", N_ITERATIONS)
        
        # 1. Caricamento Dati TSV
        train_path = os.path.join(BASE_DIR, 'ECG5000_TRAIN.tsv')
        test_path = os.path.join(BASE_DIR, 'ECG5000_TEST.tsv')
        
        X_loc_full_train, x_last_loc_full_train, reg_full_train = process_ecg_sequences(train_path)
        X_loc_test, x_last_loc_test, reg_test = process_ecg_sequences(test_path)
        
        classes = np.unique(np.concatenate([reg_full_train, reg_test]))
        mlflow.log_param("n_classes", len(classes))

        # 2. Suddivisione Validazione (80/20 sul set di train da 500)
        X_loc_train, X_loc_val, reg_train, reg_val = train_test_split(
            X_loc_full_train, reg_full_train, test_size=0.2, random_state=GLOBAL_SEED, stratify=reg_full_train
        )
        _, _, x_last_loc_train, x_last_loc_val = train_test_split(
            X_loc_full_train, x_last_loc_full_train, test_size=0.2, random_state=GLOBAL_SEED, stratify=reg_full_train
        )

        # 3. Lancio Valutazione MOMENT
        start = time.time()
        results = evaluate_moment_pipeline_class(
            X_train=X_loc_train, x_last_train=x_last_loc_train, reg_train=reg_train,
            X_val=X_loc_val, x_last_val=x_last_loc_val, reg_val=reg_val,
            X_train_full=X_loc_full_train, x_last_train_full=x_last_loc_full_train, reg_train_full=reg_full_train,
            X_test=X_loc_test, x_last_test=x_last_loc_test, reg_test=reg_test,
            model_size=model_size
        )
        duration = time.time() - start

        # 4. Export Risultati su MLflow
        mlflow.log_metric("val_composite_score", results['val_composite_score'])
        mlflow.log_metric("composite_score", results['composite_score'])
        mlflow.log_metric("composite_score_std", results['composite_score_std'])
        mlflow.log_metric("accuracy", results['accuracy'])
        mlflow.log_metric("accuracy_std", results['accuracy_std'])
        mlflow.log_metric("rmse_stability", results['rmse_recon'])
        mlflow.log_metric("rmse_stability_std", results['rmse_recon_std'])
        mlflow.log_metric("duration_sec", duration)
        mlflow.log_param("best_alpha", results['best_alpha'])

        print(f"\n[MLFLOW LOGGED] ECG5000"
              f" - Acc: {results['accuracy']:.2%} (+/- {results['accuracy_std']:.2%})"
              f" - Stabilità: {results['rmse_recon']:.4f}")

    print("\n" + "="*60)
    print("MOMENT BENCHMARK COMPLETED.")