import os
import time
import random
import numpy as np
import torch
import mlflow
import gc
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split

# --- Moduli custom ---
from base_deep_rc_search import BaseDeepRCSearch
from psi_DeepRC_IP import DeepReservoir     
from utils import robust_torch_ridge_fit

# =====================================================================
# CONFIGURAZIONE
# =====================================================================
BASE_DIR = '/home/mmendula/last-rush/datasets/ECG5000' # Aggiorna questo path

WINDOW_SIZE = 140
FORECAST_HORIZON = 20
N_ITERATIONS = 100

SEEDS_SEARCH = [11, 22, 33]
SEEDS_TEST   = [11, 22, 33, 44, 55]
GLOBAL_SEED  = 42

# --------- Multi-tap readout ---------
N_TAPS = 4              

# --------- RC-RevIN v5 ---------
SOFT_FLOOR_C = 0.02     
CLIP_VAL     = 4.0      

# --------- Architettura ---------
LLE_TUNING   = True     
BIAS_SCALING = 0.1      
CONCAT_UNITS = True     

HP_SPACE = {
    "tot_units": [250, 500, 1000, 1500],
    "n_layers": [1, 2, 3, 4],
    "spectral_radius": {"min": np.log(0.5), "max": np.log(1.2)},
    "leaky": [0.1, 0.3, 0.5, 1.0],
    "connectivity_recurrent": [5, 10, 20],
    "input_scaling": {"min": np.log(0.02), "max": np.log(0.5)},
    "ridge_alpha": {"min": np.log(1e-3), "max": np.log(1e4)}
}

GLOBAL_PARAMS = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "window_size": WINDOW_SIZE
}

EPS = 1e-8

# =====================================================================
# UTILITY
# =====================================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_reservoir(input_dim, params, device, seed):
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    seed_everything(seed)
    conn = int(round(params["connectivity_recurrent"]))
    model = DeepReservoir(
        input_size=input_dim,
        tot_units=int(round(params["tot_units"])),
        n_layers=int(round(params["n_layers"])),
        concat=CONCAT_UNITS,
        input_scaling=params.get("input_scaling", 0.1),
        inter_scaling=1.0,
        spectral_radius=params["spectral_radius"],
        leaky=params["leaky"],
        connectivity_recurrent=conn,
        connectivity_input=conn,
        connectivity_inter=conn,
        bias_bool=True,
        bias_scaling=BIAS_SCALING,
        effective_rescaling=True,
        nonlin='tanh',
        lle_tuning=LLE_TUNING,
        n_taps=N_TAPS,
        verbose=False,
    ).to(device)

    if hasattr(torch, "compile"):
        model = torch.compile(model, dynamic=True)

    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)

    return model

def process_ecg_sequences(filepath):
    """
    Carica i dati testuali TSV di ECG5000, estrae i label ed esegue il
    RevIN localmente su ogni singolo battito indipendente.
    """
    # FIX: Forziamo il delimitatore a tabulazione per i file .tsv
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

def add_bias(R):
    return np.hstack([R, np.ones((R.shape[0], 1), dtype=R.dtype)])

def one_hot_fixed(labels, classes):
    return (labels[:, None] == classes[None, :]).astype(np.float32)

# =====================================================================
# SEARCHER PER CLASSIFICAZIONE (NO FORECAST)
# =====================================================================
class FoundationalRCClassSearch(BaseDeepRCSearch):
    def __init__(self,
                 X_loc_train, x_last_loc_train, reg_train,
                 X_loc_val, x_last_loc_val, reg_val,
                 X_loc_test, x_last_loc_test, reg_test,
                 X_loc_full, x_last_loc_full, reg_full,
                 classes, hp_space, n_iterations, global_params, epsilon_greedy=0.3):
        super().__init__(hp_space, n_iterations, global_params, epsilon_greedy)

        self.X_loc_train, self.x_last_loc_train, self.reg_train = X_loc_train, x_last_loc_train, reg_train
        self.X_loc_val, self.x_last_loc_val, self.reg_val = X_loc_val, x_last_loc_val, reg_val
        self.X_loc_full, self.x_last_loc_full, self.reg_full = X_loc_full, x_last_loc_full, reg_full
        self.X_loc_test, self.x_last_loc_test, self.reg_test = X_loc_test, x_last_loc_test, reg_test

        self.classes = classes
        self.input_dim = self.X_loc_train.shape[2]
        self.my_best_score = -np.inf
        self.best_meta_result = {}
        self._eval_cache = {}

    def _get_reservoir_states_in_batches(self, model, X_numpy, device, batch_size=2048, noise_std=0.0):
        L1_states, LDeep_states = [], []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(X_numpy), batch_size):
                batch = torch.from_numpy(X_numpy[i:i + batch_size]).float().to(device)
                if noise_std > 0.0:
                    batch += torch.randn_like(batch) * noise_std
                _, states_last = model(batch, return_all_layers=True)
                L1_states.append(states_last[0].cpu().numpy())
                LDeep_states.append(states_last[-1].cpu().numpy())
        return np.concatenate(L1_states, axis=0), np.concatenate(LDeep_states, axis=0)

    def _run_heads(self, model, params,
                   X_loc_fit, x_last_loc_fit, reg_fit,
                   X_loc_ev, x_last_loc_ev, reg_ev, device):

        ridge_alpha = params.get("ridge_alpha", 1e-3)
        noise_std = 0.05

        R_fit_L1, R_fit_LD = self._get_reservoir_states_in_batches(model, X_loc_fit, device)
        R_ev_L1, R_ev_LD = self._get_reservoir_states_in_batches(model, X_loc_ev, device)
        R_fit_L1n, _ = self._get_reservoir_states_in_batches(model, X_loc_fit, device, noise_std=noise_std)
        R_ev_L1n, _ = self._get_reservoir_states_in_batches(model, X_loc_ev, device, noise_std=noise_std)

        # ---- Head 1: Classificazione (MULTI-SCALE READOUT) ----
        # Concateniamo Layer 1 (micro-dinamiche, QRS) e Layer Deep (macro-dinamiche, onda T)
        C_fit_raw = np.hstack([R_fit_L1, R_fit_LD])
        C_ev_raw  = np.hstack([R_ev_L1, R_ev_LD])
        
        C_fit = add_bias(C_fit_raw)
        C_ev  = add_bias(C_ev_raw)

        y_reg_oh = one_hot_fixed(reg_fit[:len(C_fit)], self.classes)
        W_reg = robust_torch_ridge_fit(C_fit, y_reg_oh, ridge_alpha, device)
        
        pred_idx = np.argmax((torch.from_numpy(C_ev).float().to(device) @ W_reg).cpu().numpy(), axis=1)
        pred_labels = self.classes[pred_idx]
        acc_class = accuracy_score(reg_ev[:len(C_ev)], pred_labels)

        # ---- Head 2: Reconstruction (Stabilità) ----
        S_fit = add_bias(R_fit_L1n)
        S_ev = add_bias(R_ev_L1n)
        tgt_fit = x_last_loc_fit
        tgt_ev = x_last_loc_ev

        alpha_rec = ridge_alpha * 10.0 
        W_recon = robust_torch_ridge_fit(S_fit, tgt_fit, alpha_rec, device)
        pred_recon = (torch.from_numpy(S_ev).float().to(device) @ W_recon).cpu().numpy()
        rmse_recon = np.sqrt(mean_squared_error(tgt_ev, pred_recon))

        # ---- Scoring (Orientato alla Classificazione) ----
        s_reg = acc_class
        s_stab = 1.0 / (1.0 + rmse_recon)
        score = 0.8 * s_reg + 0.2 * s_stab  # Dominato dall'accuratezza

        metrics = {
            "score": score,
            "accuracy": acc_class,
            "rmse_recon": rmse_recon,
        }
        readouts = {"class": W_reg, "reconstruct": W_recon}
        n_features = C_fit.shape[1]

        del R_fit_L1, R_fit_LD, R_ev_L1, R_ev_LD, R_fit_L1n, R_ev_L1n
        del C_fit, C_ev, S_fit, S_ev
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return metrics, readouts, n_features

    def evaluate(self, full_params):
        for ip in ["tot_units", "n_layers", "connectivity_recurrent"]:
            if ip in full_params:
                full_params[ip] = int(round(float(full_params[ip])))

        param_key = tuple(sorted([(k, v) for k, v in full_params.items()]))
        if param_key in self._eval_cache:
            c_score, c_readouts, c_n_features = self._eval_cache[param_key]
            return c_score, c_readouts, None, c_n_features

        device = full_params["device"]
        per_seed = []
        readouts, n_features = None, None

        for seed in SEEDS_SEARCH:
            model = build_reservoir(self.input_dim, full_params, device, seed)
            m, r, n_features = self._run_heads(
                model, full_params,
                self.X_loc_train, self.x_last_loc_train, self.reg_train,
                self.X_loc_val, self.x_last_loc_val, self.reg_val,
                device)
            per_seed.append(m)
            readouts = r
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mean_score = float(np.mean([m["score"] for m in per_seed]))
        std_score = float(np.std([m["score"] for m in per_seed]))

        meta = {
            "score": mean_score,
            "score_std_seeds": std_score,
            "accuracy": float(np.mean([m["accuracy"] for m in per_seed])),
            "rmse_recon": float(np.mean([m["rmse_recon"] for m in per_seed])),
        }

        if mean_score > self.my_best_score:
            self.my_best_score = mean_score
            self.best_meta_result = meta

        self._eval_cache[param_key] = (mean_score, readouts, n_features)

        return mean_score, readouts, None, n_features

    def evaluate_final_test_set(self, best_params):
        device = self.global_params["device"]
        results = []

        for seed in SEEDS_TEST:
            model = build_reservoir(self.input_dim, best_params, device, seed)
            m, _, _ = self._run_heads(
                model, best_params,
                self.X_loc_full, self.x_last_loc_full, self.reg_full,
                self.X_loc_test, self.x_last_loc_test, self.reg_test,
                device)
            results.append(m)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        agg = {}
        for k in ["score", "accuracy", "rmse_recon"]:
            vals = [m[k] for m in results]
            agg[k] = float(np.mean(vals))
            agg[k + "_std"] = float(np.std(vals))
        return agg

# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    seed_everything(GLOBAL_SEED)

    print("--- ESECUZIONE CLASSIFICAZIONE ECG5000 ---")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("IEEE_ECG5000_Classification")

    with mlflow.start_run(run_name="DeepRC_ECG5000_Classification"):
        mlflow.log_param("architecture", "DeepRC_Ridge_MultiTap_EffRescale_LLE")
        mlflow.log_param("dataset", "ECG5000")
        mlflow.log_param("window_size", WINDOW_SIZE)
        
        # 1. Caricamento Dati
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

        # 3. Lancio del Searcher
        searcher = FoundationalRCClassSearch(
            X_loc_train=X_loc_train, x_last_loc_train=x_last_loc_train, reg_train=reg_train,
            X_loc_val=X_loc_val, x_last_loc_val=x_last_loc_val, reg_val=reg_val,
            X_loc_test=X_loc_test, x_last_loc_test=x_last_loc_test, reg_test=reg_test,
            X_loc_full=X_loc_full_train, x_last_loc_full=x_last_loc_full_train, reg_full=reg_full_train,
            classes=classes, 
            hp_space=HP_SPACE, n_iterations=N_ITERATIONS,
            global_params=GLOBAL_PARAMS, epsilon_greedy=0.3
        )

        start = time.time()
        searcher.search()
        duration = time.time() - start

        # 4. Estrazione ed Export Risultati
        best_val_meta = searcher.best_meta_result
        clean_params = searcher.best_params.copy()
        for ip in ["tot_units", "n_layers", "connectivity_recurrent"]:
            if ip in clean_params:
                clean_params[ip] = int(round(float(clean_params[ip])))

        test_res = searcher.evaluate_final_test_set(clean_params)

        mlflow.log_params(clean_params)
        mlflow.log_metric("val_composite_score", best_val_meta['score'])
        mlflow.log_metric("composite_score", test_res['score'])
        mlflow.log_metric("accuracy", test_res['accuracy'])
        mlflow.log_metric("accuracy_std", test_res['accuracy_std'])
        mlflow.log_metric("rmse_stability", test_res['rmse_recon'])
        mlflow.log_metric("duration_sec", duration)

        history_data = []
        for param_tuple, (score, _, n_feat) in searcher._eval_cache.items():
            row = dict(param_tuple)
            row['val_composite_score'] = score
            row['n_features_readout'] = n_feat
            history_data.append(row)
            
        df_history = pd.DataFrame(history_data).sort_values(by='val_composite_score', ascending=False)
        csv_filename = "hp_history_ECG5000.csv"
        df_history.to_csv(csv_filename, index=False)
        mlflow.log_artifact(csv_filename)
        os.remove(csv_filename)

        print(f"\n[MLFLOW LOGGED] ECG5000"
              f" - Acc: {test_res['accuracy']:.2%} (+/- {test_res['accuracy_std']:.2%})"
              f" - Stabilità: {test_res['rmse_recon']:.4f}")