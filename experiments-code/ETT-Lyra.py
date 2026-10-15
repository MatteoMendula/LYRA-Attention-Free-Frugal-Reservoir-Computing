import os
import time
import random
import numpy as np
import pandas as pd
import torch
import mlflow
import gc
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error, accuracy_score

# --- Moduli custom ---
from base_deep_rc_search import BaseDeepRCSearch
from psi_DeepRC_IP import DeepReservoir     
from utils import robust_torch_ridge_fit

# =====================================================================
# CONFIGURAZIONE & DRY RUN TOGGLE
# =====================================================================
BASE_DIR = '/home/mmendula/last-rush/datasets/ett'

DRY_RUN = False

WINDOW_SIZE = 96
HORIZONS = [96] if DRY_RUN else [96, 192, 336, 720]
N_ITERATIONS = 5 if DRY_RUN else 100

SEEDS_SEARCH = [11] if DRY_RUN else [11, 22, 33]
SEEDS_TEST   = [42] if DRY_RUN else [11, 22, 33, 44, 55]
GLOBAL_SEED  = 42

# --------- Multi-tap readout ---------
N_TAPS = 4              

# --------- RC-RevIN v5: soft floor + clipping difensivo ---------
SOFT_FLOOR_C = 0.02     
CLIP_VAL     = 4.0      

# --------- Architettura ---------
LLE_TUNING   = True     
BIAS_SCALING = 0.1      
CONCAT_UNITS = True     

HP_SPACE = {
    "tot_units": [250, 500, 1000],
    "n_layers": [1, 2, 3, 4, 5],
    "spectral_radius": {"min": np.log(0.5), "max": np.log(1.2)},
    "leaky": [0.1, 0.3, 0.5, 1.0],
    "connectivity_recurrent": [5, 10, 20],
    "input_scaling": {"min": np.log(0.02), "max": np.log(0.5)},
    # Ridge Alpha molto ampio per permettere il collasso sul naive se serve
    "ridge_alpha": {"min": np.log(1e-3), "max": np.log(1e4)},
    "use_deep_for_forecast": [0, 1], # 1 = ABL1A (Deep States ON), 0 = ABL1B
}

GLOBAL_PARAMS = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "window_size": WINDOW_SIZE
    # forecast_horizon viene sovrascritto dinamicamente nel ciclo, 
    # quindi non è necessario dichiararlo staticamente qui.
}

EPS = 1e-8

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
    """Loads ETT csv, drops the date column, returns numpy array."""
    df = pd.read_csv(filepath)
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    return df.values.astype(np.float32)

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

def add_bias(R):
    return np.hstack([R, np.ones((R.shape[0], 1), dtype=R.dtype)])

def one_hot_fixed(labels, classes):
    return (labels[:, None] == classes[None, :]).astype(np.float32)

def naive_forecast_rmse(x_last_glob, y_glob):
    naive = np.repeat(x_last_glob[:, np.newaxis, :], y_glob.shape[1], axis=1)
    return np.sqrt(mean_squared_error(
        y_glob.reshape(len(y_glob), -1), naive.reshape(len(naive), -1)))

# =====================================================================
# SEARCHER ARCHITETTURA UNIFICATA
# =====================================================================
class FoundationalRCSearch(BaseDeepRCSearch):
    def __init__(self,
                 X_loc_train, y_loc_train, y_glob_train, x_last_loc_train, mu_train, sd_train, reg_train,
                 X_loc_val, y_loc_val, y_glob_val, x_last_loc_val, x_last_glob_val, mu_val, sd_val, reg_val,
                 X_loc_test, y_loc_test, y_glob_test, x_last_loc_test, x_last_glob_test, mu_test, sd_test, reg_test,
                 X_loc_full, y_loc_full, y_glob_full, x_last_loc_full, mu_full, sd_full, reg_full,
                 classes, scaler,
                 hp_space, n_iterations, global_params, epsilon_greedy=0.3):
        super().__init__(hp_space, n_iterations, global_params, epsilon_greedy)

        self.X_loc_train, self.y_loc_train, self.y_glob_train, self.x_last_loc_train, self.mu_train, self.sd_train, self.reg_train = X_loc_train, y_loc_train, y_glob_train, x_last_loc_train, mu_train, sd_train, reg_train
        self.X_loc_val, self.y_loc_val, self.y_glob_val, self.x_last_loc_val, self.x_last_glob_val, self.mu_val, self.sd_val, self.reg_val = X_loc_val, y_loc_val, y_glob_val, x_last_loc_val, x_last_glob_val, mu_val, sd_val, reg_val
        self.X_loc_full, self.y_loc_full, self.y_glob_full, self.x_last_loc_full, self.mu_full, self.sd_full, self.reg_full = X_loc_full, y_loc_full, y_glob_full, x_last_loc_full, mu_full, sd_full, reg_full
        self.X_loc_test, self.y_loc_test, self.y_glob_test, self.x_last_loc_test, self.x_last_glob_test, self.mu_test, self.sd_test, self.reg_test = X_loc_test, y_loc_test, y_glob_test, x_last_loc_test, x_last_glob_test, mu_test, sd_test, reg_test

        self.classes = classes
        self.scaler = scaler
        self.input_dim = self.X_loc_train.shape[2]
        self.my_best_score = -np.inf
        self.best_meta_result = {}
        self._eval_cache = {}

        self.naive_rmse_val = naive_forecast_rmse(self.x_last_glob_val, self.y_glob_val)
        self.naive_rmse_test = naive_forecast_rmse(self.x_last_glob_test, self.y_glob_test)

    def _get_reservoir_states_in_batches(self, model, X_numpy, device, batch_size=2048, noise_std=0.0):
        L1_states, LDeep_states = [], []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(X_numpy), batch_size):
                batch = torch.from_numpy(X_numpy[i:i + batch_size]).float().pin_memory().to(device, non_blocking=True)
                if noise_std > 0.0:
                    batch += torch.randn_like(batch) * noise_std
                _, states_last = model(batch, return_all_layers=True)
                L1_states.append(states_last[0].cpu().numpy())
                LDeep_states.append(states_last[-1].cpu().numpy())
        return np.concatenate(L1_states, axis=0), np.concatenate(LDeep_states, axis=0)

    def _run_heads(self, model, params,
                   X_loc_fit, y_loc_fit, y_glob_fit, x_last_loc_fit, mu_fit, sd_fit, reg_fit,
                   X_loc_ev, y_loc_ev, y_glob_ev, x_last_loc_ev, mu_ev, sd_ev, reg_ev, naive_rmse, device):

        ridge_alpha = params.get("ridge_alpha", 1e-3)
        noise_std = 0.05
        use_deep_for_forecast = params.get("use_deep_for_forecast", 0)

        R_fit_L1, R_fit_LD = self._get_reservoir_states_in_batches(model, X_loc_fit, device)
        R_ev_L1, R_ev_LD = self._get_reservoir_states_in_batches(model, X_loc_ev, device)
        R_fit_L1n, _ = self._get_reservoir_states_in_batches(model, X_loc_fit, device, noise_std=noise_std)
        R_ev_L1n, _ = self._get_reservoir_states_in_batches(model, X_loc_ev, device, noise_std=noise_std)

        # ---- Head 1: Forecasting (Residual / Delta) ----
        if use_deep_for_forecast:
            F_fit_raw = np.hstack([R_fit_L1, R_fit_LD, x_last_loc_fit])
            F_ev_raw  = np.hstack([R_ev_L1, R_ev_LD, x_last_loc_ev])
        else:
            F_fit_raw = np.hstack([R_fit_L1, x_last_loc_fit])
            F_ev_raw  = np.hstack([R_ev_L1, x_last_loc_ev])

        F_fit = add_bias(F_fit_raw)
        F_ev  = add_bias(F_ev_raw)

        H, F_dim = y_loc_fit.shape[1], y_loc_fit.shape[2]
        baseline_fit = np.repeat(x_last_loc_fit[:, None, :], H, axis=1)
        baseline_ev  = np.repeat(x_last_loc_ev[:, None, :], H, axis=1)

        target_delta_fit = y_loc_fit - baseline_fit
        target_delta_fit_flat = target_delta_fit.reshape(len(target_delta_fit), -1)

        W_fore = robust_torch_ridge_fit(F_fit, target_delta_fit_flat, ridge_alpha, device)
        pred_delta_flat = (torch.from_numpy(F_ev).float().to(device) @ W_fore).cpu().numpy()

        pred_delta = pred_delta_flat.reshape(len(y_loc_ev), H, F_dim)
        pred_loc = baseline_ev + pred_delta
        pred_glob = pred_loc * sd_ev[:, None, :] + mu_ev[:, None, :]

        rmse_fore = np.sqrt(mean_squared_error(
            y_glob_ev.reshape(len(y_glob_ev), -1),
            pred_glob.reshape(len(pred_glob), -1)
        ))

        # ---- Head 2: Regime Classification ----
        C_fit = add_bias(R_fit_LD)
        C_ev = add_bias(R_ev_LD)

        alpha_reg = ridge_alpha * 100.0 
        y_reg_oh = one_hot_fixed(reg_fit[:len(C_fit)], self.classes)
        W_reg = robust_torch_ridge_fit(C_fit, y_reg_oh, alpha_reg, device)
        
        pred_idx = np.argmax((torch.from_numpy(C_ev).float().to(device) @ W_reg).cpu().numpy(), axis=1)
        pred_labels = self.classes[pred_idx]
        acc_reg = accuracy_score(reg_ev[:len(C_ev)], pred_labels)

        # ---- Head 3: Reconstruction ----
        S_fit = add_bias(R_fit_L1n)
        S_ev = add_bias(R_ev_L1n)
        tgt_fit = x_last_loc_fit
        tgt_ev = x_last_loc_ev

        alpha_rec = ridge_alpha * 10.0
        W_recon = robust_torch_ridge_fit(S_fit, tgt_fit, alpha_rec, device)
        pred_recon = (torch.from_numpy(S_ev).float().to(device) @ W_recon).cpu().numpy()
        rmse_recon = np.sqrt(mean_squared_error(tgt_ev, pred_recon))

        s_fore = 1.0 / (1.0 + rmse_fore / (naive_rmse + EPS))
        s_reg = acc_reg
        s_stab = 1.0 / (1.0 + rmse_recon)
        score = 0.3 * s_fore + 0.4 * s_reg + 0.3 * s_stab

        metrics = {
            "score": score, "rmse_forecast": rmse_fore,
            "rmse_forecast_naive": naive_rmse, "acc_regime": acc_reg, "rmse_recon": rmse_recon,
        }
        readouts = {"forecast": W_fore, "regime": W_reg, "reconstruct": W_recon}
        n_features = F_fit.shape[1]
        outputs = {"pred_glob": pred_glob}

        del R_fit_L1, R_fit_LD, R_ev_L1, R_ev_LD, R_fit_L1n, R_ev_L1n
        del F_fit, F_ev, C_fit, C_ev, S_fit, S_ev
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

        return metrics, readouts, n_features, outputs

    def evaluate(self, full_params):
        # FIX: Cast immediato degli iperparametri interi
        for ip in ["tot_units", "n_layers", "connectivity_recurrent", "use_deep_for_forecast"]:
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
            m, r, n_features, _ = self._run_heads(
                model, full_params,
                self.X_loc_train, self.y_loc_train, self.y_glob_train, self.x_last_loc_train, self.mu_train, self.sd_train, self.reg_train,
                self.X_loc_val, self.y_loc_val, self.y_glob_val, self.x_last_loc_val, self.mu_val, self.sd_val, self.reg_val,
                self.naive_rmse_val, device)
            per_seed.append(m)
            readouts = r
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        mean_score = float(np.mean([m["score"] for m in per_seed]))
        std_score = float(np.std([m["score"] for m in per_seed]))

        meta = {
            "score": mean_score, "score_std_seeds": std_score,
            "rmse_forecast": float(np.mean([m["rmse_forecast"] for m in per_seed])),
            "acc_regime": float(np.mean([m["acc_regime"] for m in per_seed])),
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
        rmse_denorm_list = []
        H = self.y_glob_test.shape[1]
        F_dim = self.y_glob_test.shape[2]

        for seed in SEEDS_TEST:
            model = build_reservoir(self.input_dim, best_params, device, seed)
            m, r, _, outputs = self._run_heads(
                model, best_params,
                self.X_loc_full, self.y_loc_full, self.y_glob_full, self.x_last_loc_full, self.mu_full, self.sd_full, self.reg_full,
                self.X_loc_test, self.y_loc_test, self.y_glob_test, self.x_last_loc_test, self.mu_test, self.sd_test, self.reg_test,
                self.naive_rmse_test, device)
            results.append(m)

            pred_glob = outputs["pred_glob"]
            pred_physical = self.scaler.inverse_transform(pred_glob.reshape(-1, F_dim)).reshape(len(pred_glob), H, F_dim)
            y_physical = self.scaler.inverse_transform(self.y_glob_test.reshape(-1, F_dim)).reshape(len(self.y_glob_test), H, F_dim)
            rmse_denorm_list.append(np.sqrt(mean_squared_error(
                y_physical.reshape(len(y_physical), -1), pred_physical.reshape(len(pred_physical), -1))))

            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()

        agg = {}
        for k in ["score", "rmse_forecast", "acc_regime", "rmse_recon"]:
            vals = [m[k] for m in results]
            agg[k] = float(np.mean(vals))
            agg[k + "_std"] = float(np.std(vals))
        agg["rmse_forecast_denorm"] = float(np.mean(rmse_denorm_list))
        agg["rmse_forecast_naive"] = self.naive_rmse_test
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

    print(f"--- STARTING ETT ABLATION SUITE (DRY_RUN={DRY_RUN}) ---")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("IEEE_ETT_Horizon_Depth_Ablation")

    for exp in experiments:
        for current_h in HORIZONS:
            seed_everything(GLOBAL_SEED)
            
            run_name = f"DeepRC_{exp['name']}_H{current_h}"
            print("\n" + "=" * 60)
            print(f"RUNNING EXPERIMENT: {run_name}")
            print("=" * 60)

            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("architecture", "DeepRC_Ridge_MultiTap_EffRescale_LLE")
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

                print(f"-> Ottimizzazione conclusa. Regimi identificati: {best_k} (BIC: {lowest_bic:.0f})")
                regimes_raw = best_gmm.predict(scaled_data)
                
                mlflow.log_param("n_regimes_adaptive", best_k)
                mlflow.log_metric("regime_bic_score", lowest_bic)

                classes = np.arange(best_k)

                # 4. SEQUENZE E RC-RevIN (Uso la logica testata di SMD)
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

                # 6. RUN SEARCHER
                dynamic_global_params = GLOBAL_PARAMS.copy()
                dynamic_global_params["forecast_horizon"] = current_h

                searcher = FoundationalRCSearch(
                    X_loc_train=X_loc_train, y_loc_train=y_loc_train, y_glob_train=y_glob_train, x_last_loc_train=x_last_loc_train, mu_train=mu_train, sd_train=sd_train, reg_train=regime_train,
                    X_loc_val=X_loc_val, y_loc_val=y_loc_val, y_glob_val=y_glob_val, x_last_loc_val=x_last_loc_val, x_last_glob_val=x_last_glob_val, mu_val=mu_val, sd_val=sd_val, reg_val=regime_val,
                    X_loc_test=X_loc_test, y_loc_test=y_loc_test, y_glob_test=y_glob_test, x_last_loc_test=x_last_loc_test, x_last_glob_test=x_last_glob_test, mu_test=mu_test, sd_test=sd_test, reg_test=regime_test,
                    X_loc_full=X_loc_full, y_loc_full=y_loc_full, y_glob_full=y_glob_full, x_last_loc_full=x_last_loc_full, mu_full=mu_full, sd_full=sd_full, reg_full=regime_full,
                    classes=classes, scaler=scaler,
                    hp_space=HP_SPACE, n_iterations=N_ITERATIONS,
                    global_params=dynamic_global_params, epsilon_greedy=0.3
                )
                
                start = time.time()
                searcher.search()
                duration = time.time() - start

                # 7. LOG RESULTS & ARTIFACTS
                best_val_meta = searcher.best_meta_result
                clean_params = searcher.best_params.copy()
                for ip in ["tot_units", "n_layers", "connectivity_recurrent", "use_deep_for_forecast"]:
                    if ip in clean_params:
                        clean_params[ip] = int(round(clean_params[ip]))

                test_res = searcher.evaluate_final_test_set(clean_params)
                
                mlflow.log_params(clean_params)
                mlflow.log_metric("val_composite_score", best_val_meta['score'])
                mlflow.log_metric("composite_score", test_res['score'])
                mlflow.log_metric("accuracy_regime", test_res['acc_regime'])
                mlflow.log_metric("rmse_forecast", test_res['rmse_forecast'])
                mlflow.log_metric("rmse_forecast_naive", test_res['rmse_forecast_naive'])
                mlflow.log_metric("rmse_forecast_denorm", test_res['rmse_forecast_denorm'])
                mlflow.log_metric("rmse_stability", test_res['rmse_recon'])
                mlflow.log_metric("duration_sec", duration)

                # Export History
                history_data = []
                for param_tuple, (score, _, n_feat) in searcher._eval_cache.items():
                    row = dict(param_tuple)
                    row['val_composite_score'] = score
                    row['n_features_readout'] = n_feat
                    history_data.append(row)
                    
                df_history = pd.DataFrame(history_data).sort_values(by='val_composite_score', ascending=False)
                csv_filename = f"hp_history_{run_name}.csv"
                df_history.to_csv(csv_filename, index=False)
                mlflow.log_artifact(csv_filename)
                os.remove(csv_filename)

                # Cleanup
                del X_loc_train, X_loc_val, X_loc_full, X_loc_test
                del searcher
                gc.collect()

                print(f"[{run_name}] COMPLETED. Acc: {test_res['acc_regime']:.2%} | RMSE (Denorm): {test_res['rmse_forecast_denorm']:.4f}")

    print("\n" + "="*60)
    print("ALL ETT EXPERIMENTS COMPLETED.")