import os
import time
import random
import numpy as np
import torch
import mlflow
import gc
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import pandas as pd

# --- Moduli custom ---
from base_deep_rc_search import BaseDeepRCSearch
from psi_DeepRC_IP import DeepReservoir     
from utils import load_machine_data, robust_torch_ridge_fit

# =====================================================================
# CONFIGURAZIONE
# =====================================================================
BASE_DIR = '/home/mmendula/last-rush/datasets/ServerMachineDataset'

WINDOW_SIZE = 60
FORECAST_HORIZON = 5
N_ITERATIONS = 100

SEEDS_SEARCH = [11, 22, 33]
SEEDS_TEST   = [11, 22, 33, 44, 55]
GLOBAL_SEED  = 42

# --------- Multi-tap readout ---------
N_TAPS = 4              # stati a 4 istanti equispaziati nella finestra

# --------- Intrinsic Plasticity ---------
IP_ETA        = 5e-4    # learning rate della regola IP
IP_EPOCHS     = 0       # passate sul sotto-campione
IP_BATCH      = 512
IP_N_SAMPLES  = 1024    # finestre usate per l'adattamento IP

# --------- RC-RevIN v5: soft floor + clipping difensivo ---------
SOFT_FLOOR_C = 0.02     # in unita' MinMax: sd_eff = sqrt(sd^2 + C^2)
CLIP_VAL     = 4.0      # clipping degli input locali (NON dei target)

# --------- Architettura (semantiche della classe originale) ---------
LLE_TUNING   = True     # loop di accettazione edge-of-chaos (pre-IP)
BIAS_SCALING = 0.1      # disaccoppiato da input_scaling (che ora e' piccolo)
CONCAT_UNITS = True     # tot_units DIVISO tra i layer (come la v1 standalone);
                        # con False ogni layer avrebbe tot_units neuroni

HP_SPACE = {
    "tot_units": [50, 100, 250, 500, 1000],
    "n_layers": [1, 2, 3, 4, 5],
    "spectral_radius": {"min": np.log(0.5), "max": np.log(1.2)},
    "leaky": [0.1, 0.3, 0.5, 1.0],
    "connectivity_recurrent": [5, 10, 20],
    # FIX: range ricalibrato per input z-scored (prima era tarato su [0,1]
    # post-MinMax e arrivava a 2.0, garantendo saturazione con input +/-3)
    "input_scaling": {"min": np.log(0.02), "max": np.log(0.5)},
    "ridge_alpha": {"min": np.log(1e-3), "max": np.log(1e4)},
    # IP: la deviazione target della distribuzione degli stati e' essa
    # stessa un HP (controlla il punto di lavoro della tanh)
    "ip_target_std": [0.1, 0.15, 0.2, 0.3],
    # ResESN rimosso: niente piu' alpha/beta
    "use_deep_for_forecast": [0, 1], # 1 = ABL1A, 0 = ABL1B
}

GLOBAL_PARAMS = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "window_size": WINDOW_SIZE,
    "forecast_horizon": FORECAST_HORIZON
}

EPS = 1e-8

# =====================================================================
# UTILITY: SEED, STABILITA', E RC-RevIN V5
# =====================================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def stabilize_spectral_radius(rho, leaky):
    eff = (1.0 - leaky) + leaky * rho
    if eff >= 0.99:
        rho = max((0.99 - (1.0 - leaky)) / max(leaky, EPS), 0.1)
    return rho

def build_reservoir(input_dim, params, device, seed, ip_data=None):
    """Costruisce il reservoir e, se ip_data e' fornito, esegue la
    pre-fase di Intrinsic Plasticity. Tutto avviene dentro uno snapshot
    RNG per non disturbare la random walk del searcher (FIX #1 originale).
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    seed_everything(seed)

    # NB: niente stabilize_spectral_radius qui. La cell con
    # effective_rescaling=True applica la condizione ESP corretta per
    # leaky-ESN (su W_eff = leaky*W + (1-leaky)*I); l'euristica del main
    # sarebbe una doppia compensazione che sposta il rho effettivamente
    # cercato. Il loop LLE fa da ulteriore rete di sicurezza.
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
        # mode="reduce-overhead" is specifically designed for speeding up 
        # small/medium matrix multiplications in recurrent loops.
        model = torch.compile(model, dynamic=True)

    if ip_data is not None:
        model.ip_pretrain(
            ip_data,
            target_std=params.get("ip_target_std", 0.2),
            eta=IP_ETA, epochs=IP_EPOCHS,
            batch_size=IP_BATCH, device=device,
        )

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

    # FIX (problema 2): SOFT FLOOR al posto del np.where binario.
    # Degrada gradualmente verso "nessuna normalizzazione" sui canali
    # piatti invece di amplificare il rumore x1000 quando sd ~ 1e-3.
    sd = np.sqrt(sd ** 2 + SOFT_FLOOR_C ** 2)

    X_loc = (X_glob - mu) / sd
    y_loc = (y_glob - mu) / sd

    # FIX (clipping difensivo): elimina gli outlier residui che nessun
    # floor cattura. Si clippano SOLO gli input, mai i target, per non
    # distorcere la ground truth della Ridge.
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

def make_ip_subsample(X, n_samples, seed):
    """Sotto-campione FISSO per l'adattamento IP, indipendente dallo stato
    RNG globale: tutte le configurazioni vedono gli stessi dati IP."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    return np.ascontiguousarray(X[np.sort(idx)])

# =====================================================================
# SEARCHER
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

        # Sotto-campioni fissi per IP (search: solo train; test: train+val)
        self.ip_sample_search = make_ip_subsample(self.X_loc_train, IP_N_SAMPLES, GLOBAL_SEED)
        self.ip_sample_final  = make_ip_subsample(self.X_loc_full,  IP_N_SAMPLES, GLOBAL_SEED)

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

        # ---- Head 1: Forecasting (Residual / Delta Prediction) ----
        if use_deep_for_forecast: # ABL1A
            F_fit_raw = np.hstack([R_fit_L1, R_fit_LD, x_last_loc_fit])
            F_ev_raw  = np.hstack([R_ev_L1, R_ev_LD, x_last_loc_ev])
        else:                     # ABL1B
            F_fit_raw = np.hstack([R_fit_L1, x_last_loc_fit])
            F_ev_raw  = np.hstack([R_ev_L1, x_last_loc_ev])

        # NESSUNA STANDARDIZZAZIONE: i dati sono già limitati da RevIN e tanh
        F_fit = add_bias(F_fit_raw)
        F_ev  = add_bias(F_ev_raw)

        H, F_dim = y_loc_fit.shape[1], y_loc_fit.shape[2]
        baseline_fit = np.repeat(x_last_loc_fit[:, None, :], H, axis=1)
        baseline_ev  = np.repeat(x_last_loc_ev[:, None, :], H, axis=1)

        # Addestriamo sulla differenza (Delta)
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

        alpha_reg = ridge_alpha * 100.0  # Regolarizzazione aggressiva per il transfer learning
        
        y_reg_oh = one_hot_fixed(reg_fit[:len(C_fit)], self.classes)
        W_reg = robust_torch_ridge_fit(C_fit, y_reg_oh, alpha_reg, device)
        
        pred_idx = np.argmax((torch.from_numpy(C_ev).float().to(device) @ W_reg).cpu().numpy(), axis=1)
        pred_labels = self.classes[pred_idx]
        acc_reg = accuracy_score(reg_ev[:len(C_ev)], pred_labels)

        # ---- Head 3: Reconstruction (Stability Probe) ----
        S_fit = add_bias(R_fit_L1n)
        S_ev = add_bias(R_ev_L1n)
        tgt_fit = x_last_loc_fit
        tgt_ev = x_last_loc_ev

        alpha_rec = ridge_alpha * 10.0 # Un po' di margine anche per la reconstruction
        W_recon = robust_torch_ridge_fit(S_fit, tgt_fit, alpha_rec, device)
        pred_recon = (torch.from_numpy(S_ev).float().to(device) @ W_recon).cpu().numpy()
        rmse_recon = np.sqrt(mean_squared_error(tgt_ev, pred_recon))

        # ---- Scoring ----
        s_fore = 1.0 / (1.0 + rmse_fore / (naive_rmse + EPS))
        s_reg = acc_reg
        s_stab = 1.0 / (1.0 + rmse_recon)
        score = 0.3 * s_fore + 0.4 * s_reg + 0.3 * s_stab

        metrics = {
            "score": score,
            "rmse_forecast": rmse_fore,
            "rmse_forecast_naive": naive_rmse,
            "acc_regime": acc_reg,
            "rmse_recon": rmse_recon,
        }
        readouts = {"forecast": W_fore, "regime": W_reg, "reconstruct": W_recon}
        n_features = F_fit.shape[1]
        outputs = {"pred_glob": pred_glob}

        del R_fit_L1, R_fit_LD, R_ev_L1, R_ev_LD, R_fit_L1n, R_ev_L1n
        del F_fit, F_ev, C_fit, C_ev, S_fit, S_ev
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return metrics, readouts, n_features, outputs

    def evaluate(self, full_params):
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
            model = build_reservoir(self.input_dim, full_params, device, seed,
                                    ip_data=self.ip_sample_search)
            m, r, n_features, _ = self._run_heads(
                model, full_params,
                self.X_loc_train, self.y_loc_train, self.y_glob_train, self.x_last_loc_train, self.mu_train, self.sd_train, self.reg_train,
                self.X_loc_val, self.y_loc_val, self.y_glob_val, self.x_last_loc_val, self.mu_val, self.sd_val, self.reg_val,
                self.naive_rmse_val, device)
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
            model = build_reservoir(self.input_dim, best_params, device, seed,
                                    ip_data=self.ip_sample_final)
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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
# CARICAMENTO DATI PER MACCHINA
# =====================================================================
def load_and_process_machine(raw_data, machine_name, split_type, scaler):
    p_label = os.path.join(BASE_DIR, 'regime_labels', f"{split_type}_{machine_name.replace('.txt', '.npy')}")
    labels_raw = np.load(p_label)
    scaled_data = scaler.transform(raw_data)
    X_loc, y_loc, y_glob, x_last_loc, x_last_glob, mu, sd = create_sequences_rcrevin(scaled_data, WINDOW_SIZE, FORECAST_HORIZON)
    labels = labels_raw[WINDOW_SIZE - 1 : WINDOW_SIZE - 1 + len(X_loc)]
    return X_loc, y_loc, y_glob, x_last_loc, x_last_glob, mu, sd, labels

# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    seed_everything(GLOBAL_SEED)

    experiments = [
        {"name": "WP1_LowDiv_2_Machines",
         "corpus": ['machine-1-1.txt', 'machine-1-2.txt'],
         "val_machine": 'machine-1-2.txt',
         "target": 'machine-3-11.txt'},
        {"name": "WP1_MedDiv_4_Machines",
         "corpus": ['machine-1-1.txt', 'machine-1-2.txt', 'machine-2-1.txt', 'machine-2-2.txt'],
         "val_machine": 'machine-1-2.txt',
         "target": 'machine-3-11.txt'},
        {"name": "WP1_HighDiv_6_Machines",
         "corpus": ['machine-1-1.txt', 'machine-1-2.txt', 'machine-2-1.txt',
                    'machine-2-2.txt', 'machine-3-1.txt', 'machine-3-2.txt'],
         "val_machine": 'machine-1-2.txt',
         "target": 'machine-3-11.txt'},
        {"name": "WP2_LODO_Groups_1_and_2",
         "corpus": ['machine-1-1.txt', 'machine-1-2.txt', 'machine-1-3.txt',
                    'machine-2-1.txt', 'machine-2-2.txt', 'machine-2-3.txt'],
         "val_machine": 'machine-2-3.txt',
         "target": 'machine-3-11.txt'},
        {"name": "WP3_Random_Mix_1",
         "corpus": ['machine-1-3.txt', 'machine-1-5.txt', 'machine-2-4.txt',
                    'machine-2-7.txt', 'machine-3-4.txt', 'machine-3-8.txt'],
         "val_machine": 'machine-2-7.txt',
         "target": 'machine-3-11.txt'},
        {"name": "WP3_Random_Mix_2",
         "corpus": ['machine-1-4.txt', 'machine-1-7.txt', 'machine-2-3.txt',
                    'machine-2-5.txt', 'machine-3-5.txt', 'machine-3-9.txt'],
         "val_machine": 'machine-2-5.txt',
         "target": 'machine-3-11.txt'},
        {"name": "WP4_Target_1-8",
         "corpus": ['machine-1-1.txt', 'machine-1-2.txt', 'machine-2-1.txt',
                    'machine-2-2.txt', 'machine-3-1.txt', 'machine-3-2.txt'],
         "val_machine": 'machine-2-2.txt',
         "target": 'machine-1-8.txt'},
        {"name": "WP4_Target_2-9",
         "corpus": ['machine-1-1.txt', 'machine-1-2.txt', 'machine-2-1.txt',
                    'machine-2-2.txt', 'machine-3-1.txt', 'machine-3-2.txt'],
         "val_machine": 'machine-2-2.txt',
         "target": 'machine-2-9.txt'},
    ]

    print(f"--- PHASE 1 EXPERIMENT SUITE ({len(experiments)} run) ---")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("IEEE_COMPARISON_MOMENT_BENCHMARK-v13-noIP")

    for exp in experiments:
        seed_everything(GLOBAL_SEED)

        print("\n" + "=" * 60)
        print(f"RUNNING EXPERIMENT: {exp['name']}")
        print("=" * 60)

        with mlflow.start_run(run_name=f"DeepRC_IP_{exp['name']}"):
            mlflow.log_param("architecture", "DeepRC_Ridge_MultiTap_IP_EffRescale_LLE")
            mlflow.log_param("lle_tuning", LLE_TUNING)
            mlflow.log_param("bias_scaling", BIAS_SCALING)
            mlflow.log_param("concat_units", CONCAT_UNITS)
            mlflow.log_param("target_machine", exp['target'])
            mlflow.log_param("val_machine", exp['val_machine'])
            mlflow.log_param("corpus_size", len(exp['corpus']))
            mlflow.log_param("corpus_list", ",".join(exp['corpus']))
            mlflow.log_param("seeds_search", ",".join(map(str, SEEDS_SEARCH)))
            mlflow.log_param("normalization", "Dual_Stream_RC_RevIN_v5_SoftFloor")
            mlflow.log_param("scaler_fit_scope", "train_machines_only")
            mlflow.log_param("soft_floor_c", SOFT_FLOOR_C)
            mlflow.log_param("clip_val", CLIP_VAL)
            mlflow.log_param("n_taps", N_TAPS)
            mlflow.log_param("ip_eta", IP_ETA)
            mlflow.log_param("ip_epochs", IP_EPOCHS)
            mlflow.log_param("ip_n_samples", IP_N_SAMPLES)
            mlflow.log_param("resesn", "removed")

            val_machine = exp["val_machine"]
            train_machines = [m for m in exp["corpus"] if m != val_machine]
            assert len(train_machines) >= 1, "Il corpus deve avere almeno 2 macchine."

            raw_tr_dict = {}
            for m in train_machines:
                raw_tr_dict[m] = load_machine_data(os.path.join(BASE_DIR, 'train', m))

            raw_val = load_machine_data(os.path.join(BASE_DIR, 'train', val_machine))
            raw_test = load_machine_data(os.path.join(BASE_DIR, 'test', exp['target']))

            scaler = MinMaxScaler()
            scaler.fit(np.vstack(list(raw_tr_dict.values())))

            tr_Xl, tr_yl, tr_yg, tr_xll, tr_m, tr_s, tr_reg = [], [], [], [], [], [], []
            for m in train_machines:
                Xl, yl, yg, xll, _, mu, sd, lab = load_and_process_machine(raw_tr_dict[m], m, 'train', scaler)
                tr_Xl.append(Xl); tr_yl.append(yl); tr_yg.append(yg); tr_xll.append(xll)
                tr_m.append(mu); tr_s.append(sd); tr_reg.append(lab)

            X_loc_train = np.concatenate(tr_Xl, axis=0)
            y_loc_train = np.concatenate(tr_yl, axis=0)
            y_glob_train = np.concatenate(tr_yg, axis=0)
            x_last_loc_train = np.concatenate(tr_xll, axis=0)
            mu_train = np.concatenate(tr_m, axis=0)
            sd_train = np.concatenate(tr_s, axis=0)
            regime_train = np.concatenate(tr_reg, axis=0)
            del tr_Xl, tr_yl, tr_yg, tr_xll, tr_m, tr_s, tr_reg

            X_loc_val, y_loc_val, y_glob_val, x_last_loc_val, x_last_glob_val, mu_val, sd_val, regime_val = load_and_process_machine(raw_val, val_machine, 'train', scaler)

            X_loc_full = np.concatenate([X_loc_train, X_loc_val], axis=0)
            y_loc_full = np.concatenate([y_loc_train, y_loc_val], axis=0)
            y_glob_full = np.concatenate([y_glob_train, y_glob_val], axis=0)
            x_last_loc_full = np.concatenate([x_last_loc_train, x_last_loc_val], axis=0)
            mu_full = np.concatenate([mu_train, mu_val], axis=0)
            sd_full = np.concatenate([sd_train, sd_val], axis=0)
            regime_full = np.concatenate([regime_train, regime_val], axis=0)

            classes = np.unique(regime_full)
            mlflow.log_param("n_regime_classes", len(classes))

            X_loc_test, y_loc_test, y_glob_test, x_last_loc_test, x_last_glob_test, mu_test, sd_test, regime_test = load_and_process_machine(raw_test, exp['target'], 'test', scaler)

            searcher = FoundationalRCSearch(
                X_loc_train=X_loc_train, y_loc_train=y_loc_train, y_glob_train=y_glob_train, x_last_loc_train=x_last_loc_train, mu_train=mu_train, sd_train=sd_train, reg_train=regime_train,
                X_loc_val=X_loc_val, y_loc_val=y_loc_val, y_glob_val=y_glob_val, x_last_loc_val=x_last_loc_val, x_last_glob_val=x_last_glob_val, mu_val=mu_val, sd_val=sd_val, reg_val=regime_val,
                X_loc_test=X_loc_test, y_loc_test=y_loc_test, y_glob_test=y_glob_test, x_last_loc_test=x_last_loc_test, x_last_glob_test=x_last_glob_test, mu_test=mu_test, sd_test=sd_test, reg_test=regime_test,
                X_loc_full=X_loc_full, y_loc_full=y_loc_full, y_glob_full=y_glob_full, x_last_loc_full=x_last_loc_full, mu_full=mu_full, sd_full=sd_full, reg_full=regime_full,
                classes=classes, scaler=scaler,
                hp_space=HP_SPACE, n_iterations=N_ITERATIONS,
                global_params=GLOBAL_PARAMS, epsilon_greedy=0.3
            )

            start = time.time()
            searcher.search()
            duration = time.time() - start

            best_val_meta = searcher.best_meta_result
            clean_params = searcher.best_params.copy()
            for ip in ["tot_units", "n_layers", "connectivity_recurrent", "use_deep_for_forecast"]:
                if ip in clean_params:
                    clean_params[ip] = int(round(clean_params[ip]))
            # spectral_radius NON viene piu' corretto qui: effective_rescaling
            # nella cell gestisce la condizione ESP per leaky-ESN.

            test_res = searcher.evaluate_final_test_set(clean_params)

            mlflow.log_params(clean_params)
            mlflow.log_metric("val_composite_score", best_val_meta['score'])
            mlflow.log_metric("val_score_std_seeds", best_val_meta['score_std_seeds'])

            mlflow.log_metric("composite_score", test_res['score'])
            mlflow.log_metric("composite_score_std", test_res['score_std'])
            mlflow.log_metric("accuracy_regime", test_res['acc_regime'])
            mlflow.log_metric("accuracy_regime_std", test_res['acc_regime_std'])
            mlflow.log_metric("rmse_forecast", test_res['rmse_forecast'])
            mlflow.log_metric("rmse_forecast_std", test_res['rmse_forecast_std'])
            mlflow.log_metric("rmse_forecast_naive", test_res['rmse_forecast_naive'])
            mlflow.log_metric("rmse_forecast_denorm", test_res['rmse_forecast_denorm'])
            mlflow.log_metric("rmse_stability", test_res['rmse_recon'])
            mlflow.log_metric("rmse_stability_std", test_res['rmse_recon_std'])
            mlflow.log_metric("duration_sec", duration)

            history_data = []
            # _eval_cache salva le chiavi come tuple di (param_name, param_value)
            for param_tuple, (score, _, n_feat) in searcher._eval_cache.items():
                row = dict(param_tuple)
                row['val_composite_score'] = score
                row['n_features_readout'] = n_feat
                history_data.append(row)
                
            df_history = pd.DataFrame(history_data)
            # Ordiniamo dalla configurazione migliore alla peggiore
            df_history = df_history.sort_values(by='val_composite_score', ascending=False)
            
            # Salviamo il CSV localmente e lo agganciamo a MLflow
            csv_filename = f"hp_history_{exp['name']}.csv"
            df_history.to_csv(csv_filename, index=False)
            mlflow.log_artifact(csv_filename)
            os.remove(csv_filename) # Pulizia del file locale

            print(f"\n[MLFLOW LOGGED] {exp['name']}"
                  f" - Acc: {test_res['acc_regime']:.2%} (+/- {test_res['acc_regime_std']:.2%})"
                  f" - RMSE (Glob): {test_res['rmse_forecast']:.4f} (+/- {test_res['rmse_forecast_std']:.4f})"
                  f" - Naive (Glob): {test_res['rmse_forecast_naive']:.4f}"
                  f" - RMSE (Physical): {test_res['rmse_forecast_denorm']:.4f}")

            del X_loc_train, y_loc_train, y_glob_train, x_last_loc_train, mu_train, sd_train, regime_train
            del X_loc_val, y_loc_val, y_glob_val, x_last_loc_val, x_last_glob_val, mu_val, sd_val, regime_val
            del X_loc_full, y_loc_full, y_glob_full, x_last_loc_full, mu_full, sd_full, regime_full
            del X_loc_test, y_loc_test, y_glob_test, x_last_loc_test, x_last_glob_test, mu_test, sd_test, regime_test
            del searcher
            gc.collect()

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED.")