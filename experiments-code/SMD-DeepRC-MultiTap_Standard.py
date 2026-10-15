import os
import time
import numpy as np
import pandas as pd
import torch
import mlflow
import gc
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge, RidgeClassifier, RidgeCV, RidgeClassifierCV
from sklearn.metrics import mean_squared_error, accuracy_score

# --- Import your custom modules ---
# Ensure base_deep_rc_search.py and psi_DeepRC.py are in the folder
from base_deep_rc_search import BaseDeepRCSearch
from psi_DeepRC import DeepReservoir

from utils import load_machine_data, create_sequences, torch_ridge_fit

# --- CONFIGURATION ---
BASE_DIR = '/home/workstation163/Documents/datasets/SMD/OmniAnomaly-master/ServerMachineDataset'

# 1. THE FOUNDATION CORPUS (Representative Mixture)
# We mix groups to learn "Universal Server Physics"
TRAIN_CORPUS = [
    'machine-1-1.txt', 'machine-1-2.txt',  # Group 1 (Web/App)
    'machine-2-1.txt', 'machine-2-2.txt',  # Group 2 (DB/Backend)
    'machine-3-1.txt', 'machine-3-2.txt'   # Group 3 (Compute/Batch)
]

# 2. THE UNSEEN TARGET (Zero-Shot Test)
# A machine NOT in the training corpus
TEST_TARGET = 'machine-3-11.txt'

# Hyperparameters
WINDOW_SIZE = 60
FORECAST_HORIZON = 5
N_ITERATIONS = 500
# N_ITERATIONS = 60

HP_SPACE = {
    "tot_units": [50, 100, 250, 500, 1000],
    # "tot_units": [50, 100, 250, 500, 1000],
    "n_layers": [2, 3, 4, 5],
    "spectral_radius": {"min": np.log(0.8), "max": np.log(1.5)},
    "leaky": [0.1, 0.3, 0.5, 1.0],
    "connectivity_recurrent": [5, 10, 20],
    "input_scaling": {"min": np.log(0.1), "max": np.log(2.0)},
    "ridge_alpha": {"min": np.log(1e-6), "max": np.log(1e-2)},

    # "alpha": [0.90, 0.95, 0.99, 0.999],  # Fading Memory Conduit
    # "beta": [0.001, 0.01, 0.1, 0.5, 1.0] # Nonlinear Abstraction Processor
}

GLOBAL_PARAMS = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "window_size": WINDOW_SIZE,
    "forecast_horizon": FORECAST_HORIZON
}

# --- THE SEARCHER CLASS ---
class FoundationalRCSearch(BaseDeepRCSearch):
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, 
                 regime_train, regime_val, regime_test, 
                 X_train_full, y_train_full, regime_train_full, 
                 hp_space, n_iterations, global_params, epsilon_greedy=0.3):
            super().__init__(hp_space, n_iterations, global_params, epsilon_greedy)
            
            # Used strictly for evaluate() (Validation Tuning)
            self.X_train = X_train
            self.y_train = y_train
            self.regime_train = regime_train
            
            self.X_val = X_val
            self.y_val = y_val
            self.regime_val = regime_val

            # <--- 2. BIND FULL CORPUS FOR evaluate_final_test_set()
            self.X_train_full = X_train_full
            self.y_train_full = y_train_full
            self.regime_train_full = regime_train_full

            # Completely isolated Test set
            self.X_test = X_test
            self.y_test = y_test
            self.regime_test = regime_test
            
            self.input_dim = self.X_train.shape[2]
            self.my_best_score = -np.inf
            self.best_meta_result = {}

    def _get_reservoir_states_in_batches(self, model, X_numpy, device, batch_size=2048, noise_std=0.0):
        L1_states = []
        LDeep_states = []
        model.eval()
        
        with torch.no_grad():
            for i in range(0, len(X_numpy), batch_size):
                batch = torch.from_numpy(X_numpy[i:i+batch_size]).float().to(device)
                if noise_std > 0.0:
                    batch += torch.randn_like(batch) * noise_std
                
                # Use the new return_all_layers flag from psi_DeepRC
                _, states_last = model(batch, return_all_layers=True)
                
                L1_states.append(states_last[0].cpu().numpy())    # The Micro View (Layer 1)
                LDeep_states.append(states_last[-1].cpu().numpy()) # The Macro View (Layer L)
                
        return np.concatenate(L1_states, axis=0), np.concatenate(LDeep_states, axis=0)

    def evaluate(self, full_params):
        device = full_params["device"]
        ridge_alpha = full_params.get("ridge_alpha", 1e-3)

        # 2. Run Reservoir
        # Note: We remove explicit **full_params to avoid double-init errors
        model = DeepReservoir(
            input_size=self.input_dim,
            tot_units=int(round(full_params["tot_units"])),
            n_layers=int(round(full_params["n_layers"])),
            spectral_radius=full_params["spectral_radius"],
            leaky=full_params["leaky"],
            connectivity_recurrent=int(round(full_params["connectivity_recurrent"])),
            input_scaling=full_params.get("input_scaling", 1.0),
            alpha=0,
            beta=1.0,
            bias_bool=True
        ).to(device)
        
        R_train_clean_L1, R_train_clean_LDeep = self._get_reservoir_states_in_batches(model, self.X_train, device)
        R_test_clean_L1, R_test_clean_LDeep  = self._get_reservoir_states_in_batches(model, self.X_val, device)
        R_train_noisy_L1, _ = self._get_reservoir_states_in_batches(model, self.X_train, device, noise_std=0.05)
        R_test_noisy_L1, _  = self._get_reservoir_states_in_batches(model, self.X_val, device, noise_std=0.05)

        # --- Head 1: Forecasting (Taps Layer 1) ---
        y_train_flat = self.y_train.reshape(len(self.y_train), -1)
        y_test_flat = self.y_val.reshape(len(self.y_val), -1)
        W_fore = torch_ridge_fit(R_train_clean_L1, y_train_flat, ridge_alpha, device)
        
        R_test_clean_L1_t = torch.from_numpy(R_test_clean_L1).float().to(device)
        pred_fore = (R_test_clean_L1_t @ W_fore).cpu().numpy()
        rmse_fore = np.sqrt(mean_squared_error(y_test_flat, pred_fore))

        # --- Head 2: Regime Classification (Taps Deepest Layer) ---
        y_reg_train = self.regime_train[:len(R_train_clean_LDeep)]
        y_reg_test = self.regime_val[:len(R_test_clean_LDeep)]
        y_reg_train_oh = pd.get_dummies(y_reg_train).astype(float).values
        
        W_reg = torch_ridge_fit(R_train_clean_LDeep, y_reg_train_oh, ridge_alpha, device)
        R_test_clean_LDeep_t = torch.from_numpy(R_test_clean_LDeep).float().to(device)
        pred_reg = np.argmax((R_test_clean_LDeep_t @ W_reg).cpu().numpy(), axis=1)
        acc_reg = accuracy_score(y_reg_test, pred_reg)

        # --- Head 3: Reconstruction / Stability (Taps Layer 1 Noisy) ---
        target_recon_train = self.X_train[:len(R_train_noisy_L1), -1, :]
        target_recon_test = self.X_val[:len(R_test_noisy_L1), -1, :]
        
        W_recon = torch_ridge_fit(R_train_noisy_L1, target_recon_train, ridge_alpha, device)
        R_test_noisy_L1_t = torch.from_numpy(R_test_noisy_L1).float().to(device)
        pred_recon = (R_test_noisy_L1_t @ W_recon).cpu().numpy()
        rmse_recon = np.sqrt(mean_squared_error(target_recon_test, pred_recon))

        # --- Head 3: Reconstruction / Stability (Taps Layer 1 Noisy) ---
        target_recon_train = self.X_train[:len(R_train_noisy_L1), -1, :]
        target_recon_test = self.X_val[:len(R_test_noisy_L1), -1, :]
        
        W_recon = torch_ridge_fit(R_train_noisy_L1, target_recon_train, ridge_alpha, device)
        
        R_test_noisy_L1_t = torch.from_numpy(R_test_noisy_L1).float().to(device)
        pred_recon = (R_test_noisy_L1_t @ W_recon).cpu().numpy()
        rmse_recon = np.sqrt(mean_squared_error(target_recon_test, pred_recon))

        # --- Scoring & Logging ---
        s_fore = np.clip(1.0 - rmse_fore, 0, 1)
        s_reg = acc_reg
        s_stab = np.clip(1.0 - (rmse_recon * 5), 0, 1) 

        final_score = (0.3 * s_fore) + (0.4 * s_reg) + (0.3 * s_stab)

        meta = {
            "score": final_score,
            "rmse_forecast": rmse_fore,
            "acc_regime": acc_reg,
            "rmse_recon": rmse_recon,
            "units": R_train_clean_L1.shape[1] # Cache units
        }

        if final_score > self.my_best_score:
            self.my_best_score = final_score
            self.best_meta_result = meta
            
        readouts = {
            "forecast": W_fore, 
            "regime": W_reg, 
            "reconstruct": W_recon
        }

        # Cache shape safely before deletion
        r_train_shape = R_train_clean_L1.shape[1]
        
        # CLEANUP
        del R_train_clean_L1, R_train_clean_LDeep, R_test_clean_L1, R_test_clean_LDeep
        del R_train_noisy_L1, R_test_noisy_L1
        del R_test_clean_L1_t, R_test_clean_LDeep_t, R_test_noisy_L1_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return final_score, readouts, None, r_train_shape

    def evaluate_final_test_set(self, best_params):
        """Runs a strict single pass over the isolated test set using the winning HPs."""
        device = self.global_params["device"]
        ridge_alpha = best_params.get("ridge_alpha", 1e-3)
        noise_std = 0.05

        model = DeepReservoir(
            input_size=self.input_dim,
            tot_units=int(round(best_params["tot_units"])),
            n_layers=int(round(best_params["n_layers"])),
            spectral_radius=best_params["spectral_radius"],
            leaky=best_params["leaky"],
            connectivity_recurrent=int(round(best_params["connectivity_recurrent"])),
            input_scaling=best_params.get("input_scaling", 1.0),
            alpha=0.0,
            beta=1.0, 
            bias_bool=True
        ).to(device)

        # Extract Taps (L1 and LDeep)
        R_train_clean_L1, R_train_clean_LDeep = self._get_reservoir_states_in_batches(model, self.X_train_full, device)
        R_test_clean_L1, R_test_clean_LDeep  = self._get_reservoir_states_in_batches(model, self.X_test, device)
        R_train_noisy_L1, _ = self._get_reservoir_states_in_batches(model, self.X_train_full, device, noise_std=noise_std)
        R_test_noisy_L1, _  = self._get_reservoir_states_in_batches(model, self.X_test, device, noise_std=noise_std)

        # Score Forecasting (Taps Layer 1)
        y_train_flat = self.y_train_full.reshape(len(self.y_train_full), -1) 
        y_test_flat = self.y_test.reshape(len(self.y_test), -1)
        W_fore = torch_ridge_fit(R_train_clean_L1, y_train_flat, ridge_alpha, device)
        rmse_fore = np.sqrt(mean_squared_error(y_test_flat, (torch.from_numpy(R_test_clean_L1).float().to(device) @ W_fore).cpu().numpy()))
        
        # Score Regime (Taps Deepest Layer)
        y_reg_train_oh = pd.get_dummies(self.regime_train_full[:len(R_train_clean_LDeep)]).astype(float).values 
        W_reg = torch_ridge_fit(R_train_clean_LDeep, y_reg_train_oh, ridge_alpha, device)
        pred_reg = np.argmax((torch.from_numpy(R_test_clean_LDeep).float().to(device) @ W_reg).cpu().numpy(), axis=1)
        acc_reg = accuracy_score(self.regime_test[:len(R_test_clean_LDeep)], pred_reg)

        # Score Reconstruction (Taps Layer 1 Noisy)
        target_recon_train = self.X_train_full[:len(R_train_noisy_L1), -1, :]
        target_recon_test = self.X_test[:len(R_test_noisy_L1), -1, :]
        W_recon = torch_ridge_fit(R_train_noisy_L1, target_recon_train, ridge_alpha, device)
        rmse_recon = np.sqrt(mean_squared_error(target_recon_test, (torch.from_numpy(R_test_noisy_L1).float().to(device) @ W_recon).cpu().numpy()))

        final_score = (0.3 * np.clip(1.0 - rmse_fore, 0, 1)) + (0.4 * acc_reg) + (0.3 * np.clip(1.0 - (rmse_recon * 5), 0, 1))

        return {
            "score": final_score, "rmse_forecast": rmse_fore, 
            "acc_regime": acc_reg, "rmse_recon": rmse_recon
        }

# --- MAIN EXECUTION (PHASE 1 WITH MLFLOW) ---
if __name__ == "__main__":
    # Define the experiments for Phase 1
    experiments = [
        {"name": "WP1_LowDiv_2_Machines", "corpus": ['machine-1-1.txt', 'machine-1-2.txt'], "target": 'machine-3-11.txt'},
        {"name": "WP1_MedDiv_4_Machines", "corpus": ['machine-1-1.txt', 'machine-1-2.txt', 'machine-2-1.txt', 'machine-2-2.txt'], "target": 'machine-3-11.txt'},
        {"name": "WP1_HighDiv_6_Machines", "corpus": ['machine-1-1.txt', 'machine-1-2.txt', 'machine-2-1.txt', 'machine-2-2.txt', 'machine-3-1.txt', 'machine-3-2.txt'], "target": 'machine-3-11.txt'},
        {"name": "WP2_LODO_Groups_1_and_2", "corpus": ['machine-1-1.txt', 'machine-1-2.txt', 'machine-1-3.txt', 'machine-2-1.txt', 'machine-2-2.txt', 'machine-2-3.txt'], "target": 'machine-3-11.txt'},
        {"name": "WP3_Random_Mix_1", "corpus": ['machine-1-3.txt', 'machine-1-5.txt', 'machine-2-4.txt', 'machine-2-7.txt', 'machine-3-4.txt', 'machine-3-8.txt'], "target": 'machine-3-11.txt'},
        {"name": "WP3_Random_Mix_2", "corpus": ['machine-1-4.txt', 'machine-1-7.txt', 'machine-2-3.txt', 'machine-2-5.txt', 'machine-3-5.txt', 'machine-3-9.txt'], "target": 'machine-3-11.txt'},
        {"name": "WP4_Target_1-8", "corpus": ['machine-1-1.txt', 'machine-1-2.txt', 'machine-2-1.txt', 'machine-2-2.txt', 'machine-3-1.txt', 'machine-3-2.txt'], "target": 'machine-1-8.txt'},
        {"name": "WP4_Target_2-9", "corpus": ['machine-1-1.txt', 'machine-1-2.txt', 'machine-2-1.txt', 'machine-2-2.txt', 'machine-3-1.txt', 'machine-3-2.txt'], "target": 'machine-2-9.txt'}
    ]

    print(f"--- STARTING PHASE 1 EXPERIMENT SUITE ({len(experiments)} Runs) ---")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("IEEE_COMPARISON_MOMENT_BENCHMARK-large-search")

    for exp in experiments:
        print("\n" + "="*60)
        print(f"RUNNING EXPERIMENT: {exp['name']}")
        print("="*60)

        # Start an MLflow run for this specific experiment
        with mlflow.start_run(run_name=f"DeepRC_{exp['name']}"):
            mlflow.log_param("architecture", "DeepRC_Ridge_MultiTap_Standard")
            
            # --- LOG SETUP PARAMETERS ---
            mlflow.log_param("target_machine", exp['target'])
            mlflow.log_param("corpus_size", len(exp['corpus']))
            mlflow.log_param("corpus_list", ",".join(exp['corpus']))
            
            # 1. LOAD & CONCATENATE TRAINING CORPUS
            X_raw_list, regime_list = [], []
            for machine in exp['corpus']:
                p_data = os.path.join(BASE_DIR, 'train', machine)
                p_label = os.path.join(BASE_DIR, 'regime_labels', f"train_{machine.replace('.txt', '.npy')}")
                X_raw_list.append(load_machine_data(p_data))
                regime_list.append(np.load(p_label))

            # 2. FIT GLOBAL SCALER
            scaler = MinMaxScaler()
            scaler.fit(np.vstack(X_raw_list))
            
            # 3. CREATE SEQUENCES FOR TRAINING AND VALIDATION
            X_train_sub_list, y_train_sub_list, reg_train_sub_list = [], [], []
            X_val_list, y_val_list, reg_val_list = [], [], []
            
            for i, data in enumerate(X_raw_list):
                d_scaled = scaler.transform(data)
                X, y = create_sequences(d_scaled, WINDOW_SIZE, FORECAST_HORIZON)
                l_aligned = regime_list[i][WINDOW_SIZE-1 : WINDOW_SIZE-1 + len(X)]
                
                # --- NEW: Slice 80/20 per machine ---
                m_val_size = int(len(X) * 0.2)
                
                X_train_sub_list.append(X[:-m_val_size])
                y_train_sub_list.append(y[:-m_val_size])
                reg_train_sub_list.append(l_aligned[:-m_val_size])
                
                X_val_list.append(X[-m_val_size:])
                y_val_list.append(y[-m_val_size:])
                reg_val_list.append(l_aligned[-m_val_size:])

            # Concatenate the balanced subsets
            X_train_sub = np.concatenate(X_train_sub_list, axis=0)
            y_train_sub = np.concatenate(y_train_sub_list, axis=0)
            regime_train_sub = np.concatenate(reg_train_sub_list, axis=0)
            
            X_val = np.concatenate(X_val_list, axis=0)
            y_val = np.concatenate(y_val_list, axis=0)
            regime_val = np.concatenate(reg_val_list, axis=0)
            
            # We still need the FULL training set for the final evaluation fit
            X_train_full = np.concatenate([X_train_sub, X_val], axis=0)
            y_train_full = np.concatenate([y_train_sub, y_val], axis=0)
            regime_train_full = np.concatenate([regime_train_sub, regime_val], axis=0)

            # 4. PREPARE TEST TARGET
            p_test_data = os.path.join(BASE_DIR, 'test', exp['target'])
            p_test_label = os.path.join(BASE_DIR, 'regime_labels', f"test_{exp['target'].replace('.txt', '.npy')}")
            
            test_scaled = scaler.transform(load_machine_data(p_test_data))
            X_test, y_test = create_sequences(test_scaled, WINDOW_SIZE, FORECAST_HORIZON)
            regime_test = np.load(p_test_label)[WINDOW_SIZE-1 : WINDOW_SIZE-1 + len(X_test)]

            # 5. RUN SEARCH
            searcher = FoundationalRCSearch(
                # Train the search loop strictly on the 80% subset
                X_train=X_train_sub, y_train=y_train_sub, regime_train=regime_train_sub,
                
                # Validate on the 20% subset
                X_val=X_val, y_val=y_val, regime_val=regime_val,
                
                # Isolate Test
                X_test=X_test, y_test=y_test, regime_test=regime_test,
                
                # NEW: Pass the full 100% corpus exclusively for the final test fit
                X_train_full=X_train_full, y_train_full=y_train_full, regime_train_full=regime_train_full,
                
                hp_space=HP_SPACE, n_iterations=N_ITERATIONS,
                global_params=GLOBAL_PARAMS, epsilon_greedy=0.4
            )
            
            start = time.time()
            searcher.search()
            duration = time.time() - start

            # 6. LOG RESULTS TO MLFLOW
            best_val_meta = searcher.best_meta_result
            clean_params = searcher.best_params.copy()

            for int_param in ["tot_units", "n_layers", "connectivity_recurrent"]:
                if int_param in clean_params:
                    clean_params[int_param] = int(round(clean_params[int_param]))

            true_test_results = searcher.evaluate_final_test_set(clean_params)
            
            # Log the best hyperparameters found by the searcher
            mlflow.log_params(clean_params)
            mlflow.log_metric("val_composite_score", best_val_meta['score'])
            
            # Log the final performance metrics
            mlflow.log_metric("accuracy_regime", true_test_results['acc_regime'])
            mlflow.log_metric("rmse_forecast", true_test_results['rmse_forecast'])
            mlflow.log_metric("rmse_stability", true_test_results['rmse_recon'])
            mlflow.log_metric("composite_score", true_test_results['score'])
            mlflow.log_metric("duration_sec", duration)

            del X_train_sub, y_train_sub, regime_train_sub
            del X_val, y_val, regime_val
            del X_train_full, y_train_full, regime_train_full
            del X_test, y_test, regime_test, searcher
            gc.collect()

            print(f"\n[MLFLOW LOGGED] {exp['name']} - Acc: {true_test_results['acc_regime']:.2%}, RMSE: {true_test_results['rmse_forecast']:.4f}")

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED. Run 'mlflow ui' in the terminal to view results.")
