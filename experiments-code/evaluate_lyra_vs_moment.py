#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to evaluate Lyra (DeepRC) vs MOMENT-1-small on the entire SMD Dataset.
Task: Regime Classification.
Readout: Ridge Classifier adapted exclusively on the train set.
"""

import os
import gc
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# --- Custom Modules (Lyra) ---
from psi_DeepRC_IP import DeepReservoir     
from utils import robust_torch_ridge_fit

# --- MOMENT Library ---
try:
    from momentfm import MOMENTPipeline
except ImportError:
    print("Warning: 'momentfm' not installed. MOMENT evaluation will fail. Run: pip install momentfm")

# =====================================================================
# CONFIGURATION
# =====================================================================
BASE_DIR = '../datasets/SMD/OmniAnomaly-master/ServerMachineDataset'
LABELS_DIR = os.path.join(BASE_DIR, 'regime_labels')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 96  # Standard window size, adjust if your experiments used 60 or 512
N_CLASSES = 4     # From regime labeling (0, 1, 2, 3)

# --- RevIN v5 Configuration ---
SOFT_FLOOR_C = 0.02
CLIP_VAL = 4.0

# --- Lyra (DeepRC) HPs ---
LYRA_PARAMS = {
    "tot_units": 500,               # Replace with your best search HP
    "n_layers": 3,                  # Replace with your best search HP
    "spectral_radius": 0.9,         # Replace with your best search HP
    "leaky": 0.5,                   # Replace with your best search HP
    "connectivity_recurrent": 10,   # Replace with your best search HP
    "input_scaling": 0.1,           # Replace with your best search HP
    "ridge_alpha": 10.0,            # Lyra's Ridge Alpha
}

# --- MOMENT HPs ---
MOMENT_ALPHA = 0.005233480488540088

# =====================================================================
# DATA LOADING & NORMALIZATION
# =====================================================================

def load_and_preprocess_split(split_name='train'):
    """
    Loads all data and labels for a given split (train/test),
    creates windows, and applies Dual_Stream_RC_RevIN_v5_SoftFloor.
    """
    print(f"\n--- Loading {split_name.upper()} Set ---")
    data_dir = os.path.join(BASE_DIR, split_name)
    machine_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    X_loc_all = []
    reg_labels_all = []
    
    for f_name in tqdm(machine_files, desc=f"Processing {split_name} machines"):
        # Load raw data
        data_path = os.path.join(data_dir, f_name)
        data = pd.read_csv(data_path, header=None).values.astype(np.float32)
        
        # Load pre-computed regime labels
        label_file = f"{split_name}_{f_name.replace('.txt', '.npy')}"
        label_path = os.path.join(LABELS_DIR, label_file)
        
        if not os.path.exists(label_path):
            continue
            
        labels = np.load(label_path)
        
        # Windowing
        n_windows = len(data) - WINDOW_SIZE + 1
        if n_windows <= 0:
            continue
            
        X_glob = np.stack([data[i:i + WINDOW_SIZE] for i in range(n_windows)])
        y_labels = labels[WINDOW_SIZE - 1 : WINDOW_SIZE - 1 + n_windows]
        
        # Dual_Stream_RC_RevIN_v5_SoftFloor Normalization
        mu = X_glob.mean(axis=1, keepdims=True)
        sd = X_glob.std(axis=1, keepdims=True)
        sd = np.sqrt(sd ** 2 + SOFT_FLOOR_C ** 2)
        
        X_loc = (X_glob - mu) / sd
        np.clip(X_loc, -CLIP_VAL, CLIP_VAL, out=X_loc)
        
        X_loc_all.append(X_loc)
        reg_labels_all.append(y_labels)

    return np.concatenate(X_loc_all, axis=0), np.concatenate(reg_labels_all, axis=0)

def add_bias(R):
    return np.hstack([R, np.ones((R.shape[0], 1), dtype=R.dtype)])

def one_hot_encode(labels, n_classes):
    return (labels[:, None] == np.arange(n_classes)[None, :]).astype(np.float32)

# =====================================================================
# LYRA (DEEPRC) PIPELINE
# =====================================================================

def build_lyra_model(input_dim):
    conn = int(round(LYRA_PARAMS["connectivity_recurrent"]))
    model = DeepReservoir(
        input_size=input_dim,
        tot_units=LYRA_PARAMS["tot_units"],
        n_layers=LYRA_PARAMS["n_layers"],
        concat=True,                       # FIXED from config
        input_scaling=LYRA_PARAMS["input_scaling"],
        inter_scaling=1.0,
        spectral_radius=LYRA_PARAMS["spectral_radius"],
        leaky=LYRA_PARAMS["leaky"],
        connectivity_recurrent=conn,
        connectivity_input=conn,
        connectivity_inter=conn,
        bias_bool=True,
        bias_scaling=0.1,                  # FIXED from config
        effective_rescaling=True,
        nonlin='tanh',
        lle_tuning=True,                   # FIXED from config
        n_taps=4,                          # FIXED from config
        verbose=False,
    ).to(DEVICE)
    return model

def extract_lyra_features(model, X_numpy, batch_size=2048):
    model.eval()
    LDeep_states = []
    with torch.no_grad():
        for i in range(0, len(X_numpy), batch_size):
            batch = torch.from_numpy(X_numpy[i:i + batch_size]).float().to(DEVICE)
            # states_last holds the readout features
            _, states_last = model(batch, return_all_layers=True)
            # Use deep states for regime classification as per previous scripts
            LDeep_states.append(states_last[-1].cpu().numpy())
    return np.concatenate(LDeep_states, axis=0)

def evaluate_lyra(X_train, y_train, X_test, y_test):
    print("\n--- Evaluating LYRA ---")
    input_dim = X_train.shape[2]
    model = build_lyra_model(input_dim)
    
    print("1. Extracting Features...")
    R_train = add_bias(extract_lyra_features(model, X_train))
    R_test = add_bias(extract_lyra_features(model, X_test))
    
    print("2. Adapting Ridge on Train Set...")
    y_train_oh = one_hot_encode(y_train, N_CLASSES)
    alpha = LYRA_PARAMS["ridge_alpha"]
    
    W_reg = robust_torch_ridge_fit(R_train, y_train_oh, alpha, DEVICE)
    
    print("3. Predicting on Test Set...")
    pred_scores = (torch.from_numpy(R_test).float().to(DEVICE) @ W_reg).cpu().numpy()
    pred_labels = np.argmax(pred_scores, axis=1)
    
    acc = accuracy_score(y_test, pred_labels)
    print(f"Lyra Overall Test Accuracy: {acc:.2%}")
    print("\nLyra Classification Report:")
    print(classification_report(y_test, pred_labels))
    
    del model, R_train, R_test
    torch.cuda.empty_cache()
    gc.collect()

# =====================================================================
# MOMENT PIPELINE
# =====================================================================

def extract_moment_features(model, X_numpy, batch_size=256):
    """
    Extracts embeddings from MOMENT-1-small.
    MOMENT usually expects shape (batch, channels, seq_len).
    """
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(X_numpy), batch_size), desc="MOMENT Inference"):
            # X_numpy shape: (batch, seq_len, channels)
            batch = torch.from_numpy(X_numpy[i:i + batch_size]).float().to(DEVICE)
            
            # Rearrange to (batch, channels, seq_len) for MOMENT
            batch = batch.permute(0, 2, 1)
            
            outputs = model(batch)
            # Get pooled embeddings. Depending on MOMENT API, it's outputs.embeddings
            emb = outputs.embeddings.mean(dim=2) # Average pooling over time if needed, or use specific token
            embeddings.append(emb.cpu().numpy())
            
    return np.concatenate(embeddings, axis=0)

def evaluate_moment(X_train, y_train, X_test, y_test):
    print("\n--- Evaluating MOMENT-1-small ---")
    
    try:
        # Load MOMENT Pipeline
        model = MOMENTPipeline.from_pretrained(
            "Autonlab/MOMENT-1-small", 
            model_kwargs={"is_training": False}
        ).to(DEVICE)
    except Exception as e:
        print(f"Failed to load MOMENT: {e}")
        return

    print("1. Extracting Features (This may take a while)...")
    R_train = add_bias(extract_moment_features(model, X_train))
    R_test = add_bias(extract_moment_features(model, X_test))
    
    print("2. Adapting Ridge on Train Set...")
    y_train_oh = one_hot_encode(y_train, N_CLASSES)
    alpha = MOMENT_ALPHA # Your specific search HP
    
    W_reg = robust_torch_ridge_fit(R_train, y_train_oh, alpha, DEVICE)
    
    print("3. Predicting on Test Set...")
    pred_scores = (torch.from_numpy(R_test).float().to(DEVICE) @ W_reg).cpu().numpy()
    pred_labels = np.argmax(pred_scores, axis=1)
    
    acc = accuracy_score(y_test, pred_labels)
    print(f"MOMENT Overall Test Accuracy: {acc:.2%}")
    print("\nMOMENT Classification Report:")
    print(classification_report(y_test, pred_labels))
    
    del model, R_train, R_test
    torch.cuda.empty_cache()
    gc.collect()

# =====================================================================
# MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    
    # 1. Load entire datasets
    X_train, y_train = load_and_preprocess_split('train')
    X_test, y_test = load_and_preprocess_split('test')
    
    print(f"\nGlobal Train Set Shape: {X_train.shape}")
    print(f"Global Test Set Shape:  {X_test.shape}")
    
    # 2. Evaluate Lyra
    evaluate_lyra(X_train, y_train, X_test, y_test)
    
    # 3. Evaluate MOMENT
    evaluate_moment(X_train, y_train, X_test, y_test)
    
    print("\n--- Benchmark Completed ---")