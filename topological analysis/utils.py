import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

def robust_torch_ridge_fit(X_np, y_np, alpha, device):
    """
    Solves Ridge Regression using float64 to prevent tiny alpha penalties 
    from being truncated by massive diagonal values in float32 space.
    """
    # Upgrade to float64 for mathematical stability
    X_t = torch.from_numpy(X_np).to(torch.float64).to(device)
    y_t = torch.from_numpy(y_np).to(torch.float64).to(device)
    
    XTX = X_t.T @ X_t
    XTY = X_t.T @ y_t
    I = torch.eye(X_t.shape[1], device=device, dtype=torch.float64)
    
    try:
        # Standard Cholesky/LU solve
        W = torch.linalg.solve(XTX + alpha * I, XTY)
    except RuntimeError:
        # Bulletproof fallback to Pseudo-Inverse if still singular
        W = torch.linalg.lstsq(XTX + alpha * I, XTY).solution
        
    # Cast back to float32 to save RAM for the rest of the pipeline
    return W.to(torch.float32)

def load_machine_data(filepath):
    return pd.read_csv(filepath, header=None).values

def create_sequences(data, window, horizon, stride=5): # Add stride
    X, y = [], []
    for i in range(0, len(data) - window - horizon + 1, stride):
        X.append(data[i : i+window])
        y.append(data[i+window : i+window+horizon])
    return np.array(X), np.array(y)

def torch_ridge_fit(X, y, alpha, device):
    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).float().to(device)
    
    # If y is 1D, make it 2D (N, 1)
    if len(y_t.shape) == 1:
        y_t = y_t.unsqueeze(1)
        
    d = X_t.shape[1]
    xtx = X_t.T @ X_t
    xty = X_t.T @ y_t
    identity = torch.eye(d, device=device)
    
    # Using solve is numerically more stable than inv()
    W = torch.linalg.solve(xtx + alpha * identity, xty)

    del X_t, y_t, xtx, xty, identity
    
    return W

# --- 0. Helper Functions (Unchanged) ---

def calculate_nrmse(y_pred, y_true):
    """Calculates the Normalized Root Mean Squared Error (NRMSE)."""
    mse = np.mean((y_pred - y_true)**2)
    rmse = np.sqrt(mse)
    std_dev_true = np.std(y_true)
    if std_dev_true == 0:
        return np.inf
    nrmse = rmse / std_dev_true
    return nrmse

def plot_eigenvalues(eigvals, title, sr_setting, sr_name="sr"):
    """Plots eigenvalues on the complex plane with the unit circle."""
    plt.figure(figsize=(7, 7))
    unit_circle = plt.Circle((0., 0.), 1., color='black', fill=False, linestyle='--', label="Unit Circle (Stability)")
    plt.gca().add_artist(unit_circle)
    sr_circle = plt.Circle((0., 0.), sr_setting, color='red', fill=False, linestyle=':', label=f"Target {sr_name} ({sr_setting:.3f})")
    plt.gca().add_artist(sr_circle)
    plt.scatter(eigvals.real, eigvals.imag, alpha=0.7, label=f"Eigenvalues (Max: {np.max(np.abs(eigvals)):.3f})")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.xlim([-1.8, 1.8])
    plt.ylim([-1.8, 1.8])
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.title(f"Jacobian Eigenvalues at Origin - {title}")
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.savefig(f"{title}_eigenvalues.png")
    plt.close()
    print(f"Generated eigenvalue plot: {title}_eigenvalues.png")

# --- 1. Custom DeepReservoir Implementation (Provided) ---

def sparse_eye_init(M: int) -> torch.FloatTensor:
    return torch.eye(M, dtype=torch.float)

def sparse_tensor_init(M: int, N: int, C: int = 1) -> torch.FloatTensor:
    dense_shape = torch.Size([M, N])
    indices = torch.zeros((M * C, 2), dtype=torch.long)
    k = 0
    for i in range(M):
        idx = np.random.choice(N, size=C, replace=False)
        for j in range(C):
            indices[k, 0] = i
            indices[k, 1] = idx[j]
            k += 1
    values = 2 * torch.rand(M * C) - 1
    return torch.sparse_coo_tensor(indices.T, values, dense_shape).to_dense().float()

def sparse_recurrent_tensor_init(M: int, C: int = 1, distrib: str = 'uniform') -> torch.FloatTensor:
    assert M >= C, "Number of units must be greater than or equal to connectivity."
    dense_shape = torch.Size([M, M])
    indices = torch.zeros((M * C, 2), dtype=torch.long)
    k = 0
    for i in range(M):
        idx = np.random.choice(M, size=C, replace=False)
        for j in range(C):
            indices[k, 0] = idx[j]
            indices[k, 1] = i
            k += 1
    if distrib == 'uniform':
        values = 2 * torch.rand(M * C) - 1
    elif distrib == 'normal':
        values = torch.randn(M * C) / np.sqrt(C)
    else:
        raise ValueError(f"Invalid distribution '{distrib}'. Only 'uniform' and 'normal' are allowed.")
    return torch.sparse_coo_tensor(indices.T, values, dense_shape).to_dense().float()

def spectral_norm_scaling(W: torch.FloatTensor, rho_desired: float) -> torch.FloatTensor:
    e, _ = np.linalg.eig(W.cpu().numpy())
    rho_current = max(abs(e))
    return W * (rho_desired / rho_current)

def fast_spectral_rescaling(W: torch.FloatTensor, rho_desired: float) -> torch.FloatTensor:
    units = W.shape[0]
    value = (rho_desired / np.sqrt(units)) * (6 / np.sqrt(12))
    return value * W

def skewsymmetric(units: int, recur_scaling: float) -> torch.FloatTensor:
    W = recur_scaling * (2 * torch.rand(units, units) - 1)
    return W - W.T

def calculate_mle(reservoir_cell, inputs, steps=1000, warmup=200):
    """
    Calculates the Maximal Lyapunov Exponent (MLLE) for a given reservoir cell.

    Args:
        reservoir_cell (ReservoirCell): The PyTorch module for the reservoir.
        inputs (np.ndarray): The input data to drive the reservoir.
        steps (int): The number of steps to average the exponent over.
        warmup (int): The number of initial steps to discard.

    Returns:
        float: The estimated Maximal Lyapunov Exponent.
    """
    # Use the device the model is on (e.g., 'cuda' or 'cpu')
    device = reservoir_cell.recurrent_kernel.device
    
    # Extract model parameters and move to the correct device
    W_h = reservoir_cell.recurrent_kernel
    W_in = reservoir_cell.kernel
    bias = reservoir_cell.bias
    lr = reservoir_cell.leaky
    n_units = reservoir_cell.units
    
    # Prepare inputs
    inputs_t = torch.from_numpy(inputs).float().to(device)
    
    # Initialize state and tangent vector
    h = torch.zeros(1, n_units, device=device)
    v = torch.randn(n_units, 1, device=device)
    v = v / torch.linalg.norm(v)
    
    log_norms = []
    
    total_steps = min(len(inputs_t), warmup + steps)

    # Run the reservoir step-by-step
    for t in range(total_steps):
        u = inputs_t[t].unsqueeze(0)
        
        # --- Main state update ---
        pre_activation = h @ W_h.T + u @ W_in + bias
        
        # --- Tangent vector evolution ---
        if t >= warmup:
            # Derivative of tanh is (1 - tanh^2)
            # This creates the diagonal matrix for the time-varying Jacobian
            d_tanh = 1.0 - torch.tanh(pre_activation)**2
            
            # The full time-varying Jacobian J(t) is composed of two parts
            # J(t) = (1-lr)*I + lr * diag(d_tanh) @ W_h
            # We apply it to v without explicitly forming the large matrix J(t)
            
            # Part 1: Effect of the recurrent weights
            v_recurrent = W_h.T @ (d_tanh.T * v)
            
            # Part 2: Combine with leak rate and identity term
            v_next = (1 - lr) * v + lr * v_recurrent
            
            # Calculate norm and re-normalize
            norm = torch.linalg.norm(v_next)
            if norm > 1e-9: # Avoid division by zero
                v = v_next / norm
                log_norms.append(torch.log(norm).item())

        # Update the state for the next step
        h = (1 - lr) * h + lr * torch.tanh(pre_activation)
        
    # The MLLE is the average of the logarithms of the expansion rates
    if not log_norms:
        return -np.inf # Return a very negative number if no steps were measured
        
    return np.mean(log_norms)

def estimate_largest_lyapunov(reservoir_layer, time_steps=1000, warmup=100, epsilon=1e-6):
    """
    Estimates the Largest Lyapunov Exponent (LLE) without computing the Jacobian.
    
    Args:
        reservoir_layer: An instance of your ReservoirLayer (or DeepReservoir)
        time_steps: How long to run the simulation
        warmup: Steps to discard to let transients die out
        epsilon: Magnitude of the initial perturbation
    """
    device = next(reservoir_layer.parameters()).device
    units = reservoir_layer.cell.units
    
    # 1. Create a dummy input (e.g., zero input or random noise)
    # The LLE is an intrinsic property, but often depends on the input drive.
    # We'll use random input to probe the driven dynamics.
    input_dim = reservoir_layer.cell.kernel.shape[0] # infer input size
    inputs = torch.randn(1, time_steps, input_dim, device=device)
    
    # 2. Run Reference Trajectory (h_ref)
    # We run step-by-step to control the state manually
    h_ref = reservoir_layer.init_hidden(1, device)
    
    # Warmup (get onto the attractor)
    with torch.no_grad():
        for t in range(warmup):
            _, h_ref = reservoir_layer.cell(inputs[:, t], h_ref)
            
        # 3. Create Perturbed Trajectory (h_pert)
        # Add tiny noise to the state
        perturbation = torch.randn_like(h_ref)
        perturbation = perturbation / torch.norm(perturbation) * epsilon
        h_pert = h_ref + perturbation
        
        lyapunov_sums = []
        
        # 4. Iterate and Measure Divergence
        for t in range(warmup, time_steps):
            xt = inputs[:, t]
            
            # Update both trajectories
            _, h_ref = reservoir_layer.cell(xt, h_ref)
            _, h_pert = reservoir_layer.cell(xt, h_pert)
            
            # Calculate distance
            diff = h_pert - h_ref
            dist = torch.norm(diff)
            
            # Log the expansion rate
            # If dist grows > epsilon, log > 0 (Chaos)
            # If dist shrinks < epsilon, log < 0 (Stability)
            expansion_rate = torch.log(dist / epsilon)
            lyapunov_sums.append(expansion_rate.item())
            
            # 5. Rescale (Renormalize) the perturbed state
            # This is CRITICAL: We must keep the perturbation small (linear regime)
            # We pull h_pert back towards h_ref so distance is exactly epsilon again
            h_pert = h_ref + (diff / dist) * epsilon

    # The LLE is the average of these instantaneous expansion rates
    lle = np.mean(lyapunov_sums)
    return lle