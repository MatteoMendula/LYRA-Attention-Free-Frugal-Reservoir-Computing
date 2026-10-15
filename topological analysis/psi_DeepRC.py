import numpy as np
import torch
import torch.nn as nn

import utils as my_utils

class ReservoirCell(nn.Module):
    def __init__(self, input_size, units, input_scaling=1., spectral_radius=0.99,
                 leaky=1, connectivity_input=10, connectivity_recurrent=10,
                 bias_bool=True, alpha=0, beta=1, distrib='uniform', nonlin='tanh',
                 effective_rescaling=True, bias_scaling=None):
        super().__init__()
        self.input_size = input_size
        self.units = units
        self.input_scaling = input_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky
        self.kernel = my_utils.sparse_tensor_init(input_size, self.units,
                                         connectivity_input) * self.input_scaling
        self.kernel = nn.Parameter(self.kernel, requires_grad=False)
        W = my_utils.sparse_recurrent_tensor_init(self.units, C=connectivity_recurrent, distrib=distrib)
        if effective_rescaling and self.leaky != 1:
            I = my_utils.sparse_eye_init(self.units)
            W_eff = W * self.leaky + (I * (1 - self.leaky))
            W_eff_scaled = my_utils.spectral_norm_scaling(W_eff, spectral_radius)
            self.recurrent_kernel = (W_eff_scaled + I * (self.leaky - 1)) * (1 / self.leaky)
        else:
            if distrib == 'normal':
                W = my_utils.spectral_radius * W
            elif distrib == 'uniform' and connectivity_recurrent == units:
                W = my_utils.fast_spectral_rescaling(W, spectral_radius)
            else:
                W = my_utils.spectral_norm_scaling(W, spectral_radius)
            self.recurrent_kernel = W
        self.recurrent_kernel = nn.Parameter(self.recurrent_kernel, requires_grad=False)
        if bias_bool:
            bias_val = bias_scaling if bias_scaling is not None else self.input_scaling
            self.bias = (2 * torch.rand(self.units) - 1) * bias_val
        else:
            self.bias = torch.zeros(self.units)
        self.bias = nn.Parameter(self.bias, requires_grad=False)
        self.alpha = torch.tensor(alpha)
        if self.alpha > 0:
            Q, _ = torch.linalg.qr(2 * torch.rand(units, units) - 1)
            self.ortho = nn.Parameter(Q, requires_grad=False)
            self.beta = nn.Parameter(torch.tensor(beta), requires_grad=False)
        self.alpha = nn.Parameter(self.alpha, requires_grad=False)
        if nonlin not in ['tanh', 'identity']:
            raise ValueError(f"Invalid nonlinearity '{nonlin}'. Only 'tanh' and 'identity' allowed.")
        self.nonlin = torch.tanh if nonlin == 'tanh' else (lambda x: x)

    def forward(self, xt, h_prev):
        input_part = torch.mm(xt, self.kernel)
        state_part = torch.mm(h_prev, self.recurrent_kernel)
        pre_activation = input_part + self.bias + state_part
        output = self.nonlin(pre_activation)
        if self.alpha > 0:
            h_prev_shuffled = torch.mm(h_prev, self.ortho)
            ht = h_prev_shuffled * self.alpha + output * self.beta
        else:
            ht = h_prev * (1 - self.leaky) + output * self.leaky
        return ht, ht

class ReservoirLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        is_euler = kwargs.pop('Euler', False)
        if is_euler:
            raise NotImplementedError("EulerReservoirCell is not implemented in this version.")
        else:
            self.cell = ReservoirCell(*args, **kwargs)

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.cell.units, device=device)

    def forward(self, x, h_prev=None):
        device = next(self.parameters()).device
        x = x.to(device)
        if h_prev is None:
            h_prev = self.init_hidden(x.shape[0], x.device)
        hs = []
        for t in range(x.shape[1]):
            xt = x[:, t]
            _, h_prev = self.cell(xt, h_prev)
            # MOVE TO CPU: This is the critical line. 
            # It frees up GPU memory immediately after the calculation.
            hs.append(h_prev.detach().cpu())

        # This stacking now happens in System RAM (CPU-side)
        hs = torch.stack(hs, dim=1)
        return hs, h_prev

class DeepReservoir(nn.Module):
    def __init__(self, input_size=1, tot_units=100, n_layers=1, concat=False,
                 input_scaling=1, inter_scaling=1, spectral_radius=0.99, leaky=1,
                 connectivity_recurrent=10, connectivity_input=10,
                 connectivity_inter=10, seed=None, 
                 target_lle_range=(-0.1, 0.05), # Target: slightly negative to near-zero
                 max_trials=10,                  # Don't try forever
                 **kwargs):
        super().__init__()
        self.n_layers = n_layers
        self.concat = concat
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        if concat:
            self.layers_units = int(tot_units / n_layers)
            remainder = tot_units % n_layers
        else:
            self.layers_units = tot_units
            remainder = 0
        reservoir_layers = []
        current_input_size = input_size
        for i in range(n_layers):
            units = self.layers_units + (remainder if i == 0 else 0)
            scaling = input_scaling if i == 0 else inter_scaling
            conn_in = connectivity_input if i == 0 else connectivity_inter

            trials = 0
            is_lle_acceptable = False
            current_sr = spectral_radius
            while not is_lle_acceptable and trials < max_trials:
                res_layer = ReservoirLayer(
                    input_size=current_input_size,
                    units=units,
                    input_scaling=scaling,
                    spectral_radius=current_sr,
                    leaky=leaky,
                    connectivity_input=conn_in,
                    connectivity_recurrent=connectivity_recurrent,
                    **kwargs
                )
                lle = my_utils.estimate_largest_lyapunov(res_layer)
                if target_lle_range[0] <= lle <= target_lle_range[1]:
                    is_lle_acceptable = True
                    print(f"Layer {i+1}: Accepted LLE={lle:.4f} after {trials+1} trials.")
                else:
                    # REACT: Adjust the spectral radius for the next attempt
                    if lle > target_lle_range[1]:
                        adjustment_factor = 0.9  # Decrease SR to reduce chaos
                        print(f"Layer {i+1}: LLE={lle:.4f} too high, decreasing SR.")
                    else:
                        adjustment_factor = 1.1  # Increase SR to enhance dynamics
                        print(f"Layer {i+1}: LLE={lle:.4f} too low, increasing SR.")

                    current_sr *= adjustment_factor
                    trials += 1
                    print(f"Layer {i+1}: Rejected LLE={lle:.4f}. Adjustment factor={adjustment_factor}. Retrying... ({trials}/{max_trials})")

            if not is_lle_acceptable:
                print(f"Layer {i+1}: Max trials reached. Using last LLE={lle:.4f}.")

            reservoir_layers.append(res_layer)
            current_input_size = units
        self.reservoir = nn.ModuleList(reservoir_layers)

    def forward(self, X, initial_states=None):
        all_layer_states = []
        states_last = []
        current_input = X
        for i, res_layer in enumerate(self.reservoir):
            h_prev = initial_states[i] if initial_states else None
            current_input, h_last = res_layer(current_input, h_prev=h_prev)
            all_layer_states.append(current_input)
            states_last.append(h_last)
        if self.concat:
            states = torch.cat(all_layer_states, dim=2)
        else:
            states = all_layer_states[-1]
        return states, states_last
    
# ==================================================
# FOUNDATIONAL EXPERIMENT COMPLETED (517.6s)
# ------------------------------
# ZERO-SHOT TRANSFER METRICS:
#   Combined Score:      0.6606
#   1. Regime Accuracy:  70.73%  <-- (The Universal Feature)
#   2. Forecasting RMSE: 0.1934
#   3. Stability RMSE:   0.1095
# ==================================================

# PHASE 3: FEW-SHOT CALIBRATION (1000 Samples)
# Layer 1: Accepted LLE=-0.0400 after 1 trials.

# RESULTS AFTER CALIBRATION:
#   Forecasting RMSE: 56.5255 (Zero-Shot was 0.1934)
#   Regime Accuracy:  43.03% (Zero-Shot was 70.73%)
# ==================================================