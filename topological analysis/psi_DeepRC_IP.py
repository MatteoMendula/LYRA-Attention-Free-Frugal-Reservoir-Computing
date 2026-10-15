"""
psi_DeepRC_IP_v2.py
===================
MERGE della DeepReservoir originale (psi_DeepRC.py) con Intrinsic Plasticity
e multi-tap TEMPORALE. Mantiene tutto cio' che della versione originale era
corretto e va preservato:

  - effective_rescaling: condizione ESP corretta per leaky-ESN
    (riscala W_eff = leaky*W + (1-leaky)*I, non W da solo)
  - loop di accettazione LLE (edge-of-chaos) con estimate_largest_lyapunov
  - connectivity_input / connectivity_inter separate, kernel input sparso
  - bias_scaling disaccoppiato da input_scaling
  - nonlin in {'tanh', 'identity'}
  - offload CPU degli stati per-timestep (memoria GPU sotto controllo)
  - semantica concat per la divisione di tot_units tra i layer

Cosa cambia:

  RIMOSSO  ResESN (alpha/beta + matrice ortogonale): il layer aggiorna
           sempre con leaky integration standard.

  AGGIUNTO Intrinsic Plasticity (Schrauwen et al. 2008 / Steil 2007):
           ogni neurone ha gain a e bias b nella tanh: y = tanh(a*z + b),
           adattati con regola locale unsupervised verso N(mu_t, sigma_t^2).
           Pre-fase su un sotto-campione, poi tutto torna congelato:
           NESSUN BPTT, il training resta la sola Ridge.
           IP e' attiva solo con nonlin='tanh' (la regola e' derivata per
           la tanh); con 'identity' a=1, b=0 restano fissi.

  AGGIUNTO multi-tap TEMPORALE: states_last[l] = concatenazione degli stati
           del layer l a n_taps istanti equispaziati nella finestra (incluso
           l'ultimo). Con n_taps=1 si recupera ESATTAMENTE il comportamento
           originale (solo h(T) per layer).

  FIX      ramo distrib=='normal': `my_utils.spectral_radius * W` era un
           AttributeError latente -> `spectral_radius * W`.

  NOTA     LLE e IP: il loop LLE accetta il reservoir PRIMA dell'adattamento
           IP. IP riduce i gain (sistema piu' contrattivo), quindi
           l'accettazione pre-IP resta un bound conservativo; per verifica
           post-IP usare model.lle_post_ip() se utils lo consente.

INTERFACCIA (invariata rispetto al main):
    _, states_last = model(batch, return_all_layers=True)
    model.ip_pretrain(X_numpy, target_std=0.2, eta=5e-4, epochs=2,
                      batch_size=512, device="cuda")
"""

import numpy as np
import torch
import torch.nn as nn

import utils as my_utils

torch.set_float32_matmul_precision('high')

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# =====================================================================
# CELL
# =====================================================================
class IPReservoirCell(nn.Module):
    """ReservoirCell originale, senza ResESN, con gain/bias IP per-neurone."""

    def __init__(self, input_size, units, input_scaling=1., spectral_radius=0.99,
                 leaky=1, connectivity_input=10, connectivity_recurrent=10,
                 bias_bool=True, distrib='uniform', nonlin='tanh',
                 effective_rescaling=True, bias_scaling=None):
        super().__init__()
        self.input_size = input_size
        self.units = units
        self.input_scaling = input_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky

        self.kernel = my_utils.sparse_tensor_init(
            input_size, self.units, connectivity_input) * self.input_scaling
        self.kernel = nn.Parameter(self.kernel, requires_grad=False)

        W = my_utils.sparse_recurrent_tensor_init(
            self.units, C=connectivity_recurrent, distrib=distrib)
        if effective_rescaling and self.leaky != 1:
            I = my_utils.sparse_eye_init(self.units)
            W_eff = W * self.leaky + (I * (1 - self.leaky))
            W_eff_scaled = my_utils.spectral_norm_scaling(W_eff, spectral_radius)
            self.recurrent_kernel = (W_eff_scaled + I * (self.leaky - 1)) * (1 / self.leaky)
        else:
            if distrib == 'normal':
                # FIX: era `my_utils.spectral_radius * W` (AttributeError)
                W = spectral_radius * W
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

        if nonlin not in ['tanh', 'identity']:
            raise ValueError(f"Invalid nonlinearity '{nonlin}'. Only 'tanh' and 'identity' allowed.")
        self.nonlin_name = nonlin
        self.nonlin = torch.tanh if nonlin == 'tanh' else (lambda x: x)

        # ---- Intrinsic Plasticity: gain/bias per-neurone (buffer, no grad) ----
        self.ip_active = (nonlin == 'tanh')
        self.register_buffer("a_ip", torch.ones(self.units))
        self.register_buffer("b_ip", torch.zeros(self.units))

    # ------------------------------------------------------------------
    def _pre_activation(self, xt, h_prev):
        return torch.mm(xt, self.kernel) + self.bias + torch.mm(h_prev, self.recurrent_kernel)

    def forward(self, xt, h_prev):
        pre = self._pre_activation(xt, h_prev)
        if self.ip_active:
            output = torch.tanh(self.a_ip * pre + self.b_ip)
        else:
            output = self.nonlin(pre)
        ht = h_prev * (1 - self.leaky) + output * self.leaky
        return ht, ht

    # ------------------------------------------------------------------
    @torch.no_grad()
    def ip_update(self, pre, eta, mu_t, var_t):
        """Un passo della regola IP (Schrauwen et al. 2008) mediato sul batch.
        Ritorna l'output y calcolato con i parametri PRE-update (regola online
        standard), che il chiamante usa per far avanzare lo stato."""
        y = torch.tanh(self.a_ip * pre + self.b_ip)
        db = -eta * (-(mu_t / var_t)
                     + (y / var_t) * (2.0 * var_t + 1.0 - y * y + mu_t * y))
        da = eta / self.a_ip + db * pre
        self.b_ip += db.mean(dim=0)
        self.a_ip += da.mean(dim=0)
        self.a_ip.clamp_(min=0.05, max=10.0)
        return y


# =====================================================================
# LAYER
# =====================================================================
class ReservoirLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        is_euler = kwargs.pop('Euler', False)
        if is_euler:
            raise NotImplementedError("EulerReservoirCell is not implemented in this version.")
        self.cell = IPReservoirCell(*args, **kwargs)

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.cell.units, device=device)

    def forward(self, x, h_prev=None, offload_cpu=False):
        """Forward standard. offload_cpu=True replica il comportamento
        originale (stati accumulati in RAM di sistema per liberare la GPU,
        necessario con batch 2048 x T=60 x 1000 unita').
        offload_cpu=False tiene tutto su device (usato dalla fase IP, che
        lavora su batch piccoli)."""
        device = next(self.parameters()).device
        x = x.to(device)
        B, T, _ = x.shape
        if h_prev is None:
            h_prev = self.init_hidden(x.shape[0], x.device)
        hs = torch.empty((B, T, self.cell.units), device=device)

        for t in range(T):
            xt = x[:, t]
            _, h_prev = self.cell(xt, h_prev)
            hs[:, t, :] = h_prev
        return hs, h_prev

    @torch.no_grad()
    def ip_pass(self, x, eta, mu_t, var_t):
        """Una passata IP online su un batch di sequenze (B, T, F) su device."""
        device = next(self.parameters()).device
        x = x.to(device)
        h = self.init_hidden(x.shape[0], device)
        for t in range(x.shape[1]):
            pre = self.cell._pre_activation(x[:, t], h)
            y = self.cell.ip_update(pre, eta, mu_t, var_t)
            h = h * (1 - self.cell.leaky) + y * self.cell.leaky


# =====================================================================
# DEEP RESERVOIR
# =====================================================================
class DeepReservoir(nn.Module):
    def __init__(self, input_size=1, tot_units=100, n_layers=1, concat=False,
                 input_scaling=1, inter_scaling=1, spectral_radius=0.99, leaky=1,
                 connectivity_recurrent=10, connectivity_input=10,
                 connectivity_inter=10, seed=None,
                 target_lle_range=(-0.1, 0.05),
                 max_trials=10,
                 lle_tuning=True,
                 n_taps=1,           # n_taps=1 == comportamento originale
                 verbose=True,
                 **kwargs):
        super().__init__()
        self.n_layers = n_layers
        self.concat = concat
        self.n_taps = int(n_taps)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # ResESN rimosso: alpha/beta eventualmente passati vengono ignorati
        kwargs.pop('alpha', None)
        kwargs.pop('beta', None)

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

            capped_conn_in = min(conn_in, current_input_size, units)
            capped_conn_rec = min(connectivity_recurrent, units)

            if not lle_tuning:
                res_layer = ReservoirLayer(
                    input_size=current_input_size, units=units,
                    input_scaling=scaling, spectral_radius=spectral_radius,
                    leaky=leaky, connectivity_input=capped_conn_in,
                    connectivity_recurrent=capped_conn_rec, **kwargs)
            else:
                # Loop di accettazione LLE originale (edge-of-chaos).
                # NB: misura PRE-IP; IP rende il sistema piu' contrattivo,
                # quindi l'accettazione resta un bound conservativo.
                trials = 0
                is_lle_acceptable = False
                current_sr = spectral_radius
                res_layer = None
                while not is_lle_acceptable and trials < max_trials:
                    res_layer = ReservoirLayer(
                        input_size=current_input_size, units=units,
                        input_scaling=scaling, spectral_radius=current_sr,
                        leaky=leaky, connectivity_input=capped_conn_in,
                        connectivity_recurrent=capped_conn_rec, **kwargs)
                    lle = my_utils.estimate_largest_lyapunov(res_layer)
                    if target_lle_range[0] <= lle <= target_lle_range[1]:
                        is_lle_acceptable = True
                        if verbose:
                            print(f"Layer {i+1}: Accepted LLE={lle:.4f} after {trials+1} trials.")
                    else:
                        current_sr *= 0.9 if lle > target_lle_range[1] else 1.1
                        trials += 1
                        if verbose:
                            print(f"Layer {i+1}: Rejected LLE={lle:.4f}. Retrying... ({trials}/{max_trials})")

            reservoir_layers.append(res_layer)
            current_input_size = units
        self.reservoir = nn.ModuleList(reservoir_layers)

    # ------------------------------------------------------------------
    def _tap_indices(self, T):
        n = max(1, min(self.n_taps, T))
        if n == 1:
            return np.array([T - 1])  # solo l'ultimo stato == comportamento originale
        return np.unique(np.round(np.linspace(0, T - 1, n)).astype(int))

    # ------------------------------------------------------------------
    def forward(self, X, initial_states=None, return_all_layers=True):
        """Identico all'originale, ma states_last[l] e' il multi-tap
        temporale del layer l: (B, units * n_tap). Con n_taps=1 coincide
        con il solo h(T) dell'implementazione originale."""
        all_layer_states = []
        states_last = []
        current_input = X
        tap_idx = None

        for i, res_layer in enumerate(self.reservoir):
            h_prev = initial_states[i] if initial_states else None
            current_input, _ = res_layer(current_input, h_prev=h_prev)
            if tap_idx is None:
                tap_idx = self._tap_indices(current_input.shape[1])
            all_layer_states.append(current_input)
            B = current_input.shape[0]
            states_last.append(
                current_input[:, tap_idx, :].reshape(B, -1).contiguous())

        if return_all_layers:
            return all_layer_states, states_last

        if self.concat:
            states = torch.cat(all_layer_states, dim=2)
        else:
            states = all_layer_states[-1]
        return states, states_last

    # ------------------------------------------------------------------
    @torch.no_grad()
    def ip_pretrain(self, X, target_mean=0.0, target_std=0.2,
                    eta=5e-4, epochs=2, batch_size=512, device="cpu"):
        """Adattamento IP layer-per-layer dal basso verso l'alto.

        X: ndarray (N, T, F) nello spazio locale RC-RevIN.
        Il layer l viene adattato usando come input gli stati dei layer
        inferiori GIA' adattati e congelati (calcolati on-device, senza
        offload: i batch IP sono piccoli). Nessun gradiente.
        I layer con nonlin='identity' vengono saltati (IP e' definita
        per la tanh)."""
        var_t = float(target_std) ** 2
        mu_t = float(target_mean)
        N = len(X)
        for l, layer in enumerate(self.reservoir):
            if not layer.cell.ip_active:
                continue
            for _ in range(int(epochs)):
                perm = torch.randperm(N)
                for s in range(0, N, batch_size):
                    sel = perm[s:s + batch_size].numpy()
                    seq = torch.from_numpy(X[sel]).float().to(device)
                    for j in range(l):
                        seq, _ = self.reservoir[j](seq, offload_cpu=False)
                    layer.ip_pass(seq, eta=eta, mu_t=mu_t, var_t=var_t)
        return self

    # ------------------------------------------------------------------
    def lle_post_ip(self):
        """Hook per la verifica edge-of-chaos POST-adattamento IP.
        Funziona se estimate_largest_lyapunov usa il forward della cell
        (che ora include a_ip/b_ip); utile per l'appendice del paper."""
        return [float(my_utils.estimate_largest_lyapunov(layer))
                for layer in self.reservoir]