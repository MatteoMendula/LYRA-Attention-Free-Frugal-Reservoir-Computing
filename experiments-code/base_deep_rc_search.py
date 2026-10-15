import numpy as np
import random
from collections import deque
import copy
import traceback
import torch

class BaseDeepRCSearch:
    def __init__(self, hp_space, n_iterations, global_params, epsilon_greedy=0.3):
        """
        Base class for Epsilon-Greedy Hyperparameter Search using psi_deepRC.
        
        Args:
            hp_space (dict): Dictionary defining the search space. 
                             Keys should match DeepReservoir args (tot_units, inter_scaling, etc.)
            n_iterations (int): Total search iterations.
            global_params (dict): Fixed parameters (e.g., n_layers, input_dim, device).
            epsilon_greedy (float): Probability of exploring random params vs exploiting memory.
        """
        self.hp_space = hp_space
        self.n_iterations = n_iterations
        self.global_params = global_params
        self.epsilon_greedy = epsilon_greedy

        # Memory for exploitation
        self.topk_memory = deque(maxlen=5)
        self.history = deque(maxlen=n_iterations)

        self.best_score = -np.inf
        self.best_params = None
        self.best_model_state_dict = None # Store PyTorch state_dict
        self.best_readout_model = None    # Store the readout (e.g. SVM, MLP, Ridge)
        self.best_output_dim = None

    def _log_uniform_sample(self, param_range):
        """Sample from log-uniform distribution defined by {'min': log_val, 'max': log_val}."""
        return np.exp(np.random.uniform(param_range["min"], param_range["max"]))

    def _perturb_log_param(self, base_value, param_range, noise_factor=0.1):
        """Perturb a value in log-space."""
        log_base = np.log(base_value + 1e-9)
        log_min = param_range["min"]
        log_max = param_range["max"]
        log_noise = np.random.normal(0, noise_factor * (log_max - log_min))
        perturbed_log = np.clip(log_base + log_noise, log_min, log_max)
        return np.exp(perturbed_log)

    def _sample_params(self):
        """Generates a single dictionary of parameters for DeepReservoir."""
        params = {}
        space = self.hp_space
        
        # 1. Handle Units (tot_units)
        if "tot_units" in space:
            if isinstance(space["tot_units"], list):
                # Bias towards smaller networks for efficiency during random search
                available = space["tot_units"]
                weights = [1/(u_val+1e-9) for u_val in available]
                weights = np.array(weights) / np.sum(weights)
                params["tot_units"] = np.random.choice(available, p=weights)
            else:
                params["tot_units"] = space["tot_units"] # Fixed value

        # 2. Handle Cell Type (Standard vs Euler)
        # DeepReservoir expects `Euler=True/False` in kwargs
        if "cell_type" in space:
            ctype = random.choice(space["cell_type"])
            if ctype.lower() == "euler":
                params["Euler"] = True
            else:
                params["Euler"] = False
        
        # 3. Handle Other Hyperparameters
        for key, value_range in space.items():
            if key in ["tot_units", "cell_type"]: continue # Already handled
            
            # Log-Uniform
            if isinstance(value_range, dict) and "min" in value_range and "max" in value_range:
                params[key] = self._log_uniform_sample(value_range)
            
            # Uniform Continuous (List of 2 floats)
            elif isinstance(value_range, list) and len(value_range) == 2 and all(isinstance(i, (int, float)) for i in value_range) and not isinstance(value_range[0], bool):
                 params[key] = random.uniform(value_range[0], value_range[1])
            
            # Categorical (List of options)
            elif isinstance(value_range, list):
                params[key] = random.choice(value_range)

        return params

    def _perturb_params(self, base_params):
        """Perturbs an existing set of parameters (Exploitation step)."""
        perturbed = copy.deepcopy(base_params)
        space = self.hp_space

        # Possibility to switch cell type
        if "cell_type" in space and random.random() < 0.1:
            ctype = random.choice(space["cell_type"])
            perturbed["Euler"] = (ctype.lower() == "euler")

        for key, base_val in base_params.items():
            if key not in space or key == "cell_type": continue
            
            # 30% chance to perturb any specific parameter
            if random.random() > 0.3: continue

            val_range = space[key]
            
            # Perturb tot_units (categorical-like)
            if key == "tot_units" and isinstance(val_range, list):
                perturbed[key] = random.choice(val_range)
            
            # Perturb Log-Uniform
            elif isinstance(val_range, dict) and "min" in val_range:
                perturbed[key] = self._perturb_log_param(base_val, val_range)
            
            # Perturb Uniform Continuous
            elif isinstance(val_range, list) and len(val_range) == 2 and isinstance(val_range[0], (int, float)):
                noise = np.random.normal(0, 0.1 * (val_range[1] - val_range[0]))
                perturbed[key] = np.clip(base_val + noise, val_range[0], val_range[1])
            
            # Perturb Categorical
            elif isinstance(val_range, list):
                perturbed[key] = random.choice(val_range)
        
        return perturbed

    def get_next_params(self):
        """Decides whether to explore or exploit and returns params."""
        exploit = len(self.topk_memory) > 0 and random.random() > self.epsilon_greedy
        
        if exploit:
            # Pick a random high-performing config from memory and perturb it
            base = random.choice(list(self.topk_memory))
            print("  [Exploiting memory...]")
            return self._perturb_params(base)
        else:
            # Explore new random params
            print("  [Exploring new params...]")
            return self._sample_params()

    def search(self):
        print(f"Starting DeepRC Epsilon-Greedy Search ({self.n_iterations} iterations)...")
        
        for i in range(self.n_iterations):
            print(f"\n--- Iteration {i+1}/{self.n_iterations} ---")
            current_params = self.get_next_params()
            
            # Merge with global fixed params (e.g. input_dim, n_layers)
            # This ensures DeepReservoir gets everything it needs
            full_params = {**self.global_params, **current_params}

            try:
                # --- EVALUATE STEP ---
                # Subclass must implement evaluate(full_params)
                # It should return: score, trained_readout, deep_rc_model_instance, output_dim, etc.
                score, readout, deep_rc_model, out_dim = self.evaluate(full_params)
            
            except NotImplementedError:
                raise
            except Exception as e:
                print(f"  [Error] Evaluation failed: {e}")
                traceback.print_exc()
                score = -np.inf
                readout, deep_rc_model, out_dim = None, None, None

            # Log history
            self.history.append({"params": current_params, "score": score})

            # Check if best
            if score > self.best_score:
                print(f"  >>> NEW BEST SCORE: {score:.5f}")
                self.best_score = score
                self.best_params = copy.deepcopy(current_params)
                
                # Store PyTorch state dict to save memory/pickle ability
                if deep_rc_model is not None:
                    # Move to CPU before storage to save GPU memory
                    self.best_model_state_dict = {k: v.cpu() for k, v in deep_rc_model.state_dict().items()}
                
                self.best_readout_model = copy.deepcopy(readout)
                self.best_output_dim = out_dim
                
                # Add to memory for exploitation
                self.topk_memory.append(copy.deepcopy(current_params))
            else:
                print(f"  Score: {score:.5f} (Best: {self.best_score:.5f})")

            # --- GARBAGE COLLECTION ---
            # Free up GPU memory after each iteration
            # ----------------------------------
            del deep_rc_model, readout  # 1. Delete local references
            import gc; gc.collect()      # 2. Force Python garbage collection
            torch.cuda.empty_cache()    # 3. Clear PyTorch's internal cache
            # ----------------------------------

        print("\nSearch finished.")
        print(f"Best Score: {self.best_score:.5f}")

    def evaluate(self, full_params):
        """
        Abstract method. 
        Must accept a dictionary of parameters, build the DeepReservoir, 
        perform training/validation, and return performance metrics.
        
        Returns:
            score (float): Metric to maximize.
            readout_model (object): The trained readout (e.g., sklearn model, torch linear layer).
            deep_rc_model (nn.Module): The instantiated DeepReservoir model.
            output_dim (int): The size of the reservoir output.
        """
        raise NotImplementedError("Subclasses must implement the evaluate method.")