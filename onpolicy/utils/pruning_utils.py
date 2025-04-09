# Gradual Pruning Utils

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def compute_sparsity(model):
    """
    Compute the percentage of zero weights in a model.

    Args:
        model (nn.Module): The PyTorch model to compute its sparsity.

    Returns:
        float: Sparsity percentage.
    """
    total_params = 0
    zero_params = 0

    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()
            zero_params += torch.sum(param == 0).item()

    return (zero_params / total_params) * 100 if total_params > 0 else 0

class HarmonicSparsityScheduler:
    """
    Pruning schedule that combines a monotonic "base" schedule with
    an added harmonic oscillation, allowing prune-and-regrow dynamics while
    still tending toward a final sparsity. Generalization of 'cyclical' schedule.

    Args:
        total_episodes (int): Total number of training episodes (including warmup).
        warmup_episodes (int): Number of initial warmup episodes with no pruning.
        initial_sparsity (float): Initial sparsity after warmup.
        final_sparsity (float): Target sparsity level.
        A0 (float): Initial amplitude of the sinusoidal component.
        lambda_decay (float): Decay rate for the amplitude.
        T0 (float): Initial period of the sinusoidal component.
        T_increase_rate (float): Rate at which the period increases over time.
        base_schedule (str): Monotonic base schedule: ["linear", "cosine", "polynomial", "exponential", "step"].
        lock_progress_threshold (float): Percentage (0-1) of total episodes after which to lock sparsity.
    """

    def __init__(
        self,
        total_episodes,
        warmup_episodes,
        initial_sparsity,
        final_sparsity,
        A0,
        lambda_decay,
        T0,
        T_increase_rate,
        base_schedule="linear",
        lock_progress_threshold=0.9,
    ):
        self.total_episodes = total_episodes
        self.warmup_episodes = warmup_episodes
        self.total_pruning_steps = total_episodes - warmup_episodes
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.A0 = A0
        self.lambda_decay = lambda_decay
        self.T0 = T0
        self.T_increase_rate = T_increase_rate
        self.base_schedule = base_schedule
        self.lock_threshold_episode = int(lock_progress_threshold * total_episodes)
        self.locked = False

    def __call__(self, global_episode):
        if global_episode < self.warmup_episodes:
            return 0.0
        if self.locked:
            return self.final_sparsity

        # Progress w.r.t. pruning phase (0 -> 1 after warmup)
        t = global_episode - self.warmup_episodes
        progress = t / self.total_pruning_steps

        # 1) Base monotonic schedule
        if self.base_schedule == "linear":
            base = self.initial_sparsity + (self.final_sparsity - self.initial_sparsity) * progress
        elif self.base_schedule == "cosine":
            base = self.final_sparsity + 0.5 * (self.initial_sparsity - self.final_sparsity) * (1 + np.cos(np.pi * progress))
        elif self.base_schedule == "polynomial":
            base = (self.initial_sparsity - self.final_sparsity) * (1 - progress) ** 3 + self.final_sparsity
        elif self.base_schedule == "exponential":
            base = self.final_sparsity + (self.initial_sparsity - self.final_sparsity) * np.exp(-5 * progress)
        elif self.base_schedule == "step":
            steps = 5 # TODO: pass the number of steps as param
            step_idx = (progress * steps).astype(int)
            step_idx = np.minimum(step_idx, steps - 1)
            fraction = step_idx / (steps - 1)
            base = self.initial_sparsity + (self.final_sparsity - self.initial_sparsity) * fraction
        else:
            base = self.initial_sparsity + (self.final_sparsity - self.initial_sparsity) * progress # linear by default

        # 2) Harmonic oscillation
        A_t = self.A0 * np.exp(-self.lambda_decay * t)
        T_t = self.T0 * (1 + self.T_increase_rate * t)
        harmonic = A_t * np.sin(2 * np.pi * t / T_t)

        # 3) Combine base schedule + harmonic oscillations
        sparsity = base + harmonic
        sparsity = np.clip(sparsity, self.initial_sparsity, self.final_sparsity)

        # 4) Lock sparsity after x% once it reaches final_sparsity
        if global_episode >= self.lock_threshold_episode and sparsity >= self.final_sparsity:
            self.locked = True
            return self.final_sparsity

        return sparsity

def apply_gradual_schedule_pruning(model, current_sparsity, pruning_type='l1'):
    """
    Apply gradual schedule sparsification to a model using either L1 (magnitude-based) or random unstructured pruning.
    This version makes the pruning permanent and removes masks in order for weights to regrow (prune-and-regrow).
    
    The function handles pruning for:
    1. Linear layers (fc1, fc2 in MLPLayer)
    2. GRU layers (rnn in RNNLayer) - change to LSTM for LSTM networks
    
    For GRU/LSTM layers, both input-to-hidden (weight_ih_l0) and hidden-to-hidden (weight_hh_l0) weights are pruned.
    Biases are not pruned to maintain baseline activation levels.
    
    Args:
        model (nn.Module): The PyTorch model to prune (actor network).
        current_sparsity (float): Current target sparsity level (0-1).
        pruning_type (str): Type of pruning ('l1' or 'random').
    """
    if current_sparsity <= 0:  # dense
        return

    # collect all the weight parameters across the model
    parameters_to_prune = []
    for name, module in model.named_modules():
        # handle both Linear layers (in MLPBase) and GRU/LSTM layers (in RNNLayer)
        if isinstance(module, (nn.Linear, nn.GRU)):
            if isinstance(module, nn.Linear):
                # prune weights in Linear layers (fc1, fc2 in MLPLayer)
                parameters_to_prune.append((module, 'weight'))
            elif isinstance(module, nn.GRU):
                # prune both weight matrices in GRU/LSTM
                parameters_to_prune.append((module, 'weight_ih_l0'))  # input-to-hidden weights
                parameters_to_prune.append((module, 'weight_hh_l0'))  # hidden-to-hidden weights

    if not parameters_to_prune:
        print("no params to prune found")
        return

    # apply **global** unstructured pruning with a target sparsity
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=(
            prune.L1Unstructured if pruning_type == 'l1'
            else prune.RandomUnstructured
        ),
        amount=current_sparsity
    )

    # make the pruning permanent (and remove masks) -> then the weights can regrow in PPO optimization
    for module, param in parameters_to_prune:
        for name, _ in list(module.named_buffers()):
            if f"{param}_mask" in name:
                prune.remove(module, param)

def get_pruning_schedule(schedule_type, episode, num_episodes, initial_sparsity, final_sparsity, warmup_episodes, harmonic_scheduler=None):
    """Calculate the target sparsity for the current episode based on the selected schedule."""
    if episode < warmup_episodes:  # warm-up
        return 0.0
    
    progress = (episode - warmup_episodes) / (num_episodes - warmup_episodes)
    if schedule_type == 'linear':
        return initial_sparsity + (final_sparsity - initial_sparsity) * progress
    elif schedule_type == 'cosine':
        return final_sparsity + 0.5 * (initial_sparsity - final_sparsity) * (1 + np.cos(np.pi * progress))
    elif schedule_type == 'polynomial':
        return final_sparsity + (initial_sparsity - final_sparsity) * (1 - progress) ** 3
    elif schedule_type == 'exponential':
        return final_sparsity + (initial_sparsity - final_sparsity) * np.exp(-5 * progress)
    elif schedule_type == 'cyclical':
        base = final_sparsity + 0.5 * (initial_sparsity - final_sparsity) * (1 + np.cos(np.pi * progress))
        cycle = 0.1 * np.sin(2 * np.pi * episode / 200)  # 200-episode cycle length, 10% amplitude
        return np.clip(base + cycle, initial_sparsity, final_sparsity)
    elif schedule_type == 'harmonic':
        if harmonic_scheduler is None:
            raise ValueError("Must provide `harmonic_scheduler` when using 'harmonic' schedule_type.")
        return harmonic_scheduler(episode)  # uses global episode here!
    elif schedule_type == 'step':
        steps = 5 # TODO: pass the number of steps as param
        step_idx = (progress * steps).astype(int)
        step_idx = np.minimum(step_idx, steps - 1)
        fraction = step_idx / (steps - 1)
        return initial_sparsity + (final_sparsity - initial_sparsity) * fraction
    else:  # default to linear
        return initial_sparsity + (final_sparsity - initial_sparsity) * progress
