import torch
import torch.nn as nn
import numpy as np

def get_batch(data, batch_index, batch_size):
    batch = []
    for dat in data:
        # Assuming dat is already a torch tensor or numpy array
        if isinstance(dat, np.ndarray):
            dat = torch.from_numpy(dat)
        
        # Simple slicing, shuffling is usually handled outside or by DataLoader
        batch.append(dat[batch_index * batch_size : (batch_index + 1) * batch_size])
    return batch

def shuffle(x, y, seed=1):
    torch.manual_seed(seed)
    idx = torch.randperm(len(x))
    x = x[idx]
    y = y[idx]
    return x, y

class UnitGaussianNormalizer(object):
    def __init__(self, x, axis=0, eps=1e-7):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        self.mean = torch.mean(x, dim=axis, keepdim=True)
        self.std = torch.std(x, dim=axis, keepdim=True)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x):
        std = self.std + self.eps
        mean = self.mean
        x = (x * std) + mean
        return x

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

# PyTorch implementation of the cosine annealing schedule might be better handled 
# using torch.optim.lr_scheduler. 
# However, if I want to mimic optax.join_schedules, I might need a custom scheduler.
# For now, I'll skip the exact implementation of cosine_annealing as it's highly optax-specific.
# Standard PyTorch training loops often use StepLR or CosineAnnealingLR.


import math
import torch
from torch.optim.lr_scheduler import LRScheduler

class CyclicalCosineLRScheduler(LRScheduler):
    def __init__(
        self, 
        optimizer, 
        total_steps: int, 
        warmup_frac: float = 0.3, 
        peak_value: float = 3e-4, 
        num_cycles: int = 3, 
        gamma: float = 0.7, 
        down: float = 1e4,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.total_steps = total_steps
        self.warmup_frac = warmup_frac
        self.peak_value = peak_value
        self.num_cycles = num_cycles
        self.gamma = gamma
        self.down = down
        
        # Calculate the duration of each cycle block
        self.decay_steps = total_steps / num_cycles
        self.warmup_steps = self.decay_steps * warmup_frac
        
        # Pre-calculate baseline targets to avoid re-computing complex bounds in loops
        # PyTorch LRSchedulers scale the *base_lr* of the optimizer. We will compute 
        # absolute values matching your Optax code, assuming the optimizer's initial LR
        # is just a placeholder or set to 1.0.
        super().__init__(optimizer, last_epoch, verbose)

    def _get_single_cycle_lr(self, step, cycle_idx):
        """Helper to compute the exact cosine decay for a specific step inside a given cycle."""
        # Calculate the localized step within this cycle block
        local_step = step - (cycle_idx * self.decay_steps)
        
        # Determine the peak, init, and end values for this specific cycle based on gamma decay
        if cycle_idx < self.num_cycles - 1:
            # Standard cycles
            c_peak = self.peak_value * (self.gamma ** cycle_idx)
            c_init = (self.peak_value / 10) * (self.gamma ** cycle_idx)
            c_end = (self.peak_value / 10) * (self.gamma ** (cycle_idx + 1))
        else:
            # Final cycle (The 'down' cycle)
            # Matches: peak_value = init_value from your code
            c_init = (self.peak_value / 10) * (self.gamma ** cycle_idx)
            c_peak = c_init 
            c_end = ((self.peak_value / 10) * (self.gamma ** cycle_idx)) / self.down

        # --- Optax warmup_cosine_decay_schedule logic mapped to PyTorch ---
        if local_step < self.warmup_steps:
            # Linear Warmup phase: from c_init to c_peak
            pct = local_step / self.warmup_steps
            return c_init + pct * (c_peak - c_init)
        else:
            # Cosine Decay phase: from c_peak down to c_end
            # Restrict local step to not overrun the cycle boundary
            cosine_step = min(local_step, self.decay_steps)
            
            # Map step progress to a fraction of the remaining decay window
            decay_window = self.decay_steps - self.warmup_steps
            pct = (cosine_step - self.warmup_steps) / decay_window
            pct = min(max(pct, 0.0), 1.0) # Clamp between 0 and 1
            
            # Cosine interpolation formula used by Optax
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * pct))
            return c_end + (c_peak - c_end) * cosine_decay

    def get_lr(self):
        # self.last_epoch tracks the current training step count in PyTorch
        step = self.last_epoch
        
        # Clip step to maximum total steps to prevent array out-of-bounds evaluation
        if step >= self.total_steps:
            step = self.total_steps - 1
            
        # Determine which cycle block we are currently sitting in
        current_cycle = int(step // self.decay_steps)
        current_cycle = min(current_cycle, self.num_cycles - 1)
        
        # Calculate the absolute target learning rate
        target_lr = self._get_single_cycle_lr(step, current_cycle)
        
        # Return the target LR for every single parameter group in the optimizer
        return [target_lr for _ in self.base_lrs]