import torch


class UnitGaussianNormalizerTorch(object):

    def __init__(self, x, axis=0, eps=1e-7):
        self.mean = torch.mean(x, axis=axis, keepdims=True)
        self.std = torch.std(x, axis=axis, keepdims=True)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x):
        std = self.std + self.eps  # n
        mean = self.mean
        x = (x * std) + mean
        return x
    
    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


import math
from torch.optim.lr_scheduler import LRScheduler
class CosineAnnealingWarmupRestarts(LRScheduler):
    """
    PyTorch translation of the custom Optax warmup cosine decay schedule.
    """
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer, 
        total_steps: int,
        warmup_frac: float = 0.3,
        peak_value: float = 3e-4,
        num_cycles: int = 3,
        gamma: float = 0.7,
        down: float = 1e4,
        last_epoch: int = -1
    ):
        self.total_steps = total_steps
        self.warmup_frac = warmup_frac
        self.peak_value = peak_value
        self.num_cycles = num_cycles
        self.gamma = gamma
        self.down = down
        
        # Calculate cycle boundaries
        self.cycle_steps = total_steps / num_cycles
        self.warmup_steps = self.cycle_steps * warmup_frac
        self.decay_steps = self.cycle_steps - self.warmup_steps
        
        self._precompute_cycles()
        
        # Initialize the base class (this will call get_lr() for the first time)
        super().__init__(optimizer, last_epoch)

    def _precompute_cycles(self):
        """Precomputes the init, peak, and end values for each cycle to match the JAX logic."""
        self.cycle_params = []
        init_val = self.peak_value / 10
        end_val = self.peak_value / 10
        peak_val = self.peak_value
        
        # Intermediate cycles
        for _ in range(self.num_cycles - 1):
            self.cycle_params.append({
                'init': init_val,
                'peak': peak_val,
                'end': end_val,
            })
            init_val = end_val
            end_val = end_val * self.gamma
            peak_val = peak_val * self.gamma
            
        # The final cycle: flat during warmup (peak == init), then drops steeply
        self.cycle_params.append({
            'init': init_val,
            'peak': init_val,  
            'end': end_val / self.down,
        })

    def get_lr(self):
        t = self.last_epoch 
        
        # If training continues past total_steps, hold the final learning rate
        if t >= self.total_steps:
            final_lr = self.cycle_params[-1]['end']
            return [final_lr for _ in self.base_lrs]
            
        # Determine the current cycle and the step within that cycle
        cycle = int(t // self.cycle_steps)
        cycle = min(cycle, self.num_cycles - 1)
        t_in_cycle = t - (cycle * self.cycle_steps)
        
        params = self.cycle_params[cycle]
        
        # Phase 1: Linear Warmup
        if t_in_cycle <= self.warmup_steps:
            if self.warmup_steps == 0:
                lr = params['peak']
            else:
                progress = t_in_cycle / self.warmup_steps
                lr = params['init'] + (params['peak'] - params['init']) * progress
                
        # Phase 2: Cosine Decay
        else:
            if self.decay_steps == 0:
                lr = params['end']
            else:
                t_decay = t_in_cycle - self.warmup_steps
                progress = t_decay / self.decay_steps
                # Standard cosine annealing formula
                lr = params['end'] + 0.5 * (params['peak'] - params['end']) * (1 + math.cos(math.pi * progress))
                
        # Apply the calculated absolute learning rate to all parameter groups
        return [lr for _ in self.base_lrs]
    