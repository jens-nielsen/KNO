### 
from jax import numpy as jnp, random as jr
import jax
from functools import partial
import optax
import equinox as eqx
import numpy as np
import torch

DTYPE=jnp.float32

### shuffle and slice each array in data tuple
@partial(jax.jit, static_argnums=-1)
def get_batch(epoch_key, data, batch_index, batch_size):
    batch = []
    for dat in data:
        dat_perm = jr.permutation(epoch_key, dat)
        batch.append(jax.lax.dynamic_slice_in_dim(
            dat_perm,
            batch_index * batch_size,
            batch_size,
        ))
    return batch

def get_batch_torch(epoch_seed: int, data: tuple[torch.Tensor, ...], batch_index: int, batch_size: int):
    """
    Shuffles and slices a tuple of tensors consistently across an epoch.
    """
    # Assume all tensors in the tuple share the same length (e.g., inputs and labels)
    num_samples = data[0].size(0)
    device = data[0].device
    
    # Create a local generator seeded by the epoch. 
    # This mimics JAX's stateless PRNG key: calling get_batch with the same 
    # epoch_seed guarantees the exact same shuffled layout for every batch index.
    gen = torch.Generator(device=device)
    gen.manual_seed(epoch_seed)
    
    # Calculate the permutation indices exactly once. 
    # This maps to JAX using the *same* key on multiple arrays in the loop.
    perm = torch.randperm(num_samples, generator=gen)
    
    start_idx = batch_index * batch_size
    end_idx = start_idx + batch_size
    
    batch = []
    for dat in data:
        # Apply the identical shuffle and slice dynamically
        dat_perm = dat[perm]
        batch.append(dat_perm[start_idx:end_idx])
        
    return batch

def is_trainable(x):
    return eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating)

### making an 'ensemble layer', which we can eqx.filter_vmap over
def create_lifted_module(base_layer, lift_dim, key):
    keys = jr.split(key, lift_dim)
    return eqx.filter_vmap(lambda key: base_layer(key=key))(keys)

### making an 'ensemble layer', which we can eqx.filter_vmap over
def create_lifted_module_torch(base_layer, lift_dim):
    return torch.nn.ModuleList([base_layer() for _ in range(lift_dim)])
    return eqx.filter_vmap(lambda key: base_layer(key=key))(keys)

def shuffle(x,y, seed=1):
    np.random.seed(seed)
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    x = x[idx]
    y = y[idx]
    return x,y

class UnitGaussianNormalizer(object):

    def __init__(self, x, axis=0, eps=1e-7):
        self.mean = jnp.mean(x, axis=axis, keepdims=True)
        self.std = jnp.std(x, axis=axis, keepdims=True)
        self.eps = eps

    @partial(jax.jit, static_argnums=(0,))
    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    @partial(jax.jit, static_argnums=(0,))
    def decode(self, x):
        std = self.std + self.eps  # n
        mean = self.mean
        x = (x * std) + mean
        return x

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

### lr schedule
def cosine_annealing(
    total_steps,
    warmup_frac=0.3,
    peak_value=3e-4,
    num_cycles=3,
    gamma=0.7,
    down=1e4
):
    init_value, end_value = peak_value/10, peak_value/10
    decay_steps = total_steps / num_cycles
    schedules = []
    boundaries = []
    boundary = 0

    for cycle in range(num_cycles -1):
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=init_value,
            warmup_steps=decay_steps * warmup_frac,
            peak_value=peak_value,
            decay_steps=decay_steps,
            end_value=end_value,
            exponent=1,
        )
        boundary = decay_steps + boundary
        boundaries.append(boundary)
        init_value = end_value
        end_value = end_value * gamma
        peak_value = peak_value * gamma
        schedules.append(schedule)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=init_value,
        warmup_steps=decay_steps * warmup_frac,
        peak_value=init_value,
        decay_steps=decay_steps,
        end_value=end_value/down,
        exponent=1,
    )
    boundary = decay_steps + boundary
    boundaries.append(boundary)
    schedules.append(schedule)

    return optax.join_schedules(schedules=schedules, boundaries=boundaries)

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