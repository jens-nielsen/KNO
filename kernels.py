from jax import random as jr, numpy as jnp
import jax
import equinox as eqx
from typing import List, Callable
from functools import partial
from abc import ABC, abstractmethod

import torch

class KernelBaseClass(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass     

    ### to be implemented by each subclass, evaluate the kernel on a single pair of coordinates x,y, of shape (ndims,)
    @abstractmethod
    def eval(self, x, y):
        pass
    
    ### make kernel/Gram matrix given vectors x,y of coordinates, same for all kernels
    def __call__(self, 
                x, 
                y):
        if x.ndim == 1 or y.ndim == 1:
            ndims = 1
        else:
            ndims = x.shape[-1]
        X,Y = x.reshape(-1, ndims), y.reshape(-1, ndims)
        k_xy = jax.vmap(jax.vmap(self.eval, (0, None)), (None, 0))(Y,X)
        return k_xy
    
class GreensSecondOrderKernel(eqx.Module, KernelBaseClass):
    phi: eqx.Module
    psi: eqx.Module
    ndims: int

    def __init__(
        self,
        *,
        ndims: int,
        latent_dim: int,
        key,
        **kwargs):
        key,_ = jr.split(key)
        self.ndims = ndims
        self.phi = eqx.nn.MLP(key=key, 
                                  in_size=ndims * 2, 
                                  out_size=1, 
                                  width_size=latent_dim, 
                                  depth=1, 
                                  activation=jax.nn.gelu, 
                                  )


        self.psi = eqx.nn.MLP(key=key, 
                                  in_size= ndims * 2, 
                                  out_size=1, 
                                  width_size=latent_dim, 
                                  depth=1, 
                                  activation=jax.nn.gelu, 
                                  )
        
    def singularity_func(self, x, y):
        if self.ndims == 2:
            r = ((x-y)**2).mean(axis=-1) + 1e-7
            return jnp.log(r)
        elif self.ndims == 1:
            r = jnp.absolute(x-y) + 1e-7
            return r
        else:
            raise NotImplementedError("Only 1D and 2D supported for GreensSecondOrderKernel")
        
    def eval(self, x, y):
        phi, psi = self.phi(jnp.concatenate([x, y])), self.psi(jnp.concatenate([x, y]))
        out = phi * self.singularity_func(x,y) + psi
        out = jnp.squeeze(out)
        
        return out    
    

class GreensSecondOrderKernelTorch(torch.nn.Module, KernelBaseClass):
    phi: torch.nn.Module
    psi: torch.nn.Module
    ndims: int

    def __init__(
        self,
        *,
        ndims: int,
        latent_dim: int,
        **kwargs):
        super().__init__()
        self.ndims = ndims
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(ndims*2, latent_dim),
            torch.nn.GELU(),
            torch.nn.Linear(latent_dim, 1)
        )
        self.psi = torch.nn.Sequential(
            torch.nn.Linear(ndims*2, latent_dim),
            torch.nn.GELU(),
            torch.nn.Linear(latent_dim, 1)
        )
        
    def singularity_func(self, x, y):
        if self.ndims == 2:
            r = ((x-y)**2).mean(axis=-1, keepdims=True) + 1e-7
            return torch.log(r)
        elif self.ndims == 1:
            r = torch.absolute(x-y) + 1e-7
            return r
        else:
            raise NotImplementedError("Only 1D and 2D supported for GreensSecondOrderKernel")
        
    def eval(self, x):
        # X_expanded = x.expand(-1, y.shape[1], -1)  
        # Y_expanded = y.expand(x.shape[0], -1, -1)  
        # input = torch.concatenate([X_expanded, Y_expanded], dim=-1)
        phi, psi = self.phi(x), self.psi(x)

        out = phi * self.singularity_func(x[..., 0], x[..., 1]) + psi
        out = torch.squeeze(out)
        
        return out    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
            # if x.ndim == 1 or y.ndim == 1:
            #     ndims = 1
            # else:
            #     ndims = x.shape[-1]
                
            # # Reshape to (-1, ndims)
            # X = x.reshape(-1, ndims)
            # Y = y.reshape(-1, ndims)
            
            # # Unsqueeze to align dimensions for broadcasting:
            # # X becomes (1, num_x, ndims)
            # # Y becomes (num_y, 1, ndims)
            # # self.eval will output a matrix of shape (num_y, num_x)
            # k_xy = self.eval(Y.unsqueeze(1), X.unsqueeze(0))
            k_xy = self.eval(x)
            
            return k_xy


class Diagonal1DBlockMatrix(torch.nn.Module):
    def __init__(self, size, block_length):
        super().__init__()
        self.diag = torch.nn.Parameter(torch.randn(size, block_length))
        self.size = size
        self.block_length = block_length

    def forward(self, x):
        # We assume x.shape = (n, m, size * block_length) 
        inp = x.reshape(x.shape[0], x.shape[1], self.size, self.block_length)
        out = torch.einsum('nmij,ij->nmi', inp, self.diag) # Sum over block length
        return out
class FastGreensSecondOrderKernelTorch(torch.nn.Module, KernelBaseClass):
    phi: torch.nn.Module
    psi: torch.nn.Module
    ndims: int

    def __init__(
        self,
        *,
        ndims: int,
        latent_dim: int,
        output_dim: int,
        **kwargs):
        super().__init__()
        self.ndims = ndims
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(ndims*2, latent_dim*output_dim),
            torch.nn.GELU(),
            Diagonal1DBlockMatrix(size=output_dim, block_length=latent_dim)
        )

        self.psi = torch.nn.Sequential(
            torch.nn.Linear(ndims*2, latent_dim*output_dim),
            torch.nn.GELU(),
            Diagonal1DBlockMatrix(size=output_dim, block_length=latent_dim)
        )
        
    def singularity_func(self, x, y):
        if self.ndims == 2:
            r = ((x-y)**2).mean(axis=-1, keepdims=True) + 1e-7
            return torch.log(r)
        elif self.ndims == 1:
            r = torch.absolute(x-y) + 1e-7
            return r
        else:
            raise NotImplementedError("Only 1D and 2D supported for GreensSecondOrderKernel") 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
            
            phi, psi = self.phi(x), self.psi(x)
            out = phi * self.singularity_func(x[..., 0], x[..., 1]) + psi
            out = torch.squeeze(out)
            
            return out


class GaussianKernel(eqx.Module, KernelBaseClass):
    scale: jax.Array
    
    def __init__(self, *, key, **kwargs):
        key,_ = jr.split(key)
        self.scale = jr.uniform(key,minval=-3.,maxval=0.0)

    def eval(self, x, y,):
        return (jnp.exp(- 0.5*(x - y)**2 / jnp.exp(2*self.scale))).sum() 


class AnisotropicGaussianKernel(eqx.Module, KernelBaseClass):
    scale: jax.Array
    ndims: int
    place: Callable

    def __init__(self, *, ndims, key, **kwargs):
        keys = jr.split(key) 
        self.scale = jr.uniform(keys[0], (int(ndims*(ndims+1)/2),), minval=-3., maxval=0.)
        self.ndims = ndims
        self.place = lambda vals: jnp.zeros((ndims,ndims)).at[jnp.tril_indices(ndims)].set(vals)

    ### one pair of pts
    def eval(self, x, y,):
        L = self.place(self.scale)
        scale = L @ L.T
        r_scaled = (x-y) @ scale @ (x-y)
        return jnp.exp(-1/2 * r_scaled)
    
class NonstationaryGaussianKernel(eqx.Module, KernelBaseClass):
    scale: eqx.Module
    ndims: int
    def __init__(self, *, ndims, latent_dim, key, **kwargs):
        keys = jr.split(key)
        self.scale = eqx.nn.Sequential(
            [
                eqx.nn.Linear(ndims, latent_dim, key=keys[0]),
                eqx.nn.Lambda(jax.nn.selu),
                eqx.nn.Linear(latent_dim, 1, key=keys[1]),
                eqx.nn.Lambda(jax.nn.softplus),
            ]
        )
        self.ndims = ndims

    def eval(self, x, y,):
        sx = self.scale(x).squeeze()
        sy = self.scale(y).squeeze()

        k_gibbs_t1 = jnp.sqrt((2 * sx * sy) / (sx**2 + sy**2))
        k_gibbs_t2 =  jnp.exp(
            -jnp.sum((x - y) ** 2) / (sx**2 + sy**2)
        )
        k_gibbs = k_gibbs_t1 * k_gibbs_t2
        return k_gibbs
    
class GaussianSpectralMixtureKernel(eqx.Module, KernelBaseClass):
    q: int
    weights: jax.Array
    freqs: jax.Array
    scales: jax.Array

    def __init__(self, *, base_kernel, q, ndims, key, **kwargs):
        key1, key2, key3 = jr.split(key, 3)
        self.q = q

        self.weights = jr.uniform(key1, (q,1), maxval=0.1, minval=-3.)
        self.freqs =  jr.uniform(key2, (q,ndims), maxval=0.1, minval=-3.)
        self.scales = jr.uniform(key, (q,), maxval=0.1, minval=-3,)

    def eval(self, x,y):
        weights = jax.nn.softplus(self.weights)
        freqs = jax.nn.softplus(self.freqs)
        scales = jax.nn.softplus(self.scales)
        tau = x-y
        cos = jnp.cos(freqs @ tau)
        gauss = eqx.filter_vmap(lambda scale: (jnp.exp(- 0.5*tau**2 / jnp.exp(2*scale))).sum())(scales)
        return jnp.sum(weights * cos * gauss)

class NonstationaryGaussianSpectralMixtureKernel(eqx.Module, KernelBaseClass):
    weights: eqx.Module
    q: int

    def __init__(
        self,
        *,
        ndims: int,
        q: int,
        latent_dim: int,
        key,
        **kwargs):
        self.q = q
        key,_ = jr.split(key)
        self.weights = eqx.nn.MLP(key=key, 
                                  in_size=ndims, 
                                  out_size=q + q + (q*ndims), 
                                  width_size=latent_dim, 
                                  depth=1, 
                                  activation=jax.nn.selu, 
                                  final_activation=jax.nn.softplus,
                                  )
        
    def eval(self, x, y):
        q = self.q
        all_x, all_y = self.weights(x), self.weights(y)
        wx, wy = all_x[:q], all_y[:q]
        sx,sy = all_x[q:2*q], all_y[q:2*q]
        fx,fy = all_x[2*q:].reshape(q,-1), all_y[2*q:].reshape(q,-1)
        k_gibbs = (jnp.sqrt(2 * sx * sy) / (sx**2 + sy**2)) * jnp.exp(
            -(jnp.sum((x - y) ** 2)) / (sx**2 + sy**2)
        )
        cosine = jnp.cos(2 * jnp.pi * (fx @ x - fy @ y))
        k_xy = (wx * wy * k_gibbs * cosine).sum()  # sum over mixtures
        return k_xy    
    
kernels = {'g': GaussianKernel,
           'a_g': AnisotropicGaussianKernel,
           'ns_g': partial(NonstationaryGaussianKernel, latent_dim=8),
           'gsm': partial(GaussianSpectralMixtureKernel, base_kernel=GaussianKernel, q=2),
           'ns_gsm': partial(NonstationaryGaussianSpectralMixtureKernel, latent_dim=8, q=2),
           'green': partial(GreensSecondOrderKernel, latent_dim=8),
           'green_torch': partial(GreensSecondOrderKernelTorch, latent_dim=8),
           'fast_green_torch': partial(FastGreensSecondOrderKernelTorch, latent_dim=8)
           }