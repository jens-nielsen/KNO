from jax import random as jr, numpy as jnp
import torch
import jax
import equinox as eqx
from typing import List, Callable
from functools import partial
from abc import ABC, abstractmethod

class KernelBaseClass(ABC, torch.nn.Module):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass     

    ### to be implemented by each subclass, evaluate the kernel on a single pair of coordinates x,y, of shape (ndims,)
    @abstractmethod
    def eval(self, x, y):
        pass
    
    ### make kernel/Gram matrix given vectors x,y of coordinates, same for all kernels
    def forward(self, 
                x, 
                y):
        if x.ndim == 1 or y.ndim == 1:
            ndims = 1
        else:
            ndims = x.shape[-1]
        X,Y = x.reshape(-1, ndims), y.reshape(-1, ndims)
        k_xy = torch.vmap(torch.vmap(self.eval,  (0, None)), (None, 0))(Y,X)
        return k_xy

class  GaussianKernel(KernelBaseClass):
    scale: torch.Tensor
    
    def __init__(self, *, key, **kwargs):
        key,_ = jr.split(key)
        self.scale = jr.uniform(key,minval=-3.,maxval=0.0)

    def eval(self, x, y,):
        return (jnp.exp(- 0.5*(x - y)**2 / jnp.exp(2*self.scale))).sum() 


class AnisotropicGaussianKernel(KernelBaseClass):
    scale: torch.Tensor
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
    
class NonstationaryGaussianKernel(KernelBaseClass):
    scale: torch.nn.Module
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
    
class GaussianSpectralMixtureKernel(KernelBaseClass):
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

class NonstationaryGaussianSpectralMixtureKernel(KernelBaseClass):
    weights: torch.nn.Module
    q: int

    def __init__(
        self,
        *,
        ndims: int,
        q: int,
        latent_dim: int,
        **kwargs):
        super().__init__()
        self.q = q
        # key,_ = jr.split(key)

        self.weights = torch.nn.Sequential(
                                    torch.nn.Linear(ndims, latent_dim),
                                    torch.nn.SELU(),
                                    torch.nn.Linear(latent_dim, q + q + (q*ndims)),
                                    torch.nn.Softplus())
        
    def eval(self, x, y):
        q = self.q
        all_x, all_y = self.weights(x), self.weights(y)
        wx, wy = all_x[:q], all_y[:q]
        sx,sy = all_x[q:2*q], all_y[q:2*q]
        fx,fy = all_x[2*q:].reshape(q,-1), all_y[2*q:].reshape(q,-1)
        k_gibbs = (torch.sqrt(2 * sx * sy) / (sx**2 + sy**2)) * torch.exp(
            -(torch.sum((x - y) ** 2)) / (sx**2 + sy**2)
        )
        cosine = torch.cos(2 * torch.pi * (fx @ x - fy @ y))
        k_xy = (wx * wy * k_gibbs * cosine).sum()  # sum over mixtures
        return k_xy    
    
kernels = {'g': GaussianKernel,
           'a_g': AnisotropicGaussianKernel,
           'ns_g': partial(NonstationaryGaussianKernel, latent_dim=8),
           'gsm': partial(GaussianSpectralMixtureKernel, base_kernel=GaussianKernel, q=2),
           'ns_gsm': partial(NonstationaryGaussianSpectralMixtureKernel, latent_dim=8, q=2),
           }


if __name__ == "__main__":
    kernel = kernels['ns_gsm'](ndims=1)
    print(kernel)
    x = torch.linspace(0,1,10).unsqueeze(1)
    k_xy = kernel(x,x)