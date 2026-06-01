
from functools import partial

import torch


class GreensSecondOrderKernelTorch(torch.nn.Module):
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
class FastGreensSecondOrderKernelTorch(torch.nn.Module):
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
            r = ((x-y)**2).mean(axis=-1, keepdims=True) + 1e-20
            return torch.log(r)
        elif self.ndims == 1:
            r = torch.absolute(x-y) + 1e-20
            return r
        else:
            raise NotImplementedError("Only 1D and 2D supported for GreensSecondOrderKernel") 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
            
            phi, psi = self.phi(x), self.psi(x)
            out = phi * self.singularity_func(x[..., 0:self.ndims], x[..., self.ndims:2*self.ndims]) + psi
            out = torch.squeeze(out)
            return out
    
class Diagonal2DBlockMatrix(torch.nn.Module):
    def __init__(self, size, output_length, block_length):
        # 
        super().__init__()
        self.diag = torch.nn.Parameter(torch.randn(size, output_length, block_length))
        self.size = size
        self.output_length = output_length
        self.block_length = block_length

    def forward(self, x):
        # We assume x.shape = (n, m, input_size * block_length) -> (n, m, input_size, block_length) 
        # we want self.diag to be (output_size, input_size, block_length)
        inp = x.reshape(x.shape[0], x.shape[1], self.size, self.block_length)
        out = torch.einsum('nmij,ioj->nmio', inp, self.diag) # Sum over block length
        return out


class NonstationaryGaussianSpectralMixtureKernelTorch(torch.nn.Module):
    weights: torch.nn.Module
    q: int

    def __init__(
        self,
        *,
        ndims: int,
        q: int,
        latent_dim: int,
        output_dim: int,
        **kwargs):
        super().__init__()
        self.q = q
        self.ndims = ndims
        self.output_dim = output_dim
        self.weights = torch.nn.Sequential(
            torch.nn.Linear(ndims, latent_dim*output_dim),
            torch.nn.SELU(),
            Diagonal2DBlockMatrix(size=output_dim , output_length=(q + q + (q*ndims)), block_length=latent_dim),
            torch.nn.Softplus(),
        )

        
    def forward(self, x):
        x_ = x[..., 0:self.ndims]

        y_ = x[..., self.ndims:2*self.ndims]
        q = self.q

        all_x = self.weights(x_).reshape(*x.shape[0:2], self.output_dim, q + q + (q*self.ndims))
        all_y =  self.weights(y_).reshape(*x.shape[0:2], self.output_dim, q + q + (q*self.ndims))
        wx, wy = all_x[..., :q], all_y[..., :q]
        sx,sy = all_x[..., q:2*q], all_y[..., q:2*q]
        fx,fy = all_x[..., 2*q:].reshape(*all_x.shape[0:2], self.output_dim, q, self.ndims), all_y[..., 2*q:].reshape(*all_x.shape[0:2], self.output_dim, q, self.ndims)
        
        k_gibbs = (torch.sqrt(2 * sx * sy) / (sx**2 + sy**2)) * torch.exp(
            -(torch.sum((x_ - y_) ** 2, dim=-1, keepdim=True).unsqueeze(-1).expand_as(sx)) / (sx**2 + sy**2)
        )
        cosine = torch.cos(2 * torch.pi * (torch.einsum('ijcqn, ijn -> ijcq',fx, x_) - torch.einsum('ijcqn, ijn -> ijcq',fy, y_)))
        k_xy = torch.sum(wx * wy * k_gibbs * cosine, dim=-1)  # sum over mixtures

        return k_xy    
    


    
kernels = {
           'ns_gsm_torch': partial(NonstationaryGaussianSpectralMixtureKernelTorch, latent_dim=8, q=2),
           'green_torch': partial(GreensSecondOrderKernelTorch, latent_dim=8),
           'fast_green_torch': partial(FastGreensSecondOrderKernelTorch, latent_dim=8)
           }