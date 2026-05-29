import torch
from pykeops.torch import LazyTensor

class Diagonal1DBlockMatrix(torch.nn.Module):
    def __init__(self, size, block_length):
        super().__init__()
        self.diag = torch.nn.Parameter(torch.randn(size, block_length))
        self.size = size
        self.block_length = block_length

    def forward(self, x):
        # Preserves any leading structural dimensions (M, N, etc.)
        orig_shape = x.shape[:-1]
        inp = x.reshape(*orig_shape, self.size, self.block_length)
        # Replaces the heavy einsum matrix contraction step
        out = (inp * self.diag).sum(dim=-1) 
        return out


class FastGreensSecondOrderKernelKeOps(torch.nn.Module):
    def __init__(self, *, ndims: int, latent_dim: int, output_dim: int):
        super().__init__()
        self.ndims = ndims
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.phi = torch.nn.Sequential(
            torch.nn.Linear(ndims * 2, latent_dim * output_dim),
            torch.nn.GELU(),
            Diagonal1DBlockMatrix(size=output_dim, block_length=latent_dim)
        )

        self.psi = torch.nn.Sequential(
            torch.nn.Linear(ndims * 2, latent_dim * output_dim),
            torch.nn.GELU(),
            Diagonal1DBlockMatrix(size=output_dim, block_length=latent_dim)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, density_f: torch.Tensor) -> torch.Tensor:
        """
        Computes the integral: Int( K(x, y) * f(y) dy ) across a batch.
        
        Parameters:
        -----------
        x : torch.Tensor of shape (M, ndims)        -> Target evaluation grid points
        y : torch.Tensor of shape (N, ndims)        -> Source integration grid points
        w : torch.Tensor of shape (N,)              -> Quadrature weights
        density_f : torch.Tensor of shape (B, N, C) -> Batch of density evaluations f(y)
                                                       (where C matches self.output_dim)
        """
        B, N, C = density_f.shape
        M = x.shape[0]

        # --- STEP 1: Compute coordinate features ONCE for all batch elements ---
        # Instead of allocating a huge global matrix, broadcast lazily via unsqueeze
        x_expanded = x.unsqueeze(1).expand(M, N, self.ndims)  # (M, N, ndims)
        y_expanded = y.unsqueeze(0).expand(M, N, self.ndims)  # (M, N, ndims)
        
        # Combine pairs into the shape the MLPs expect: (M, N, 2*ndims)
        xy_pairs = torch.cat([x_expanded, y_expanded], dim=-1)

        # Run through the layers. Outputs are (M, N, channels)
        phi_out = self.phi(xy_pairs)
        psi_out = self.psi(xy_pairs)

        # --- STEP 2: Compute Singularity Function Separately ---
        if self.ndims == 2:
            r = ((x_expanded - y_expanded) ** 2).mean(dim=-1, keepdims=True) + 1e-7
            singularity = torch.log(r)  # (M, N, 1)
        elif self.ndims == 1:
            singularity = torch.absolute(x_expanded - y_expanded) + 1e-7  # (M, N, 1)
        else:
            raise NotImplementedError("Only 1D and 2D supported.")

        # --- STEP 3: Construct the Static Base Kernel Matrix ---
        # Shape: (M, N, channels)
        K_matrix = phi_out * singularity + psi_out

        # --- STEP 4: Stream the Integration via PyKeOps LazyTensors ---
        # Wrap our calculated kernel features into symbolic LazyTensors
        # K_matrix is (M, N, C) -> we treat rows as index 'i' and columns as index 'j'
        # PyKeOps expects shapes (M, 1, C) for 'i' variables and (1, N, C) for 'j' variables
        K_lazy = LazyTensor(K_matrix.unsqueeze(1))  # Symbolic shape: (M, 1, N, channels)

        outputs = []
        for b in range(B):
            # Extract and wrap the specific batch density f(y)
            # density_f[b] is (N, C) -> make it a column-variable 'j'
            density_lazy = LazyTensor(density_f[b].unsqueeze(0)) # Symbolic shape: (1, N, channels)

            # Symbolic element-wise multiplication
            integrand_lazy = K_lazy * density_lazy * w.view(1, N, 1)  # Broadcast weights to match channels

            # Reduce along axis 1 (the 'N' / column dimension representing the 'y' points)
            # This streams the reduction in GPU registers, bypassing the 150MB allocation entirely!
            integrated_chunk = integrand_lazy.sum(dim=1)  # Output Shape: (M, channels)
            outputs.append(integrated_chunk)

        # Stack the micro-batch pieces back together -> (batch_size, M, channels)
        return torch.stack(outputs, dim=0)