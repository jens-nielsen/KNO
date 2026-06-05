
import torch
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import spdiags, lil_matrix
from scipy.sparse.linalg import spsolve

def assemble_variable_darcy_matrix(coef):
    """
    Solve the elliptic PDE: -div(coef*grad(p)) = F
    
    Parameters
    ----------
    coef : ndarray
        Diffusion coefficient field of shape (K, K)
    F : ndarray
        Forcing function field of shape (K, K)
        
    Returns
    -------
    P : ndarray
        Solution to the PDE of shape (K, K)
    """
    K = coef.shape[0]
    
    coef_interp = coef

    # Build finite difference matrix for interior points
    n = K - 2
    A = lil_matrix((n*n, n*n))  # 4D array to hold sparse matrix entries
    
    for j in range(1, K-1):
        
        
        data = [
            [*(-(coef_interp[1:K-2, j] + coef_interp[2:K-1, j]) / 2), 0], 
            (coef_interp[0:K-2, j] + coef_interp[1:K-1, j]) / 2 + 
             (coef_interp[2:K, j] + coef_interp[1:K-1, j]) / 2 +
             (coef_interp[1:K-1, j-1] + coef_interp[1:K-1, j]) / 2 +
             (coef_interp[1:K-1, j+1] + coef_interp[1:K-1, j]) / 2, 
            [0, *(-(coef_interp[1:K-2, j] + coef_interp[2:K-1, j]) / 2)]
        ]

        diags = [-1, 0, 1]

        A[(j -1)*n:(j)*n, (j -1)*n:(j)*n] = spdiags(data, diags, n, n)

        if j != K-2:
            A[(j -1)*n:(j)*n, (j)*n:(j+1)*n] = spdiags([-(coef_interp[1:K-1, j] + coef_interp[1:K-1, j+1]) / 2], [0], n, n)
            A[(j)*n:(j+1)*n, (j -1)*n:(j)*n] = spdiags([-(coef_interp[1:K-1, j] + coef_interp[1:K-1, j+1]) / 2], [0], n, n)
    
    A = A.tocsr()
    A = A * (K - 1)**2
    
    return A

# --- Example Usage Validation ---
if __name__ == "__main__":
    res = 32
    # Create a dummy variable coefficient vector (e.g., all ones for standard Laplacian)
    a_coeff = torch.ones((res * res, 1), dtype=torch.float32)
    
    A_mat = assemble_variable_darcy_matrix(a_coeff, res=res, h=1.0)
    print("Assembled Matrix Shape:", A_mat.shape)
    print("\nVisual Matrix Map (Rows 0 & 5):")
    print("Row 0 (Boundary node, should be identity):", A_mat[0])
    print("Row 5 (Interior node, should be standard stencil):", A_mat[5])

    u_grid = torch.rand((res * res, 1), dtype=torch.float32)
    f_vector = A_mat @ u_grid

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(12, 5))
    axs[0].imshow(a_coeff.view(res, res).cpu(), origin='lower')
    axs[0].set_title('Variable Coefficient a(x,y)')
    axs[1].imshow(u_grid.view(res, res).cpu(), origin='lower')        
    axs[1].set_title('Input Solution u')
    axs[2].imshow(f_vector.view(res, res).cpu(), origin='lower')
    axs[2].set_title('Resulting Forcing f = A u')
    plt.tight_layout() 
    plt.show()