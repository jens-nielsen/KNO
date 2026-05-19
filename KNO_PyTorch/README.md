# Kernel Neural Operator (PyTorch Version)

This is a PyTorch rewrite of the original Kernel Neural Operator (KNO) codebase.

## Project Structure

- `models.py`: PyTorch implementations of various KNO architectures.
- `kernels.py`: PyTorch implementations of stationary and non-stationary kernels.
- `quadratures.py`: Quadrature rules for triangles.
- `utils.py`: Normalizers and data loading utilities.
- `burgers.py`: Training script for 1D Burgers' equation.
- `darcy_2d.py`: Training script for 2D Darcy flow on uniform grids.
- `darcy_pwc.py`: Training script for 2D Darcy flow with piecewise constant coefficients.
- `darcy_triangle.py`: Training script for 2D Darcy flow on triangular meshes.
- `diffusion_reaction.py`: Training script for 3D diffusion-reaction equations.
- `ns_3d.py`: Training script for 3D Navier-Stokes.
- `ns_pipe.py`: Training script for Navier-Stokes in a pipe.

## How to Run

To run a training script, use:

```bash
python -m KNO_PyTorch.burgers --epochs 1000 --batch-size 10
```

Make sure you are in the root directory of the project and that the `datasets` folder is available.

## Requirements

- PyTorch
- NumPy
- SciPy
- tqdm
- wandb
