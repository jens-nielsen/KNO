import torch
from darcy_do import assemble_variable_darcy_matrix

from torch_utils import UnitGaussianNormalizerTorch
import numpy as np

import matplotlib.pyplot as plt


# Fundamental solution for 2D Darcy Flow
def divergence_inverse_2d(grid, q_weights, f):
    # grid: (m, n, 4) concatenated grid of integration and evalutation points
    # q_weights: (n, 1)
    # f: (b, n, 1)
    # Compute the pairwise differences between grid points
    diff_x = grid[..., 0:1] - grid[..., 2:3]  # (m, n, 1)
    diff_y = grid[..., 1:2] - grid[..., 3:4]  # (m, n, 1)
    abs_sqr_diff = diff_x**2 + diff_y**2 + 1e-22 # (m, n, 1) 

    kernel_x = diff_x / (2 * torch.pi * abs_sqr_diff)  # (m, n, 1)
    kernel_y = diff_y / (2 * torch.pi * abs_sqr_diff)  # (m, n, 1)

    kernel = torch.cat([kernel_x, kernel_y], dim=-1)  # (m, n, 2)

    weighted_f = f * q_weights  # (b, n, 1)

    output = torch.einsum('mnk,bn->bmk', kernel, weighted_f.squeeze(-1)) # (b, m, 2)

    return output


def gradient_inverse_2d(grid, q_weights, f):
    # grid: (m, n, 4) concatenated grid of integration and evalutation points
    # q_weights: (n, 1)
    # f: (b, n, 2)
    # output: (b, m)

    # Compute the pairwise differences between grid points
    diff_x = grid[..., 0:1] - grid[..., 2:3]  # (m, n, 1)
    diff_y = grid[..., 1:2] - grid[..., 3:4]  # (m, n, 1)
    abs_sqr_diff = diff_x**2 + diff_y**2 + 1e-22  # (m, n, 1)

    kernel_x = diff_x / (2 * torch.pi * abs_sqr_diff)  # (m, n, 1)
    kernel_y = diff_y / (2 * torch.pi * abs_sqr_diff)  # (m, n, 1)

    kernel = torch.cat([kernel_x, kernel_y], dim=-1)  # (m, n, 2)

    weighted_f = f * q_weights  # (b, n, 2)

    output = torch.einsum('mnk,bnk->bm', kernel, weighted_f) # (b, m)

    return output


def darcy_flow_inverse(grid, q_weights, a, f=1.0):
    # grid: (m, n, 4) concatenated grid of integration and evalutation points
    # q_weights: (n, 1)
    # a: (b, n, 1)
    # f: (b, n, 1)
    # output: (b, m)

    if f==1.0:
        f = torch.ones_like(a, dtype=grid.dtype, device=grid.device)

    output = divergence_inverse_2d(grid, q_weights, f)  # (b, m, 2)
    output = output / a  # (b, m, 2)
    output = gradient_inverse_2d(grid, q_weights, output)  # (b, m)

    return output


class DarcyFlowGTModel(torch.nn.Module):
    def __init__(self, grid, q_weights):
        super().__init__()
        self.q_weights = q_weights # (n,)

        self.concat_grid = torch.cat([grid.unsqueeze(0).expand( grid.shape[0], -1, -1), grid.unsqueeze(1).expand(-1, grid.shape[0], -1)], dim=-1) # (n, n, 4)

    def forward(self, a):
        a = a.reshape(a.shape[0], -1, 1) # (b, n, 1)
        return darcy_flow_inverse(self.concat_grid, self.q_weights, a)





DTYPE = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
# key = jr.PRNGKey(args.seed)

### load data
data = np.load('./datasets/darcy_pwc.npz')
x, y = torch.tensor(data["x"], dtype=DTYPE), torch.tensor(data["y"], dtype=DTYPE)
res_1d = 29
domain_dims = 2
codomain_dims = 1
y = y.reshape(1200, -1)
x = x.reshape(1200, res_1d, res_1d, 1)

x_grid_1d = torch.linspace(0, 1, 29, dtype=DTYPE)
x_grid = torch.stack(torch.meshgrid(x_grid_1d, x_grid_1d, indexing='ij')).permute(1, 2, 0).to(device)

ntrain = 1000
ntest = 200
x_train, x_test = x[:ntrain], x[-ntest:]
y_train, y_test = y[:ntrain], y[-ntest:]

# x_normalizer = UnitGaussianNormalizerTorch(x_train)
# x_train = x_normalizer.encode(x_train)
# x_test = x_normalizer.encode(x_test)
# y_normalizer = UnitGaussianNormalizerTorch(y_train)

# x_normalizer.to(device)
# y_normalizer.to(device)

## 2D Trapezoidal rule weights
h = x_grid[1,0,0] - x_grid[0,0,0]
w = torch.ones((res_1d, res_1d)).to(device) * h*h
w[0,0] = h*h/4
w[0,-1] = h*h/4
w[-1,0] = h*h/4
w[-1,-1] = h*h/4
w[0,1:-1] = h*h/2
w[-1,1:-1] = h*h/2
w[1:-1,0] = h*h/2
w[1:-1,-1] = h*h/2
q_weights = w.reshape(-1,1).to(device)

# Class Differential Operator


def eval(model, batch,):
    x,y = batch
    y_pred = model(x)
    
    i = 10
    fig, axes = plt.subplots(1, 3, figsize=(10,5))
    axes[0].imshow(y[i].reshape(res_1d, res_1d).cpu(), origin='lower')
    axes[0].set_title('Ground Truth')
    axes[1].imshow(y_pred[i].reshape(res_1d, res_1d).detach().cpu(), origin='lower')
    axes[1].set_title('Prediction')
    axes[2].imshow(x[i].reshape(res_1d, res_1d).cpu(), origin='lower')
    axes[2].set_title('Input')
    plt.show()

    y_pred = y_pred.reshape(ntest,-1)
    y_pred = y_normalizer.decode(y_pred)
    test_l2 = ((y - y_pred)**2).sum(axis=-1).mean()
    test_rel_l2 =  (torch.linalg.norm(y-y_pred, axis=1) / torch.linalg.norm(y, axis=1)).mean()
    return test_l2.item(), test_rel_l2.item()

model = DarcyFlowGTModel(x_grid.reshape(-1, 2), q_weights).to(device)

test_l2, test_rel_l2 = eval(model, (x_test.to(device), y_test.to(device)))
print(f'test rel_l2: {test_rel_l2*100:.3f}')



# eqx.tree_serialise_leaves(f"./saved_models/DarcyPWC_{args.int_kernel}.eqx", model)
