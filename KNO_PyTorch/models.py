import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable

import models
from .kernels import kernels

from torch.func import stack_module_state, functional_call, vmap


def create_lifted_module(base_layer, num_models):
    # 1. Create a base layer (e.g., a simple MLP)
    models = torch.nn.ModuleList([base_layer() for _ in range(num_models)])
    params, buffers = stack_module_state(models)
    params = {k: v.to(device) for k, v in params.items()}
    buffers = {k: v.to(device) for k, v in buffers.items()}
    base_model = base_layer().to("meta")


    def functional_forward(params, buffers, x, y):
        return functional_call(base_model, (params, buffers), (x, y))
    

    # parallel_mlps = vmap(functional_forward, in_dims=(0, 0, None))

    # x_input = torch.randn(32, 10) # A batch of 32 data samples

    # output = parallel_mlps(params, buffers, x_input)
    # print(output.shape)

    
    return params, buffers, base_model, functional_forward


# 2. Instantiate identical structures in a list

# 3. CRITICAL EFFICIENCY STEP: Stacking the state
# This extracts all weights and combines them. 
# For example, a single model weight of [64, 10] becomes a stacked [16, 64, 10].

# Create a "meta" backbone to use as a skeleton blueprint (saves memory)

# 4. Define a purely functional representation of the forward pass

# 5. Vectorize using vmap
# (0, 0, None) means: parallelize over dim 0 of params and buffers, but share the same x

# 6. Execute in parallel inside a single kernel call

class KNO_REG_GRID_1D(torch.nn.Module):
    integration_kernels: List[torch.nn.Module]
    proj_layers: List[torch.nn.Module]
    pointwise_layers: List[torch.nn.Module]
    lift_kernel: torch.nn.Module
    lift_dim: int
    depth: int
    activation: Callable

    def to(self, device):
        super().to(device)
        self.params = [{k: v.to(device) for k, v in params.items()} for params in self.params]
        self.buffers = [{k: v.to(device) for k, v in buffers.items()} for buffers in self.buffers]
        return self

    def __init__(self, integration_kernel, lift_dim, depth, in_feats):
        super().__init__()

        # Initialise a list of integration kernels, one per layer, each with lift_dim parallel models inside
        self.integration_kernels = [
            torch.nn.ModuleList([integration_kernel() for _ in range(lift_dim)]) for _ in range(depth)
        ]

        states = [(stack_module_state(tmp_models)) for tmp_models in self.integration_kernels]

        self.params = [{k: v for k, v in params.items()} for params, _ in states]
        self.buffers = [{k: v for k, v in buffers.items()} for _, buffers in states]
        
        self.meta_holder = {"base_model": integration_kernel().to("meta")} # This is a dummy module just to hold the structure of the base model for functional_call, we never actually run it directly so it can be on meta device
        #

        self.pointwise_layers = torch.nn.ModuleList([torch.nn.Conv1d(lift_dim, lift_dim, 1)for _ in range(depth)])

        self.proj_layers = torch.nn.ModuleList([torch.nn.Linear(lift_dim, lift_dim), 
                                                torch.nn.Linear(lift_dim, lift_dim), 
                                                torch.nn.Linear(lift_dim, 1)])
        self.lift_kernel = torch.nn.Linear(in_feats, lift_dim)
        
        self.activation = torch.nn.functional.gelu
        self.lift_dim = lift_dim
        self.depth = depth



    def functional_forward(self, single_params, single_buffers, x_in, y_in):
            return functional_call(self.meta_holder["base_model"], (single_params, single_buffers), (x_in, y_in))

    def __call__(self, 
                 f_x, ### input fn, note no batch dim 
                 x_grid, 
                 q_weights,
                 ):

        def integration_transform(
                single_params, 
                single_buffers,
                q_nodes, ### quad nodes
                q_weights,     ### quad weights
                f_q):
            
            G = (self.functional_forward(single_params=single_params, single_buffers=single_buffers, x_in=q_nodes, y_in=q_nodes)) * q_weights.T
            f_q = torch.einsum('bq,kq->bk',f_q, G)
            return f_q
        
        f_q = f_x ### already at quad nodes # (B, N, 1)
        q_nodes = x_grid # (N, 1)
        f_q = torch.concatenate((f_q, q_nodes.unsqueeze(0).expand(f_q.shape[0], -1, -1)), axis=-1) 
        f_q = self.lift_kernel(f_q)
        f_q = self.activation(f_q)

        for i in range(self.depth-1):
            f_q_skip = self.pointwise_layers[i](f_q.permute(0, 2, 1)).permute(0, 2, 1) # (B, N, lift_dim) -> (B, N, lift_dim)

            single_params = self.params[i]
            single_buffers = self.buffers[i]

            f_q = torch.vmap(lambda params, buffers, f: integration_transform(params, buffers, q_nodes, q_weights, f), in_dims=(0, 0, 2), out_dims=2)(single_params, single_buffers,f_q)
                                                                                                
            f_q = f_q_skip + f_q
            f_q = self.activation(f_q)
        
        f_q_skip = self.pointwise_layers[-1](f_q.permute(0, 2, 1)).permute(0, 2, 1)
        
        single_params = self.params[-1]
        single_buffers = self.buffers[-1]
        f_q = torch.vmap(lambda params, buffers, f: integration_transform(params, buffers, q_nodes, q_weights, f), in_dims=(0, 0, 2), out_dims=2)(single_params, single_buffers,f_q)
        f_q = f_q_skip + f_q
        
        f_q = self.activation(self.proj_layers[0](f_q))
        f_q = self.activation(self.proj_layers[1](f_q))
        f_q = self.proj_layers[2](f_q)
        f_q = f_q.squeeze()
        return f_q
    

class KNO_DIFFUSION_REACTION(nn.Module):
    def __init__(self, output_kernel_type, integration_kernel_type, lift_dim, depth, in_feats, **kwargs):
        super().__init__()
        self.lift_dim = lift_dim
        self.depth = depth
        self.activation = F.gelu
        
        self.integration_kernels = nn.ModuleList([
            kernels[integration_kernel_type](ensemble_size=lift_dim, ndims=3) # Assuming 3D based on comment
            for _ in range(depth)
        ])
        
        self.pointwise_layers = nn.ModuleList([
            nn.Conv1d(lift_dim, lift_dim, 1) for _ in range(depth)
        ])
        
        self.lift_kernel = nn.Linear(in_feats, lift_dim)
        self.proj_layers = nn.Sequential(
            nn.Linear(lift_dim, lift_dim),
            nn.GELU(),
            nn.Linear(lift_dim, lift_dim),
            nn.GELU(),
            nn.Linear(lift_dim, 1)
        )
        
        self.output_kernel = kernels[output_kernel_type](ensemble_size=1, ndims=3)

    def forward(self, f_x, y_grid, q_nodes, q_weights):
        # f_x: (B, N_in, in_feats)
        # q_nodes: (N_q, 3)
        # y_grid: (N_y, 3)
        
        B = f_x.shape[0]
        
        f_x = torch.cat([f_x, q_nodes.unsqueeze(0).expand(B, -1, -1)], dim=-1)
        f_q = self.lift_kernel(f_x)
        f_q = self.activation(f_q)
        
        for i in range(self.depth):
            f_q_skip = self.pointwise_layers[i](f_q.transpose(1, 2)).transpose(1, 2)
            
            G = self.integration_kernels[i](q_nodes, q_nodes)
            G = G * q_weights.T
            f_q_int = torch.einsum('lnm,bml->bnl', G, f_q)
            
            f_q = f_q_skip + f_q_int
            if i < self.depth - 1:
                f_q = self.activation(f_q)
                
        f_q = self.proj_layers(f_q) # (B, N_q, 1)
        
        # Move to grid using the output kernel (GP interpolation)
        # Kqq = self.output_kernel(q_nodes,q_nodes) + (jnp.eye(len(q_nodes)) * 1e-5)
        # Kqy = self.output_kernel(q_nodes, y_grid)
        # KyqKqqInv = jnp.linalg.solve(Kqq, Kqy).T
        # f_y = jnp.einsum('mc,qm->qc', f_q,  KyqKqqInv) 
        
        Kqq = self.output_kernel(q_nodes, q_nodes).squeeze(0) + torch.eye(len(q_nodes), device=f_q.device) * 1e-5
        Kqy = self.output_kernel(q_nodes, y_grid).squeeze(0)
        
        # KyqKqqInv: (N_y, N_q)
        KyqKqqInv = torch.linalg.solve(Kqq, Kqy).T
        
        f_y = torch.einsum('bnc,mn->bmc', f_q, KyqKqqInv)
        return f_y

class KNO_DARCY_PWC(nn.Module):
    def __init__(self, integration_kernel_type, depth, lift_dim, ndims, in_feats, **kwargs):
        super().__init__()
        self.lift_dim = lift_dim
        self.depth = depth
        self.ndims = ndims
        self.in_feats = in_feats
        self.activation = F.gelu
        
        self.lift_kernel = nn.Linear(in_feats, lift_dim)
        
        # Factorized kernels: (kernel_x, kernel_y)
        self.integration_kernels = nn.ModuleList([
            nn.ModuleList([
                kernels[integration_kernel_type](ensemble_size=lift_dim, ndims=1)
                for _ in range(ndims)
            ]) for _ in range(depth)
        ])
        
        self.pointwise_layers = nn.ModuleList([
            nn.Conv1d(lift_dim, lift_dim, 1) for _ in range(depth)
        ])
        
        self.proj_layers = nn.Sequential(
            nn.Linear(lift_dim, lift_dim),
            nn.GELU(),
            nn.Linear(lift_dim, lift_dim),
            nn.GELU(),
            nn.Linear(lift_dim, 1)
        )

    def forward(self, f_x, x_grid, q_weights):
        # f_x: (B, N, N, 1)
        # x_grid: (N, N, 2)
        # q_weights: (N, 1)
        
        B, N, _, _ = f_x.shape
        q_nodes_1d = x_grid[:, 0, 0] # (N,)
        
        f_x = torch.cat([f_x, x_grid.unsqueeze(0).expand(B, -1, -1, -1)], dim=-1)
        f_x = f_x.view(B, -1, self.in_feats)
        f_q = self.lift_kernel(f_x) # (B, N*N, L)
        f_q = f_q.view(B, N, N, self.lift_dim).permute(0, 3, 1, 2) # (B, L, N, N)
        
        for i in range(self.depth):
            f_q_skip = self.pointwise_layers[i](f_q.reshape(B, self.lift_dim, -1)).view(B, self.lift_dim, N, N)
            
            # Factorized integration
            G1 = self.integration_kernels[i][0](q_nodes_1d, q_nodes_1d) # (L, N, N)
            G2 = self.integration_kernels[i][1](q_nodes_1d, q_nodes_1d) # (L, N, N)
            G1 = G1 * q_weights.T
            G2 = G2 * q_weights.T
            
            # f_q: (B, L, N, N)
            # result = G1 @ f_q + f_q @ G2.T
            # for each l: f_q[b, l] = G1[l] @ f_q[b, l] + f_q[b, l] @ G2[l].T
            f_q_int = torch.einsum('lnm,blmk->blnk', G1, f_q) + torch.einsum('blnm,lkm->blnk', f_q, G2)
            
            f_q = f_q_skip + f_q_int
            if i < self.depth - 1:
                f_q = self.activation(f_q)
                
        f_q = f_q.permute(0, 2, 3, 1).reshape(B, -1, self.lift_dim)
        f_y = self.proj_layers(f_q)
        return f_y

class KNO_DARCY_TRIANGLE(nn.Module):
    def __init__(self, input_kernel_type, output_kernel_type, integration_kernel_type, lift_dim, depth, in_feats, **kwargs):
        super().__init__()
        self.lift_dim = lift_dim
        self.depth = depth
        self.activation = F.gelu
        
        self.lift_kernel = nn.Linear(in_feats, lift_dim)
        
        self.integration_kernels = nn.ModuleList([
            kernels[integration_kernel_type](ensemble_size=lift_dim, ndims=2)
            for _ in range(depth)
        ])
        
        self.pointwise_layers = nn.ModuleList([
            nn.Conv1d(lift_dim, lift_dim, 1) for _ in range(depth)
        ])
        
        self.proj_layers = nn.Sequential(
            nn.Linear(lift_dim, lift_dim),
            nn.GELU(),
            nn.Linear(lift_dim, lift_dim),
            nn.GELU(),
            nn.Linear(lift_dim, 1)
        )
        
        self.input_kernel = kernels[input_kernel_type](ensemble_size=1, ndims=2)
        self.output_kernel = kernels[output_kernel_type](ensemble_size=1, ndims=2)

    def forward(self, f_x, x_grid, y_grid, q_nodes, q_weights):
        B = f_x.shape[0]
        
        f_x = torch.cat([f_x, x_grid.unsqueeze(0).expand(B, -1, -1)], dim=-1)
        f_x = self.lift_kernel(f_x) # (B, N_x, L)
        
        # Interpolate from x_grid to q_nodes
        Kxx = self.input_kernel(x_grid, x_grid).squeeze(0) + torch.eye(len(x_grid), device=f_x.device) * 1e-5
        Kxq = self.input_kernel(x_grid, q_nodes).squeeze(0)
        KqxKinv = torch.linalg.solve(Kxx, Kxq).T # (N_q, N_x)
        
        f_q = torch.einsum('bnl,mn->bml', f_x, KqxKinv)
        f_q = self.activation(f_q)
        
        for i in range(self.depth):
            f_q_skip = self.pointwise_layers[i](f_q.transpose(1, 2)).transpose(1, 2)
            
            G = self.integration_kernels[i](q_nodes, q_nodes)
            G = G * q_weights.T
            f_q_int = torch.einsum('lnm,bml->bnl', G, f_q)
            
            f_q = f_q_skip + f_q_int
            if i < self.depth - 1:
                f_q = self.activation(f_q)
                
        f_q = self.proj_layers(f_q)
        
        # Interpolate from q_nodes to y_grid
        Kqq = self.output_kernel(q_nodes, q_nodes).squeeze(0) + torch.eye(len(q_nodes), device=f_q.device) * 1e-5
        Kqy = self.output_kernel(q_nodes, y_grid).squeeze(0)
        KyqKqqInv = torch.linalg.solve(Kqq, Kqy).T
        
        f_y = torch.einsum('bnc,mn->bmc', f_q, KyqKqqInv)
        return f_y

class KNO_NS_PIPE(nn.Module):
    def __init__(self, integration_kernel_type, depth, lift_dim, ndims, in_feats, res_1d, **kwargs):
        super().__init__()
        self.lift_dim = lift_dim
        self.depth = depth
        self.ndims = ndims
        self.res_1d = res_1d
        self.activation = F.gelu
        
        self.lift_kernel = nn.Linear(in_feats, lift_dim)
        
        self.integration_kernels = nn.ModuleList([
            nn.ModuleList([
                kernels[integration_kernel_type](ensemble_size=lift_dim, ndims=1)
                for _ in range(ndims)
            ]) for _ in range(depth)
        ])
        
        self.pointwise_layers = nn.ModuleList([
            nn.Conv1d(lift_dim, lift_dim, 1) for _ in range(depth)
        ])
        
        self.proj_layers = nn.Sequential(
            nn.Linear(lift_dim, lift_dim),
            nn.GELU(),
            nn.Linear(lift_dim, lift_dim),
            nn.GELU(),
            nn.Linear(lift_dim, 1)
        )

    def forward(self, q_grid, wx, wy):
        # q_grid: (N, N, 2)
        B = 1 # JAX code seems to handle batching via vmap, here we assume at least 1
        N = self.res_1d
        
        grid_1d_y = q_grid[0, :, 1]
        grid_1d_x = q_grid[:, 0, 0]
        
        q_flat = q_grid.view(-1, 2)
        f_q = self.lift_kernel(q_flat).view(N, N, self.lift_dim)
        # Add batch dim for consistency
        f_q = f_q.unsqueeze(0).permute(0, 3, 1, 2) # (1, L, N, N)
        
        for i in range(self.depth):
            f_q_skip = self.pointwise_layers[i](f_q.reshape(1, self.lift_dim, -1)).view(1, self.lift_dim, N, N)
            
            G1 = self.integration_kernels[i][0](grid_1d_x, grid_1d_x) * wx.T
            G2 = self.integration_kernels[i][1](grid_1d_y, grid_1d_y) * wy.T
            
            f_q_int = torch.einsum('lnm,blmk->blnk', G1, f_q) + torch.einsum('blnm,lkm->blnk', f_q, G2)
            
            f_q = f_q_skip + f_q_int
            if i < self.depth - 1:
                f_q = self.activation(f_q)
                
        f_q = f_q.permute(0, 2, 3, 1).reshape(-1, self.lift_dim)
        f_y = self.proj_layers(f_q)
        return f_y

class KNO_NS_3D(nn.Module):
    def __init__(self, integration_kernel_type, depth, lift_dim, ndims, in_feats, res_1d, **kwargs):
        super().__init__()
        self.lift_dim = lift_dim
        self.depth = depth
        self.ndims = ndims
        self.in_feats = in_feats
        self.res_1d = res_1d
        self.activation = F.gelu
        
        self.lift_kernel = nn.Linear(in_feats, lift_dim)
        
        self.integration_kernels = nn.ModuleList([
            nn.ModuleList([
                kernels[integration_kernel_type](ensemble_size=lift_dim, ndims=1)
                for _ in range(ndims)
            ]) for _ in range(depth)
        ])
        
        self.pointwise_layers = nn.ModuleList([
            nn.Conv1d(lift_dim, lift_dim, 1) for _ in range(depth)
        ])
        
        self.proj_layers = nn.Sequential(
            nn.Linear(lift_dim, lift_dim),
            nn.GELU(),
            nn.Linear(lift_dim, lift_dim),
            nn.GELU(),
            nn.Linear(lift_dim, 1)
        )

    def forward(self, f_x, x_grid, q_weights):
        # f_x: (B, N, N, N, 1)
        # x_grid: (N, N, N, 3)
        # q_weights: (N, 1)
        
        B, N, _, _, _ = f_x.shape
        q_nodes_1d = x_grid[:, 0, 0, 1] # Based on JAX code
        
        f_x = torch.cat([f_x, x_grid.unsqueeze(0).expand(B, -1, -1, -1, -1)], dim=-1)
        f_x = f_x.view(B, -1, self.in_feats)
        f_q = self.lift_kernel(f_x) # (B, N^3, L)
        f_q = f_q.view(B, N, N, N, self.lift_dim).permute(0, 4, 1, 2, 3) # (B, L, N, N, N)
        
        for i in range(self.depth):
            f_q_skip = self.pointwise_layers[i](f_q.reshape(B, self.lift_dim, -1)).view(B, self.lift_dim, N, N, N)
            
            G1 = self.integration_kernels[i][0](q_nodes_1d, q_nodes_1d) * q_weights.T
            G2 = self.integration_kernels[i][1](q_nodes_1d, q_nodes_1d) * q_weights.T
            G3 = self.integration_kernels[i][2](q_nodes_1d, q_nodes_1d) * q_weights.T
            
            # f_q: (B, L, N, N, N)
            # f_q_int = G1 @ f_q + f_q @ G2 + ... (3D tensor contractions)
            f_q_int = torch.einsum('lnm,blmjk->blnjk', G1, f_q) + \
                      torch.einsum('lnm,blimk->blink', G2, f_q) + \
                      torch.einsum('lnm,blijm->blijn', G3, f_q)
            
            f_q = f_q_skip + f_q_int
            if i < self.depth - 1:
                f_q = self.activation(f_q)
                
        f_q = f_q.permute(0, 2, 3, 4, 1).reshape(B, -1, self.lift_dim)
        f_y = self.proj_layers(f_q)
        return f_y
