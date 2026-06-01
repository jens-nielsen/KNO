import torch
from typing import List, Tuple


class KNO_UNET_GREEN_2D(torch.nn.Module):
    integration_kernels: List[torch.nn.Module]
    lift_kernel: torch.nn.Module
    depth: int
    proj_layers: torch.nn.Module
    pointwise_layers: List[torch.nn.Module]
    d: int
    lift_dim: int
    in_feats: int
  

    def __init__(self,
                 integration_kernel,
                 depth,
                 lift_dim,
                 ndims,
                 in_feats,
                 device
    ):  
        super().__init__()
        # keys = jr.split(key, 7)
        
        self.lift_dim = lift_dim
        self.d = ndims

        self.proj_layers = torch.nn.ModuleList([
            torch.nn.Linear(lift_dim, lift_dim),
            torch.nn.Linear(lift_dim, lift_dim),
            torch.nn.Linear(lift_dim, 1)
        ])

        # self.pointwise_layers = [eqx.nn.Conv(1, lift_dim, lift_dim, 1, key=key) for key in jr.split(keys[3], depth)]
        self.pointwise_layers = torch.nn.ModuleList([torch.nn.Conv2d(lift_dim, lift_dim, kernel_size=1, padding=0) for _ in range(depth)])

        # self.lift_kernel = eqx.nn.Linear(in_feats,lift_dim,key=keys[4])
        self.lift_kernel = torch.nn.Linear(in_feats, lift_dim)

        self.skip_convs = torch.nn.ModuleList([torch.nn.Conv2d(lift_dim*2, lift_dim, kernel_size=1, padding=0) for _ in range(depth//2 + depth % 2)])

    
        # self.integration_kernels = [clm(integration_kernel, lift_dim, k) for k in jr.split(keys[5],depth)]
        self.integration_kernels = torch.nn.ModuleList([integration_kernel() for _ in range(depth)])

        self.in_feats = in_feats
        self.depth = depth
    
    def trapezoidal_rule_weights(self, int_grid):
        ## 2D Trapezoidal rule weights
        h = int_grid[1,0,0] - int_grid[0,0,0]
        w = torch.ones((int_grid.shape[0], int_grid.shape[1])) * h*h
        w[0,0] = h*h/4
        w[0,-1] = h*h/4
        w[-1,0] = h*h/4
        w[-1,-1] = h*h/4
        w[0,1:-1] = h*h/2
        w[-1,1:-1] = h*h/2
        w[1:-1,0] = h*h/2
        w[1:-1,-1] = h*h/2
        q_weights = w.flatten()
        return q_weights

    def register_grid(self, input_grid_shape: Tuple[int, int], device):
        ## We assume input grid is regular!!!

        self.int_grids: List[torch.Tensor] = []
        self.eval_grids: List[torch.Tensor] = []
        self.q_weights: List[torch.Tensor] = []
        self.input_grids: List[torch.Tensor] = [] # Concatenated grid pairs for kernel input, shape (M, N, 2*d) where M is number of eval points and N is number of integration points

        int_grid_shape = (input_grid_shape[0], input_grid_shape[1])

        self.int_grid_shapes = []
        
        for i in range(self.depth):
            if i < self.depth // 2:
                eval_grid_shape = (int_grid_shape[0]//2, int_grid_shape[1]//2)
                self.int_grid_shapes.append(int_grid_shape)

            elif i == self.depth // 2 and self.depth % 2 == 1:
                eval_grid_shape = int_grid_shape

            else:
                eval_grid_shape = self.int_grid_shapes.pop()

            int_grid = torch.stack(torch.meshgrid(torch.linspace(0,1,int_grid_shape[0]), torch.linspace(0,1,int_grid_shape[1]), indexing='ij'), dim=-1)
            eval_grid = torch.stack(torch.meshgrid(torch.linspace(0,1,eval_grid_shape[0]), torch.linspace(0,1,eval_grid_shape[1]), indexing='ij'), dim=-1)
            q_weights = self.trapezoidal_rule_weights(int_grid)


            int_grid_shape = eval_grid_shape
            
            self.int_grids.append(int_grid.to(device))
            self.eval_grids.append(eval_grid.to(device))
            self.q_weights.append(q_weights.to(device))

            # We only need the inputs to the kernel, so we can precompute the concatenated grid pairs
            X = eval_grid.reshape(-1, eval_grid.shape[-1]).unsqueeze(1)  # shape (M, 1, d)
            Y = int_grid.reshape(-1, int_grid.shape[-1]).unsqueeze(0)  # shape (1, N, d)
            X_expanded = X.expand(-1, Y.shape[1], -1)  
            Y_expanded = Y.expand(X.shape[0], -1, -1)  
            self.input_grids.append(torch.concatenate([X_expanded, Y_expanded], dim=-1).to(device))
        
        # For debugging
        # print('Registered grid shapes for KNO_UNET_GREEN_2D:')
        # for i, grid in enumerate(self.input_grids):
        #     print(f'  Grid {i}: {grid.shape}')
        #     print(f'    Integration grid: {self.int_grids[i].shape}')
        #     print(f'    Evaluation grid: {self.eval_grids[i].shape}')
        #     print(f'    Quadrature weights: {self.q_weights[i].shape}')

    def __call__(self, 
                 f_x
                 ):
        f_x = self.lift_kernel(f_x)
        f_q = f_x


        layer_skip_connections = []
        for i in range(self.depth):
            if i <= self.depth // 2 - 1 + self.depth % 2:

                layer_skip_connections.append(f_q)
                # print(f'Added skip connection at layer {i}, shape: {f_q.shape}')

            elif i > self.depth // 2:
                skip_connection = layer_skip_connections.pop()
                # print(f'Using skip connection at layer {i}, shape: {skip_connection.shape}')
                f_q = torch.concatenate([f_q, skip_connection], dim=-1)
                f_q = self.skip_convs[i - self.depth // 2 - 1](f_q.permute(0,3,1,2)).permute(0,2,3,1)

            # Generate features (B, C, H, W) and normalized coordinates (-1 to 1)
            f_q_skip = torch.nn.functional.grid_sample(f_q.permute(0,3,1,2), (self.eval_grids[i].expand(f_q.shape[0], -1, -1, -1) * 2 - 1), mode='bilinear', align_corners=True)

            f_q_skip = self.pointwise_layers[i](f_q_skip).permute(0,2,3,1)

            # f_q = self.integration_kernels[i](x_grid.flatten(end_dim=-2), eval_grid.flatten(end_dim=-2), q_weights.flatten(), f_q.flatten(start_dim=1,end_dim=2))
            weighted_int_kernel = torch.einsum('eil,i->eil', self.integration_kernels[i](self.input_grids[i]), self.q_weights[i])
            f_q = torch.einsum('eil,bil->bel', weighted_int_kernel, f_q.flatten(start_dim=1, end_dim=2)).reshape(f_q.shape[0], self.eval_grids[i].shape[0], self.eval_grids[i].shape[1], self.lift_dim)


            f_q = f_q_skip + f_q


            if i < self.depth - 1:
                f_q = torch.nn.functional.gelu(f_q)
        

        skip_connection = layer_skip_connections.pop()
        # print(f'Using skip connection at layer {i+1}, shape: {skip_connection.shape}')
        f_q = torch.concatenate([f_q, skip_connection], dim=-1)
        f_q = self.skip_convs[-1](f_q.permute(0,3,1,2)).permute(0,2,3,1)

        f_q = torch.nn.functional.gelu(self.proj_layers[0](f_q))
        f_q = torch.nn.functional.gelu(self.proj_layers[1](f_q))
        f_q = self.proj_layers[2](f_q)
        f_y = f_q
        return f_y
    