import jax
from jax import numpy as jnp, random as jr, scipy as jsp
import equinox as eqx
from typing import Callable, List

from utils import create_lifted_module as clm


### 2d factorized model for regular grid
class KNO_DARCY_PWC_INVERTIBLE(eqx.Module):
    integration_kernels: List[eqx.Module]
    lift_kernel: eqx.Module
    depth: int
    proj_layers: eqx.Module
    pointwise_layers: List[eqx.Module]
    d: int
    lift_dim: int
    in_feats: int
    activation: Callable

    def __init__(self,
                 integration_kernel,
                 depth,
                 lift_dim,
                 ndims,
                 in_feats,
                 key,
    ):  
        
        keys = jr.split(key, 7)
        
        self.lift_dim = lift_dim
        self.d = ndims

        self.proj_layers = [eqx.nn.Linear(lift_dim, lift_dim, key=keys[0]),
                            eqx.nn.Linear(lift_dim, lift_dim, key=keys[1]),
                            eqx.nn.Linear(lift_dim, 1, key=keys[2])]
        
        self.pointwise_layers = [eqx.nn.Conv(1, lift_dim, lift_dim, 1, key=key) for key in jr.split(keys[3], depth)]

        self.lift_kernel = eqx.nn.Linear(in_feats,lift_dim,key=keys[4])
        self.integration_kernels = [(clm(integration_kernel, lift_dim, k1), 
                                     clm(integration_kernel, lift_dim, k2)) for k in jr.split(keys[5],depth) for k1,k2 in [jr.split(k, ndims)]]

        self.in_feats = in_feats
        self.depth = depth
        self.activation = jax.nn.gelu
    def forward(self, 
                 f_x, ### input fn, note no batch dim 
                 x_grid, 
                 q_weights,
                 ):

        def integration_transform(int_kernel,
                q, ### quad nodes
                w,     ### quad weights
                f_q):
            G1 = int_kernel[0](q,q) * w.T
            G2 = int_kernel[1](q,q) * w.T
            f_q = (G1 + G2) @ f_q
            # print(q.shape, G1.shape, w.shape, f_q.shape, G2.shape)
            # (29,) (29, 29) (29, 1) (29, 29) (29, 29)
            return f_q
        
        q_nodes = x_grid[:,0,0] ## grab 1d x grid

        f_x = jnp.concatenate((f_x,x_grid), axis=-1) 
        print(f_x.shape)
        f_x = f_x.reshape(-1,self.in_feats)
        print(f_x.shape)
        f_x = eqx.filter_vmap(self.lift_kernel)(f_x)
        f_x = f_x.reshape(len(q_nodes), len(q_nodes), self.lift_dim).transpose(2,0,1)
        print(f_x.shape)
        f_q = f_x

        for i in range(self.depth-1):

            f_q_skip = self.pointwise_layers[i](f_q.reshape(self.lift_dim, -1))
            f_q_skip = f_q_skip.reshape(f_q.shape)

            f_q = eqx.filter_vmap(lambda int_kernel, f: integration_transform(int_kernel,q_nodes,q_weights,f), 
                                 in_axes=(eqx.if_array(0),0), 
                                 out_axes=0)(self.integration_kernels[i],
                                             f_q)
            f_q = f_q_skip + f_q
            f_q = self.activation(f_q)

        f_q_skip = self.pointwise_layers[-1](f_q.reshape(self.lift_dim, -1))
        f_q_skip = f_q_skip.reshape(f_q.shape)

        f_q = eqx.filter_vmap(lambda int_kernel, f: integration_transform(int_kernel,q_nodes,q_weights,f), 
                             in_axes=(eqx.if_array(0),0), 
                             out_axes=0)(self.integration_kernels[-1],
                                         f_q)
        f_q = f_q + f_q_skip

        f_q = f_q.transpose(1,2,0).reshape(-1,self.lift_dim)
        f_q = self.activation(eqx.filter_vmap(self.proj_layers[0])(f_q))
        f_q = self.activation(eqx.filter_vmap(self.proj_layers[1])(f_q))
        f_q = eqx.filter_vmap(self.proj_layers[2])(f_q)
        f_y = f_q
        return f_y
    
    def __call__(self, 
                 f_x, ### input fn, note no batch dim 
                 x_grid, 
                 q_weights):
        

        q_nodes = x_grid[:,0,0] ## grab 1d x grid

        # Assume for now self.lift kernel is linear and invertible, so we can directly apply the inverse to f_x to get f_q  
        # f_q = inverse(self.proj_layers)(f_x)
        f_x = jnp.concatenate((f_x,x_grid), axis=-1) 
        f_x = f_x.reshape(-1,self.in_feats)
        f_x = eqx.filter_vmap(self.lift_kernel)(f_x)
        f_x = f_x.reshape(len(q_nodes), len(q_nodes), self.lift_dim).transpose(2,0,1)
        f_q = f_x
        
        f_q = self._inverse_skip_connection(f_q) 

        for i in range(self.depth-1, -1, -1):
            f_q = self._inverse_activation(f_q)
            f_q = self._inverse_skip_connection(f_q, self.integration_kernels[i], self.pointwise_layers[i], q_nodes, q_weights)
        
        # Assume for now self.proj_layers are linear and invertible, so we can directly apply the inverse to f_q to get f_x
        # f_q = inverse(self.lift_kernel)(f_q)
        f_q = f_q.transpose(1,2,0).reshape(-1,self.lift_dim)
        f_q = self.activation(eqx.filter_vmap(self.proj_layers[0])(f_q))
        f_q = self.activation(eqx.filter_vmap(self.proj_layers[1])(f_q))
        f_q = eqx.filter_vmap(self.proj_layers[2])(f_q)
        f_y = f_q

        return f_y
    
    def _inverse_activation(self, f_q):
        if self.activation == jax.nn.softplus:
            return jnp.log(jnp.exp(f_q) - 1)
        else:
            raise NotImplementedError("Inverse activation not implemented for this activation function.")

    
    def _inverse_skip_connection(self, int_kernel,
                                 pointwise_layer,
                q_nodes,
                q_weights,
                f_q):
        '''
        Inverse of skip connection f_q = f_q_skip + f_q,
            f_q_skip = self.pointwise_layers[i](f_q.reshape(self.lift_dim, -1))
            f_q_new = eqx.filter_vmap(lambda int_kernel, f: integration_transform(int_kernel,q_nodes,q_weights,f), 
                                 in_axes=(eqx.if_array(0),0), 
                                 out_axes=0)(self.integration_kernels[i],
                                             f_q)

            Rewritten:
            ConvMatrix = (out_dim, in_dim, 1)
            ConvBias = (out_dim, 1)
            f_q_skip = ConvMatrix @ f_q.reshape(out_dim, -1) + ConvBias
            IntKernels = (in_dim==out_dim, q_len, q_len) # Assume Diagonal Kernel Matrix
            f_q = int_kernel @ f_q
            f_q_new = (int_kernel + ConvMatrix) @ f_q + ConvBias
            We can solve for f_q:
            f_q = (int_kernel + ConvMatrix)^(-1) @ (f_q_new - ConvBias)
        '''

        conv_matrix = pointwise_layer.weight.squeeze(-1)
        conv_bias = pointwise_layer.bias.reshape(-1,1)

        def inverse_int_kernel(int_kernel,
                q, ### quad nodes
                w,     ### quad weights
        ):
            G1 = int_kernel[0](q,q) * w.T
            G2 = int_kernel[1](q,q) * w.T
            return G1 + G2
            

        int_kernel = eqx.filter_vmap(lambda int_kernel: inverse_int_kernel(int_kernel,q_nodes,q_weights), 
                             in_axes=(eqx.if_array(0)), 
                             out_axes=0)(self.integration_kernels[-1]) # (in_dim == out_dim, q_len, q_len)
        
        A = int_kernel.unsqueeze(0).expand(conv_matrix.shape[0], -1, -1, -1) + conv_matrix.unsqueeze(-1).reshape(conv_matrix.shape[0], conv_matrix.shape[1], int_kernel.shape[-2], int_kernel.shape[-1])
        b = f_q.reshape(f_q.shape[0], -1) - conv_bias.reshape(-1, f_q.shape[1]*f_q.shape[2]) 
        inverse_f_q = eqx.filter_vmap(lambda A, b: jsp.linalg.solve(A, b), 
                                      in_axes=(0, 0), 
                                      out_axes=0)(A, b)
        
        inverse_f_q = inverse_f_q.reshape(f_q.shape)
        return inverse_f_q

