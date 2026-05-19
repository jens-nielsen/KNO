import jax
from jax import numpy as jnp, random as jr
import equinox as eqx
from typing import List, Callable
from utils import create_lifted_module as clm


class KNO_REG_GRID_1D(eqx.Module):
    integration_kernels: List[eqx.Module]
    proj_layers: List[eqx.Module]
    pointwise_layers: List[eqx.Module]
    lift_kernel: eqx.Module
    lift_dim: int
    depth: int
    activation: Callable

    def __init__(self, integration_kernel, lift_dim, depth, in_feats, *, key):

        keys = jr.split(key,2)
        self.integration_kernels = [clm(integration_kernel, lift_dim=lift_dim, key=k) for k in jr.split(keys[0], depth)]
        self.pointwise_layers = [eqx.nn.Conv(1, lift_dim, lift_dim, 1, key=key) for key in jr.split(keys[1], depth)]

        keys = jr.split(keys[0],4)
        self.proj_layers = [eqx.nn.Linear(lift_dim, lift_dim, key=keys[0]), 
                            eqx.nn.Linear(lift_dim, lift_dim, key=keys[1]), 
                            eqx.nn.Linear(lift_dim, 1, key=keys[2])]
        self.lift_kernel = eqx.nn.Linear(in_feats, lift_dim, key=keys[3])
        
        self.activation = jax.nn.gelu
        self.lift_dim = lift_dim
        self.depth = depth

    def __call__(self, 
                 f_x, ### input fn, note no batch dim 
                 x_grid, 
                 q_weights,
                 ):

        def integration_transform(int_kernel,
                q_nodes, ### quad nodes
                q_weights,     ### quad weights
                f_q):
            
            G = (int_kernel(q_nodes,q_nodes)) * q_weights.T
            f_q = jnp.einsum('q,kq->k',f_q, G)
            return f_q
        
        q_nodes = x_grid # (N, 1)
        f_q = f_x ### already at quad nodes
        f_q = jnp.concatenate((f_q,q_nodes), axis=-1) 
        f_q = eqx.filter_vmap(self.lift_kernel)(f_q)
        f_q = self.activation(f_q)
        
        # f_q shape is (N, lift_dim) where N is number of quad nodes, we want to keep this ordering for conv layers but need to move lift_dim to end for integration transform

        for i in range(self.depth-1):
            # q_weights is (N, 1)
            f_q_skip = self.pointwise_layers[i](f_q.T).T
            f_q = eqx.filter_vmap(lambda int_kernel, f: integration_transform(int_kernel,q_nodes,q_weights,f), 
                                     in_axes=(eqx.if_array(0),1), out_axes=1)(self.integration_kernels[i], 
                                                                              f_q)         
                                                                                                
            f_q = f_q_skip + f_q
            f_q = self.activation(f_q)
        
        f_q_skip = self.pointwise_layers[-1](f_q.T).T
        f_q = eqx.filter_vmap(lambda int_kernel, f: integration_transform(int_kernel,q_nodes,q_weights,f), 
                             in_axes=(eqx.if_array(0),1), out_axes=1)(self.integration_kernels[-1],
                                                                      f_q)
        f_q = f_q_skip + f_q
                
        f_q = self.activation(eqx.filter_vmap(self.proj_layers[0])(f_q))
        f_q = self.activation(eqx.filter_vmap(self.proj_layers[1])(f_q))
        f_q = eqx.filter_vmap(self.proj_layers[2])(f_q)
        f_q = f_q.squeeze()
        return f_q
    
### 3d non-factorized model with interpolant on the backend
class KNO_DIFFUSION_REACTION(eqx.Module):
    output_kernel: eqx.Module
    integration_kernels: List[eqx.Module]
    proj_layers: List[eqx.Module]
    pointwise_layers: List[eqx.Module]
    lift_kernel: eqx.Module
    lift_dim: int
    depth: int
    activation: Callable

    def __init__(self, output_kernel, integration_kernel, lift_dim, depth, in_feats, *, key):

        keys = jr.split(key)
        self.integration_kernels = [clm(integration_kernel, lift_dim=lift_dim, key=k) for k in jr.split(keys[0], depth)]
        self.pointwise_layers = [eqx.nn.Conv(1, lift_dim, lift_dim, 1, key=key) for key in jr.split(keys[1], depth)]

        keys = jr.split(keys[0],4)
        self.proj_layers = [eqx.nn.Linear(lift_dim, lift_dim, key=keys[0]), 
                            eqx.nn.Linear(lift_dim, lift_dim, key=keys[1]), 
                            eqx.nn.Linear(lift_dim, 1, key=keys[2])]
        
        self.lift_kernel = eqx.nn.Linear(in_feats, lift_dim, key=keys[3])
        
        keys = jr.split(keys[0])
        self.output_kernel = output_kernel(key=keys[0])

        self.activation = jax.nn.gelu
        self.lift_dim = lift_dim
        self.depth = depth

    def __call__(self, 
                 f_x, ### input fn, note no batch dim 
                 y_grid,
                 q_nodes,
                 q_weights,
                 ):

        def integration_transform(int_kernel,
                q_nodes, ### quad nodes
                q_weights,     ### quad weights
                f_q):
            G = (int_kernel(q_nodes,q_nodes)) * q_weights.T
            f_q = jnp.einsum('q,kq->k',f_q, G)
            return f_q
        
        f_x = jnp.concatenate((f_x,q_nodes), axis=-1) 
        f_q = eqx.filter_vmap(self.lift_kernel)(f_x)
        f_q = self.activation(f_q)

        for i in range(self.depth-1):

            f_q_skip = self.pointwise_layers[i](f_q.T).T
            f_q = eqx.filter_vmap(lambda int_kernel, f: integration_transform(int_kernel,q_nodes,q_weights,f), 
                                 in_axes=(eqx.if_array(0),1), 
                                 out_axes=1)(self.integration_kernels[i],
                                             f_q)
                                                                                                               
            f_q = f_q_skip + f_q
            f_q = self.activation(f_q)
        
        f_q_skip = self.pointwise_layers[-1](f_q.T).T
        f_q = eqx.filter_vmap(lambda int_kernel, f: integration_transform(int_kernel,q_nodes,q_weights,f), 
                             in_axes=(eqx.if_array(0),1), 
                             out_axes=1)(self.integration_kernels[-1],
                                         f_q)
        f_q = f_q_skip + f_q
                
        f_q = self.activation(eqx.filter_vmap(self.proj_layers[0])(f_q))
        f_q = self.activation(eqx.filter_vmap(self.proj_layers[1])(f_q))
        f_q = eqx.filter_vmap(self.proj_layers[2])(f_q)
        f_q = f_q.reshape(len(q_nodes),1)

        ### move to grid
        Kqq = self.output_kernel(q_nodes,q_nodes) + (jnp.eye(len(q_nodes)) * 1e-5)
        Kqy = self.output_kernel(q_nodes, y_grid)
        KyqKqqInv = jnp.linalg.solve(Kqq, Kqy).T
        f_y = jnp.einsum('mc,qm->qc', f_q,  KyqKqqInv) 

        return f_y
    

### 2d factorized model for regular grid
class KNO_DARCY_PWC(eqx.Module):
    integration_kernels: List[eqx.Module]
    lift_kernel: eqx.Module
    depth: int
    proj_layers: eqx.Module
    pointwise_layers: List[eqx.Module]
    d: int
    lift_dim: int
    in_feats: int

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

    def __call__(self, 
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
            f_q = (G1 @ f_q) + (f_q @ G2.T)
            return f_q
        
        q_nodes = x_grid[:,0,0] ## grab 1d x grid

        f_x = jnp.concatenate((f_x,x_grid), axis=-1) 
        f_x = f_x.reshape(-1,self.in_feats)
        f_x = eqx.filter_vmap(self.lift_kernel)(f_x)
        f_x = f_x.reshape(len(q_nodes), len(q_nodes), self.lift_dim).transpose(2,0,1)
        f_q = f_x

        for i in range(self.depth-1):

            f_q_skip = self.pointwise_layers[i](f_q.reshape(self.lift_dim, -1))
            f_q_skip = f_q_skip.reshape(f_q.shape)

            f_q = eqx.filter_vmap(lambda int_kernel, f: integration_transform(int_kernel,q_nodes,q_weights,f), 
                                 in_axes=(eqx.if_array(0),0), 
                                 out_axes=0)(self.integration_kernels[i],
                                             f_q)
            f_q = f_q_skip + f_q
            f_q = jax.nn.gelu(f_q)

        f_q_skip = self.pointwise_layers[-1](f_q.reshape(self.lift_dim, -1))
        f_q_skip = f_q_skip.reshape(f_q.shape)

        f_q = eqx.filter_vmap(lambda int_kernel, f: integration_transform(int_kernel,q_nodes,q_weights,f), 
                             in_axes=(eqx.if_array(0),0), 
                             out_axes=0)(self.integration_kernels[-1],
                                         f_q)
        f_q = f_q + f_q_skip

        f_q = f_q.transpose(1,2,0).reshape(-1,self.lift_dim)
        f_q = jax.nn.gelu(eqx.filter_vmap(self.proj_layers[0])(f_q))
        f_q = jax.nn.gelu(eqx.filter_vmap(self.proj_layers[1])(f_q))
        f_q = eqx.filter_vmap(self.proj_layers[2])(f_q)
        f_y = f_q
        return f_y
    
### 2D non-factorized model with trainable interpolant on both ends
class KNO_DARCY_TRIANGLE(eqx.Module):
    input_kernel: eqx.Module
    output_kernel: eqx.Module
    integration_kernels: List[eqx.Module]
    proj_layers: List[eqx.Module]
    pointwise_layers: List[eqx.Module]
    lift_kernel: eqx.Module
    lift_dim: int
    depth: int
    activation: Callable

    def __init__(self, input_kernel, output_kernel, integration_kernel, lift_dim, depth, in_feats, *, key):

        keys = jr.split(key,2)
        self.integration_kernels = [clm(integration_kernel, lift_dim=lift_dim, key=k) for k in jr.split(keys[0], depth)]
        self.pointwise_layers = [eqx.nn.Conv(1, lift_dim, lift_dim, 1, key=key) for key in jr.split(keys[1], depth)]

        keys = jr.split(keys[0],4)
        self.proj_layers = [eqx.nn.Linear(lift_dim, lift_dim, key=keys[0]), 
                            eqx.nn.Linear(lift_dim, lift_dim, key=keys[1]), 
                            eqx.nn.Linear(lift_dim, 1, key=keys[2])]
        self.lift_kernel = eqx.nn.Linear(in_feats, lift_dim, key=keys[3])
        
        keys = jr.split(keys[0], 2)
        self.input_kernel = input_kernel(key=keys[0])
        self.output_kernel = output_kernel(key=keys[1])

        self.activation = jax.nn.gelu
        self.lift_dim = lift_dim
        self.depth = depth

    def __call__(self, 
                 f_x, ### input fn, note no batch dim 
                 x_grid, 
                 y_grid,
                 q_nodes,
                 q_weights,
                 ):

        def integration_transform(int_kernel,
                q_nodes, ### quad nodes
                q_weights,     ### quad weights
                f_q):
            G = (int_kernel(q_nodes,q_nodes)) * q_weights.T
            f_q = jnp.einsum('q,kq->k',f_q, G)
            return f_q
        
        f_x = jnp.concatenate((f_x,x_grid), axis=-1) 
        f_x = eqx.filter_vmap(self.lift_kernel)(f_x)
        f_x = f_x.reshape(len(x_grid),self.lift_dim)

        Kxx = self.input_kernel(x_grid, x_grid) + (jnp.eye(len(x_grid)) * 1e-5)
        Kxq = self.input_kernel(x_grid, q_nodes)
        KqxKinv = jnp.linalg.solve(Kxx, Kxq).T
        f_q = jnp.einsum('mc,qm->qc', f_x, KqxKinv) 

        f_q = self.activation(f_q)
        
        for i in range(self.depth-1):
            f_q_skip = self.pointwise_layers[i](f_q.T).T
            f_q = eqx.filter_vmap(lambda int_kernel, f: integration_transform(int_kernel,q_nodes,q_weights,f), 
                                 in_axes=(eqx.if_array(0),1), out_axes=1)(self.integration_kernels[i],
                                                                          f_q)
                                                                                                               
            f_q = f_q_skip + f_q
            f_q = self.activation(f_q)
        
        f_q_skip = self.pointwise_layers[-1](f_q.T).T

        f_q = eqx.filter_vmap(lambda int_kernel, f: integration_transform(int_kernel,q_nodes,q_weights,f), 
                             in_axes=(eqx.if_array(0),1), out_axes=1)(self.integration_kernels[-1],
                                                                      f_q)
        f_q = f_q_skip + f_q
  
        f_q = self.activation(eqx.filter_vmap(self.proj_layers[0])(f_q))
        f_q = self.activation(eqx.filter_vmap(self.proj_layers[1])(f_q))
        f_q = eqx.filter_vmap(self.proj_layers[2])(f_q)

        Iqq = jnp.eye(len(q_nodes)) * 1e-5
        Kqq = self.output_kernel(q_nodes,q_nodes) + Iqq
        Kqy = self.output_kernel(q_nodes, y_grid)
        KyqKqqInv = jnp.linalg.solve(Kqq, Kqy).T
        f_y = jnp.einsum('mc,qm->qc', f_q,  KyqKqqInv) 

        return f_y
    
### 2D factorized model where each dim has a slightly different quad rule
class KNO_NS_PIPE(eqx.Module):
    integration_kernels: List[eqx.Module]
    lift_kernel: eqx.Module
    depth: int
    proj_layers: eqx.Module
    pointwise_layers: List[eqx.Module]
    d: int
    lift_dim: int
    res_1d: int
    activation: Callable

    def __init__(self,
                 integration_kernel,
                 depth,
                 lift_dim,
                 ndims,
                 in_feats,
                 res_1d,
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

        self.integration_kernels = [(clm(integration_kernel, lift_dim, k1), clm(integration_kernel, lift_dim, k2)) for k in jr.split(keys[5],depth) for k1,k2 in [jr.split(k, ndims)]]

        self.depth = depth
        self.res_1d = res_1d
        self.activation = jax.nn.gelu

    def __call__(self,q,wx,wy):

        grid_1d_y = q[0, :, 1]
        grid_1d_x = q[:, 0, 0]

        def integration_transform(int_kernel,
                f_q):

            G1 = int_kernel[0](grid_1d_x,grid_1d_x) * wx.T
            G2 = int_kernel[1](grid_1d_y,grid_1d_y) * wy.T
            f_q = jnp.einsum('ij,ki->kj',f_q, G1) +  jnp.einsum('ij,kj->ik',f_q, G2)
            return f_q


        q = q.reshape(-1,2)
        f_x = eqx.filter_vmap(self.lift_kernel)(q)
        f_x = f_x.reshape(self.res_1d,self.res_1d,self.lift_dim)
        f_q = f_x
        for i in range(self.depth-1):

            f_q_skip = self.pointwise_layers[i](f_q.reshape(-1,self.lift_dim).T).T
            f_q_skip = f_q_skip.reshape(f_q.shape)

            f_q = eqx.filter_vmap(lambda int_kernel, f: integration_transform(int_kernel,f), in_axes=(eqx.if_array(0),self.d), out_axes=self.d)(self.integration_kernels[i],
                                                                                                                              f_q)
            f_q = f_q_skip + f_q
            f_q = self.activation(f_q)

        f_q_skip = self.pointwise_layers[-1](f_q.reshape(-1,self.lift_dim).T).T
        f_q_skip = f_q_skip.reshape(f_q.shape)
        f_q = eqx.filter_vmap(lambda int_kernel, f: integration_transform(int_kernel,f), in_axes=(eqx.if_array(0),self.d), out_axes=self.d)(self.integration_kernels[i+1],f_q)
        f_q = f_q + f_q_skip

        f_q = f_q.reshape(-1,self.lift_dim)
        f_q = self.activation(eqx.filter_vmap(self.proj_layers[0])(f_q))
        f_q = self.activation(eqx.filter_vmap(self.proj_layers[1])(f_q))
        f_y = eqx.filter_vmap(self.proj_layers[2])(f_q)
        return f_y
    
### 3D factorized model 
class KNO_NS_3D(eqx.Module):
    integration_kernels: List[eqx.Module]
    lift_kernel: eqx.Module
    depth: int
    proj_layers: eqx.Module
    pointwise_layers: List[eqx.Module]
    d: int
    lift_dim: int
    in_feats: int
    res_1d: int
    activation: Callable

    def __init__(self,
                 integration_kernel,
                 depth,
                 lift_dim,
                 ndims,
                 in_feats,
                 res_1d, 
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
        self.integration_kernels = [(clm(integration_kernel, lift_dim, k1), clm(integration_kernel, lift_dim, k2), 
                                     clm(integration_kernel, lift_dim, k3)) for k in jr.split(keys[5],depth) for k1,k2,k3 in [jr.split(k, ndims)]]

        self.in_feats = in_feats
        self.depth = depth
        self.res_1d = res_1d
        self.activation = jax.nn.gelu

    def __call__(self, 
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
            G3 = int_kernel[2](q,q) * w.T
            f_q = jnp.einsum('ijk,li->ljk', f_q, G1) \
                    + jnp.einsum('ijk,lj->ilk', f_q, G2) \
                    + jnp.einsum('ijk,lk->ijl', f_q, G3)
            return f_q
        
        q_nodes = x_grid[:,0,0,1]
        f_x = jnp.concatenate((f_x,x_grid), axis=-1) 
        f_x = f_x.reshape(-1,self.in_feats)
        f_x = eqx.filter_vmap(self.lift_kernel)(f_x)
        f_q = f_x.reshape(len(q_nodes), len(q_nodes), len(q_nodes), self.lift_dim)

        for i in range(self.depth-1):
            f_q_skip = self.pointwise_layers[i](f_q.reshape(-1,self.lift_dim).T).T
            f_q_skip = f_q_skip.reshape(f_q.shape)

            f_q = eqx.filter_vmap(lambda int_kernel, f: integration_transform(int_kernel,q_nodes,q_weights,f), 
                                 in_axes=(eqx.if_array(0),self.d), out_axes=self.d)(self.integration_kernels[i],
                                                                                    f_q)
            f_q = f_q_skip + f_q
            f_q = self.activation(f_q)

        f_q_skip = self.pointwise_layers[-1](f_q.reshape(-1,self.lift_dim).T).T
        f_q_skip = f_q_skip.reshape(f_q.shape)

        f_q = eqx.filter_vmap(lambda int_kernel, f: integration_transform(int_kernel,q_nodes,q_weights,f), 
                             in_axes=(eqx.if_array(0),self.d), out_axes=self.d)(self.integration_kernels[-1],
                                                                                f_q)
        f_q = f_q + f_q_skip

        f_q = f_q.reshape(-1,self.lift_dim)
        f_q = self.activation(eqx.filter_vmap(self.proj_layers[0])(f_q))
        f_q = self.activation(eqx.filter_vmap(self.proj_layers[1])(f_q))
        f_y = eqx.filter_vmap(self.proj_layers[2])(f_q)
        return f_y