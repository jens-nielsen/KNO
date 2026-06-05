import jax
from jax import numpy as jnp, random as jr, scipy as jsp
import equinox as eqx
from typing import Callable, List

from utils import create_lifted_module as clm, partial

from kernels import kernels

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_sylvester

@jax.custom_vjp
def differentiable_sylvester(A, B, C):
    """Solves AX + XB = C stably and supports JAX differentiation."""
    return jsp.linalg.solve_sylvester(A, B, C, method='schur')

def differentiable_sylvester_fwd(A, B, C):
    X = differentiable_sylvester(A, B, C)
    # Save inputs and outputs needed for the backward pass
    return X, (A, B, X)

def differentiable_sylvester_bwd(res, g):
    A, B, X = res
    # g is the incoming gradient (cotangent) matrix w.r.t X
    
    # The adjoint equation is: A.T @ Y + Y @ B.T = g
    # We use .swapaxes(-1, -2) instead of .T to cleanly support batching (vmap)
    Y = jsp.linalg.solve_sylvester(A.swapaxes(-1, -2), B.swapaxes(-1, -2), g, method='schur')
    
    # Analytical gradients derived via the Implicit Function Theorem
    grad_A = -Y @ X.swapaxes(-1, -2)
    grad_B = -X.swapaxes(-1, -2) @ Y
    grad_C = Y
    
    return grad_A, grad_B, grad_C

# Register the forward and backward paths with JAX
differentiable_sylvester.defvjp(differentiable_sylvester_fwd, differentiable_sylvester_bwd)

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
    inverse_proj_layers: List[eqx.Module]
    inverse_lift_kernel: eqx.Module
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
        self.activation = jax.nn.softplus

        # Placeholders for inverse layers, will be set up after initialization
        self.inverse_setup(lift_dim, in_feats, key)
        
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
            f_q = (G1 + G2) @ f_q
            # print(q.shape, G1.shape, w.shape, f_q.shape, G2.shape)
            # (29,) (29, 29) (29, 1) (29, 29) (29, 29)
            return f_q
        
        q_nodes = x_grid[:,0,0] ## grab 1d x grid

        f_x = jnp.concatenate((f_x,x_grid), axis=-1) 
        f_x = f_x.reshape(-1,self.in_feats)
        f_x = eqx.filter_vmap(self.lift_kernel)(f_x)
        f_x = f_x.reshape(len(q_nodes), len(q_nodes), self.lift_dim).transpose(2,0,1)
        f_q = f_x

        for i in range(self.depth-1):

            # f_q_skip = self.pointwise_layers[i](f_q.reshape(self.lift_dim, -1))
            # f_q_skip = f_q_skip.reshape(f_q.shape)

            f_q = eqx.filter_vmap(lambda int_kernel, f: integration_transform(int_kernel,q_nodes,q_weights,f), 
                                 in_axes=(eqx.if_array(0),0), 
                                 out_axes=0)(self.integration_kernels[i],
                                             f_q)
            # f_q = f_q_skip + f_q
            f_q = self.activation(f_q)

        # f_q_skip = self.pointwise_layers[-1](f_q.reshape(self.lift_dim, -1))
        # f_q_skip = f_q_skip.reshape(f_q.shape)

        f_q = eqx.filter_vmap(lambda int_kernel, f: integration_transform(int_kernel,q_nodes,q_weights,f), 
                             in_axes=(eqx.if_array(0),0), 
                             out_axes=0)(self.integration_kernels[-1],
                                         f_q)
        # f_q = f_q + f_q_skip

        f_q = f_q.transpose(1,2,0).reshape(-1,self.lift_dim)
        f_q = self.activation(eqx.filter_vmap(self.proj_layers[0])(f_q))
        f_q = self.activation(eqx.filter_vmap(self.proj_layers[1])(f_q))
        f_q = eqx.filter_vmap(self.proj_layers[2])(f_q)
        f_y = f_q
        return f_y
    
    def inverse_call(self, 
                 f_x, ### input fn, note no batch dim 
                 x_grid, 
                 q_weights):
        

        q_nodes = x_grid[:,0,0] ## grab 1d x grid

        # Assume for now self.lift kernel is linear and invertible, so we can directly apply the inverse to f_x to get f_q  
        # f_q = inverse(self.proj_layers)(f_x)
        # f_x = jnp.concatenate((f_x,x_grid), axis=-1) 
        f_x = f_x.reshape(-1,1)
        f_x = self._inverse_activation(eqx.filter_vmap(self.inverse_proj_layers[2])(f_x-self.proj_layers[2].bias))
        f_x = self._inverse_activation(eqx.filter_vmap(self.inverse_proj_layers[1])(f_x-self.proj_layers[1].bias))
        f_x = eqx.filter_vmap(self.inverse_proj_layers[0])(f_x - self.proj_layers[0].bias)
        f_x = f_x.reshape(len(q_nodes), len(q_nodes), self.lift_dim).transpose(2,0,1)
        f_q = f_x
        
        f_q = self._inverse_skip_connection(self.integration_kernels[-1],
                                            self.pointwise_layers[-1],
                                            q_nodes,
                                            q_weights,
                                            f_q) 

        for i in range(self.depth-2, -1, -1):
            f_q = self._inverse_activation(f_q)
            f_q = self._inverse_skip_connection(self.integration_kernels[i], self.pointwise_layers[i], q_nodes, q_weights,f_q)
        
        # Assume for now self.proj_layers are linear and invertible, so we can directly apply the inverse to f_q to get f_x
        # f_q = inverse(self.lift_kernel)(f_q)
        f_q = f_q.transpose(1,2,0).reshape(-1,self.lift_dim)
        f_q = eqx.filter_vmap(self.inverse_lift_kernel[0])(f_q-self.lift_kernel.bias)
        f_y = f_q

        f_y = f_y.reshape(len(q_nodes), len(q_nodes), self.in_feats)

        return f_y
    
    def inverse_setup(self, lift_dim, in_feats, key):
    
        def update_weights(linear_layer, new_weight):
            # Swap the weight out-of-place (since Equinox models are immutable)
            where = lambda l: l.weight
            modified_linear = eqx.tree_at(where, linear_layer, new_weight)

        keys = jr.split(key, 7)

        self.inverse_proj_layers =[eqx.nn.Linear(lift_dim, lift_dim, key=keys[0], use_bias = False),
                            eqx.nn.Linear(lift_dim, lift_dim, key=keys[1], use_bias = False),
                            eqx.nn.Linear(1, lift_dim, key=keys[2], use_bias = False)]

        for i, layer in enumerate(self.inverse_proj_layers):
            update_weights(layer, jnp.linalg.pinv(self.proj_layers[i].weight))

        self.inverse_lift_kernel = [eqx.nn.Linear(lift_dim, in_feats, key=keys[3], use_bias = False)]
        update_weights(self.inverse_lift_kernel[0], jnp.linalg.pinv(self.lift_kernel.weight))


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

        conv_matrix = pointwise_layer.weight
        conv_bias = pointwise_layer.bias

        def calc_int_kernel(int_kernel,
                q, ### quad nodes
        ):
            G1 = int_kernel[0](q,q)
            G2 = int_kernel[1](q,q)
            return G1, G2
            

        G1, G2 = eqx.filter_vmap(lambda int_kernel: calc_int_kernel(int_kernel,q_nodes), 
                             in_axes=(eqx.if_array(0)), 
                             out_axes=0)(self.integration_kernels[-1]) # (in_dim == out_dim, q_len, q_len)
        # (29, 29)
        # print(differentiable_sylvester(G1[0], G2[0].T, f_q[0]).shape)
        # (64, 29, 29) (64, 64, 1) (64, 1) (29, 1) (64, 29, 29)
        # print(G1.shape, G2.shape, conv_matrix.shape, conv_bias.shape, q_weights.shape, f_q.shape)
        
        # We assume for now that there is no convolutional skip, i.e. ConvMatrix = 0, so we can directly apply the inverse of the integration kernel to f_q to get f_q_new

        inverse_f_q = eqx.filter_vmap(lambda G1, G2, f_q: jsp.linalg.solve_sylvester(G1, G2, f_q, method='eigen'), 
                                      in_axes=(0, 0, 0), 
                                      out_axes=0)(G1, G2, f_q) # (64, 29, 29)
        # (64, 29, 29)
        # print(inverse_f_q.shape)
        return inverse_f_q

    

if __name__ == "__main__":
    key = jr.PRNGKey(0)
    int_kernel = kernels['ns_gsm']
    int_kernel = partial(int_kernel, ndims=1, discontinuous=False)
    lift_dim = 16
    in_feats = 3
    model = KNO_DARCY_PWC_INVERTIBLE(int_kernel, 2, lift_dim, 2, in_feats, key)
    print("MODEL INITIALIZED")
    # model.inverse_setup(lift_dim, in_feats, key)
    x_grid = jnp.linspace(0,1,29)
    q_weights = jnp.ones_like(x_grid) * (1/29)
    x_grid = jnp.stack(jnp.meshgrid(x_grid, x_grid), axis=-1)
    f_x = jnp.sin(jnp.pi * x_grid[...,0:1]) * jnp.sin(jnp.pi * x_grid[...,1:2])
    print(f_x.shape, x_grid.shape, q_weights.shape)
    f_y = model(f_x, x_grid, q_weights)
    f_x_recon = model.inverse_call(f_y, x_grid, q_weights)
    print(jnp.linalg.norm(f_x - f_x_recon))

