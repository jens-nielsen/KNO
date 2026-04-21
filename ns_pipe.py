import jax
import optax
from jax import numpy as jnp, random as jr
import jax.random as jr
from utils import *
import equinox as eqx
from kernels import *
from models import KNO_NS_PIPE as model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--lr-max', type=float, default=0.001)
parser.add_argument('--lift-dim', type=int, default=128)
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--test-batch-size', type=int, default=1)
parser.add_argument('--int-kernel', type=str, default='ns_gsm', choices=['g', 'a_g','ns_g', 'gsm', 'ns_gsm'])
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--print-every', type=int, default=1)
parser.add_argument('--eval-every', type=int, default=5)

args = parser.parse_args()
print(args)

DTYPE = jnp.float32
key = jr.PRNGKey(args.seed)

### load data
fp = f'./datasets/ns_pipe.npz'
data = jnp.load(fp)

print(dict(data).keys())
y_grid = data['y_grid'].astype(jnp.float32).reshape(2310,-1,2)
y = data['y'].astype(jnp.float32).reshape(2310,-1)

y_mu, y_std = jnp.mean(y_grid, axis=(0,1), keepdims=True), jnp.std(y_grid, axis=(0,1), keepdims=True)
y_grid = y_grid - y_mu
y_grid = y_grid / y_std

y_grid = y_grid.reshape(-1,129,129,2)

y_h = y_grid[0,0,1,1] - y_grid[0,0,0,1]
x_h = y_grid[0,1,0,0] - y_grid[0,0,0,0]

grid_1d_y = y_grid[:, 0, :, 1]
grid_1d_x = y_grid[:, :, 0, 0]

wx = jnp.zeros((129,))
wx = wx.at[0].set(x_h/2)
wx = wx.at[-1].set(x_h/2)
wx = wx.at[1:-1].set(x_h)
wx = wx.reshape(-1,1)

wy = jnp.zeros((129,))
wy = wy.at[0].set(y_h/2)
wy = wy.at[-1].set(y_h/2)
wy = wy.at[1:-1].set(y_h)
wy = wy.reshape(-1,1)
key,_ = jr.split(key)

q_nodes = y_grid

domain_dims = 2
codomain_dims = 0
ntrain = 1000
ntest = 200

q_train, q_test = q_nodes[: ntrain], q_nodes[-ntest:]
y_train, y_test = y[: ntrain], y[-ntest:]

### data config 
num_train_batches = len(q_train) // args.batch_size
num_steps = args.epochs * num_train_batches

## kernel setup
integration_kernel = kernels[args.int_kernel]
integration_kernel = partial(integration_kernel, ndims=1)

### preprocess data
y_normalizer = UnitGaussianNormalizer(y_train)
in_feats = domain_dims + codomain_dims
model = model(integration_kernel, 
              args.depth, 
              args.lift_dim,
              domain_dims,
              in_feats,
              129,
              key=key) 

lr_schedule = cosine_annealing(args.epochs*num_train_batches, peak_value=args.lr_max)
optimizer = optax.adam(lr_schedule)
opt_state = optimizer.init(eqx.filter([model], is_trainable))

param_count = sum(x.size for x in jax.tree.leaves(eqx.filter(model, is_trainable)))
print(f'{param_count=}')

@eqx.filter_jit
def train_step(model, opt_state, optimizer, batch, ):
    q, y = batch
    def loss(model):
        y_pred = eqx.filter_vmap(lambda q: model(q, wx, wy))(q)
        y_pred = y_pred.reshape(args.batch_size, -1)
        y_pred = y_normalizer.decode(y_pred)
        l2 =  ((y - y_pred)**2).sum(axis=-1).mean()
        rel_l2 = (jnp.linalg.norm(y-y_pred, axis=1) / jnp.linalg.norm(y, axis=1)).mean()
        return l2, rel_l2
    (train_loss,rel_l2), grads = eqx.filter_value_and_grad(loss, has_aux=True)(model)
    updates,opt_state = optimizer.update([grads], 
                                            opt_state, 
                                            eqx.filter([model], is_trainable))
    model = eqx.apply_updates(model, updates[0])
    return model, opt_state, train_loss, rel_l2

@eqx.filter_jit
def eval(model, batch,):
    q, y = batch
    def loss(model):
        y_pred = jax.lax.map(lambda xs: model(xs, wx,wy),q , batch_size=args.test_batch_size)
        y_pred = y_pred.reshape(ntest,-1)
        y_pred = y_normalizer.decode(y_pred)
        return (jnp.linalg.norm(y-y_pred, axis=1) / jnp.linalg.norm(y, axis=1)).mean()
    
    rel_l2 = loss(model)
    return rel_l2

test_l2_best = 100.
for epoch in range(args.epochs):
    epoch_key,_ = jr.split(key)

    for i in range(num_train_batches): 
        batch = get_batch(epoch_key, (q_train, y_train), i, args.batch_size)
        model, opt_state, train_loss, rel_l2 = train_step(model, opt_state, optimizer, batch)

    if (epoch % args.print_every) == 0 or (epoch == args.epochs - 1):
        print(f'{epoch=}, train rel_l2: {rel_l2.item()*100:.3f}')
        
    if (epoch % args.eval_every) == 0 or (epoch == args.epochs - 1):
        test_rel_l2 = eval(model, (q_test, y_test))
        print(f'test rel_l2: {test_rel_l2.item()*100:.3f}')
