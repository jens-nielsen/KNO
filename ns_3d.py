import jax
import optax
from jax import numpy as jnp, random as jr
import jax.random as jr
from utils import *
import equinox as eqx
from kernels import *
from models import KNO_NS_3D as model
import argparse

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch-size', type=int, default=5)
parser.add_argument('--lr-max', type=float, default=0.001)
parser.add_argument('--lift-dim', type=int, default=64)
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--test-batch-size', type=int, default=5)
parser.add_argument('--int-kernel', type=str, default='gsm', choices=['g', 'a_g','ns_g', 'gsm', 'ns_gsm'])
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--print-every', type=int, default=1)
parser.add_argument('--eval-every', type=int, default=5)

args = parser.parse_args()
print(args)
DTYPE = jnp.float32
key = jr.PRNGKey(args.seed)

DTYPE = np.float32
key = jax.random.PRNGKey(seed=args.seed)

ntrain = 90
ntest = 10

res_1d = 64
n_1d = 128
ndims = 3
data = np.load('./datasets/ns_pdebench_pres_t0_t1_mach_1.0.npz')
x = data['x'].astype(DTYPE)[...,None]
y = data['y'].astype(DTYPE)
x = x[:,::2,::2,::2] 
y = y[:,::2,::2,::2].reshape(ntrain + ntest, -1)
domain_dims = 3
codomain_dims = 1

x_grid = data['x_grid'].astype(DTYPE)
x_grid = jnp.asarray(jnp.meshgrid(x_grid, x_grid, x_grid)).transpose(1,2,3,0)
x_grid = x_grid[::2,::2,::2]

x,y = shuffle(x,y)
x_train, x_test = x[:ntrain], x[-ntest:]
y_train, y_test = y[:ntrain], y[-ntest:]

print(f'{x_train.shape=}, {x_test.shape=}, {y_train.shape=}, {y_test.shape=}')

num_train_batches = len(x_train) // args.batch_size
train_batch_size = args.batch_size

## kernel setup
integration_kernel = kernels[args.int_kernel]
integration_kernel = partial(integration_kernel, ndims=1)

### preprocess data
x_normalizer = UnitGaussianNormalizer(x_train, axis=None)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)
y_normalizer = UnitGaussianNormalizer(y_train.reshape(ntrain,-1), axis=None)

in_feats = codomain_dims + domain_dims
model = model(integration_kernel, 
            args.depth,
            args.lift_dim, 
            domain_dims,
            in_feats,
            res_1d,
            key=key) 

h = x_grid[1,0,0,1] - x_grid[0,0,0,1]
w = jnp.zeros((64,))
w = w.at[0].set(h/2)
w = w.at[-1].set(h/2)
q_weights = w.at[1:-1].set(h)

lr_schedule = cosine_annealing(args.epochs*num_train_batches, peak_value=args.lr_max)
optimizer = optax.adam(lr_schedule)
opt_state = optimizer.init(eqx.filter([model], is_trainable))

param_count = sum(x.size for x in jax.tree.leaves(eqx.filter(model, is_trainable)))
print(f'{param_count=}')

@eqx.filter_jit
def train_step(model, opt_state, optimizer, batch, ):
    x,y = batch

    def loss(model):
        y_pred = eqx.filter_vmap(lambda x: model(x,
                                                x_grid,
                                                q_weights))(x)
        y_pred = y_pred.reshape(train_batch_size, -1)
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
    x,y = batch
    def loss(model):
        y_pred = jax.lax.map(lambda x: model(x,x_grid,
                                                q_weights),x, batch_size=args.test_batch_size) 
        y_pred = y_pred.reshape(ntest,-1)
        y_pred = y_normalizer.decode(y_pred)
        return (jnp.linalg.norm(y-y_pred, axis=1) / jnp.linalg.norm(y, axis=1)).mean()
    
    rel_l2 = loss(model)
    return rel_l2

for epoch in range(args.epochs):
    key,_ = jr.split(key)

    for batch_index in range(num_train_batches): 
        batch = get_batch(key, (x_train, y_train), batch_index, args.batch_size)
        model, opt_state, train_loss, rel_l2 = train_step(model, opt_state, optimizer, batch)
        print(rel_l2)
    if (epoch % args.print_every) == 0 or (epoch == args.epochs - 1):
        print(f'{epoch=}, train rel_l2: {rel_l2.item()*100:.3f}')
        
    if (epoch % args.eval_every) == 0 or (epoch == args.epochs - 1):
        test_rel_l2 = eval(model, (x_test, y_test))
        print(f'test rel_l2: {test_rel_l2.item()*100:.3f}')
