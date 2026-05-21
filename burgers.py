import jax
import optax
from jax import numpy as jnp, random as jr
from tqdm import tqdm
from utils import *
import equinox as eqx
from kernels import *
from models import KNO_REG_GRID_1D as model

import os
os.environ["JAX_PLATFORMS"] = "metal"


import argparse
import wandb
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--batch-size', type=int, default=10)
parser.add_argument('--lr-max', type=float, default=0.001)
parser.add_argument('--lift-dim', type=int, default=64)
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--test-batch-size', type=int, default=200)
parser.add_argument('--int-kernel', type=str, default='ns_gsm', choices=['g', 'a_g','ns_g', 'gsm', 'ns_gsm', 'green'])
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--print-every', type=int, default=5)
parser.add_argument('--eval-every', type=int, default=5)
parser.add_argument('--wandb', action='store_true')

args = parser.parse_args()
print(args)

DTYPE = jnp.float32
key = jr.PRNGKey(args.seed)

### load data
fp = './datasets/burgers.npz'
data = jnp.load(fp)
dataset = fp.split('/')[-1].split('.')[0]
x, x_grid, y = data["x"].astype(DTYPE), data["x_grid"].astype(DTYPE), data["y"].astype(DTYPE)
y = y.reshape(1200, -1)
codomain_dims, domain_dims = 1,1

ntrain = 1000
ntest = 200

x_train, x_test = x[:ntrain], x[-ntest:]
y_train, y_test = y[:ntrain], y[-ntest:]

num_train_batches = len(x_train) // args.batch_size
num_steps = args.epochs * num_train_batches

## kernel setup
integration_kernel = kernels[args.int_kernel]
integration_kernel = partial(integration_kernel, ndims=1)

x_normalizer = UnitGaussianNormalizer(x_train)  
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)
y_normalizer = UnitGaussianNormalizer(y_train)

in_feats = domain_dims + codomain_dims
model = model(integration_kernel, 
              args.lift_dim, 
              args.depth,
              in_feats,
              key=key) 

lr_schedule = cosine_annealing(num_steps, peak_value=args.lr_max)
optimizer=optax.adam(lr_schedule)
opt_state = optimizer.init(eqx.filter([model], is_trainable))

### load trapezoidal quadrature rule
w = jnp.zeros_like(x_grid)
w = w.at[0].set((x_grid[1] - x_grid[0])/2)
w = w.at[-1].set((x_grid[1] - x_grid[0])/2)
q_weights = w.at[1:-1].set(x_grid[1] - x_grid[0])

param_count = sum(x.size for x in jax.tree.leaves(eqx.filter(model, is_trainable)))
print(f'{param_count=}')

@eqx.filter_jit
def train_step(model, opt_state, optimizer, batch, ):
    x,y = batch

    def loss(model):
        y_pred = eqx.filter_vmap(lambda x: model(x,
                                                x_grid,
                                                q_weights))(x)
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
        y_pred = eqx.filter_vmap(lambda x: model(x,
                                                x_grid,
                                                q_weights))(x)
        y_pred = y_normalizer.decode(y_pred)
        l2 = ((y - y_pred)**2).sum(axis=-1).mean()
        rel_l2 =(jnp.linalg.norm(y-y_pred, axis=1) / jnp.linalg.norm(y, axis=1)).mean()
        return l2, rel_l2
    
    test_l2, test_rel_l2 = loss(model)
    return test_l2, test_rel_l2

test_l2_best = 100.

if args.wandb:

    wandb.init(
        project="KNO",
        config=vars(args),
        name="BurgerKNO_" + args.int_kernel,
    )

for epoch in tqdm(range(args.epochs)):
    key,_ = jr.split(key)

    total_train_loss = 0.
    for batch_index in range(num_train_batches): 
        batch = get_batch(key, (x_train, y_train), batch_index, args.batch_size)
        model, opt_state, train_loss, rel_l2 = train_step(model, opt_state, optimizer, batch)

    if (epoch % args.print_every) == 0 or (epoch == args.epochs - 1):
        print(f'{epoch=}, train rel_l2: {rel_l2.item()*100:.3f}')
        
    if (epoch % args.eval_every) == 0 or (epoch == args.epochs - 1):
        test_l2, test_rel_l2 = eval(model, (x_test, y_test))
        print(f'test rel_l2: {test_rel_l2.item()*100:.3f}')
    
    if args.wandb:
        wandb.log({"Train L2": train_loss, "Test L2": test_l2, "Train Rel L2": rel_l2, "Test Rel L2": test_rel_l2})


if args.wandb:
    wandb.finish()

eqx.tree_serialise_leaves(f"./saved_models/BurgersKNO_{args.int_kernel}.eqx", model)
# new_model = eqx.tree_deserialise_leaves("./saved_models/model.eqx", model)

