import jax
from jax import numpy as jnp, random as jr
import optax
from utils import *
from kernels import *
import equinox as eqx
from models import KNO_DARCY_PWC as model
import argparse

import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10_000)
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--lr-max', type=float, default=0.001)
parser.add_argument('--lift-dim', type=int, default=64)
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--test-batch-size', type=int, default=200)
parser.add_argument('--int-kernel', type=str, default='ns_gsm', choices=['g', 'a_g','ns_g', 'gsm', 'ns_gsm'])
parser.add_argument('--seed', type=int, default=4)
parser.add_argument('--print-every', type=int, default=1)
parser.add_argument('--eval-every', type=int, default=5)

args = parser.parse_args()
print(args)

DTYPE = jnp.float32
key = jr.PRNGKey(args.seed)

### load data
data = jnp.load('./datasets/darcy_pwc.npz')
x, y = data["x"].astype(DTYPE),data["y"].astype(DTYPE)
res_1d = 29
domain_dims = 2
codomain_dims = 1
y = y.reshape(1200, -1)
x = x.reshape(1200, res_1d, res_1d, 1)

x_grid_1d = jnp.linspace(0,1,29)
x_grid = jnp.asarray(jnp.meshgrid(x_grid_1d, x_grid_1d, indexing='ij')).transpose(1,2,0).astype(DTYPE)

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

in_feats = codomain_dims + domain_dims
model = model(integration_kernel, 
              args.depth,
              args.lift_dim, 
              domain_dims,
              in_feats,
              key=key) 

lr_schedule = cosine_annealing(num_steps, peak_value=args.lr_max)
optimizer=optax.adam(lr_schedule)
opt_state = optimizer.init(eqx.filter([model], is_trainable))

### trapezoidal quad rule for res_1d, which is same as grid the function is on
h = x_grid[1,0,0] - x_grid[0,0,0] 
w = jnp.ones((res_1d,)) * h
w = w.at[0].set(h/2)
w = w.at[-1].set(h/2)
q_weights = w.reshape(-1,1)

param_count = sum(x.size for x in jax.tree.leaves(eqx.filter(model, is_trainable)))
print(f'{param_count=}')

@eqx.filter_jit
def train_step(model, opt_state, optimizer, batch, ):
    x,y = batch

    def loss(model):
        y_pred = eqx.filter_vmap(lambda x: model(x,
                                                x_grid,
                                                q_weights))(x)
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
    x,y = batch
    def loss(model):
        y_pred = jax.lax.map(lambda x: model(x, x_grid, q_weights),x, batch_size=args.test_batch_size) 
        y_pred = y_pred.reshape(ntest,-1)
        y_pred = y_normalizer.decode(y_pred)
        test_l2 = ((y - y_pred)**2).sum(axis=-1).mean()
        rel_l2 =  (jnp.linalg.norm(y-y_pred, axis=1) / jnp.linalg.norm(y, axis=1)).mean()
        return test_l2, rel_l2
    
    test_l2, test_rel_l2 = loss(model)
    return test_l2, test_rel_l2


wandb.init(
    project="KNO_PWC",
    config=vars(args),
    name="DarcyPWC_KNO",
)


for epoch in range(args.epochs):
    key,_ = jr.split(key)

    for batch_index in range(num_train_batches): 
        batch = get_batch(key, (x_train, y_train), batch_index, args.batch_size)
        model, opt_state, train_loss, rel_l2 = train_step(model, opt_state, optimizer, batch)

    if (epoch % args.print_every) == 0 or (epoch == args.epochs - 1):
        print(f'{epoch=}, train rel_l2: {rel_l2.item()*100:.3f}')
        
    if (epoch % args.eval_every) == 0 or (epoch == args.epochs - 1):
        test_l2, test_rel_l2 = eval(model, (x_test, y_test))
        print(f'test rel_l2: {test_rel_l2.item()*100:.3f}')

    wandb.log({"Train L2": train_loss, "Test L2": test_l2, "Train Rel L2": rel_l2, "Test Rel L2": test_rel_l2})

wandb.finish()

eqx.tree_serialise_leaves("./saved_models/DarcyPWC_KNO.eqx", model)
