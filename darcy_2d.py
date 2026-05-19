
import os
os.environ["JAX_PLATFORMS"] = "METAL"


import jax
from jax import numpy as jnp, random as jr
import optax
from utils import *
from kernels import *
import equinox as eqx
from tqdm import tqdm
from models import KNO_DARCY_PWC as model
import argparse
import wandb


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch-size', type=int, default=10)
parser.add_argument('--lr-max', type=float, default=0.001)
parser.add_argument('--lift-dim', type=int, default=64)
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--test-batch-size', type=int, default=10)
parser.add_argument('--int-kernel', type=str, default='ns_gsm', choices=['g', 'a_g','ns_g', 'gsm', 'ns_gsm'])
parser.add_argument('--seed', type=int, default=4)
parser.add_argument('--print-every', type=int, default=1)
parser.add_argument('--eval-every', type=int, default=5)

args = parser.parse_args()
print(args)

DTYPE = jnp.float32
key = jr.PRNGKey(args.seed)

### load data
data = jnp.load('./datasets/darcy_2d_uniform_data.npz')
x, y = data["a"].astype(DTYPE),data["u"].astype(DTYPE) 

r = 4
s = int(((421 - 1) / r) + 1) 
x = x[:, :: r, :: r][:, : s, :s]
y = y[:, :: r, :: r][:, : s, :s]

# To overcome jax issues, we need to go from (:, 105, 105) to (:, 96, 96) 
res = 96
x = x[:, :res, :res]
y = y[:, :res, :res]

domain_dims = 2
codomain_dims = 1
y = y.reshape(y.shape[0], -1)
x = x.reshape(x.shape[0], res, res, 1)

x_grid_1d = jnp.linspace(0,1,res)
x_grid = jnp.asarray(jnp.meshgrid(x_grid_1d, x_grid_1d, indexing='ij')).transpose(1,2,0).astype(DTYPE)

ntrain = 1024
ntest = 100
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
w = jnp.ones((res,)) * h
w = w.at[0].set(h/2)
w = w.at[-1].set(h/2)
q_weights = w.reshape(-1,1)

param_count = sum(x.size for x in jax.tree.leaves(eqx.filter(model, is_trainable)))
print(f'{param_count=}')

@eqx.filter_jit
def train_step(model, opt_state, optimizer, batch, ):
    x,y = batch

    def loss(model):
        y_pred = jax.lax.map(lambda x: model(x,
                                            x_grid,
                                            q_weights), x)
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
        l2 = ((y - y_pred)**2).sum(axis=-1).mean()
        rel_l2 =(jnp.linalg.norm(y-y_pred, axis=1) / jnp.linalg.norm(y, axis=1)).mean()
        return l2, rel_l2
    
    test_l2, test_rel_l2 = loss(model)
    return test_l2, test_rel_l2


wandb.init(
    project="KNO",
    config=vars(args),
    name="Darcy2DUniformKNO",
)

for epoch in tqdm(range(args.epochs)):
    key,_ = jr.split(key)

    total_train_loss = 0.
    total_rel_train_loss = 0.
    for batch_index in range(num_train_batches): 
        batch = get_batch(key, (x_train, y_train), batch_index, args.batch_size)
        model, opt_state, train_loss, rel_l2 = train_step(model, opt_state, optimizer, batch)

        total_train_loss += train_loss.item()
        total_rel_train_loss += rel_l2.item()

    if (epoch % args.print_every) == 0 or (epoch == args.epochs - 1):
        print(f'{epoch=}, train rel_l2: {rel_l2.item()*100:.3f}')
        
    if (epoch % args.eval_every) == 0 or (epoch == args.epochs - 1):
        test_l2, test_rel_l2 = eval(model, (x_test, y_test))
        print(f'test rel_l2: {test_rel_l2.item()*100:.3f}')


    wandb.log({"Train L2": total_train_loss / ntrain, "Test L2": test_l2 / ntest, "Train Rel L2": total_rel_train_loss / ntrain, "Test Rel L2": test_rel_l2 / ntest})


wandb.finish()

eqx.tree_serialise_leaves("./saved_models/darcy2d_uniform_model.eqx", model)
# new_model = eqx.tree_deserialise_leaves("./saved_models/model.eqx", model)

