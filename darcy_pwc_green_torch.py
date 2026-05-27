import torch
import optax
from utils import *
from kernels import *
import equinox as eqx
from models import KNO_DARCY_PWC_GREEN_TORCH as model_cls
import matplotlib.pyplot as plt
import argparse

import wandb
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10_000)
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--lr-max', type=float, default=0.001)
parser.add_argument('--lift-dim', type=int, default=64)
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--test-batch-size', type=int, default=200)
parser.add_argument('--int-kernel', type=str, default='green_torch', choices=['g', 'a_g','ns_g', 'gsm', 'ns_gsm', 'green', 'green_torch'])
parser.add_argument('--seed', type=int, default=4)
parser.add_argument('--print-every', type=int, default=5)
parser.add_argument('--eval-every', type=int, default=5)
parser.add_argument('--wandb', action='store_true')

args = parser.parse_args()
print(args)

DTYPE = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
# key = jr.PRNGKey(args.seed)

### load data
data = np.load('./datasets/darcy_pwc.npz')
x, y = torch.tensor(data["x"], dtype=DTYPE), torch.tensor(data["y"], dtype=DTYPE)
res_1d = 29
domain_dims = 2
codomain_dims = 1
y = y.reshape(1200, -1)
x = x.reshape(1200, res_1d, res_1d, 1)

x_grid_1d = torch.linspace(0, 1, 29, dtype=DTYPE)
x_grid = torch.stack(torch.meshgrid(x_grid_1d, x_grid_1d, indexing='ij')).permute(1, 2, 0).to(device)

ntrain = 1000
ntest = 200
x_train, x_test = x[:ntrain], x[-ntest:]
y_train, y_test = y[:ntrain], y[-ntest:]

# For visualization only
# print(f'{x_train.shape=}, {y_train.shape=}')

# y_train = y_train.reshape(ntrain, res_1d, res_1d, codomain_dims)

# fig, axes = plt.subplots(1, 2, figsize=(12, 10))
# im1 = axes[0].contourf(x_grid[:,:,0], x_grid[:,:,1], x_train[1,:,:,0], levels=20, cmap='viridis')
# plt.colorbar(im1, ax=axes[0])

# im2 = axes[1].contourf(x_grid[:,:,0], x_grid[:,:,1], y_train[1,:,:,0], levels=20, cmap='viridis')
# plt.colorbar(im2, ax=axes[1])

# plt.show()
# assert False

num_train_batches = len(x_train) // args.batch_size
num_steps = args.epochs * num_train_batches

## kernel setup
integration_kernel = kernels[args.int_kernel]
integration_kernel = partial(integration_kernel, ndims=2)

x_normalizer = UnitGaussianNormalizerTorch(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)
y_normalizer = UnitGaussianNormalizerTorch(y_train)

x_normalizer.to(device)
y_normalizer.to(device)

in_feats = codomain_dims + domain_dims
model = model_cls(integration_kernel, 
              args.depth,
              args.lift_dim, 
              domain_dims,
              in_feats, 
              device).to(device)


optimizer=torch.optim.Adam(model.parameters(), lr=args.lr_max)
lr_schedule = cosine_annealing(num_steps, peak_value=args.lr_max)
lr_scheduler = CosineAnnealingWarmupRestarts(
    optimizer=optimizer,
    total_steps=100_000,   # Example total steps
    warmup_frac=0.3,
    peak_value=3e-4,
    num_cycles=3,
    gamma=0.7,
    down=1e4
)

## evaluation grid
eval_grid_n = 30 
x_eval_grid_1d = torch.linspace(0,1,eval_grid_n, dtype=DTYPE)
x_eval_grid = torch.stack(torch.meshgrid(x_eval_grid_1d, x_eval_grid_1d, indexing='ij')).permute(1, 2, 0)
y_eval_train = torch.nn.functional.interpolate(y_train.reshape(ntrain, res_1d, res_1d, codomain_dims).permute(0,3,1,2), size=(eval_grid_n, eval_grid_n), mode='bicubic').permute(0,2,3,1)
y_eval_test = torch.nn.functional.interpolate(y_test.reshape(ntest, res_1d, res_1d, codomain_dims).permute(0,3,1,2), size=(eval_grid_n, eval_grid_n), mode='bicubic').permute(0,2,3,1)

## 2D Trapezoidal rule weights
h = x_eval_grid[1,0,0] - x_eval_grid[0,0,0]
w = torch.ones((res_1d, res_1d)) * h*h
w[0,0] = h*h/4
w[0,-1] = h*h/4
w[-1,0] = h*h/4
w[-1,-1] = h*h/4
w[0,1:-1] = h*h/2
w[-1,1:-1] = h*h/2
w[1:-1,0] = h*h/2
w[1:-1,-1] = h*h/2
q_weights = w.reshape(-1,1).to(device)

# param_count = sum(x.size for x in jax.tree.leaves(eqx.filter(model, is_trainable)))
param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'{param_count=}')

def train_step(model, optimizer, batch, device):
    x,y = batch
    x = x.to(device)
    y = y.to(device)

    def loss(model):
        y_pred = model(x, x_grid, x_grid, q_weights)
        y_pred = y_pred.reshape(args.batch_size, -1)
        y_pred = y_normalizer.decode(y_pred)
        l2 =  ((y - y_pred)**2).sum(axis=-1).mean()
        rel_l2 = (torch.linalg.norm(y-y_pred, axis=1) / torch.linalg.norm(y, axis=1)).mean()
        return l2, rel_l2
    
    train_l2, train_rel_l2 = loss(model)
    optimizer.zero_grad()
    train_l2.backward()
    optimizer.step()
    lr_scheduler.step()
    return model, train_l2, train_rel_l2

def eval(model, batch,):
    x,y = batch
    def loss(model):
        y_pred = model(x, x_grid, x_grid, q_weights)
        # y_pred = jax.lax.map(lambda x: model(x, x_grid, x_eval_grid, q_weights),x, batch_size=args.test_batch_size) 
        y_pred = y_pred.reshape(ntest,-1)
        y_pred = y_normalizer.decode(y_pred)
        test_l2 = ((y - y_pred)**2).sum(axis=-1).mean()
        rel_l2 =  (torch.linalg.norm(y-y_pred, axis=1) / torch.linalg.norm(y, axis=1)).mean()
        return test_l2, rel_l2
    
    test_l2, test_rel_l2 = loss(model)
    return test_l2, test_rel_l2


if args.wandb:

    wandb.init(
        project="KNO_PWC",
        config=vars(args),
        name="DarcyPWC_KNO_" + args.int_kernel,
    )


for epoch in tqdm(range(args.epochs)):
    for batch_index in range(num_train_batches): 
        batch = get_batch_torch(epoch, (x_train, y_train), batch_index, args.batch_size)
        model, train_l2, train_rel_l2 = train_step(model, optimizer, batch, device)

    if (epoch % args.print_every) == 0 or (epoch == args.epochs - 1):
        print(f'{epoch=}, train rel_l2: {train_rel_l2.item()*100:.3f}')
        
    if (epoch % args.eval_every) == 0 or (epoch == args.epochs - 1):
        test_l2, test_rel_l2 = eval(model, (x_test, y_test))
        print(f'test rel_l2: {test_rel_l2.item()*100:.3f}')

    assert False
    if args.wandb:
        wandb.log({"Train L2": train_l2, "Test L2": test_l2, "Train Rel L2": train_rel_l2, "Test Rel L2": test_rel_l2})

if args.wandb:
    wandb.finish()

# eqx.tree_serialise_leaves(f"./saved_models/DarcyPWC_{args.int_kernel}.eqx", model)
