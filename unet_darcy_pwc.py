from functools import partial

import torch
from torch_utils import CosineAnnealingWarmupRestarts, UnitGaussianNormalizerTorch
from torch_kernels import kernels
import numpy as np
from unet_models import KNO_UNET_GREEN_2D
import argparse

import wandb
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10_000)
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--lr-max', type=float, default=0.001)
parser.add_argument('--lift-dim', type=int, default=64)
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--test-batch-size', type=int, default=1)
parser.add_argument('--int-kernel', type=str, default='fast_green_torch', choices=['g', 'a_g','ns_g', 'gsm', 'ns_gsm', 'ns_gsm_torch', 'green', 'green_torch', 'fast_green_torch'])
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

num_train_batches = len(x_train) // args.batch_size
num_steps = args.epochs * num_train_batches

## kernel setup
integration_kernel = kernels[args.int_kernel]
if args.int_kernel in ['green', 'green_torch', 'fast_green_torch', "ns_gsm_torch"]:
    integration_kernel = partial(integration_kernel, ndims=2, output_dim=args.lift_dim)
else:
    integration_kernel = partial(integration_kernel, ndims=2)

x_normalizer = UnitGaussianNormalizerTorch(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)
y_normalizer = UnitGaussianNormalizerTorch(y_train)

x_normalizer.to(device)
y_normalizer.to(device)

model_cls = KNO_UNET_GREEN_2D

in_feats = codomain_dims + domain_dims
model = model_cls(integration_kernel, 
              args.depth,
              args.lift_dim, 
              domain_dims,
              in_feats, 
              device).to(device)


optimizer=torch.optim.Adam(model.parameters(), lr=args.lr_max)
lr_scheduler = CosineAnnealingWarmupRestarts(
    optimizer=optimizer,
    total_steps=100_000,   # Example total steps
    warmup_frac=0.3,
    peak_value=3e-4,
    num_cycles=3,
    gamma=0.7,
    down=1e4
)

# param_count = sum(x.size for x in jax.tree.leaves(eqx.filter(model, is_trainable)))
param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'{param_count=}')

def train_step(model, optimizer, batch, device):
    x,y = batch
    x = x.to(device)
    y = y.to(device)
    x = torch.concatenate((x,x_grid.unsqueeze(0).expand(*x.shape[0:-1], x_grid.shape[-1])), axis=-1)
    y_pred = model(x)
    y_pred = y_pred.reshape(args.batch_size, -1)
    y_pred = y_normalizer.decode(y_pred)
    train_l2 =  ((y - y_pred)**2).sum(axis=-1).mean()
    train_rel_l2 = (torch.linalg.norm(y-y_pred, axis=1) / torch.linalg.norm(y, axis=1)).mean()
    optimizer.zero_grad()
    train_l2.backward()

    optimizer.step()
    lr_scheduler.step()

    loss_val = train_l2.item() 
    rel_loss_val = train_rel_l2.item()
    
    return loss_val, rel_loss_val

def eval(model, batch,):
    x,y = batch
    x = torch.concatenate((x,x_grid.unsqueeze(0).expand(*x.shape[0:-1], x_grid.shape[-1])), axis=-1)
    y_pred = model(x)
    y_pred = y_pred.reshape(ntest,-1)
    y_pred = y_normalizer.decode(y_pred)
    test_l2 = ((y - y_pred)**2).sum(axis=-1).mean()
    test_rel_l2 =  (torch.linalg.norm(y-y_pred, axis=1) / torch.linalg.norm(y, axis=1)).mean()
    return test_l2.item(), test_rel_l2.item()


if args.wandb:

    wandb.init(
        project="KNO_PWC",
        config=vars(args),
        name="DarcyPWC_KNO_" + args.int_kernel,
    )

model.register_grid(x_grid.shape, device)

trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)

for epoch in tqdm(range(args.epochs)):
    for batch in trainloader: 
        train_l2, train_rel_l2 = train_step(model, optimizer, batch, device)

    if (epoch % args.print_every) == 0 or (epoch == args.epochs - 1):
        print(f'{epoch=}, train rel_l2: {train_rel_l2*100:.3f}')
        
    if (epoch % args.eval_every) == 0 or (epoch == args.epochs - 1):
        test_l2, test_rel_l2 = eval(model, (x_test.to(device), y_test.to(device)))
        print(f'test rel_l2: {test_rel_l2*100:.3f}')

    if args.wandb:
        wandb.log({"Train L2": train_l2, "Test L2": test_l2, "Train Rel L2": train_rel_l2, "Test Rel L2": test_rel_l2})

if args.wandb:
    wandb.finish()

torch.save(model.state_dict(), f"./saved_models/DarcyPWC_{args.int_kernel}.pth")
