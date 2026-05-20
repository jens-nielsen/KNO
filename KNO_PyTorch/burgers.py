from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
# import wandb
import os

import wandb


from .utils import CyclicalCosineLRScheduler,cosine_annealing,  UnitGaussianNormalizer, get_batch, shuffle
from .models import KNO_REG_GRID_1D
from .kernels import kernels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--lr-max', type=float, default=0.001)
    parser.add_argument('--lift-dim', type=int, default=64)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--test-batch-size', type=int, default=200)
    parser.add_argument('--int-kernel', type=str, default='ns_gsm', choices=['g', 'a_g','ns_g', 'gsm', 'ns_gsm'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print-every', type=int, default=1)
    parser.add_argument('--eval-every', type=int, default=1)
    parser.add_argument('--device', type=str, default='mps' if torch.backends.mps.is_available() else 'cpu')

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # load data
    fp = './datasets/burgers.npz' # Adjusted path
    data = np.load(fp)
    x = torch.from_numpy(data["x"]).float()
    x_grid = torch.from_numpy(data["x_grid"]).float()
    y = torch.from_numpy(data["y"]).float().view(1200, -1)
    
    ntrain = 1000
    ntest = 200

    x_train, x_test = x[:ntrain], x[-ntest:]
    y_train, y_test = y[:ntrain], y[-ntest:]

    # kernel setup
    
    integration_kernel = kernels[args.int_kernel]
    integration_kernel = partial(integration_kernel, ndims=1)

    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    
    y_normalizer = UnitGaussianNormalizer(y_train)
    y_normalizer.to(device)

    in_feats = 2 # domain_dims (1) + codomain_dims (1)
    model = KNO_REG_GRID_1D(
        integration_kernel=integration_kernel,
        lift_dim=args.lift_dim,
        depth=args.depth,
        in_feats=in_feats
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr_max)
    # Simple scheduler to mimic cosine annealing if needed, 
    # but for now let's use a standard one.

    num_train_batches = len(x_train) // args.batch_size
    num_steps = args.epochs * num_train_batches
    scheduler = cosine_annealing(optimizer=optimizer, total_steps=num_steps, peak_value=args.lr_max)
    # scheduler = CyclicalCosineLRScheduler(optimizer, total_steps=num_steps, peak_value=args.lr_max)

    # trapezoidal quadrature rule
    w = torch.zeros_like(x_grid.squeeze())
    dx = x_grid[1, 0] - x_grid[0, 0]
    w[0] = dx / 2
    w[-1] = dx / 2
    w[1:-1] = dx
    q_weights = w.unsqueeze(1).to(device) # (N, 1)
    x_grid = x_grid.to(device) # (N, 1)

    wandb.init(
        project="KNO",
        config=vars(args),
        name="BurgerKNO_PyTorch",
    )

    num_train_batches = ntrain // args.batch_size

    for epoch in tqdm(range(args.epochs)):
        model.train()
        # Shuffle train data
        x_train, y_train = shuffle(x_train, y_train, seed=args.seed + epoch)
        
        total_rel_l2 = 0
        for i in range(num_train_batches):
            batch_x, batch_y = get_batch((x_train, y_train), i, args.batch_size)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # model expects (B, N, in_feats)
            # batch_x: (B, N), x_grid: (N, 1)
            output = torch.vmap(lambda x: model(x, x_grid, q_weights))(batch_x)
            output = y_normalizer.decode(output)
            
            loss = ((batch_y - output)**2).sum(axis=-1).mean()
            train_rel_l2 = (torch.norm(batch_y - output, dim=1) / torch.norm(batch_y, dim=1)).mean()
            
            loss.backward()
            optimizer.step()
            
            total_rel_l2 += train_rel_l2.item()
        
            scheduler.step()
        
        avg_rel_l2 = total_rel_l2 / num_train_batches
        
        if epoch % args.print_every == 0:
            print(f"Epoch {epoch}, Train Rel L2: {avg_rel_l2:.4f}")
            
        if epoch % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                test_rel_l2_total = 0
                num_test_batches = ntest // args.test_batch_size
                for i in range(num_test_batches):
                    bx, by = get_batch((x_test, y_test), i, args.test_batch_size)
                    bx, by = bx.to(device), by.to(device)
                    
                    out = torch.vmap(lambda x: model(x, x_grid, q_weights))(bx)
                    out = y_normalizer.decode(out)
                    
                    test_l2 = ((by - out)**2).sum(axis=-1).mean().item()
                    test_rel_l2 = (torch.norm(by - out, dim=1) / torch.norm(by, dim=1)).mean()
                    test_rel_l2_total += test_rel_l2.item()
                
                avg_test_rel_l2 = test_rel_l2_total / num_test_batches
                print(f"Test Rel L2: {avg_test_rel_l2:.4f}")
                wandb.log({"Train L2": loss, "Test L2": test_l2, "Train Rel L2": train_rel_l2, "Test Rel L2": test_rel_l2})

    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/burgers_model.pt")

if __name__ == "__main__":
    main()



