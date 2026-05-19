import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
import wandb
import os

from .utils import UnitGaussianNormalizer, get_batch, shuffle
from .models import KNO_DARCY_PWC

def main():
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
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # load data
    data = np.load('./datasets/darcy_2d_uniform_data.npz')
    x_raw = data["a"]
    y_raw = data["u"]

    r = 4
    s = int(((421 - 1) / r) + 1)
    x_raw = x_raw[:, ::r, ::r][:, :s, :s]
    y_raw = y_raw[:, ::r, ::r][:, :s, :s]

    res = 96
    x_raw = x_raw[:, :res, :res]
    y_raw = y_raw[:, :res, :res]

    x = torch.from_numpy(x_raw).float().unsqueeze(-1) # (B, N, N, 1)
    y = torch.from_numpy(y_raw).float().view(y_raw.shape[0], -1) # (B, N*N)

    x_grid_1d = torch.linspace(0, 1, res)
    grid_y, grid_x = torch.meshgrid(x_grid_1d, x_grid_1d, indexing='ij')
    x_grid = torch.stack([grid_x, grid_y], dim=-1).float() # (N, N, 2)

    ntrain = 1024
    ntest = 100
    x_train, x_test = x[:ntrain], x[-ntest:]
    y_train, y_test = y[:ntrain], y[-ntest:]

    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    
    y_normalizer = UnitGaussianNormalizer(y_train)
    y_normalizer.to(device)

    in_feats = 1 + 2 # codomain_dims + domain_dims
    model = KNO_DARCY_PWC(
        integration_kernel_type=args.int_kernel,
        depth=args.depth,
        lift_dim=args.lift_dim,
        ndims=2,
        in_feats=in_feats
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr_max)
    num_train_batches = ntrain // args.batch_size
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    h = x_grid[1, 0, 0] - x_grid[0, 0, 0]
    w = torch.ones(res) * h
    w[0] = h / 2
    w[-1] = h / 2
    q_weights = w.unsqueeze(1).to(device) # (res, 1)
    x_grid = x_grid.to(device)

    wandb.init(
        project="KNO_PyTorch",
        config=vars(args),
        name="Darcy2DUniformKNO_PyTorch",
    )

    for epoch in tqdm(range(args.epochs)):
        model.train()
        x_train, y_train = shuffle(x_train, y_train, seed=args.seed + epoch)
        
        total_rel_l2 = 0
        for i in range(num_train_batches):
            batch_x, batch_y = get_batch((x_train, y_train), i, args.batch_size)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x, x_grid, q_weights)
            output = output.view(args.batch_size, -1)
            output = y_normalizer.decode(output)
            
            loss = F.mse_loss(output, batch_y)
            rel_l2 = (torch.norm(batch_y - output, dim=1) / torch.norm(batch_y, dim=1)).mean()
            
            loss.backward()
            optimizer.step()
            total_rel_l2 += rel_l2.item()
            
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
                    
                    out = model(bx, x_grid, q_weights)
                    out = out.view(args.test_batch_size, -1)
                    out = y_normalizer.decode(out)
                    
                    rel_l2 = (torch.norm(by - out, dim=1) / torch.norm(by, dim=1)).mean()
                    test_rel_l2_total += rel_l2.item()
                
                avg_test_rel_l2 = test_rel_l2_total / num_test_batches
                print(f"Test Rel L2: {avg_test_rel_l2:.4f}")
                wandb.log({"Train Rel L2": avg_rel_l2, "Test Rel L2": avg_test_rel_l2})

    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/darcy2d_model.pt")

if __name__ == "__main__":
    main()
