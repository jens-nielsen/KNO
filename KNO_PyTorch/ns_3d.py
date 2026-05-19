import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
import os

from .utils import UnitGaussianNormalizer, get_batch, shuffle
from .models import KNO_NS_3D

def main():
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
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # load data
    data = np.load('./datasets/ns_3d_.npz')
    x_raw = data['x']
    y_raw = data['y']
    x_grid_raw = data['x_grid']
    
    ntrain = 90
    ntest = 10
    res_1d = 64
    
    x = torch.from_numpy(x_raw).float()
    y = torch.from_numpy(y_raw).float()
    x_grid = torch.from_numpy(x_grid_raw).float().to(device)

    x_train, x_test = x[:ntrain], x[-ntest:]
    y_train, y_test = y[:ntrain], y[-ntest:]

    x_normalizer = UnitGaussianNormalizer(x_train, axis=None)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    
    y_normalizer = UnitGaussianNormalizer(y_train.reshape(ntrain, -1), axis=None)
    y_normalizer.to(device)

    in_feats = 1 + 3 # codomain_dims + domain_dims
    model = KNO_NS_3D(
        integration_kernel_type=args.int_kernel,
        depth=args.depth,
        lift_dim=args.lift_dim,
        ndims=3,
        in_feats=in_feats,
        res_1d=res_1d
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr_max)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    h = x_grid[1, 0, 0, 1] - x_grid[0, 0, 0, 1]
    w = torch.zeros(res_1d)
    w[0] = h / 2
    w[-1] = h / 2
    w[1:-1] = h
    q_weights = w.unsqueeze(1).to(device)

    num_train_batches = ntrain // args.batch_size

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
            
            loss = F.mse_loss(output, batch_y.view(args.batch_size, -1))
            rel_l2 = (torch.norm(batch_y.view(args.batch_size, -1) - output, dim=1) / torch.norm(batch_y.view(args.batch_size, -1), dim=1)).mean()
            
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
                    
                    rel_l2 = (torch.norm(by.view(args.test_batch_size, -1) - out, dim=1) / torch.norm(by.view(args.test_batch_size, -1), dim=1)).mean()
                    test_rel_l2_total += rel_l2.item()
                
                avg_test_rel_l2 = test_rel_l2_total / num_test_batches
                print(f"Test Rel L2: {avg_test_rel_l2:.4f}")

    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/ns_3d_model.pt")

if __name__ == "__main__":
    main()
