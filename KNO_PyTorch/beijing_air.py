import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
import os
import pickle

from .utils import UnitGaussianNormalizer, get_batch, shuffle
from .models import KNO_REG_GRID_1D

def get_beijing(seed=42):
    Ntr, Nte = 5000, 1000
    with open('./datasets/beijing_data.pickle', 'rb') as handle:
        d = pickle.load(handle)
    X, Y = d["x"][:Ntr+Nte], d["y"][:Ntr+Nte]
    X, Y = shuffle(torch.from_numpy(X).float(), torch.from_numpy(Y).float(), seed=seed)
    Xtr, Xte = X[:Ntr], X[Ntr:]
    Ytr, Yte = Y[:Ntr], Y[Ntr:]
    return Xtr, Xte, Ytr, Yte

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--lr-max', type=float, default=0.001)
    parser.add_argument('--lift-dim', type=int, default=256)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--test-batch-size', type=int, default=200)
    parser.add_argument('--int-kernel', type=str, default='gsm', choices=['g', 'a_g','ns_g', 'gsm', 'ns_gsm'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print-every', type=int, default=1)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    x_train, x_test, y_train, y_test = get_beijing(seed=args.seed)
    y_train, y_test = y_train.squeeze(), y_test.squeeze()
    
    res = x_train.shape[1]
    x_grid_1d = torch.linspace(0, 1, res)
    x_grid = x_grid_1d.unsqueeze(1).float().to(device)

    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    
    y_normalizer = UnitGaussianNormalizer(y_train)
    y_normalizer.to(device)

    in_feats = 1 + 5 # domain_dims + codomain_dims
    model = KNO_REG_GRID_1D(
        integration_kernel_type=args.int_kernel,
        lift_dim=args.lift_dim,
        depth=args.depth,
        in_feats=in_feats
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr_max)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    dx = x_grid_1d[1] - x_grid_1d[0]
    w = torch.zeros(res)
    w[0] = dx / 2
    w[-1] = dx / 2
    w[1:-1] = dx
    q_weights = w.unsqueeze(1).to(device)

    num_train_batches = len(x_train) // args.batch_size

    for epoch in tqdm(range(args.epochs)):
        model.train()
        x_train, y_train = shuffle(x_train, y_train, seed=args.seed + epoch)
        
        total_rel_l2 = 0
        for i in range(num_train_batches):
            batch_x, batch_y = get_batch((x_train, y_train), i, args.batch_size)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x, x_grid, q_weights)
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
                num_test_batches = len(x_test) // args.test_batch_size
                for i in range(num_test_batches):
                    bx, by = get_batch((x_test, y_test), i, args.test_batch_size)
                    bx, by = bx.to(device), by.to(device)
                    
                    out = model(bx, x_grid, q_weights)
                    out = y_normalizer.decode(out)
                    
                    rel_l2 = (torch.norm(by - out, dim=1) / torch.norm(by, dim=1)).mean()
                    test_rel_l2_total += rel_l2.item()
                
                avg_test_rel_l2 = test_rel_l2_total / num_test_batches
                print(f"Test Rel L2: {avg_test_rel_l2:.4f}")

    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/beijing_model.pt")

if __name__ == "__main__":
    main()
