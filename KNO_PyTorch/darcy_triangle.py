import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
import os
from numpy.polynomial.legendre import leggauss

from .utils import UnitGaussianNormalizer, get_batch, shuffle
from .models import KNO_DARCY_TRIANGLE
from .quadratures import triangle_quad_rule

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--lr-max', type=float, default=0.001)
    parser.add_argument('--lift-dim', type=int, default=96)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--test-batch-size', type=int, default=10)
    parser.add_argument('--input-kernel', type=str, default='a_g', choices=['g', 'a_g','ns_g', 'gsm', 'ns_gsm'])
    parser.add_argument('--output-kernel', type=str, default='a_g', choices=['g', 'a_g','ns_g', 'gsm', 'ns_gsm'])
    parser.add_argument('--int-kernel', type=str, default='ns_gsm', choices=['g', 'a_g','ns_g', 'gsm', 'ns_gsm'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quadrature-res', type=int, default=8)
    parser.add_argument('--print-every', type=int, default=1)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # load data
    data = np.load('./datasets/darcy_triangular.npz')
    x_grid_raw = data["bc_coords"]
    
    x_mu = torch.from_numpy(x_grid_raw.mean(axis=0)).float()
    x_std = torch.from_numpy(x_grid_raw.std(axis=0)).float()
    
    x_grid = (torch.from_numpy(x_grid_raw).float() - x_mu) / x_std
    y_grid_raw = data["mesh_grid"]
    y_grid = (torch.from_numpy(y_grid_raw).float() - x_mu) / x_std
    
    x_raw = data["k"]
    y_raw = data["h"]
    
    x = torch.from_numpy(x_raw).float().view(2000, -1, 1)
    y = torch.from_numpy(y_raw).float().view(2000, -1)

    ntrain = 1900
    ntest = 100
    x_train, x_test = x[:ntrain], x[-ntest:]
    y_train, y_test = y[:ntrain], y[-ntest:]

    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    
    y_normalizer = UnitGaussianNormalizer(y_train)
    y_normalizer.to(device)

    in_feats = 2 + 1 # domain_dims + codomain_dims
    model = KNO_DARCY_TRIANGLE(
        input_kernel_type=args.input_kernel,
        output_kernel_type=args.output_kernel,
        integration_kernel_type=args.int_kernel,
        lift_dim=args.lift_dim,
        depth=args.depth,
        in_feats=in_feats
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr_max)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # load quadrature rules
    q_nodes, q_weights = triangle_quad_rule(args.quadrature_res, leggauss)
    q_nodes = (q_nodes - x_mu) / x_std
    q_weights = torch.prod(x_std) * q_weights
    
    q_nodes = q_nodes.to(device)
    q_weights = q_weights.to(device)
    x_grid = x_grid.to(device)
    y_grid = y_grid.to(device)

    num_train_batches = ntrain // args.batch_size

    for epoch in tqdm(range(args.epochs)):
        model.train()
        x_train, y_train = shuffle(x_train, y_train, seed=args.seed + epoch)
        
        total_rel_l2 = 0
        for i in range(num_train_batches):
            batch_x, batch_y = get_batch((x_train, y_train), i, args.batch_size)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x, x_grid, y_grid, q_nodes, q_weights)
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
                    
                    out = model(bx, x_grid, y_grid, q_nodes, q_weights)
                    out = out.view(args.test_batch_size, -1)
                    out = y_normalizer.decode(out)
                    
                    rel_l2 = (torch.norm(by - out, dim=1) / torch.norm(by, dim=1)).mean()
                    test_rel_l2_total += rel_l2.item()
                
                avg_test_rel_l2 = test_rel_l2_total / num_test_batches
                print(f"Test Rel L2: {avg_test_rel_l2:.4f}")

    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/darcy_triangle_model.pt")

if __name__ == "__main__":
    main()
