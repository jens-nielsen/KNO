import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
import os

from .utils import UnitGaussianNormalizer, get_batch, shuffle
from .models import KNO_NS_PIPE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr-max', type=float, default=0.001)
    parser.add_argument('--lift-dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=7)
    parser.add_argument('--test-batch-size', type=int, default=1)
    parser.add_argument('--int-kernel', type=str, default='ns_gsm', choices=['g', 'a_g','ns_g', 'gsm', 'ns_gsm'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print-every', type=int, default=1)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--grad-clip', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # load data
    data = np.load('./datasets/ns_pipe.npz')
    y_grid_raw = data['y_grid'].reshape(2310, -1, 2)
    y_raw = data['y'].reshape(2310, -1)

    y_grid_tensor = torch.from_numpy(y_grid_raw).float()
    y_mu = y_grid_tensor.mean(dim=(0, 1), keepdim=True)
    y_std = y_grid_tensor.std(dim=(0, 1), keepdim=True)
    y_grid_norm = (y_grid_tensor - y_mu) / y_std
    y_grid = y_grid_norm.view(-1, 129, 129, 2)

    y_h = y_grid[0, 0, 1, 1] - y_grid[0, 0, 0, 1]
    x_h = y_grid[0, 1, 0, 0] - y_grid[0, 0, 0, 0]

    wx = torch.zeros(129)
    wx[0] = x_h / 2
    wx[-1] = x_h / 2
    wx[1:-1] = x_h
    wx = wx.unsqueeze(1).to(device)

    wy = torch.zeros(129)
    wy[0] = y_h / 2
    wy[-1] = y_h / 2
    wy[1:-1] = y_h
    wy = wy.unsqueeze(1).to(device)

    q_nodes = y_grid.to(device)
    y = torch.from_numpy(y_raw).float()

    ntrain = 1000
    ntest = 200
    q_train, q_test = q_nodes[:ntrain], q_nodes[-ntest:]
    y_train, y_test = y[:ntrain], y[-ntest:]

    y_normalizer = UnitGaussianNormalizer(y_train)
    y_normalizer.to(device)

    in_feats = 2 + 0 # domain_dims + codomain_dims
    model = KNO_NS_PIPE(
        integration_kernel_type=args.int_kernel,
        depth=args.depth,
        lift_dim=args.lift_dim,
        ndims=2,
        in_feats=in_feats,
        res_1d=129
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr_max)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    num_train_batches = ntrain // args.batch_size

    for epoch in tqdm(range(args.epochs)):
        model.train()
        q_train, y_train = shuffle(q_train, y_train, seed=args.seed + epoch)
        
        total_rel_l2 = 0
        for i in range(num_train_batches):
            batch_q, batch_y = get_batch((q_train, y_train), i, args.batch_size)
            batch_q, batch_y = batch_q.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            # KNO_NS_PIPE.forward(q, wx, wy)
            # JAX code: eqx.filter_vmap(lambda q: model(q, wx, wy))(q)
            # Batching manually for simplicity or use vmap
            output_list = []
            for j in range(args.batch_size):
                out = model(batch_q[j], wx, wy)
                output_list.append(out)
            output = torch.stack(output_list).view(args.batch_size, -1)
            output = y_normalizer.decode(output)
            
            loss = F.mse_loss(output, batch_y)
            rel_l2 = (torch.norm(batch_y - output, dim=1) / torch.norm(batch_y, dim=1)).mean()
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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
                    bq, by = get_batch((q_test, y_test), i, args.test_batch_size)
                    bq, by = bq.to(device), by.to(device)
                    
                    out_list = []
                    for j in range(args.test_batch_size):
                        out_list.append(model(bq[j], wx, wy))
                    out = torch.stack(out_list).view(args.test_batch_size, -1)
                    out = y_normalizer.decode(out)
                    
                    rel_l2 = (torch.norm(by - out, dim=1) / torch.norm(by, dim=1)).mean()
                    test_rel_l2_total += rel_l2.item()
                
                avg_test_rel_l2 = test_rel_l2_total / num_test_batches
                print(f"Test Rel L2: {avg_test_rel_l2:.4f}")

    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/ns_pipe_model.pt")

if __name__ == "__main__":
    main()
