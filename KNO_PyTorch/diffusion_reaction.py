import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
import os
from scipy.io import loadmat

from .utils import UnitGaussianNormalizer, get_batch, shuffle
from .models import KNO_DIFFUSION_REACTION

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--lr-max', type=float, default=0.001)
    parser.add_argument('--lift-dim', type=int, default=32)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--test-batch-size', type=int, default=10)
    parser.add_argument('--output-kernel', type=str, default='a_g', choices=['g', 'a_g','ns_g', 'gsm', 'ns_gsm'])
    parser.add_argument('--int-kernel', type=str, default='ns_gsm', choices=['g', 'a_g','ns_g', 'gsm', 'ns_gsm'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quad-res', default=729, type=int)
    parser.add_argument('--print-every', type=int, default=1)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # load data
    data = np.load('./datasets/diffrec_3d.npz')
    x_raw = data["x"]
    y_raw = data["y"]
    y_grid = torch.from_numpy(data["y_grid"]).float().to(device)

    const = x_raw[:, 0]
    x = torch.ones((1200, args.quad_res, 1)) * torch.from_numpy(const).view(-1, 1, 1).float()
    y = torch.from_numpy(y_raw).float().view(1200, -1)

    ntrain = 1000
    ntest = 200
    x_train, x_test = x[:ntrain], x[-ntest:]
    y_train, y_test = y[:ntrain], y[-ntest:]

    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    
    y_normalizer = UnitGaussianNormalizer(y_train)
    y_normalizer.to(device)

    in_feats = 3 + 1 # domain_dims + codomain_dims
    model = KNO_DIFFUSION_REACTION(
        output_kernel_type=args.output_kernel,
        integration_kernel_type=args.int_kernel,
        lift_dim=args.lift_dim,
        depth=args.depth,
        in_feats=in_feats
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr_max)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    qr = loadmat(f'./datasets/n_{args.quad_res}.mat')
    q_nodes = torch.from_numpy(qr['t']).float().to(device)
    q_weights = torch.from_numpy(qr['w']).float().to(device)

    num_train_batches = ntrain // args.batch_size

    for epoch in tqdm(range(args.epochs)):
        model.train()
        x_train, y_train = shuffle(x_train, y_train, seed=args.seed + epoch)
        
        total_rel_l2 = 0
        for i in range(num_train_batches):
            batch_x, batch_y = get_batch((x_train, y_train), i, args.batch_size)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            # model(f_x, y_grid, q_nodes, q_weights)
            output = model(batch_x, y_grid, q_nodes, q_weights)
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
                    
                    out = model(bx, y_grid, q_nodes, q_weights)
                    out = out.view(args.test_batch_size, -1)
                    out = y_normalizer.decode(out)
                    
                    rel_l2 = (torch.norm(by - out, dim=1) / torch.norm(by, dim=1)).mean()
                    test_rel_l2_total += rel_l2.item()
                
                avg_test_rel_l2 = test_rel_l2_total / num_test_batches
                print(f"Test Rel L2: {avg_test_rel_l2:.4f}")

    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/diffrec_model.pt")

if __name__ == "__main__":
    main()
