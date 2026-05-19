import torch
import numpy as np

def triangle_quad_rule(n, quadrature_fn, triangle=None):
    if triangle is None:
        triangle = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]], dtype=torch.float64)
    else:
        triangle = torch.as_tensor(triangle, dtype=torch.float64)

    def quad_rule_2d(n, quadrature_fn):
        quad_nodes, quad_weights = quadrature_fn(n)
        quad_nodes = torch.as_tensor(quad_nodes, dtype=torch.float64)
        quad_weights = torch.as_tensor(quad_weights, dtype=torch.float64)
        
        a, b = -1.0, 1.0  # old domain
        c, d = 0.0, 1.0  # new domain
        
        t = (((quad_nodes - a) * (d - c)) / (b - a)) + c
        det_j = (d - c) / (b - a)
        w = quad_weights * det_j
        
        # Meshgrid
        grid = torch.meshgrid(t, t, indexing='ij')
        t_2d = torch.stack([grid[0].flatten(), grid[1].flatten()], dim=1)
        w_2d = torch.outer(w, w).flatten().unsqueeze(1)
        
        return t_2d, w_2d

    quad_rule = quad_rule_2d(n, quadrature_fn)

    def coord_square_to_quadrilateral(x, quadrilateral):
        # x: (2,), quadrilateral: (4, 2)
        x1, x2, x3, x4 = quadrilateral
        xi, eta = x[0], x[1]
        
        psi_1 = (1 - xi) * (1 - eta)
        psi_2 = xi * (1 - eta)
        psi_3 = xi * eta
        psi_4 = (1 - xi) * eta
        
        return (x1 * psi_1 + x2 * psi_2 + x3 * psi_3 + x4 * psi_4)

    def detj_square_to_quadrilateral(x, quadrilateral):
        # x: (2,), quadrilateral: (4, 2)
        xi, eta = x[0], x[1]
        x1, x2, x3, x4 = quadrilateral
        
        # Manual Jacobian for bilinear mapping
        d_psi_d_xi = torch.tensor([-(1 - eta), (1 - eta), eta, -eta], dtype=torch.float64)
        d_psi_d_eta = torch.tensor([-(1 - xi), -xi, xi, (1 - xi)], dtype=torch.float64)
        
        # dx/dxi, dy/dxi
        # x is (4, 2), d_psi_d_xi is (4,)
        # J = [dx/dxi  dx/deta]
        #     [dy/dxi  dy/deta]
        
        j11 = torch.dot(quadrilateral[:, 0], d_psi_d_xi)
        j12 = torch.dot(quadrilateral[:, 0], d_psi_d_eta)
        j21 = torch.dot(quadrilateral[:, 1], d_psi_d_xi)
        j22 = torch.dot(quadrilateral[:, 1], d_psi_d_eta)
        
        return j11 * j22 - j12 * j21

    def quad_rule_square_to_quadrilateral(quad_rule, quadrilateral):
        t, w = quad_rule
        num_points = t.shape[0]
        
        updated_t = torch.zeros_like(t)
        updated_w = torch.zeros_like(w)
        
        for i in range(num_points):
            updated_t[i] = coord_square_to_quadrilateral(t[i], quadrilateral)
            updated_w[i] = w[i] * detj_square_to_quadrilateral(t[i], quadrilateral)
            
        return updated_t, updated_w

    A, B, C = triangle[0], triangle[1], triangle[2]
    O = (A + B + C) / 3
    D = (A + B) / 2
    E = (B + C) / 2
    F = (A + C) / 2
    
    quadrilaterals = torch.stack([
        torch.stack([A, D, O, F]),
        torch.stack([B, E, O, D]),
        torch.stack([C, F, O, E])
    ])

    all_t = []
    all_w = []
    for i in range(3):
        t_q, w_q = quad_rule_square_to_quadrilateral(quad_rule, quadrilaterals[i])
        all_t.append(t_q)
        all_w.append(w_q)
        
    triangle_quad_t = torch.cat(all_t, dim=0)
    triangle_quad_w = torch.cat(all_w, dim=0)
    
    return triangle_quad_t.to(torch.float32), triangle_quad_w.to(torch.float32)
