import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from encoding import get_encoder
from .renderer import NeRFRenderer

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def build_scaling_rotation_inv(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = 1./(s[:,0]+1e-8)
    L[:,1,1] = 1./(s[:,1]+1e-8)
    L[:,2,2] = 1./(s[:,2]+1e-8)

    L = R @ L
    return L

def build_covariance_from_scaling_rotation(scaling, rotation):
    assert scaling.size(-1) == 3 and rotation.size(-1) == 4 and len(scaling.shape) == len(rotation.shape)
    assert scaling.shape[:-1] == rotation.shape[:-1]
    prefix = scaling.shape[:-1]
    L = build_scaling_rotation(scaling.reshape(-1, 3), rotation.reshape(-1, 4))
    L_inv = build_scaling_rotation_inv(scaling.reshape(-1, 3), rotation.reshape(-1, 4))
    covariance = L @ L.transpose(1, 2)
    covariance_inv = L_inv @ L_inv.transpose(1, 2)
    return covariance.reshape(*prefix, 3, 3), covariance_inv.reshape(*prefix, 3, 3)

class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 bound=1,
                 **kwargs
                 ):
        super().__init__(bound, **kwargs)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + 3 + 3 + 4
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))
            if l != num_layers - 1:
                sigma_net.append(nn.ReLU())

        self.sigma_net = nn.Sequential(*sigma_net)
    
    def calc_NDF(self, scaling, cov_inv, d):
        d = torch.nn.functional.normalize(d)
        pdf = 1 / (
            torch.pi * torch.prod(scaling, dim=-1, keepdim=True) * torch.square((d[..., None, :] @ cov_inv @ d[..., :, None]).squeeze(-1))
        ).clamp_min(1e-8) # (..., 1)
        return pdf
    
    def calc_PA(self, cov, d, d_p=None):
        if d_p is None: d_p = d
        return torch.sqrt((d[..., None, :] @ cov @ d_p[..., :, None]).squeeze(-1).clamp_min(0.))

    def forward(self, x, d_i):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        assert torch.all(torch.logical_and(x > - self.bound, x < self.bound)), f"{x.min()}, {x.max()}, {self.bound}"

        # sigma
        x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        h = self.sigma_net(self.encoder(x.detach()))
        sigma = torch.exp(h[..., 0:1])
        alpha = torch.sigmoid(h[..., 1:1+3])
        scaling = torch.sigmoid(h[..., 1+3:1+3+3])
        quaternion = h[..., 1+3+3:1+3+3+4]; quaternion[..., 0] = quaternion[..., 0] + 1; quaternion = torch.nn.functional.normalize(quaternion)
        cov, cov_inv = build_covariance_from_scaling_rotation(scaling, quaternion)
        integral = self.calc_PA(cov, d_i)
        raw_sigma_t = sigma * integral
        sigma_t = torch.clamp(raw_sigma_t, 1e-3, self.sigma_majorant) # (N, 1)

        # Importance Sampling VNDF
        @torch.no_grad()
        def sample_VNDF(batch_size=1):
            omega_i = d_i
            omega_j = torch.nn.functional.normalize(torch.cross(torch.tensor([0., 0., 1.], device=d_i.device, dtype=d_i.dtype)[None, :].expand_as(omega_i), omega_i))
            omega_k = torch.nn.functional.normalize(torch.cross(omega_i, omega_j))
            S_kk = self.calc_PA(cov, omega_k, omega_k)
            S_kj = self.calc_PA(cov, omega_k, omega_j)
            S_ki = self.calc_PA(cov, omega_k, omega_i)
            S_jj = self.calc_PA(cov, omega_j, omega_j)
            S_ji = self.calc_PA(cov, omega_j, omega_i)
            S_ii = self.calc_PA(cov, omega_i, omega_i)
            S_kji = torch.concatenate((
                S_kk, S_kj, S_ki, 
                S_kj, S_jj, S_ji, 
                S_ki, S_ji, S_ii, 
            ), dim=-1).reshape(-1, 3, 3)
            M_k = torch.sqrt((torch.linalg.det(S_kji)[..., None] * (1 / (S_jj * S_ii - S_ji * S_ji))).clamp_min(0.))
            M_k = torch.concatenate((M_k, torch.zeros_like(M_k), torch.zeros_like(M_k)), dim=-1)
            M_j_1 = - (S_ki * S_ji - S_kj * S_ii) * torch.rsqrt(torch.clamp_min(S_jj * S_ii - S_ji * S_ji, 1e-8))
            M_j_2 = torch.sqrt(torch.clamp_min(S_jj * S_ii - S_ji * S_ji, 0.))
            M_j = torch.rsqrt(S_ii.clamp_min(1e-8)) * torch.concatenate((M_j_1, M_j_2, torch.zeros_like(M_j_1)), dim=-1)
            M_i = torch.rsqrt(S_ii.clamp_min(1e-8)) * torch.concatenate((S_ki, S_ji, S_ii), dim=-1)

            B = S_kk.shape[0]
            
            U_1 = torch.rand(B, batch_size, 1, device='cuda')
            U_2 = torch.rand(B, batch_size, 1, device='cuda')
            p_u = torch.sqrt(U_1) * torch.cos(2 * torch.pi * U_2)
            p_v = torch.sqrt(U_1) * torch.sin(2 * torch.pi * U_2)
            p_w = torch.sqrt(1 - p_u ** 2 - p_v ** 2)
            d_out_kji = torch.nn.functional.normalize(p_u * M_k[:, None, :] + p_v * M_j[:, None, :] + p_w * M_i[:, None, :], dim=-1) # (N, b, 3)
            return torch.nn.functional.normalize((torch.stack((omega_k, omega_j, omega_i), dim=-1) @ d_out_kji.transpose(-1, -2)).transpose(-1, -2), dim=-1).squeeze(-2) # (N, ?, 3)

        d_m = sample_VNDF() # (N, 3)
        mask = (d_m * d_i).sum(dim=-1) > 0
        d_m[mask] = -d_m[mask]
        d_o = -d_i + 2.0 * d_m * (d_m * d_i).sum(dim=-1, keepdim=True)

        fused_rho = alpha * integral * torch.reciprocal((d_m * d_i).sum(dim=-1, keepdim=True).abs() + 1E-8)

        return {
            'raw_sigma_t': raw_sigma_t.detach(), 
            'sigma_t': sigma_t, 
            'fused_rho': fused_rho, 
            'd_out': d_o, 
        }
    
    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.env_map, 'lr': lr}
        ]
        
        return params