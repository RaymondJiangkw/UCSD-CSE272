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
                out_dim = 1 + 3
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
    
    def density(self, x, d_i):
        # sigma
        x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        h = self.sigma_net(self.encoder(x.detach()))
        sigma_t = torch.exp(h[..., :1])
        return sigma_t

    def forward(self, x, d_i):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # sigma
        x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        h = self.sigma_net(self.encoder(x.detach()))
        sigma_t = torch.exp(h[..., :1])
        alpha = torch.sigmoid(h[..., 1:])

        d_o = torch.nn.functional.normalize(torch.randn_like(d_i), dim=-1)
        
        fused_rho = alpha * 2.0
        
        return {
            'raw_sigma_t': sigma_t.detach(), 
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