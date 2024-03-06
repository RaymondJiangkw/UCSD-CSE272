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

def build_covariance_from_scaling_rotation(scaling, rotation):
    assert scaling.size(-1) == 3 and rotation.size(-1) == 4 and len(scaling.shape) == len(rotation.shape)
    assert scaling.shape[:-1] == rotation.shape[:-1]
    prefix = scaling.shape[:-1]
    L = build_scaling_rotation(scaling.reshape(-1, 3), rotation.reshape(-1, 4))
    covariance = L @ L.transpose(1, 2)
    return covariance.reshape(*prefix, 3, 3)

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
                out_dim = 1 + 1 + 2 + 4 + self.geo_feat_dim
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))
            if l != num_layers - 1:
                sigma_net.append(nn.ReLU())

        self.sigma_net = nn.Sequential(*sigma_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir)
        
        color_net =  []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir + self.geo_feat_dim
            else:
                in_dim = hidden_dim_color
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim_color
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))
            if l != num_layers - 1:
                color_net.append(nn.ReLU())

        self.color_net = nn.Sequential(*color_net)
    
    def calc_NDF(self, scaling, cov, d):
        d = torch.nn.functional.normalize(d)
        pdf = 1 / (
            torch.pi * torch.prod(scaling, dim=-1, keepdim=True) * torch.square((d[..., None, :] @ torch.linalg.inv(cov.to(torch.float32)) @ d[..., :, None]).squeeze(-1))
        ).clamp_min(1e-8) # (..., 1)
        return pdf
    
    def calc_PA(self, cov, d, d_p=None):
        if d_p is None: d_p = d
        return torch.sqrt((d[..., None, :] @ cov @ d_p[..., :, None]).squeeze(-1).clamp_min(0.))

    def forward(self, x, d, M=8):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        assert torch.all(torch.logical_and(x > - self.bound, x < self.bound)), f"{x.min()}, {x.max()}, {self.bound}"

        # sigma
        x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        h = self.sigma_net(self.encoder(x.detach()))

        sigma = torch.exp(h[..., 0:1] + np.log(1e-1))
        alpha = torch.sigmoid(h[..., 1:2])
        scaling = torch.sigmoid(h[..., 2:2+2]); scaling = torch.cat((scaling, torch.ones_like(scaling[..., :1])), dim=-1)
        quaternion = h[..., 2+2:2+2+4]; quaternion[..., 0] = quaternion[..., 0] + 1; quaternion = torch.nn.functional.normalize(quaternion)
        geo_feat = h[..., 2+2+4:]
        cov = build_covariance_from_scaling_rotation(scaling, quaternion)
        integral = self.calc_PA(cov, d)
        raw_sigma_t = sigma * integral
        sigma_t = torch.nan_to_num(torch.clamp_max(raw_sigma_t, self.sigma_majorant), self.sigma_majorant, self.sigma_majorant, self.sigma_majorant) # (N, 1)
        assert torch.all(sigma_t >= 0), f'{sigma}, {integral}'
        sigma_s = alpha * sigma_t # (N, 1)

        # Importance Sampling VNDF
        @torch.no_grad()
        def sample_NDF(batch_size=1):
            omega_i = d
            omega_j = torch.nn.functional.normalize(torch.cross(torch.tensor([0., 0., 1.], device=d.device, dtype=d.dtype)[None, :].expand_as(omega_i), omega_i))
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
            M_k = torch.sqrt(torch.nan_to_num(torch.linalg.det(S_kji.to(torch.float32))[..., None] * (1 / (S_jj * S_ii - S_ji * S_ji))).clamp_min(0.))
            # assert torch.all(~torch.isnan(M_k)), f"{torch.any(torch.isnan(S_jj * S_ii - S_ji * S_ji))}, {S_kji[torch.isnan(torch.linalg.det(S_kji.to(torch.float32)))]}"
            M_k = torch.concatenate((M_k, torch.zeros_like(M_k), torch.zeros_like(M_k)), dim=-1)
            M_j_1 = - (S_ki * S_ji - S_kj * S_ii) * torch.rsqrt(torch.clamp_min(S_jj * S_ii - S_ji * S_ji, 1e-8))
            M_j_2 = torch.sqrt(torch.clamp_min(S_jj * S_ii - S_ji * S_ji, 0.))
            M_j = torch.rsqrt(S_ii.clamp_min(1e-8)) * torch.concatenate((M_j_1, M_j_2, torch.zeros_like(M_j_1)), dim=-1)
            M_i = torch.rsqrt(S_ii.clamp_min(1e-8)) * torch.concatenate((S_ki, S_ji, S_ii), dim=-1)
            assert torch.all(~torch.isnan(M_k))
            assert torch.all(~torch.isnan(M_j))
            assert torch.all(~torch.isnan(M_i))

            B = S_kk.shape[0]
            assert len(S_kk.shape) == 2 and len(M_k.shape) == 2

            U_1 = torch.rand(B, batch_size, 1, device='cuda')
            U_2 = torch.rand(B, batch_size, 1, device='cuda')
            p_u = torch.sqrt(U_1) * torch.cos(2 * torch.pi * U_2)
            p_v = torch.sqrt(U_1) * torch.sin(2 * torch.pi * U_2)
            p_w = torch.sqrt(1 - p_u ** 2 - p_v ** 2)
            d_out_kji = torch.nn.functional.normalize(p_u * M_k[:, None, :] + p_v * M_j[:, None, :] + p_w * M_i[:, None, :], dim=-1) # (N, b, 3)
            return torch.nn.functional.normalize((torch.stack((omega_k, omega_j, omega_i), dim=-1) @ d_out_kji.transpose(-1, -2)).transpose(-1, -2), dim=-1).squeeze(-2) # (N, ?, 3)

        d_m_s = sample_NDF(M + 1)
        d_m = d_m_s[:, 0, :] # (N, 3)
        d_up = torch.zeros_like(d_m); d_up[..., -1] = 1.
        r_forward = d_m
        r_up = d_up
        r_right = torch.nn.functional.normalize(torch.cross(r_forward, r_up), dim=-1)
        r_up = torch.nn.functional.normalize(torch.cross(r_forward, r_right), dim=-1)
        R = torch.stack((r_right, r_up, r_forward), axis=-1) # (N, 3, 3)
        # assert torch.allclose(torch.linalg.det(R), 1.0), f'{torch.linalg.det(R)}'
        # assert torch.allclose((R @ d_up[:, :, None]).squeeze(-1), d_m), f'{(R @ d_up[:, :, None]).squeeze(-1)}, {d_m}'
        
        _phi = 2.0 * torch.pi * torch.rand_like(d_m[:, :1]) # (N, 1)
        _tmp = torch.rand_like(_phi)
        _t0 = torch.sqrt(1 - _tmp) # (N, 1)
        _t1 = torch.sqrt(_tmp)
        _d_o = torch.cat((
            torch.cos(_phi) * _t0, torch.sin(_phi) * _t0, _t1
        ), dim=-1) # (N, 3)
        d_o = (R @ _d_o[:, :, None]).squeeze(-1)

        rho = (1 / torch.pi) * (d_m_s[:, 1:, :] * d_o[:, None, :]).abs().sum(dim=-1).mean(dim=-1, keepdim=True).clamp_min_(1e-3)
        # assert torch.all(rho > 0.0)
        pdf_d_out = ((1 / torch.pi) * (d_o * d_m).sum(dim=-1, keepdim=True) * (d * d_m).sum(dim=-1, keepdim=True).abs() * self.calc_NDF(scaling, cov, d_m) / integral).clamp_min_(1e-2)
        # assert torch.all(pdf_d_out > 0.0), f'{pdf_d_out.min()}, {pdf_d_out.max()}'

        h = torch.cat([self.encoder_dir(d), geo_feat], dim=-1)
        # sigmoid activation for rgb
        Le = torch.exp(self.color_net(h))

        return {
            'raw_sigma_t': raw_sigma_t.detach(), 
            'sigma_s': sigma_s, 
            'sigma_t': sigma_t, 
            'rho': rho, 
            'pdf_d_out': pdf_d_out,
            'd_out': d_o, 
            'Le': Le, 
        }
    
    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr},
            {'params': self.env_map, 'lr': lr}
        ]
        
        return params