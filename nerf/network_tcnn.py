import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from encoding import get_encoder
from .renderer import NeRFRenderer

BOX_OFFSETS = torch.tensor([[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]],
                               device='cuda')

def hash(coords, log2_hashmap_size):
    '''
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    '''
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i]*primes[i]

    return torch.tensor((1<<log2_hashmap_size)-1).to(xor_result.device) & xor_result

def get_voxel_vertices(xyz, bounding_box, resolution, log2_hashmap_size):
    '''
    xyz: 3D coordinates of samples. B x 3
    bounding_box: min and max x,y,z coordinates of object bbox
    resolution: number of voxels per axis
    '''
    box_min, box_max = bounding_box

    if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
        # print("ALERT: some points are outside bounding box. Clipping them!")
        xyz = torch.clamp(xyz, min=box_min, max=box_max)

    grid_size = (box_max-box_min)/resolution
    
    bottom_left_idx = torch.floor((xyz-box_min)/grid_size).int()
    voxel_min_vertex = bottom_left_idx*grid_size + box_min
    voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0,1.0,1.0], device='cuda')*grid_size

    voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS
    hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices

class HashEmbedder(nn.Module):
    def __init__(self, bounding_box, n_levels=16, n_features_per_level=2,\
                log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super(HashEmbedder, self).__init__()
        self.bounding_box = bounding_box
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.out_dim = self.n_levels * self.n_features_per_level

        self.b = torch.exp((torch.log(self.finest_resolution)-torch.log(self.base_resolution))/(n_levels-1))

        self.embeddings = nn.ModuleList([nn.Embedding(2**self.log2_hashmap_size, \
                                        self.n_features_per_level) for i in range(n_levels)])
        # custom uniform initialization
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)
            # self.embeddings[i].weight.data.zero_()
        

    def trilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        '''
        x: B x 3
        voxel_min_vertex: B x 3
        voxel_max_vertex: B x 3
        voxel_embedds: B x 8 x 2
        '''
        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = (x - voxel_min_vertex)/(voxel_max_vertex-voxel_min_vertex) # B x 3

        # step 1
        # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
        c00 = voxel_embedds[:,0]*(1-weights[:,0][:,None]) + voxel_embedds[:,4]*weights[:,0][:,None]
        c01 = voxel_embedds[:,1]*(1-weights[:,0][:,None]) + voxel_embedds[:,5]*weights[:,0][:,None]
        c10 = voxel_embedds[:,2]*(1-weights[:,0][:,None]) + voxel_embedds[:,6]*weights[:,0][:,None]
        c11 = voxel_embedds[:,3]*(1-weights[:,0][:,None]) + voxel_embedds[:,7]*weights[:,0][:,None]

        # step 2
        c0 = c00*(1-weights[:,1][:,None]) + c10*weights[:,1][:,None]
        c1 = c01*(1-weights[:,1][:,None]) + c11*weights[:,1][:,None]

        # step 3
        c = c0*(1-weights[:,2][:,None]) + c1*weights[:,2][:,None]

        return c

    def forward(self, x):
        # x is 3D point position: B x 3
        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = torch.floor(self.base_resolution * self.b**i)
            voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices = get_voxel_vertices(\
                                                x, self.bounding_box, \
                                                resolution, self.log2_hashmap_size)
            
            voxel_embedds = self.embeddings[i](hashed_voxel_indices)

            x_embedded = self.trilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
            x_embedded_all.append(x_embedded)

        return torch.cat(x_embedded_all, dim=-1)

def softexp(x: torch.Tensor):
    linear_mask = x >= 1
    # recipr_mask = x <= -1
    expont_mask = ~linear_mask # torch.logical_and(~linear_mask, ~recipr_mask)
    y = torch.zeros_like(x)
    y.masked_scatter_(linear_mask, torch.e * torch.masked_select(x, linear_mask))
    # y.masked_scatter_(recipr_mask, - 1.0 / (torch.e * torch.masked_select(x, recipr_mask)))
    y.masked_scatter_(expont_mask, torch.exp(torch.masked_select(x, expont_mask)))
    return y

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
        # self.encoder = HashEmbedder((0, 1), finest_resolution=2048 * bound)
        # self.in_dim  = self.encoder.n_levels * self.encoder.n_features_per_level

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 3 + 3 + 2 + 4
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
        return torch.sqrt((d[..., None, :] @ cov @ d_p[..., :, None]).squeeze(-1).clamp_min(1e-8))
    
    def density(self, x, d_i):
        # sigma
        x = x.contiguous()
        d_i = d_i.contiguous()
        x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        h = self.sigma_net(self.encoder(x))
        sigma = softexp(h[..., 0:3].clone())
        scaling = torch.sigmoid(h[..., 3+3:3+3+2])
        scaling = torch.cat((
            scaling, torch.ones_like(scaling[..., :1])
        ), dim=-1)
        quaternion = h[..., 3+3+2:3+3+2+4]
        quaternion = torch.cat((
            quaternion[..., :1] + 1, quaternion[..., 1:]
        ), dim=-1)
        quaternion = torch.nn.functional.normalize(quaternion)
        cov, cov_inv = build_covariance_from_scaling_rotation(scaling, quaternion)
        assert torch.all(~torch.isnan(cov))
        assert torch.all(~torch.isnan(cov_inv))
        integral = self.calc_PA(cov, d_i)
        assert torch.all(~torch.isnan(integral))
        assert torch.all(torch.isfinite(integral))
        sigma_t = sigma * integral # (N, 1)
        # assert torch.all(sigma_t >= 0), f"{sigma.min()}, {sigma.max()}, {integral.min()}, {integral.max()}"
        return sigma_t

    def forward(self, x, d_i):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # sigma
        x = x.contiguous().detach()
        d_i = d_i.contiguous().detach()
        assert torch.all(~torch.isnan(x))
        assert torch.all(~torch.isnan(d_i))
        x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        h = self.encoder(x)
        assert torch.all(~torch.isnan(h))
        h = self.sigma_net(h)
        assert torch.all(~torch.isnan(h))
        sigma = softexp(h[..., 0:3].clone())
        alpha = torch.sigmoid(h[..., 3:3+3])
        scaling = torch.sigmoid(h[..., 3+3:3+3+2])
        scaling = torch.cat((
            scaling, torch.ones_like(scaling[..., :1])
        ), dim=-1)
        quaternion = h[..., 3+3+2:3+3+2+4]
        quaternion = torch.cat((
            quaternion[..., :1] + 1, quaternion[..., 1:]
        ), dim=-1)
        quaternion = torch.nn.functional.normalize(quaternion)
        cov, cov_inv = build_covariance_from_scaling_rotation(scaling, quaternion)
        assert torch.all(~torch.isnan(cov))
        assert torch.all(~torch.isnan(cov_inv))
        integral = self.calc_PA(cov, d_i)
        assert torch.all(~torch.isnan(integral))
        sigma_t = sigma * integral # (N, 3)
        assert torch.all(~torch.isnan(sigma_t))
        # assert torch.all(sigma_t >= 0), f"{sigma.min()}, {sigma.max()}, {integral.min()}, {integral.max()}"

        # Importance Sampling VNDF
        @torch.no_grad()
        def sample_VNDF(batch_size=1):
            omega_i = d_i
            omega_j = torch.nn.functional.normalize(torch.cross(torch.tensor([0., 0., 1.], device=d_i.device, dtype=d_i.dtype)[None, :].expand_as(omega_i), omega_i, dim=-1))
            # assert torch.all((omega_i * omega_j).sum(dim=-1).abs() < 1e-3), f'{(omega_i * omega_j).sum(dim=-1).abs().min()}, {(omega_i * omega_j).sum(dim=-1).abs().max()}, {torch.linalg.vector_norm(omega_i, ord=2, dim=-1).min()}, {torch.linalg.vector_norm(omega_i, ord=2, dim=-1).max()}, {torch.linalg.vector_norm(omega_j, ord=2, dim=-1).min()}, {torch.linalg.vector_norm(omega_j, ord=2, dim=-1).min()}'
            mask = torch.linalg.vector_norm(omega_j, ord=2, dim=-1) < 1e-3
            omega_j[mask] = torch.nn.functional.normalize(torch.cross(torch.tensor([1., 0., 0.], device=d_i.device, dtype=d_i.dtype)[None, :].expand_as(omega_i[mask]), omega_i[mask], dim=-1))
            omega_k = torch.nn.functional.normalize(torch.cross(omega_i, omega_j, dim=-1))
            assert torch.all((1. - torch.linalg.vector_norm(omega_i, ord=2, dim=-1)).abs() < 1e-3)
            assert torch.all((1. - torch.linalg.vector_norm(omega_j, ord=2, dim=-1)).abs() < 1e-3)
            assert torch.all((1. - torch.linalg.vector_norm(omega_k, ord=2, dim=-1)).abs() < 1e-3)
            assert torch.all((omega_i * omega_j).sum(dim=-1).abs() < 1e-3), f'{(omega_i * omega_j).sum(dim=-1).abs().min()}, {(omega_i * omega_j).sum(dim=-1).abs().max()}, {torch.linalg.vector_norm(omega_i, ord=2, dim=-1).min()}, {torch.linalg.vector_norm(omega_i, ord=2, dim=-1).max()}, {torch.linalg.vector_norm(omega_j, ord=2, dim=-1).min()}, {torch.linalg.vector_norm(omega_j, ord=2, dim=-1).min()}'
            assert torch.all((omega_i * omega_k).sum(dim=-1).abs() < 1e-3)
            assert torch.all((omega_k * omega_j).sum(dim=-1).abs() < 1e-3)
            assert torch.all(~torch.isnan(omega_i))
            assert torch.all(~torch.isnan(omega_j))
            assert torch.all(~torch.isnan(omega_k))
            S_kk = self.calc_PA(cov, omega_k, omega_k)
            assert torch.all(torch.isfinite(S_kk))
            S_kj = self.calc_PA(cov, omega_k, omega_j)
            assert torch.all(torch.isfinite(S_kj))
            S_ki = self.calc_PA(cov, omega_k, omega_i)
            assert torch.all(torch.isfinite(S_ki))
            S_jj = self.calc_PA(cov, omega_j, omega_j)
            assert torch.all(torch.isfinite(S_jj))
            S_ji = self.calc_PA(cov, omega_j, omega_i)
            assert torch.all(torch.isfinite(S_ji))
            S_ii = self.calc_PA(cov, omega_i, omega_i)
            assert torch.all(torch.isfinite(S_ii))
            S_kji = torch.concatenate((
                S_kk, S_kj, S_ki, 
                S_kj, S_jj, S_ji, 
                S_ki, S_ji, S_ii, 
            ), dim=-1).reshape(-1, 3, 3)
            assert torch.all(~torch.isnan(S_kji))
            assert torch.all(torch.isfinite(S_kji))
            assert torch.all(~torch.isnan(torch.linalg.det(S_kji))), f'{torch.linalg.det(S_kji)}'
            assert torch.all(~torch.isnan((1 / (S_jj * S_ii - S_ji * S_ji))))
            M_k = torch.sqrt((torch.linalg.det(S_kji).clamp_min(1e-8)[..., None] * (1 / (S_jj * S_ii - S_ji * S_ji).clamp_min(1e-8))))
            assert torch.all(~torch.isnan(M_k))
            assert torch.all(torch.isfinite(M_k))
            M_k = torch.concatenate((M_k, torch.zeros_like(M_k), torch.zeros_like(M_k)), dim=-1)
            M_j_1 = - (S_ki * S_ji - S_kj * S_ii) * torch.rsqrt(torch.clamp_min(S_jj * S_ii - S_ji * S_ji, 1e-8))
            assert torch.all(~torch.isnan(M_j_1))
            assert torch.all(torch.isfinite(M_j_1))
            M_j_2 = torch.sqrt(torch.clamp_min(S_jj * S_ii - S_ji * S_ji, 1e-8))
            assert torch.all(~torch.isnan(M_j_2))
            assert torch.all(torch.isfinite(M_j_2))
            M_j = torch.rsqrt(S_ii.clamp_min(1e-8)) * torch.concatenate((M_j_1, M_j_2, torch.zeros_like(M_j_1)), dim=-1)
            assert torch.all(~torch.isnan(M_j))
            assert torch.all(torch.isfinite(M_j))
            M_i = torch.rsqrt(S_ii.clamp_min(1e-8)) * torch.concatenate((S_ki, S_ji, S_ii), dim=-1)
            assert torch.all(~torch.isnan(M_i))
            assert torch.all(torch.isfinite(M_i))

            B = S_kk.shape[0]
            
            U_1 = torch.rand(B, batch_size, 1, device='cuda')
            U_2 = torch.rand(B, batch_size, 1, device='cuda')
            p_u = torch.sqrt(U_1) * torch.cos(2 * torch.pi * U_2)
            p_v = torch.sqrt(U_1) * torch.sin(2 * torch.pi * U_2)
            p_w = torch.sqrt(1 - p_u ** 2 - p_v ** 2)
            assert torch.all(~torch.isnan(p_u)), f'{U_1.min()}, {U_1.max()}, {U_2.min()}, {U_2.max()}'
            assert torch.all(~torch.isnan(p_v)), f'{U_1.min()}, {U_1.max()}, {U_2.min()}, {U_2.max()}'
            assert torch.all(~torch.isnan(p_w)), f'{(1 - p_u ** 2 - p_v ** 2).min()}'
            d_out_kji = p_u * M_k[:, None, :] + p_v * M_j[:, None, :] + p_w * M_i[:, None, :]
            assert torch.all(~torch.isnan(d_out_kji))
            d_out_kji_2 = torch.nn.functional.normalize(d_out_kji, dim=-1) # (N, b, 3)
            assert torch.all(~torch.isnan(d_out_kji_2)), f'{torch.linalg.vector_norm(d_out_kji, ord=2, dim=-1).min()}, {torch.linalg.vector_norm(d_out_kji, ord=2, dim=-1).max()}'
            return torch.nn.functional.normalize((torch.stack((omega_k, omega_j, omega_i), dim=-1) @ d_out_kji_2.transpose(-1, -2)).transpose(-1, -2), dim=-1).squeeze(-2) # (N, ?, 3)

        d_m = sample_VNDF().detach() # (N, 3)
        assert torch.all(~torch.isnan(d_m))
        # mask = (d_m * d_i).sum(dim=-1) > 0
        # d_m[mask] = -d_m[mask]
        d_o = (-d_i + 2.0 * d_m * (d_m * d_i).sum(dim=-1, keepdim=True)).contiguous().detach()
        assert torch.all((torch.linalg.norm(d_o, dim=-1) - 1.).abs() < 1E-5), f'{torch.linalg.norm(d_o, dim=-1).min()}'
        
        rho = self.calc_NDF(scaling, cov_inv, d_m) / (4.0 * integral)
        assert torch.all(~torch.isnan(rho))
        fused_rho = alpha * sigma_t * rho / (rho.detach() + 1E-8)
        
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