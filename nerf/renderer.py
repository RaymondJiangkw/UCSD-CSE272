import math
import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import raymarching
from .utils import custom_meshgrid

def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # bins: [B, T], old_z_vals
    # weights: [B, T - 1], bin weights.
    # return: [B, n_samples], new_z_vals

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples, (cdf_g[..., 1] - cdf_g[..., 0]) * torch.reciprocal((bins_g[..., 1] - bins_g[..., 0]).clamp_min(1e-3))


def plot_pointcloud(pc, color=None):
    # pc: [N, 3]
    # color: [N, 3/4]
    print('[visualize points]', pc.shape, pc.dtype, pc.min(0), pc.max(0))
    pc = trimesh.PointCloud(pc, color)
    # axis
    axes = trimesh.creation.axis(axis_length=4)
    # sphere
    sphere = trimesh.creation.icosphere(radius=1)
    trimesh.Scene([pc, axes, sphere]).show()


class NeRFRenderer(nn.Module):
    def __init__(self,
                 bound=1,
                 cuda_ray=False,
                 density_scale=1, # scale up deltas (or sigmas), to make the density grid more sharp. larger value than 1 usually improves performance.
                 min_near=0.2,
                 density_thresh=0.01,
                 bg_radius=-1,
                 ):
        super().__init__()

        self.bound = bound
        self.cascade = 1 + math.ceil(math.log2(bound))
        self.grid_size = 128
        self.density_scale = density_scale
        self.min_near = min_near
        self.density_thresh = density_thresh
        self.bg_radius = bg_radius # radius of the background sphere.

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        # extra state for cuda raymarching
        self.cuda_ray = cuda_ray
        self.sigma_majorant = 1. # Should be dynamically adjusted
        self.env_map = torch.nn.Parameter(torch.zeros(1, 3, 128, 256))
    
    def forward(self, x, d):
        raise NotImplementedError()

    # separated density and color query (can accelerate non-cuda-ray mode.)
    def density(self, x):
        raise NotImplementedError()

    def color(self, x, d, mask=None, **kwargs):
        raise NotImplementedError()

    def sample_env_map(self, inputs):
        # inputs = inputs / self.bound # (-1, 1)
        # inputs = inputs / torch.linalg.vector_norm(inputs, ord=2, dim=-1, keepdim=True)
        # assert torch.allclose(torch.linalg.vector_norm(inputs, ord=2, dim=-1), torch.ones_like(inputs[..., 0]))
        x, y, z = inputs.unbind(dim=-1)
        theta = (1 / np.pi) * torch.arctan2(x, -z).nan_to_num()[..., None] # (N, 1) in [-1, 1]
        phi = (2 * (1 / np.pi) * torch.arccos(y).nan_to_num() - 1)[..., None] # (N, 1) in [-1, 1]
        coords = torch.stack([ theta, phi ], dim=-1)[None, :, :, :] # (1, N, 1, 2)
        # print("Coords:", coords)
        Le = torch.exp(F.grid_sample(self.env_map.expand(-1, 3, -1, -1), coords, mode='bilinear', align_corners=False).view(3, -1).permute(1, 0)) # (N, 3)
        return Le

    def run(self, rays_o, rays_d, num_steps=128, upsample_steps=128, indicator_steps=16, bg_color=None, perturb=False, max_depths=2, rr_depth=5, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_color: [3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]
        
        # choose aabb
        aabb = self.aabb_train if self.training else self.aabb_infer

        max_sigma = -1
        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3) # (N, 3)
        rays_d = rays_d.contiguous().view(-1, 3) # (N, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device
        tot = num_steps
        rgbs = torch.zeros((N * tot, 3), device=device, dtype=torch.float32)

        b_rays_o = rays_o[:, None, :].expand(-1, tot, -1).contiguous().reshape(-1, 3) # (n, 3)
        b_rays_d = rays_d[:, None, :].expand(-1, tot, -1).contiguous().reshape(-1, 3) # (n, 3)
        b_rays_i = torch.arange(N * tot).to(device).long() # (n)
        current_throughput = torch.ones(len(b_rays_o), 3, device=device) # (n, 3)

        for current_depth in range(max_depths):
            if len(current_throughput) <= 0:
                break
            
            # b_nears, b_fars = raymarching.near_far_from_aabb(b_rays_o, b_rays_d, aabb, self.min_near)
            # assert len(b_nears.shape) == 1
            # assert len(b_fars.shape) == 1
            # b_nears = torch.clamp_min(b_nears, 1e-3)
            # b_fars = torch.clamp_min(b_fars, 1e-3)
            # || o + d * t ||_2^2 == bound^2
            hit_a = torch.square(b_rays_d).sum(dim=-1) # (n, )
            hit_b = 2.0 * (b_rays_d * b_rays_o).sum(dim=-1) # (n, )
            hit_c = torch.square(b_rays_o).sum(dim=-1) - self.bound ** 2
            hit_delta = hit_b ** 2 - 4.0 * hit_a * hit_c
            invalid_mask = hit_delta <= 0.
            b_rays_o = b_rays_o[~invalid_mask].contiguous()
            b_rays_d = b_rays_d[~invalid_mask].contiguous()
            b_rays_i = b_rays_i[~invalid_mask].contiguous()
            current_throughput = current_throughput[~invalid_mask].contiguous()
            hit_a = hit_a[~invalid_mask].contiguous()
            hit_b = hit_b[~invalid_mask].contiguous()
            hit_c = hit_c[~invalid_mask].contiguous()
            hit_delta = hit_delta[~invalid_mask].contiguous()
            b_nears = ((-hit_b - torch.sqrt(hit_delta)) / (2.0 * hit_a)).clamp_min(1e-3)
            b_fars = ((-hit_b + torch.sqrt(hit_delta)) / (2.0 * hit_a)).clamp_min(1e-3)

            _z_vals = torch.linspace(0.0, 1.0, indicator_steps, device=device).unsqueeze(0) # [1, T]
            _z_vals = _z_vals.expand((len(b_rays_o), indicator_steps)) # [N, T]
            _z_vals = b_nears[:, None] + (b_fars - b_nears)[:, None] * _z_vals # [N, T], in [nears, fars]
            _sample_dist = (b_fars - b_nears)[:, None] / indicator_steps
            _z_vals = _z_vals + (torch.rand(_z_vals.shape, device=device) - 0.5) * _sample_dist
            _sigma = self.density((b_rays_o[:, None, :] + b_rays_d[:, None, :] * _z_vals[:, :, None]).reshape(-1, 3), b_rays_d[:, None, :].expand(-1, indicator_steps, -1).reshape(-1, 3)).reshape(*_z_vals.shape, -1) # [N, T, 3]
            _channel = torch.randint(_sigma.size(-1), (len(_sigma), 1, 1), device='cuda') # [N, 1, 1]
            b_rays_majorant_multi = torch.clamp_min(torch.max(_sigma, dim=1).values, 1e-3) # [N, 3]
            b_rays_majorant = torch.clamp_min(torch.max(torch.gather(
                _sigma, -1, _channel.expand(_sigma.shape[0], _sigma.shape[1], 1)
            )[:, :, 0], dim=-1).values, 1e-3) # [N, 1]
            if current_depth < max_depths - 1:
                b_rays_t = ((torch.log(1 - torch.rand(len(b_rays_o), device=device)) / -b_rays_majorant) + b_nears)[:, None] # (n, 1)
            else:
                b_rays_t = b_fars[:, None]
            # print('b_rays_t:', b_rays_t.min(), b_rays_t.max())
            hit_mask = (b_rays_t >= b_fars[:, None]).squeeze(dim=-1) # (n, )
            if hit_mask.sum() > 0:
                denom = torch.exp(- b_rays_majorant_multi[hit_mask] * (b_fars[:, None][hit_mask] - b_nears[:, None][hit_mask])) / (torch.max(b_rays_majorant_multi[hit_mask], dim=-1, keepdim=True).values + 1E-8) + 1E-8
                # print('denom:', denom.min(), denom.max())
                assert torch.all(~torch.isnan(current_throughput))
                assert torch.all(~torch.isnan(self.sample_env_map(b_rays_o[hit_mask] + b_rays_d[hit_mask] * b_fars[:, None][hit_mask])))
                assert torch.all(~torch.isnan(denom))
                assert torch.all(~torch.isnan(denom / denom.detach().mean(dim=-1, keepdim=True))), f'{denom.min()}, {denom.max()}, {denom.detach().mean(dim=-1, keepdim=True).min()}, {denom.detach().mean(dim=-1, keepdim=True).max()}'
                rgbs[b_rays_i[hit_mask]] = rgbs[b_rays_i[hit_mask]] + current_throughput[hit_mask] * (denom / denom.detach().mean(dim=-1, keepdim=True)) * self.sample_env_map(b_rays_o[hit_mask] + b_rays_d[hit_mask] * b_fars[:, None][hit_mask])

            if hit_mask.sum() == len(hit_mask):
                break
            
            b_rays_o = b_rays_o[~hit_mask].contiguous()  # (N, 3)
            b_rays_d = b_rays_d[~hit_mask].contiguous()  # (N, 3)
            b_rays_i = b_rays_i[~hit_mask].contiguous()  # (N, )
            b_rays_t = b_rays_t[~hit_mask].contiguous()  # (N, 1)
            b_rays_majorant = b_rays_majorant[~hit_mask].contiguous() # (N, )
            b_rays_majorant_multi = b_rays_majorant_multi[~hit_mask].contiguous() # (N, 3)
            b_nears  = b_nears[~hit_mask][:, None].contiguous()   # (N, 1)
            b_fars   = b_fars[~hit_mask][:, None].contiguous()    # (N, 1)
            current_throughput = current_throughput[~hit_mask].contiguous()  # (N, 3)
            _channel = _channel[~hit_mask].contiguous() # (N, 1, 1)
            transmittance = torch.exp(- b_rays_majorant_multi * (b_rays_t - b_nears) ) / torch.max(b_rays_majorant_multi, dim=-1, keepdim=True).values
            denom = (b_rays_majorant_multi * transmittance).mean(dim=-1, keepdim=True).detach()
            current_throughput = current_throughput * (transmittance / denom)
            assert torch.all(~torch.isnan(current_throughput))
            b_rays_o = (b_rays_o + b_rays_d * b_rays_t).detach()
            out = self.forward(b_rays_o, b_rays_d)
            # update max sigma
            max_sigma = max(max_sigma, b_rays_majorant.max().item())

            scatter_prob = torch.clamp_max(torch.gather(out["sigma_t"], -1, _channel[:, :, 0]).squeeze(1) / b_rays_majorant, 1.0)
            scatter_mask = torch.rand_like(scatter_prob) < scatter_prob

            # Hit Real Particles
            b_rays_d = b_rays_d.clone()
            b_rays_d[scatter_mask] = out["d_out"][scatter_mask]
            current_throughput = current_throughput.clone()
            current_throughput[scatter_mask] = (current_throughput[scatter_mask] / (scatter_prob[:, None][scatter_mask] + 1E-8)) * out["fused_rho"][scatter_mask]
            assert torch.all(~torch.isnan(current_throughput))

            # Hit Fake Particles
            current_throughput = current_throughput.clone()
            current_throughput[~scatter_mask] = current_throughput[~scatter_mask] / (1 - scatter_prob[:, None][~scatter_mask] + 1E-8) # * torch.clamp_min(b_rays_majorant[~scatter_mask][:, None] - out["sigma_t"][~scatter_mask], 0.)
            assert torch.all(~torch.isnan(current_throughput))

            # Russian roulette
            if current_depth >= rr_depth:
                rr_prob = torch.max(current_throughput, dim=-1).values
                terminate = torch.rand_like(current_throughput[:, 0]) > rr_prob
                b_rays_o = b_rays_o[~terminate]
                b_rays_d = b_rays_d[~terminate]
                b_rays_i = b_rays_i[~terminate]
                b_rays_t = b_rays_t[~terminate]
                rr_prob = rr_prob[~terminate]
                current_throughput = current_throughput[~terminate] * torch.reciprocal(rr_prob)[:, None]
        assert torch.all(~torch.isnan(rgbs))
        rgbs = rgbs.reshape(N, tot, 3).mean(dim=1)

        # calculate color
        image = rgbs # [N, 3], in [0, 1]

        image = image.view(*prefix, 3)
        
        return {
            'image': image,
            'max_sigma': max_sigma, 
        }

    def render(self, rays_o, rays_d, staged=False, max_ray_batch=4096, show_progress_bar=False, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]

        B, N = rays_o.shape[:2]
        device = rays_o.device

        image = torch.empty((B, N, 3), device=device)
        max_sigma = -1
        for b in range(B):
            head = 0
            while head < N:
                tail = min(head + max_ray_batch, N)
                results_ = self.run(rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], **kwargs)
                max_sigma = max(max_sigma, results_["max_sigma"])
                image[b:b+1, head:tail] = results_['image']
                head += max_ray_batch
        
        results = {}
        results['image'] = image

        # Dynamically adjust the majorant
        self.sigma_majorant = max_sigma

        return results