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
        self.env_map = torch.nn.Parameter(torch.randn(1, 3, 128, 256) * 0.002)
    
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
        coords = torch.stack([ phi, theta ], dim=-1)[None, :, :, :] # (1, N, 1, 2)
        # print("Coords:", coords)
        Le = torch.exp(F.grid_sample(self.env_map, coords, mode='bilinear', align_corners=False).view(3, -1).permute(1, 0)) # (N, 3)
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
        rgbs = torch.zeros((N, 3), device=device, dtype=torch.float32)

        b_rays_o = rays_o[:, None, :].expand(-1, tot, -1).reshape(-1, 3) # (n, 3)
        b_rays_d = rays_d[:, None, :].expand(-1, tot, -1).reshape(-1, 3) # (n, 3)
        
        def evaluate_radiance(b_rays_o, b_rays_d, current_depth):
            # || o + d * t ||_2^2 == bound^2
            hit_a = torch.square(b_rays_d).sum(dim=-1) # (n, )
            hit_b = 2.0 * (b_rays_d * b_rays_o).sum(dim=-1) # (n, )
            hit_c = torch.square(b_rays_o).sum(dim=-1) - self.bound ** 2
            hit_delta = torch.clamp_min(hit_b ** 2 - 4.0 * hit_a * hit_c, 0.)
            b_nears = ((-hit_b - torch.sqrt(hit_delta)) / (2.0 * hit_a))[:, None]
            b_fars = ((-hit_b + torch.sqrt(hit_delta)) / (2.0 * hit_a))[:, None]

            z_vals = torch.lerp(b_nears, b_fars, torch.linspace(0., 1., upsample_steps, device=b_rays_o.device)[None, :]) # (N, M)
            # Perturb sampled positions
            sample_dist = (b_fars - b_nears) / upsample_steps # (N, 1)
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
            # Upsample sampled positions (Importance Sampling)
            with torch.no_grad():
                # print(b_rays_o.shape, b_rays_d.shape, z_vals.shape, (b_rays_o[:, None, :] + b_rays_d[:, None, :] * z_vals[:, :, None]).shape)
                sigma = self.density((b_rays_o[:, None, :] + b_rays_d[:, None, :] * z_vals[:, :, None]).reshape(-1, 3), b_rays_d[:, None, :].expand(-1, upsample_steps, -1).reshape(-1, 3)).reshape_as(z_vals) # (N, M)
                deltas = z_vals[..., 1:] - z_vals[..., :-1] # (N, M-1)
                deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1) # (N, M)
                alphas = 1 - torch.exp(-deltas * self.density_scale * sigma) # (N, M)
                alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # (N, M+1)
                weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # (N, M)
                z_vals_mid = (z_vals[..., :-1] + 0.5 * deltas[..., :-1]) # (N, M-1)
                new_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], upsample_steps, det=not self.training)[0].detach() # (N, M)
                z_vals = torch.sort(torch.cat((z_vals, new_z_vals), dim=-1), dim=-1).values # (N, 2 * M)
            b_rays_p = (b_rays_o[:, None, :] + b_rays_d[:, None, :] * z_vals[:, :, None]) # (N, 2 * M, 3)
            if current_depth == max_depths - 1:
                sigma = self.density(b_rays_p.reshape(-1, 3), b_rays_d[:, None, :].expand(-1, 2 * upsample_steps, -1)).reshape_as(z_vals) # (N, 2 * M)
                deltas = z_vals[..., 1:] - z_vals[..., :-1] # (N, 2 * M - 1)
                deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1) # (N, 2 * M)
                alphas = 1 - torch.exp(-deltas * self.density_scale * sigma) # (N, 2 * M)
                transmittance = alphas[:, -1] * torch.prod((1 - alphas + 1e-15)[..., :-1], dim=-1) # (N, )
                b_rays_h = b_rays_o + b_rays_d * b_fars
                return transmittance[:, None] * self.sample_env_map(b_rays_h)
            else:
                out = self.forward(b_rays_p.reshape(-1, 3), b_rays_d[:, None, :].expand(-1, 2 * upsample_steps, -1).reshape(-1, 3)) # (N, 2 * M)
                sigma = out["sigma_t"].reshape_as(z_vals) # (N, 2 * M)
                deltas = z_vals[..., 1:] - z_vals[..., :-1] # (N, 2 * M - 1)
                deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1) # (N, 2 * M)
                alphas = 1 - torch.exp(-deltas * self.density_scale * sigma) # (N, 2 * M)
                alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1) # (N, 2 * M + 1)
                weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1] # (N, 2 * M)
                radiance = evaluate_radiance(b_rays_p.reshape(-1, 3), out["d_out"], current_depth + 1)
                print(weights.shape, b_rays_p.shape, radiance.shape)
                radiance = radiance.reshape(*weights.shape, 3)
                return (weights[:, :, None] * out["fused_rho"].reshape(*weights.shape, -1) * radiance).sum(dim=1)

        rgbs = evaluate_radiance(b_rays_o, b_rays_d, 0).reshape(N, tot, -1).mean(dim=1)

        # calculate color
        image = rgbs # [N, 3], in [0, 1]

        image = image.view(*prefix, 3)
        
        return {
            'image': image,
            'max_sigma': max_sigma, 
        }

    def render(self, rays_o, rays_d, staged=False, max_ray_batch=4096, **kwargs):
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