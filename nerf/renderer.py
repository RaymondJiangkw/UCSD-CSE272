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

    return samples


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
        self.sigma_majorant = 0.1 # Should be dynamically adjusted
        self.env_map = torch.nn.Parameter(torch.rand(1, 3, 16, 32))
    
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

    def run(self, rays_o, rays_d, num_steps=128, upsample_steps=128, bg_color=None, perturb=False, max_depths=2, rr_depth=5, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_color: [3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]

        BOUND = self.bound

        max_sigma = -1
        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3) # (N, 3)
        rays_d = rays_d.contiguous().view(-1, 3) # (N, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device
        tot = num_steps + upsample_steps
        rgbs = torch.zeros((N, 3), device=device, dtype=torch.float32)

        b_rays_o = rays_o[:, None, :].expand(-1, tot, -1).reshape(-1, 3) # (n, 3)
        b_rays_d = rays_d[:, None, :].expand(-1, tot, -1).reshape(-1, 3) # (n, 3)
        b_rays_i = torch.arange(N).to(device).long().reshape(-1, 1).expand(-1, tot).reshape(-1) # (n)
        current_throughput = torch.ones(len(b_rays_o), 1, device=device) # (n, 1)

        for current_depth in range(max_depths):
            if len(current_throughput) <= 0:
                break

            # Hit Environment Map Bound
            # || rays_o + rays_d * l ||^2_2 == bound ^ 2
            hit_quadratic_a = torch.square(b_rays_d).sum(dim=-1) # (N, )
            hit_quadratic_c = torch.square(b_rays_o).sum(dim=-1) - BOUND ** 2 # (N, )
            hit_quadratic_b = 2.0 * (b_rays_o * b_rays_d).sum(dim=-1) # (N, )
            hit_delta = hit_quadratic_b * hit_quadratic_b - 4.0 * hit_quadratic_a * hit_quadratic_c
            invalid_mask = hit_delta <= 0.
            hit_t = ((- hit_quadratic_b + torch.sqrt(hit_delta.clamp_min(0.))) / (2.0 * hit_quadratic_a))[:, None] # (N, 1)
            invalid_mask = torch.logical_or(invalid_mask, hit_t.squeeze(-1) < 0.)

            hit_t = hit_t[~invalid_mask]
            b_rays_o = b_rays_o[~invalid_mask]
            b_rays_d = b_rays_d[~invalid_mask]
            b_rays_i = b_rays_i[~invalid_mask]
            current_throughput = current_throughput[~invalid_mask]
            hit_pts = b_rays_o + b_rays_d * hit_t # (N, 3)

            if current_depth < max_depths - 1:
                b_rays_t = (torch.log(1 - torch.rand(len(b_rays_o), 1, device=device)) / -self.sigma_majorant) # (n, 1)
            else:
                b_rays_t = hit_t
            hit_mask = (b_rays_t >= hit_t).squeeze(dim=-1) # (n, )
            rgbs[b_rays_i[hit_mask]] = rgbs[b_rays_i[hit_mask]] + current_throughput[hit_mask] * torch.exp(-self.sigma_majorant * hit_t[hit_mask]) * self.sample_env_map(hit_pts[hit_mask])
            # print()
            # print(f"Current Depth: {current_depth} with Hitting ratio {hit_mask.sum() / len(hit_mask)}")

            if hit_mask.sum() == len(hit_mask):
                break

            hit_t = hit_t[~hit_mask]
            b_rays_o = b_rays_o[~hit_mask]
            b_rays_d = b_rays_d[~hit_mask]
            b_rays_i = b_rays_i[~hit_mask]
            b_rays_t = b_rays_t[~hit_mask]
            current_throughput = current_throughput[~hit_mask]
            assert torch.all(~torch.isnan(b_rays_o))
            assert torch.all(~torch.isnan(b_rays_d))
            assert torch.all(~torch.isnan(b_rays_t)), f"{b_rays_t}, {self.sigma_majorant}"
            b_rays_o = b_rays_o + b_rays_d * b_rays_t
            out = self.forward(b_rays_o, b_rays_d.clone())
            current_throughput = current_throughput / self.sigma_majorant  # Transmittance term cancels out
            # update max sigma
            max_sigma = max(max_sigma, out["raw_sigma_t"].max().item())
            scatter_prob = out["sigma_t"] / self.sigma_majorant # (n, 1)
            scatter_mask = (torch.rand_like(scatter_prob) < scatter_prob).squeeze(-1) # (n, )

            # Hit Real Particles
            rgbs[b_rays_i[scatter_mask]] = rgbs[b_rays_i[scatter_mask]] + current_throughput[scatter_mask] * out["Le"][scatter_mask]

            if current_depth == max_depths - 1:
                break

            current_throughput[scatter_mask] = current_throughput[scatter_mask] * out["sigma_s"][scatter_mask] * out["rho"][scatter_mask] * torch.reciprocal(out["pdf_d_out"][scatter_mask])
            b_rays_d[scatter_mask] = out["d_out"][scatter_mask].to(b_rays_d.dtype)
            assert torch.all(~torch.isnan(out["d_out"])), f"{out['d_out'].min()}, {out['d_out'].max()}"

            # Hit Fake Particles
            current_throughput[~scatter_mask] = current_throughput[~scatter_mask] * (self.sigma_majorant - out["sigma_t"][~scatter_mask])

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
        # print(rgbs, rgbs / tot)
        
        rgbs = rgbs / tot

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
        self.sigma_majorant = max(max_sigma, 1e-3)

        return results