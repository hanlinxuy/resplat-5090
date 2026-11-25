"""Fused FP32 plane-sweep sampling and channel correlation for one depth group."""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from .matching import project_depth_grid, sample_projected_features


@triton.jit
def _warp_correlation(
    Ref, Target, Grid, Partial,
    V: tl.constexpr, C: tl.constexpr, D: tl.constexpr,
    H: tl.constexpr, W: tl.constexpr,
    BLOCK_PIXELS: tl.constexpr, BLOCK_CHANNELS: tl.constexpr,
):
    pixels = tl.program_id(0) * BLOCK_PIXELS + tl.arange(0, BLOCK_PIXELS)
    channel = tl.arange(0, BLOCK_CHANNELS)
    pair = tl.program_id(1)
    spatial: tl.constexpr = H * W
    elements: tl.constexpr = D * spatial
    valid = pixels < elements
    grid_offset = (pair * elements + pixels) * 2
    x = (tl.load(Grid + grid_offset, valid, 0) + 1.0) * (0.5 * (W - 1))
    y = (tl.load(Grid + grid_offset + 1, valid, 0) + 1.0) * (0.5 * (H - 1))
    x0, y0 = tl.floor(x).to(tl.int32), tl.floor(y).to(tl.int32)
    dx, dy = x - x0, y - y0
    offset = (pair * C + channel[:, None]) * spatial
    in_bounds = (channel[:, None] < C) & valid[None, :]
    x0_ok, x1_ok = (x0 >= 0) & (x0 < W), (x0 >= -1) & (x0 < W - 1)
    y0_ok, y1_ok = (y0 >= 0) & (y0 < H), (y0 >= -1) & (y0 < H - 1)
    base = offset + (y0 * W + x0)[None, :]
    v00 = tl.load(Target + base, in_bounds & (x0_ok & y0_ok)[None, :], 0)
    v01 = tl.load(Target + base + 1, in_bounds & (x1_ok & y0_ok)[None, :], 0)
    v10 = tl.load(Target + base + W, in_bounds & (x0_ok & y1_ok)[None, :], 0)
    v11 = tl.load(Target + base + W + 1, in_bounds & (x1_ok & y1_ok)[None, :], 0)
    sampled = v00 * ((1.0 - dx) * (1.0 - dy))[None, :]
    sampled += v01 * (dx * (1.0 - dy))[None, :]
    sampled += v10 * ((1.0 - dx) * dy)[None, :]
    sampled += v11 * (dx * dy)[None, :]
    ref_offset = ((pair // V) * C + channel[:, None]) * spatial + pixels[None, :] % spatial
    reference = tl.load(Ref + ref_offset, in_bounds, 0)
    corr = tl.sum(reference * sampled, axis=0) * (C ** -0.5)
    tl.store(Partial + pair * elements + pixels, corr, valid)


class _CostVolumeChunk(torch.autograd.Function):
    @staticmethod
    def forward(ctx, reference, target, intrinsics, pose, depths, disable_cudnn):
        batch_views, neighbors, channels, height, width = target.shape
        depth_count = depths.size(1)
        grid = project_depth_grid(intrinsics, pose, depths)
        partial = reference.new_empty((batch_views, neighbors, depth_count, height, width))
        _warp_correlation[(triton.cdiv(depth_count * height * width, 32), batch_views * neighbors)](
            reference, target, grid, partial, neighbors, channels, depth_count,
            height, width, 32, triton.next_power_of_2(channels),
            num_warps=4, enable_fp_fusion=False,
        )
        ctx.save_for_backward(reference, target, intrinsics, pose, depths)
        ctx.disable_cudnn = disable_cudnn
        return partial.mean(dim=1)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        reference, target, intrinsics, pose, depths = ctx.saved_tensors
        # The original projection has no camera/depth gradient. Reuse the original
        # differentiable sampler and FP32 reduction on only this group's materialization.
        with torch.enable_grad():
            ref = reference.detach().requires_grad_()
            tgt = target.detach().requires_grad_()
            grid = project_depth_grid(intrinsics, pose, depths)
            warped = sample_projected_features(tgt.flatten(0, 1), grid, ctx.disable_cudnn)
            warped = warped.unflatten(0, target.shape[:2])
            corr = (ref[:, None, :, None] * warped).sum(dim=2) / (ref.size(1) ** 0.5)
            dref, dtgt = torch.autograd.grad(corr.mean(dim=1), (ref, tgt), gradient)
        return dref, dtgt, None, None, None, None


cost_volume_chunk = _CostVolumeChunk.apply
