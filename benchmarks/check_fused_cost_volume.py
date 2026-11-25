"""Compare the depth-blocked Triton forward and recomputation gradients to eager."""

import json

import torch

from src.model.encoder.unimatch.fused_cost_volume import cost_volume_chunk
from src.model.encoder.unimatch.matching import warp_with_pose_depth_candidates


def main():
    torch.manual_seed(321)
    torch.set_float32_matmul_precision("highest")
    bv, neighbors, channels, depth, height, width = 4, 2, 128, 19, 17, 21
    ref = torch.randn(bv, channels, height, width, device="cuda", requires_grad=True)
    tgt = torch.randn(bv, neighbors, channels, height, width, device="cuda", requires_grad=True)
    intrinsics = torch.eye(3, device="cuda").repeat(bv * neighbors, 1, 1)
    intrinsics[:, 0, 0] = 18
    intrinsics[:, 1, 1] = 16
    intrinsics[:, 0, 2] = 10
    intrinsics[:, 1, 2] = 8
    pose = torch.eye(4, device="cuda").repeat(bv * neighbors, 1, 1)
    pose[:, :2, 3] = torch.randn(bv * neighbors, 2, device="cuda") * 0.65
    depths = torch.rand(bv * neighbors, depth, height, width, device="cuda") * 3 + 0.25
    warped = warp_with_pose_depth_candidates(tgt.flatten(0, 1), intrinsics, pose, depths, grid_sample_disable_cudnn=True)
    expected = ((ref[:, None, :, None] * warped.unflatten(0, (bv, neighbors))).sum(2)
                / (channels ** 0.5)).mean(1)
    actual = torch.cat([
        cost_volume_chunk(ref, tgt, intrinsics, pose, part.contiguous(), True)
        for part in depths.split(8, dim=1)
    ], dim=1)
    gradient = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(expected, (ref, tgt), gradient)
    actual_grads = torch.autograd.grad(actual, (ref, tgt), gradient)
    metrics = {}
    for name, ref_tensor, candidate in zip(("cost", "dref", "dtgt"),
                                            (expected, *expected_grads), (actual, *actual_grads)):
        diff = candidate - ref_tensor
        metrics[name] = {
            "max_abs": diff.abs().max().item(),
            "relative_l2": (diff.norm() / ref_tensor.norm()).item(),
        }
        torch.testing.assert_close(candidate, ref_tensor, atol=4e-5, rtol=3e-5)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
