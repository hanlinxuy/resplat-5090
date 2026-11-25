"""Warm-start a 4×4 subpixel ReSplat head from a one-Gaussian checkpoint."""

import argparse
import math
from pathlib import Path

import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=True)
    weights = checkpoint["state_dict"]
    prefix = "encoder.gaussian_head."
    first = weights[prefix + "0.weight"]
    second = weights[prefix + "2.weight"]
    channels = first.shape[0]
    if second.shape != (channels, channels) or channels <= 10 or (channels - 10) % 3:
        raise ValueError("Expected a one-Gaussian head with quaternion, scale, opacity, offset and RGB SH")

    # The encoder interprets each attribute as (channel, subpixel_x, subpixel_y).
    side = 4
    copies = side**2
    weights[prefix + "0.weight"] = first.repeat_interleave(copies, dim=0)
    weights[prefix + "0.bias"] = weights[prefix + "0.bias"].repeat_interleave(copies)
    subpixel_identity = torch.eye(copies, dtype=second.dtype)
    weights[prefix + "2.weight"] = (
        second[:, None, :, None] * subpixel_identity[None, :, None, :]
    ).reshape(channels * copies, channels * copies)
    final_bias = weights[prefix + "2.bias"].clone()
    # Approximate a quarter of the original axis scale in the exponential
    # region of softplus. Keep the learned opacity and SH residuals.
    final_bias[4:7] -= math.log(side)
    weights[prefix + "2.bias"] = final_bias.repeat_interleave(copies)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": weights}, args.output)
    print(f"Expanded {channels} head channels to {channels * copies} in {args.output}")
