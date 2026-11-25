"""Time exact gsplat variants on tensors captured from a ReSplat training forward."""

import argparse
import json
from pathlib import Path
from statistics import median

import torch
from gsplat.rendering import rasterization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    torch.manual_seed(731)
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
    spec = {name: value.cuda() if isinstance(value, torch.Tensor) else value
            for name, value in fixture["args"].items()}
    names = ["means", "opacities", "colors"]
    names += ["covars"] if spec["covars"] is not None else ["quats", "scales"]
    inputs = tuple(spec[name].requires_grad_() for name in names)
    reference = None
    adjoints = None
    report = dict(device=torch.cuda.get_device_name(), visible_pairs=fixture["visible_pairs"],
                  intersections=fixture["intersections"], gaussian_shape=list(spec["means"].shape),
                  camera_shape=list(spec["viewmats"].shape), variants={})
    for packed, tile_size in ((True, 16), (False, 16), (True, 8), (True, 32)):
        with torch.cuda.nvtx.range(f"raster/packed{packed}/tile{tile_size}"):
            samples = []
            for iteration in range(2 + args.steps):
                start, forward, backward = (torch.cuda.Event(enable_timing=True) for _ in range(3))
                start.record()
                color, alpha, _ = rasterization(**{**spec, "packed": packed, "tile_size": tile_size})
                forward.record()
                if adjoints is None:
                    adjoints = (torch.randn_like(color), torch.randn_like(alpha))
                grads = torch.autograd.grad((color, alpha), inputs, adjoints)
                backward.record()
                backward.synchronize()
                if iteration == 0:
                    actual = (color.detach(), alpha.detach(), *(g.detach() for g in grads))
                    assert all(torch.isfinite(x).all().item() for x in actual)
                    if reference is None:
                        reference = actual
                    error = {
                        name: ((a - b).norm() / a.norm().clamp_min(1e-12)).item()
                        for name, a, b in zip(("color_depth", "alpha", *names), reference, actual)
                    }
                    del actual
                if iteration >= 2:
                    samples.append((start.elapsed_time(forward), forward.elapsed_time(backward),
                                    start.elapsed_time(backward)))
                del color, alpha, grads
        label = f"{'packed' if packed else 'dense'}_tile{tile_size}"
        report["variants"][label] = dict(
            forward_ms=median(row[0] for row in samples), backward_ms=median(row[1] for row in samples),
            total_ms=median(row[2] for row in samples), samples=samples, relative_l2=error,
        )
        print(label, report["variants"][label], flush=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
