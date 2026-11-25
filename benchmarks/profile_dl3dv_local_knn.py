"""Profile the DL3DV point-transformer shape on synthetic calibrated scenes."""

import argparse
import importlib.util
import json
from pathlib import Path
from statistics import median

import torch

from src.model.encoder.point_transformer.local_knn import local_knn_query


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--transformer", action="store_true")
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--packed-kv", action="store_true")
    variants.add_argument("--fused-local-rows", type=int, choices=(1, 2, 4, 8))
    parser.add_argument("--channels-last-conv", action="store_true")
    parser.add_argument("--fused-knn", action="store_true")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    b, v, h, w, c, d, k = args.batch, 8, 64, 112, 512, 64, 16
    n = v * h * w
    camera = torch.eye(4, device=device).expand(b, v, 4, 4).clone()
    camera[..., 0, 3] = torch.arange(v, device=device) * 0.028
    camera[..., 1, 3] = torch.arange(b, device=device).view(b, 1) * 0.11
    intrinsics = torch.tensor([[0.85, 0., 0.5], [0., 0.85, 0.5], [0., 0., 1.]],
                              device=device).expand(b, v, 3, 3).contiguous()
    rows, cols = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    z = 1.5 + torch.rand((b, v, h, w), device=device) * 0.75
    p = torch.stack((((cols + 0.5) / w - 0.5) * z / 0.85,
                     ((rows + 0.5) / h - 0.5) * z / 0.85, z), dim=-1)
    p = (p + camera[..., :3, 3][:, :, None, None]).reshape(-1, 3).contiguous()
    shape = (v, h, w)
    model = None
    report = dict(device=torch.cuda.get_device_name(), cc=torch.cuda.get_device_capability(),
                  torch=torch.__version__, batch=b, shape=shape, candidates=391, chunk=args.chunk,
                  fused_knn=args.fused_knn)

    def neighbors():
        if model is not None:
            return model.compute_knn(p, offsets, b=b, v=v, h=h, w=w,
                                     extrinsics=camera, intrinsics=intrinsics)
        return local_knn_query(k, p, camera, intrinsics, *shape, num_neighbor_views=32,
                               query_chunk_size=args.chunk, fused_candidates=args.fused_knn)

    if args.transformer:
        from src.model.encoder.point_transformer.layer import PlainPointTransformer

        model = PlainPointTransformer(c, k, num_blocks=6, attn_proj_channels=d, with_mv_attn=True,
                                      with_mv_attn_lowres=True, use_local_knn=True,
                                      local_knn_num_neighbor_views=32).cuda().train()
        model.fused_knn_candidates = args.fused_knn
        features = torch.randn(b * n, c, device=device, requires_grad=True)
        grad = torch.randn_like(features)
        offsets = torch.arange(1, b + 1, device=device, dtype=torch.int32) * n
        params = tuple(model.parameters())
        low = n // 16
        local_flops = 2 * n * (4*c*d + 8*c*c + 2*k*d)
        multiview_flops = 2 * (low * (32*c*c + 12*c*c) + 9*n*c*c + 2*low*low*c)
        report["useful_train_flops"] = b * 6 * 3 * (local_flops + multiview_flops)
        report["model_params"] = sum(p.numel() for p in params)

    def run(query=neighbors):
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        events[0].record()
        with torch.cuda.nvtx.range("resplat/knn"), torch.autocast("cuda", dtype=torch.bfloat16):
            idx = query()
        events[1].record()
        if model is not None:
            with torch.cuda.nvtx.range("resplat/point-fwd"), torch.autocast("cuda", dtype=torch.bfloat16):
                y = model((p, features, offsets), b=b, v=v, h=h, w=w, cached_knn_idx=idx)
            events[2].record()
            with torch.cuda.nvtx.range("resplat/point-bwd"):
                grads = torch.autograd.grad(y, (features, *params), grad)
            events[3].record()
            events[3].synchronize()
            return idx, [events[0].elapsed_time(events[i]) for i in (1, 2, 3)], (y, *grads)
        events[1].synchronize()
        return idx, [events[0].elapsed_time(events[1])], ()

    if args.original:
        spec = importlib.util.spec_from_file_location("original_knn", args.original)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref = torch.cat([original.local_knn_query(
                k, p[i*n:(i+1)*n], camera[i], intrinsics[i], *shape, num_neighbor_views=32,
            ).add_(i*n) for i in range(b)])
        new = run()[0]
        report["different_index_elements"] = (new != ref).sum().item()
        assert torch.equal(new, ref)
        del new, ref
        if model is None:
            def old():
                return original.local_knn_query(k, p[:n], camera[0], intrinsics[0],
                                                *shape, num_neighbor_views=32)

            for label, query in (("original_one_scene", old), ("new", neighbors)):
                run(query)
                start_allocated = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                times = [run(query)[1][0] for _ in range(args.steps)]
                report[label] = dict(median_ms=median(times), samples_ms=times,
                                     peak_extra_bytes=torch.cuda.max_memory_allocated() - start_allocated)

    idx, _, check = run()
    idx_by_scene = idx.view(b, n, k)
    origin = torch.arange(b, device=device).view(b, 1, 1) * n
    assert ((idx_by_scene >= origin) & (idx_by_scene < origin + n)).all().item()
    assert all(torch.isfinite(t).all().item() for t in check)
    del idx, idx_by_scene, check
    if args.packed_kv or args.fused_local_rows or args.channels_last_conv:
        if model is None:
            parser.error("attention variants require --transformer")
        _, _, reference = run()
        for block in model.blocks:
            block.attn.packed_kv = args.packed_kv
            block.attn.fused_local_rows = args.fused_local_rows
        if args.channels_last_conv:
            for block in model.mv_blocks:
                block.channels_last_conv = True
                block.conv.to(memory_format=torch.channels_last)
        _, _, candidate = run()
        errors = [
            ((a.float() - b.float()).norm() / a.float().norm().clamp_min(1e-12)).item()
            for a, b in zip(reference, candidate)
        ]
        report["attention_variant"] = "packed_kv" if args.packed_kv else (
            f"fused_rows_{args.fused_local_rows}" if args.fused_local_rows else "baseline"
        )
        report["channels_last_conv"] = args.channels_last_conv
        report["attention_validation"] = dict(output_relative_l2=errors[0],
                                              input_grad_relative_l2=errors[1],
                                              max_parameter_grad_relative_l2=max(errors[2:]))
        assert max(errors) < 0.02, report["attention_validation"]
        del reference, candidate
    run()
    torch.cuda.reset_peak_memory_stats()
    if args.capture:
        torch.cuda.profiler.start()
    samples = [run()[1] for _ in range(args.steps)]
    if args.capture:
        torch.cuda.profiler.stop()
    report["samples_ms_cumulative"] = samples
    report["knn_ms"] = median(row[0] for row in samples)
    if model is not None:
        report["point_forward_ms"] = median(row[1]-row[0] for row in samples)
        report["point_backward_ms"] = median(row[2]-row[1] for row in samples)
        report["point_total_ms"] = median(row[2] for row in samples)
        report["measured_tflops"] = report["useful_train_flops"] / (report["point_total_ms"] * 1e9)
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("samples")}), flush=True)


if __name__ == "__main__":
    main()
