"""Render held-out views from the 448×224 training checkpoint."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image

from scripts.infer_colmap import build_model
from src.evaluation.metrics import compute_psnr, compute_ssim


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    height, width = 224, 448
    encoder, decoder, shim = build_model(
        "dl3dv", args.checkpoint, 0, (height, width),
        ["model.encoder.shim_patch_size=4", "model.encoder.init_use_local_knn=true",
         "model.encoder.init_fused_local_attention_rows=4",
         "model.encoder.init_fused_knn_candidates=true",
         "model.encoder.init_channels_last_conv=true"],
        torch.device("cuda"),
    )
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
    batch = shim({
        name: {field: value.unsqueeze(0).cuda() for field, value in fixture[name].items()}
        for name in ("context", "target")
    })
    context, target = batch["context"], batch["target"]
    context["image"] = context["image"].float() / 255
    target["image"] = target["image"].float() / 255
    encoded = encoder(context, 64, deterministic=False)
    gaussian = encoded["gaussians"] if isinstance(encoded, dict) else encoded
    output = decoder.forward(
        gaussian, target["extrinsics"], target["intrinsics"], target["near"], target["far"],
        (height, width), depth_mode=None,
    ).color[0].float().clamp(0, 1)
    truth = target["image"][0].float()
    psnr, ssim = compute_psnr(truth, output), compute_ssim(truth, output)
    result = dict(scene=fixture["scene"], gaussians=gaussian.means.size(1),
                  context=context["index"][0].cpu().tolist(), target=target["index"][0].cpu().tolist(),
                  psnr=psnr.mean().item(), ssim=ssim.mean().item(),
                  psnr_per_view=psnr.cpu().tolist(), ssim_per_view=ssim.cpu().tolist())
    (args.output / "real-view-metrics.json").write_text(json.dumps(result, indent=2) + "\n")

    sheet = Image.new("RGB", (width * 2, (height + 28) * len(output)), "#f5f5f5")
    draw = ImageDraw.Draw(sheet)
    for i, (gt, predicted) in enumerate(zip(truth, output, strict=True)):
        top = i * (height + 28)
        draw.text((8, top + 7), f"Ground truth | frame {result['target'][i]}", fill="#222222")
        draw.text((width + 8, top + 7), f"ReSplat render | PSNR {psnr[i]:.1f} dB", fill="#222222")
        sheet.paste(to_pil_image(gt.cpu()), (0, top + 28))
        sheet.paste(to_pil_image(predicted.cpu()), (width, top + 28))
    sheet.save(args.output / "real-view-comparison.jpg", quality=93)

    inputs = Image.new("RGB", (4 * width // 2, 2 * (height // 2 + 22)), "#f5f5f5")
    draw = ImageDraw.Draw(inputs)
    for i, view in enumerate(context["image"][0]):
        left, top = (i % 4) * width // 2, (i // 4) * (height // 2 + 22)
        draw.text((left + 6, top + 5), f"Input {i + 1} | frame {result['context'][i]}", fill="#222222")
        thumb = to_pil_image(view.cpu()).resize((width // 2, height // 2))
        inputs.paste(thumb, (left, top + 22))
    inputs.save(args.output / "real-view-inputs.jpg", quality=93)
    frames = [to_pil_image(view.cpu()) for view in output]
    frames[0].save(args.output / "real-view-render.gif", save_all=True, append_images=frames[1:],
                   duration=350, loop=0)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
