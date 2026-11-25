"""Fixed novel-view evaluation before and after a synthetic initialization run."""

from io import BytesIO
import json
import os
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_lightning.callbacks import Callback
from torchvision.transforms.functional import to_tensor, to_pil_image

from src.dataset.shims.crop_shim import apply_crop_shim
from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim


class SyntheticViewQuality(Callback):
    def __init__(self, root, output):
        self.root = Path(root)
        self.output = Path(output)
        self.results = {}
        self.eval_steps = {
            int(step) for step in os.environ.get("RESPLAT_QUALITY_EVAL_STEPS", "").split(",") if step
        }

    def on_train_start(self, trainer, pl_module):
        self.evaluate(pl_module, "before")

    def on_train_end(self, trainer, pl_module):
        self.evaluate(pl_module, "after")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step in self.eval_steps:
            self.eval_steps.remove(step)
            self.evaluate(pl_module, f"step-{step}")

    @torch.no_grad()
    def evaluate(self, module, phase):
        was_training = module.training
        module.eval()
        available_lpips = (Path(torch.hub.get_dir()) / "checkpoints" / "vgg16-397923af.pth").is_file()
        split_results = {}
        for split in ("train", "test"):
            index = json.loads((self.root / split / "index.json").read_text())
            rows = []
            limit = 12 if split == "test" else 4
            for name, file in sorted(index.items())[:limit]:
                scene, = torch.load(self.root / split / file, map_location="cpu", weights_only=True)
                cameras = scene["cameras"]
                intrinsics = torch.eye(3).repeat(len(cameras), 1, 1)
                intrinsics[:, 0, 0], intrinsics[:, 1, 1], intrinsics[:, 0, 2], intrinsics[:, 1, 2] = cameras[:, :4].T
                extrinsics = torch.eye(4).repeat(len(cameras), 1, 1)
                extrinsics[:, :3] = cameras[:, 6:].reshape(-1, 3, 4)
                extrinsics = extrinsics.inverse()

                batch = {}
                context_ids = scene.get("eval_context", [4, 10, 16, 22, 28, 34, 40, 46])
                target_ids = scene.get("eval_target", [7, 13, 19, 25, 31, 37])
                for key, ids in (("context", context_ids), ("target", target_ids)):
                    images = torch.stack([
                        to_tensor(Image.open(BytesIO(scene["images"][i].numpy().tobytes())).convert("RGB"))
                        for i in ids
                    ])
                    batch[key] = dict(image=images, intrinsics=intrinsics[ids], extrinsics=extrinsics[ids],
                                      near=torch.full((len(ids),), scene.get("near", 0.01)),
                                      far=torch.full((len(ids),), scene.get("far", 200.0)), index=torch.tensor(ids))
                if batch["context"]["image"].shape[-2:] != (224, 448):
                    batch = apply_crop_shim(batch, (224, 448))
                batch = module.data_shim({
                    key: {field: value.unsqueeze(0).to(module.device) for field, value in views.items()}
                    for key, views in batch.items()
                })
                context, target = batch["context"], batch["target"]
                gaussians = module.encoder(context, module.global_step, deterministic=False)
                gaussian = gaussians["gaussians"] if isinstance(gaussians, dict) else gaussians
                output = module.decoder.forward(
                    gaussian, target["extrinsics"], target["intrinsics"], target["near"], target["far"],
                    (224, 448), depth_mode=None,
                ).color[0].float().clamp(0, 1)
                gt = target["image"][0].float()
                row = dict(scene=name, gaussian_count=gaussian.means.shape[1],
                           psnr=compute_psnr(gt, output).mean().item(),
                           ssim=compute_ssim(gt, output).mean().item(),
                           mse=(gt - output).square().mean().item())
                if "masks" in scene:
                    mask = torch.stack([
                        to_tensor(Image.open(BytesIO(scene["masks"][i].numpy().tobytes())).convert("L"))
                        for i in target_ids
                    ]).to(module.device)
                    mse = ((gt - output).square() * mask).sum(dim=(1, 2, 3))
                    mse = mse / (3 * mask.sum(dim=(1, 2, 3))).clamp_min(1)
                    row["foreground_psnr"] = (-10 * mse.clamp_min(1e-12).log10()).mean().item()
                    row["center_ssim"] = compute_ssim(gt[..., 112:336], output[..., 112:336]).mean().item()
                    interior = F.max_pool2d(1 - mask, 5, stride=1, padding=2) < 0.05
                    luma = gt.new_tensor([0.2126, 0.7152, 0.0722])[None, :, None, None]
                    gt_gray = (gt * luma).sum(1, keepdim=True)
                    pred_gray = (output * luma).sum(1, keepdim=True)
                    half_gray = F.interpolate(F.avg_pool2d(gt_gray, 2), size=gt_gray.shape[-2:],
                                              mode="bilinear", align_corners=False)
                    gt_high = (gt_gray - F.avg_pool2d(gt_gray, 3, stride=1, padding=1))[interior]
                    pred_high = (pred_gray - F.avg_pool2d(pred_gray, 3, stride=1, padding=1))[interior]
                    half_high = (half_gray - F.avg_pool2d(half_gray, 3, stride=1, padding=1))[interior]
                    if gt_high.numel() and gt_high.abs().mean().item() > 0:
                        row["highpass_ratio"] = (pred_high.abs().mean() / gt_high.abs().mean()).item()
                        row["half_resolution_gt_ratio"] = (half_high.abs().mean() / gt_high.abs().mean()).item()
                if available_lpips:
                    row["lpips"] = compute_lpips(gt, output).mean().item()
                if split == "test" and not rows:
                    to_pil_image(torch.cat((gt[2], output[2]), dim=-1).cpu()).save(
                        self.output / f"novel-view-{phase}.png"
                    )
                    if "masks" in scene:
                        panels = torch.cat((gt[..., 112:336], output[..., 112:336]), dim=-1)
                        montage = torch.cat([panels[i] for i in (0, 2, 5)], dim=-2)
                        to_pil_image(montage.cpu()).save(self.output / f"face-views-{phase}.jpg", quality=95)
                rows.append(row)
            split_results[split] = {
                "mean": {key: mean(row[key] for row in rows if key in row)
                         for key in sorted(set().union(*(row.keys() for row in rows)) - {"scene"})},
                "samples": rows,
            }
        self.results[phase] = dict(step=int(module.global_step), lpips_available=available_lpips, **split_results)
        (self.output / "view-quality.json").write_text(json.dumps(self.results, indent=2) + "\n")
        module.train(was_training)
