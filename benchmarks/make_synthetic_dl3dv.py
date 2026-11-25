"""Write calibrated synthetic scenes in the project's DL3DV .torch chunk format."""

import argparse
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--scenes", type=int, default=12)
    parser.add_argument("--views", type=int, default=56)
    parser.add_argument("--holdout-scenes", type=int, default=0)
    parser.add_argument("--image-shape", type=int, nargs=2, default=(270, 480))
    args = parser.parse_args()
    if args.scenes < 1 or args.views < 50 or args.holdout_scenes < 0:
        parser.error("use at least one scene and 50 camera views")
    rng = np.random.default_rng(731)
    h, w = args.image_shape
    if h < 1 or w < 1:
        parser.error("image dimensions must be positive")
    fx, fy = 0.86, 1.15
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    rayx, rayy = ((xx + 0.5) / w - 0.5) / fx, ((yy + 0.5) / h - 0.5) / fy
    (args.root / "train").mkdir(parents=True, exist_ok=True)
    (args.root / "test").mkdir(parents=True, exist_ok=True)

    index = {}
    heldout = {}
    for scene in range(args.scenes + args.holdout_scenes):
        phase = rng.uniform(-np.pi, np.pi, size=3)
        images, cameras = [], []
        for view, t in enumerate(np.linspace(-1.0, 1.0, args.views)):
            cx, cy = float(t * 0.48), float(0.12 * np.sin(t * 2.4 + phase[0]))
            foreground_x, foreground_y = 2.6 * rayx + cx, 2.6 * rayy + cy
            foreground = (foreground_x**2 + (foreground_y * 1.3)**2) < 0.52
            x = np.where(foreground, foreground_x, 4.1 * rayx + cx)
            y = np.where(foreground, foreground_y, 4.1 * rayy + cy)
            shape = 0.12 * np.sin(13 * x + phase[0]) * np.sin(12 * y)
            channels = [0.48 + 0.19 * np.sin((i + 2) * 4.1 * x + (i + 1) * 2.9 * y + phase[i])
                        + shape + 0.09 * foreground for i in range(3)]
            rgb = np.uint8(np.clip(np.stack(channels, axis=-1), 0, 1) * 255)
            stream = BytesIO()
            Image.fromarray(rgb).save(stream, format="JPEG", quality=91)
            images.append(torch.frombuffer(bytearray(stream.getvalue()), dtype=torch.uint8))
            cameras.append([fx, fy, 0.5, 0.5, 0, 0,
                            1, 0, 0, -cx, 0, 1, 0, -cy, 0, 0, 1, 0])
        record = {"key": f"synthetic_dl3dv_{scene:04d}", "cameras": torch.tensor(cameras, dtype=torch.float32),
                  "images": images}
        chunk = f"{scene:04d}.torch"
        if scene < args.scenes:
            torch.save([record], args.root / "train" / chunk)
            index[record["key"]] = chunk
        if scene >= args.scenes or (scene == 0 and not args.holdout_scenes):
            torch.save([record], args.root / "test" / chunk)
            heldout[record["key"]] = chunk
    (args.root / "test" / "index.json").write_text(json.dumps(heldout) + "\n")
    (args.root / "train" / "index.json").write_text(json.dumps(index) + "\n")


if __name__ == "__main__":
    main()
