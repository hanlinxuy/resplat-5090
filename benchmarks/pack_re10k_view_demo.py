"""Pack 8 calibrated input photos and 6 held-out views from the official Re10K demo."""

import argparse
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    height, width = 224, 448
    with ZipFile(args.archive) as archive:
        first_scene = next(name for name in sorted(archive.namelist()) if name.endswith("/000000.torch"))
        scene, = torch.load(BytesIO(archive.read(first_scene)), map_location="cpu", weights_only=True)
    cameras = scene["cameras"]
    start = (len(cameras) - 43) // 2
    context = [start + i * 6 for i in range(8)]
    target = [index + 3 for index in context[:6]]
    intrinsics = torch.eye(3).repeat(len(cameras), 1, 1)
    intrinsics[:, 0, 0], intrinsics[:, 1, 1], intrinsics[:, 0, 2], intrinsics[:, 1, 2] = cameras[:, :4].T
    extrinsics = torch.eye(4).repeat(len(cameras), 1, 1)
    extrinsics[:, :3] = cameras[:, 6:].reshape(-1, 3, 4)
    extrinsics = extrinsics.inverse()
    extrinsics = extrinsics[context[len(context) // 2]].inverse() @ extrinsics
    span = torch.linalg.vector_norm(extrinsics[context[0], :3, 3] - extrinsics[context[-1], :3, 3])
    extrinsics[:, :3, 3] *= 0.6 / span.clamp_min(1e-6)

    result = {"scene": scene["key"]}
    for label, ids in (("context", context), ("target", target)):
        images = []
        for index in ids:
            image = Image.open(BytesIO(scene["images"][index].numpy().tobytes())).convert("RGB")
            orig_w, orig_h = image.size
            scale = max(width / orig_w, height / orig_h)
            sw, sh = round(orig_w * scale), round(orig_h * scale)
            left, top = (sw - width) // 2, (sh - height) // 2
            image = image.resize((sw, sh), Image.Resampling.LANCZOS)
            images.append(to_tensor(image.crop((left, top, left + width, top + height))).mul(255).byte())
            intrinsics[index, 0] *= sw / width
            intrinsics[index, 1] *= sh / height
            intrinsics[index, 0, 2] -= left / width
            intrinsics[index, 1, 2] -= top / height
        result[label] = dict(image=torch.stack(images), intrinsics=intrinsics[ids],
                             extrinsics=extrinsics[ids], near=torch.full((len(ids),), 0.01),
                             far=torch.full((len(ids),), 200.0), index=torch.tensor(ids))
    torch.save(result, args.output)
    print(scene["key"], context, target, args.output)


if __name__ == "__main__":
    main()
