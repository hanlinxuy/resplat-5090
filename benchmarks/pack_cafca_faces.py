"""Extract a calibrated, subject-separated 448×224 research subset from Cafca v2."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO, RawIOBase
import json
from math import atan2, degrees, hypot
from pathlib import Path
from struct import unpack_from
from time import sleep
from urllib.request import Request, urlopen
from zipfile import ZipFile, ZIP_DEFLATED
import zlib

import torch
from PIL import Image

URL = "https://dataset.ait.ethz.ch/downloads/cafca_v2/mini_sample_dataset.zip"
SIZE = 4261946501
CONDITIONS = ("00000_000", "00003_001", "00007_002", "00009_000", "00011_001", "00012_002")


class RangeArchive(RawIOBase):
    def __init__(self, url, size):
        self.url = url
        self.size = size
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, pos, whence=0):
        self.pos = (0, self.pos, self.size)[whence] + pos
        return self.pos

    def read(self, length=-1):
        end = self.size if length < 0 else min(self.size, self.pos + length)
        if end <= self.pos:
            return b""
        block = 4 * 1024 * 1024
        if end - self.pos > 2 * block:
            parts = [(start, min(start + block, end)) for start in range(self.pos, end, block)]
            print(f"reading {end - self.pos:,} bytes of archive index", flush=True)
            with ThreadPoolExecutor(max_workers=8) as pool:
                data = b"".join(pool.map(lambda bounds: self.download(*bounds), parts))
        else:
            data = self.download(self.pos, end)
        self.pos = end
        return data

    def download(self, start, end):
        request = Request(f"{self.url}?part={start}-{end}", headers={"Range": f"bytes={start}-{end - 1}"})
        for attempt in range(4):
            try:
                with urlopen(request, timeout=35) as response:
                    expected = f"bytes {start}-{end - 1}/{self.size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise OSError("Dataset range response does not match requested slice")
                    data = response.read()
                if len(data) != end - start:
                    raise OSError("Truncated dataset range")
                return data
            except OSError:
                if attempt == 3:
                    raise
                sleep(attempt + 1)
        raise AssertionError("unreachable")


def read_group(source, entries):
    start = min(entry.header_offset for entry in entries)
    stop = max(entry.header_offset + entry.compress_size + 256 for entry in entries)
    raw = source.download(start, stop)
    result = {}
    for entry in entries:
        offset = entry.header_offset - start
        name_length, extra_length = unpack_from("<HH", raw, offset + 26)
        body = offset + 30 + name_length + extra_length
        data = raw[body : body + entry.compress_size]
        if entry.compress_type == ZIP_DEFLATED:
            data = zlib.decompress(data, -15)
        if zlib.crc32(data) != entry.CRC:
            raise ValueError(f"Checksum mismatch for {entry.filename}")
        result[Path(entry.filename).stem] = data
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--url", default=URL)
    parser.add_argument("--archive-size", type=int, default=SIZE)
    parser.add_argument("--first-identity", type=int, default=0)
    parser.add_argument("--identities", type=int, default=5)
    parser.add_argument("--heldout-identities", type=int, default=1)
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    args = parser.parse_args()
    if not 0 < args.heldout_identities < args.identities:
        parser.error("split must include both train and held-out identities")
    source = RangeArchive(args.url, args.archive_size)
    with ZipFile(source) as archive:
        entries = archive.NameToInfo.copy()

    prefix_for = {}
    for name in entries:
        if name.endswith("/cameras_json/C00.json"):
            prefix = name.rsplit("/", 2)[0]
            prefix_for[tuple(prefix.split("/")[-2:])] = prefix
    identities = [f"{i:05}" for i in range(args.first_identity, args.first_identity + args.identities)]
    holdout = set(identities[-args.heldout_identities:])
    jobs = [(identity, condition) for identity in identities for condition in args.conditions]
    selected_prefix = {key: prefix_for[key] for key in jobs}
    print(f"packing {len(jobs)} conditions across {len(identities)} identities", flush=True)
    indices = {"train": {}, "test": {}}
    for split in indices:
        (args.output / split).mkdir(parents=True, exist_ok=True)
    view_index = {}

    def pack(item):
        identity, condition = item
        prefix = selected_prefix[item]
        scene = f"cafca_{identity}_{condition}"
        split = "test" if identity in holdout else "train"
        destination = args.output / split / f"{scene}.torch"
        if destination.is_file():
            record, = torch.load(destination, map_location="cpu", weights_only=True)
            return split, scene, destination.name, record["eval_context"], record["eval_target"]
        names = [f"C{i:02}" for i in range(30)]
        cams = read_group(source, [entries[f"{prefix}/cameras_json/{name}.json"] for name in names])
        cams = {name: json.loads(content) for name, content in cams.items()}
        angles = {}
        for name, camera in cams.items():
            x, y, z = camera["position"]
            angles[name] = degrees(atan2(x, z)), degrees(atan2(y, hypot(x, z)))
        front = sorted(names, key=lambda n: abs(angles[n][0]) + 0.5 * abs(angles[n][1]))[:18]
        front.sort(key=lambda n: angles[n][0])
        colors = read_group(source, [entries[f"{prefix}/color_image/{name}.png"] for name in front])
        masks = read_group(source, [entries[f"{prefix}/foreground_mask/{name}.png"] for name in front])
        images, silhouettes, poses = [], [], []
        for name in front:
            plate = Image.new("RGB", (448, 224), (112, 112, 112))
            color = Image.open(BytesIO(colors[name])).convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
            mask = Image.open(BytesIO(masks[name])).convert("L").resize((224, 224), Image.Resampling.BILINEAR)
            plate.paste(color, (112, 0), mask)
            silhouette = Image.new("L", (448, 224))
            silhouette.paste(mask, (112, 0))
            stream, mask_stream = BytesIO(), BytesIO()
            plate.save(stream, format="JPEG", quality=94, subsampling=0)
            silhouette.save(mask_stream, format="PNG")
            images.append(torch.frombuffer(bytearray(stream.getvalue()), dtype=torch.uint8))
            silhouettes.append(torch.frombuffer(bytearray(mask_stream.getvalue()), dtype=torch.uint8))
            k = cams[name]["K"]
            pose = cams[name]["world2cam"]
            poses.append([k[0][0] / 1024, k[1][1] / 512,
                          (112 + k[0][2] * 224 / 512) / 448, k[1][2] / 512, 0., 0.,
                          *(v for row in pose[:3] for v in row)])
        context = [0, 2, 4, 6, 9, 11, 14, 17]
        target = [1, 3, 5, 8, 12, 15]
        record = dict(key=scene, cameras=torch.tensor(poses), images=images, masks=silhouettes,
                      eval_context=context, eval_target=target, near=0.25, far=2.)
        temporary = destination.with_suffix(".tmp")
        torch.save([record], temporary)
        temporary.replace(destination)
        print(f"{split} {scene}: {len(front)} cameras", flush=True)
        return split, scene, destination.name, context, target

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for split, scene, file, context, target in pool.map(pack, jobs):
            indices[split][scene] = file
            view_index[scene] = dict(context=context, target=target)

    for split, index in indices.items():
        (args.output / split / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    (args.output / "view-index.json").write_text(json.dumps(view_index, indent=2) + "\n")


if __name__ == "__main__":
    main()
