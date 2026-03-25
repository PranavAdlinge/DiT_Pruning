"""
MDS prepare utilities for datasets/prepare scripts.

Common schema and helpers for prepare output compatible with precompute.py.
"""

import os
from pathlib import Path
from typing import Iterator

from PIL import Image
from PIL.ImageOps import exif_transpose
from streaming import MDSWriter
from streaming.base.util import merge_index
from tqdm import tqdm

# MDS schema; precompute.py expects these columns.
MDS_T2I_COLUMNS = {
    "width": "int32",
    "height": "int32",
    "image": "jpeg",
    "caption": "str",
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def collect_images_from_dir(
    root: Path,
    *,
    suffixes: set[str] | None = None,
    min_size: int = 0,
) -> Iterator[tuple[str, int, int]]:
    """
    Yield (path, width, height) for valid images under root.
    Skips images smaller than min_size if min_size > 0.
    """
    suffixes = suffixes or IMAGE_SUFFIXES
    for p in sorted(root.iterdir()):
        if p.suffix.lower() in suffixes:
            try:
                img = Image.open(p)
                img = exif_transpose(img)
                w, h = img.size
                if min_size > 0 and min(w, h) < min_size:
                    continue
                yield str(p), w, h
            except Exception:
                continue


def load_image_rgb(path: str) -> Image.Image:
    """Load image from path, transpose EXIF, convert to RGB."""
    img = Image.open(path)
    img = exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def write_path_caption_shard(
    pairs: list[tuple[str, str]],
    local_mds_dir: str,
    worker_idx: int,
    *,
    size_limit: int = 256 * (2**20),
) -> None:
    """
    Write one MDS shard from (path, caption) pairs.
    Compatible with precompute.py / StreamingT2IDataset.
    """
    save_dir = os.path.join(local_mds_dir, str(worker_idx))
    os.makedirs(save_dir, exist_ok=True)
    writer = MDSWriter(
        out=save_dir,
        columns=MDS_T2I_COLUMNS,
        compression=None,
        size_limit=size_limit,
        max_workers=64,
    )
    for path, caption in tqdm(pairs, desc=f"Worker {worker_idx}"):
        try:
            img = load_image_rgb(path)
            w, h = img.size
            writer.write({"image": img, "caption": caption, "width": w, "height": h})
        except Exception as e:
            print(f"Skipping {path}: {e}")
    writer.finish()


def merge_mds_shards(local_mds_dir: str, num_shards: int) -> None:
    """Merge index.json from each shard subfolder into local_mds_dir."""
    shards_metadata = [
        os.path.join(local_mds_dir, str(i), "index.json") for i in range(num_shards)
    ]
    merge_index(shards_metadata, out=local_mds_dir, keep_local=True)
