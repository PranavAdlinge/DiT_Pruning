"""
SSTK prepare: images + captions from images_txt and <path>.json -> MDS.

Caption path: <image_path>.json, key "description"

Example:
  python prepare.py --images_txt ./sa1b/image_paths.txt \\
    --local_mds_dir ./sa1b/mds/ --num_proc 16 --seed 42 --size 100000 \\
    --min_size 512 --min_aspect_ratio 0.67 --max_aspect_ratio 1.33
"""

import json
import os
from argparse import ArgumentParser
from multiprocessing import Pool, current_process

import numpy as np
from PIL import Image
from streaming import MDSWriter
from tqdm import tqdm

from nexus.utils.mds_utils import MDS_T2I_COLUMNS, merge_mds_shards


def parse_arguments() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--images_txt",
        type=str,
        required=True,
        help="Path to txt file with image paths (one path per line)",
    )
    parser.add_argument(
        "--local_mds_dir",
        type=str,
        default="",
        help="Directory to store mds shards.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Number of images to sample. If None, use all images from txt.",
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=512,
        help="Minimum dimension (width/height). Set to 0 to disable.",
    )
    parser.add_argument(
        "--min_aspect_ratio",
        type=float,
        default=0.67,
        help="Minimum aspect ratio (width/height).",
    )
    parser.add_argument(
        "--max_aspect_ratio",
        type=float,
        default=1.33,
        help="Maximum aspect ratio (width/height).",
    )
    return parser.parse_args()


def _current_process_index() -> int:
    p = current_process()
    return p._identity[0] - 1


def write_images(images_path: np.ndarray, args: ArgumentParser) -> None:
    assert isinstance(images_path, np.ndarray), "images_path must be np.ndarray"
    idx = _current_process_index()
    print(f"Writing {len(images_path)} images in process {idx}")
    save_dir = os.path.join(args.local_mds_dir, str(idx))
    os.makedirs(save_dir, exist_ok=True)
    writer = MDSWriter(
        out=save_dir,
        columns=MDS_T2I_COLUMNS,
        compression=None,
        size_limit=256 * (2**20),
        max_workers=64,
    )
    for f in tqdm(images_path):
        if not f.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        try:
            img = Image.open(f)
            w, h = img.size
            if args.min_size > 0 and min(w, h) < args.min_size:
                continue
            aspect_ratio = w / h
            if aspect_ratio < args.min_aspect_ratio or aspect_ratio > args.max_aspect_ratio:
                continue
            cap_path = str(f) + ".json"
            with open(cap_path) as cf:
                cap = json.load(cf)["description"]
            if isinstance(cap, list):
                cap = json.dumps([str(c).strip() if c else "" for c in cap])
            else:
                cap = str(cap).strip() if cap else ""
            writer.write({"image": img, "caption": cap, "width": w, "height": h})
        except Exception as e:
            print(f"Skipping {f}: {e}")
    writer.finish()


def main() -> None:
    args = parse_arguments()
    os.makedirs(args.local_mds_dir, exist_ok=True)

    with open(args.images_txt) as f:
        images_path = [line.strip() for line in f if line.strip()]
    print(f"Total {len(images_path)} images in txt file")

    if args.size is not None:
        n = min(args.size, len(images_path))
        rng = np.random.default_rng(args.seed)
        rng.shuffle(images_path)
        images_path = images_path[:n]
        print(f"Sampled {n} images (seed={args.seed}, size={args.size})")
    else:
        print(f"Using all {len(images_path)} images")

    chunks = np.array_split(images_path, args.num_proc)
    with Pool(processes=args.num_proc) as pool:
        pool.starmap(write_images, [(c, args) for c in chunks])

    merge_mds_shards(args.local_mds_dir, args.num_proc)


if __name__ == "__main__":
    main()
