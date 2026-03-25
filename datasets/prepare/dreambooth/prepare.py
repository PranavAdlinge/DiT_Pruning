"""
DreamBooth prepare: instance (+ class) images -> MDS for precompute.py.

Supports HuggingFace dataset (e.g. diffusers/dog-example) or local folders.
When --generate_class_images, uses Flux2KleinPipeline to generate class images
if class_data_dir is empty or has fewer than num_class_images.

Example (dog from HuggingFace, data_dir layout instance/class subdirs):
  python prepare.py --dataset_name diffusers/dog-example --download_dir ./dreambooth/instance \\
    --instance_prompt "a photo of sks dog" --local_mds_dir ./dreambooth/mds/

Example (with prior preservation, generate class images):
  python prepare.py --dataset_name diffusers/dog-example --download_dir ./dreambooth/instance \\
    --instance_prompt "a photo of sks dog" --class_prompt "a dog" \\
    --generate_class_images --num_class_images 100 --class_data_dir ./dreambooth/class \\
    --local_mds_dir ./dreambooth/mds/
"""

import os
import sys

# Disable xet storage for HF downloads (must be set before importing huggingface_hub)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from nexus.utils.mds_utils import (
    IMAGE_SUFFIXES,
    collect_images_from_dir,
    merge_mds_shards,
    write_path_caption_shard,
)
from nexus.utils import DATA_TYPES


def _count_class_images(class_data_dir: Path) -> int:
    """Count existing image files in class_data_dir."""
    if not class_data_dir.exists():
        return 0
    return sum(
        1
        for p in class_data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def _generate_class_images(args) -> None:
    """Generate class images with Flux2KleinPipeline if needed."""
    if not args.generate_class_images:
        return
    if args.class_data_dir is None:
        args.class_data_dir = str(
            Path(args.local_mds_dir).parent / "class"
        )
        print(f"Using default class_data_dir: {args.class_data_dir}")

    class_dir = Path(args.class_data_dir)
    class_dir.mkdir(parents=True, exist_ok=True)
    cur_count = _count_class_images(class_dir)

    if cur_count >= args.num_class_images:
        print(f"Class dir has {cur_count} images >= {args.num_class_images}, skipping generation.")
        return

    num_new = args.num_class_images - cur_count
    print(f"Generating {num_new} class images with prompt: {args.class_prompt}")

    try:
        import torch
        from diffusers import Flux2KleinPipeline
        from huggingface_hub.utils import insecure_hashlib
        from tqdm import tqdm
    except ImportError as e:
        raise ImportError(
            "Class image generation requires: torch, diffusers, tqdm. "
            f"pip install torch diffusers tqdm. {e}"
        ) from e

    dtype = DATA_TYPES[args.prior_generation_precision]
    has_cuda = torch.cuda.is_available()
    has_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    device = "cuda" if has_cuda else ("mps" if has_mps else "cpu")

    pipeline = Flux2KleinPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=dtype,
    )
    pipeline.set_progress_bar_config(disable=True)
    pipeline = pipeline.to(device)

    rng = np.random.default_rng(args.seed)
    indices = np.arange(num_new)
    rng.shuffle(indices)

    for i in tqdm(range(num_new), desc="Generating class images"):
        idx = cur_count + indices[i]
        generator = torch.Generator(device=device).manual_seed(args.seed + i)
        with torch.autocast(device_type=device, dtype=dtype):
            image = pipeline(
                prompt=args.class_prompt,
                height=args.class_image_resolution,
                width=args.class_image_resolution,
                generator=generator,
            ).images[0]
        h = insecure_hashlib.sha1(image.tobytes()).hexdigest()
        out_path = class_dir / f"{idx}-{h[:8]}.jpg"
        image.save(out_path)

    del pipeline
    if has_cuda:
        torch.cuda.empty_cache()
    print(f"Saved {num_new} class images to {class_dir}")


def parse_arguments() -> ArgumentParser:
    parser = ArgumentParser(
        description="Prepare DreamBooth instance (and optional class) data for precompute.py."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="HuggingFace dataset (e.g. diffusers/dog-example). Requires --download_dir.",
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        default=None,
        help="Directory to download --dataset_name into. Required when using --dataset_name.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="Local folder with instance images. Mutually exclusive with --dataset_name.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default="a photo of sks dog",
        help="Instance prompt for DreamBooth.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        help="Local folder with class images for prior preservation. Required when --generate_class_images.",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default="a dog",
        help="Class prompt for prior preservation.",
    )
    parser.add_argument(
        "--generate_class_images",
        action="store_true",
        help="Generate class images with pipeline if class_data_dir is empty or has fewer than num_class_images.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help="Number of class images for prior preservation. Default from DreamBooth example.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="black-forest-labs/FLUX.2-klein-base-4B",
        help="Model for class image generation (Flux2KleinPipeline).",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=1,
        help="Batch size for class image generation.",
    )
    parser.add_argument(
        "--class_image_resolution",
        type=int,
        default=512,
        help="Resolution for generated class images. Default from DreamBooth example.",
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default="bfloat16",
        choices=list(DATA_TYPES.keys()),
        help="Precision for class generation. Default: bfloat16.",
    )
    parser.add_argument(
        "--local_mds_dir",
        type=str,
        required=True,
        help="Directory to store MDS shards (compatible with precompute.py).",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=256,
        help="Minimum dimension. Set to 0 to disable.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeat each instance image this many times.",
    )
    return parser.parse_args()


def _download_dataset(dataset_name: str, download_dir: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("huggingface_hub required. pip install huggingface_hub")
    snapshot_download(
        repo_id=dataset_name,
        repo_type="dataset",
        local_dir=download_dir,
        ignore_patterns=[".gitattributes"],
    )


def _collect_instance_items(args) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []

    if args.dataset_name is not None:
        if args.download_dir is None:
            raise ValueError("--download_dir is required when using --dataset_name")
        _download_dataset(args.dataset_name, args.download_dir)
        root = Path(args.download_dir)
        for path, w, h in collect_images_from_dir(root, min_size=args.min_size):
            items.append((path, args.instance_prompt))

    elif args.instance_data_dir is not None:
        root = Path(args.instance_data_dir)
        if not root.exists():
            raise FileNotFoundError(f"Instance data dir does not exist: {root}")
        for path, w, h in collect_images_from_dir(root, min_size=args.min_size):
            items.append((path, args.instance_prompt))
    else:
        raise ValueError("Specify either --dataset_name or --instance_data_dir")

    expanded: list[tuple[str, str]] = []
    for path, cap in items:
        for _ in range(args.repeats):
            expanded.append((path, cap))
    return expanded


def _collect_class_items(args) -> list[tuple[str, str]]:
    if args.class_data_dir is None:
        return []
    root = Path(args.class_data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Class data dir does not exist: {root}")
    return [
        (path, args.class_prompt)
        for path, w, h in collect_images_from_dir(root, min_size=args.min_size)
    ]


def _write_mds_subdir(
    items: list[tuple[str, str]],
    local_dir: str,
    num_proc: int,
    seed: int,
) -> None:
    """Write items to MDS shards under local_dir, then merge."""
    if not items:
        return
    rng = np.random.default_rng(seed)
    arr = np.arange(len(items))
    rng.shuffle(arr)
    partitions = np.array_split(arr, num_proc)
    partitions = [p for p in partitions if len(p) > 0]
    n_shards = len(partitions)

    def chunk(i: int) -> list[tuple[str, str]]:
        return [items[idx] for idx in partitions[i]]

    if n_shards == 1:
        write_path_caption_shard(chunk(0), local_dir, 0)
    else:
        with Pool(processes=n_shards) as pool:
            pool.starmap(
                write_path_caption_shard,
                [(chunk(i), local_dir, i) for i in range(n_shards)],
            )
    merge_mds_shards(local_dir, n_shards)


def main() -> None:
    args = parse_arguments()

    if args.dataset_name is not None and args.instance_data_dir is not None:
        raise ValueError("Specify only one of --dataset_name or --instance_data_dir")

    if args.generate_class_images and not (args.class_prompt or "").strip():
        raise ValueError("--class_prompt is required when --generate_class_images")

    _generate_class_images(args)
 
    instance_items = _collect_instance_items(args)
    class_items = _collect_class_items(args)
    if not instance_items:
        raise ValueError("No instance images found.")

    mds_root = Path(args.local_mds_dir)
    instance_dir = str(mds_root / "instance")
    class_dir = str(mds_root / "class")

    _write_mds_subdir(instance_items, instance_dir, args.num_proc, args.seed)
    _write_mds_subdir(class_items, class_dir, args.num_proc, args.seed + 1)

    n_total = len(instance_items) + len(class_items)
    print(
        f"Wrote {n_total} samples ({len(instance_items)} instance, {len(class_items)} class) "
        f"to {args.local_mds_dir} (instance/, class/)"
    )


if __name__ == "__main__":
    main()
