"""
Precompute VAE latents and text embeddings from MDS shards (prepare output).

Works with any prepare script output: datasets/prepare/sstk or datasets/prepare/dreambooth.
MDS input must have columns: image, caption, width, height.

Layout discovery:
  - Flat: shards (0, 1, 2, ...) directly under datadir -> output to savedir
  - Nested: shards under datadir/X -> output to savedir/X for each subfolder X
  Groups are processed sequentially; each group uses up to num_proc workers.

Example:
  python datasets/precompute.py \\
    --datadir ./mds/ \\
    --savedir ./mds_latents_flux2/ \\
    --num_proc 8 \\
    --dataloader_workers 4 \\
    --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-base-4B \\
    --batch_size 32
"""

import json
import os
import subprocess
import sys
import tempfile
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from streaming import MDSWriter, Stream
from streaming.base.util import merge_index
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from nexus.data.t2i_dataset import StreamingT2IDataset
from nexus.data.utils import text_preprocessing
from nexus.utils import DATA_TYPES


def _datadir_to_streams(datadir: Union[List[str], str]) -> List[Stream]:
    """Convert datadir (str or list of strs) to list of Stream for StreamingDataset."""
    paths = [datadir] if isinstance(datadir, str) else list(datadir)
    return [Stream(local=p) for p in paths]


def build_streaming_sstk_t2i_dataloader(
    datadir: Union[List[str], str],
    batch_size: int,
    resolution: int = 512,
    drop_last: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
    image_key: str = "image",
    caption_key: str = "caption",
    clean_caption: bool = True,
) -> DataLoader:
    streams = _datadir_to_streams(datadir)

    transform = transforms.Compose([
        transforms.Resize(
            resolution,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    init_kwargs: dict = {
        "streams": streams,
        "transforms_list": [transform],
        "batch_size": batch_size,
        "shuffle": shuffle,
        "image_key": image_key,
        "caption_key": caption_key,
        "clean_caption": clean_caption,
    }
    dataset = StreamingT2IDataset(**init_kwargs)

    def custom_collate(batch_items: List[Dict]) -> Dict:
        out = {k: [] for k in batch_items[0].keys()}
        for item in batch_items:
            for key, value in item.items():
                out[key].append(value)
        return out

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        collate_fn=custom_collate,
        num_workers=num_workers,
    )

    return dataloader


def _caption_sample_weights(n: int, weights_arg: Optional[List[float]]) -> np.ndarray:
    if weights_arg is None or len(weights_arg) == 0:
        return np.ones(n) / n
    w = np.array(weights_arg[:n], dtype=np.float64)
    if len(w) < n:
        rest = np.ones(n - len(w)) / (n - len(w))
        w = np.concatenate([w, rest])
    return w / w.sum()


def _sample_caption(
    captions: List[str],
    weights: np.ndarray,
    rng: np.random.Generator,
    clean: bool = True,
) -> str:
    idx = rng.choice(len(captions), p=weights)
    return text_preprocessing(captions[idx], clean)[0]


def _shards_under(root: str) -> List[str]:
    """Digit-named shard subdirs (0, 1, 2, ...) under root."""
    if not os.path.isdir(root):
        return []
    return [
        os.path.join(root, name)
        for name in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, name)) and name.isdigit()
    ]


def discover_groups(datadir: str, savedir: str) -> List[Tuple[str, List[str]]]:
    """
    Map datadir structure to output groups: [(outdir, [shard_paths]), ...].

    - Flat: shards (0, 1, 2, ...) directly under datadir -> (savedir, shards)
    - Nested: shards under datadir/X -> (savedir/X, shards) for each subfolder X

    Raises ValueError if layout is ambiguous (both direct and nested shards) or empty.
    """
    if not os.path.isdir(datadir):
        raise ValueError(f"datadir does not exist: {datadir}")

    direct = _shards_under(datadir)
    subdirs = []
    for name in sorted(os.listdir(datadir)):
        subpath = os.path.join(datadir, name)
        if os.path.isdir(subpath):
            shards = _shards_under(subpath)
            if shards:
                subdirs.append((os.path.join(savedir, name), shards))

    if direct and subdirs:
        raise ValueError(
            f"Ambiguous layout under {datadir}: has direct shards and subfolders with shards."
        )
    if direct:
        return [(savedir, direct)]
    if subdirs:
        return subdirs
    raise ValueError(
        f"No MDS shards under {datadir}: expected digit-named dirs (0, 1, 2, ...)."
    )


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--datadir", type=str, default=None, help="MDS shards from prepare.")
    parser.add_argument("--savedir", type=str, default="", help="Output path for precomputed latents.")
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="Workers per group (default 1). Groups are processed sequentially.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image resolution for resize and latent encoding.",
    )
    parser.add_argument("--save_images", default=False, action="store_true")
    parser.add_argument(
        "--model_dtype",
        type=str,
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--save_dtype", type=str, choices=("float16", "float32"), default="float16")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="black-forest-labs/FLUX.2-klein-base-4B",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vae", default=True, action="store_true")
    parser.add_argument("--no_vae", dest="vae", action="store_false")
    parser.add_argument("--text_encoder", default=True, action="store_true")
    parser.add_argument("--no_text_encoder", dest="text_encoder", action="store_false")
    parser.add_argument(
        "--text_encoder_out_layers",
        type=int,
        nargs="+",
        default=[9, 18, 27],
    )
    parser.add_argument("--max_sequence_length", type=int, default=128)
    parser.add_argument("--caption_sample_weights", type=float, nargs="+", default=None)
    parser.add_argument("--dataloader_workers", type=int, default=2)
    parser.add_argument("--worker_idx", type=int, default=None)
    parser.add_argument("--subfolder_paths", type=str, default=None)
    parser.add_argument("--args_file", type=str, default=None)
    args = parser.parse_args()

    if args.args_file is not None:
        with open(args.args_file) as f:
            data = json.load(f)
        args = Namespace(**data)
    return args


def _precompute_worker(task: Tuple[List[str], int, object]) -> None:
    subfolder_paths, worker_idx, args = task

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device_idx = worker_idx % num_gpus
        device = torch.device("cuda", device_idx)
        torch.cuda.set_device(device_idx)
    else:
        device = torch.device("cpu")

    torch.manual_seed(worker_idx + args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_idx + args.seed)
    np.random.seed(worker_idx + args.seed)
    rng = np.random.default_rng(worker_idx + args.seed)

    vae = AutoencoderKLFlux2.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=DATA_TYPES[args.model_dtype],
    ).to(device).eval()
    vae = torch.compile(vae)

    text_encoder = Qwen3ForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=DATA_TYPES[args.model_dtype],
    ).to(device).eval()
    text_encoder.requires_grad_(False)

    tokenizer = Qwen2TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )

    caption_key = "caption"
    image_key = "image"

    dataloader = build_streaming_sstk_t2i_dataloader(
        datadir=subfolder_paths,
        batch_size=args.batch_size,
        resolution=args.resolution,
        drop_last=False,
        shuffle=False,
        num_workers=args.dataloader_workers,
        image_key=image_key,
        caption_key=caption_key,
        clean_caption=True,
    )
    ds = dataloader.dataset
    n_samples = getattr(ds, "size", None)
    if n_samples is None:
        try:
            n_samples = len(ds)
        except (TypeError, NotImplementedError):
            n_samples = "?"
    print(f"Worker {worker_idx} -> subdirs: {subfolder_paths}, device={device}, samples={n_samples}")

    columns = {"caption": "str"}
    if args.vae:
        columns[f"latents_{args.resolution}"] = "bytes"
    if args.text_encoder:
        columns["text_embeds"] = "bytes"
    if args.save_images:
        columns["image"] = "jpeg"

    out_dir = os.path.join(args.savedir, str(worker_idx))
    os.makedirs(out_dir, exist_ok=True)
    writer = MDSWriter(
        out=out_dir,
        columns=columns,
        compression=None,
        size_limit=256 * (2**20),
        max_workers=64,
    )

    for batch in tqdm(dataloader, desc=f"Worker {worker_idx}", position=worker_idx):
        images = torch.stack(batch["image_0"]).to(device)
        batch_size = images.shape[0]

        try:
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=DATA_TYPES[args.model_dtype]):
                    latents_dict = {}
                    if args.vae:
                        latent_dist = vae.encode(images)
                        assert isinstance(latent_dist, AutoencoderKLOutput)
                        lat = latent_dist.latent_dist.sample().to(
                            DATA_TYPES[args.save_dtype]
                        )
                        latents_dict[args.resolution] = lat.detach().cpu().numpy()

                    captions_to_encode = []
                    for i in range(batch_size):
                        c = batch["caption"][i]
                        if isinstance(c, list) and len(c) > 1:
                            weights = _caption_sample_weights(
                                len(c), args.caption_sample_weights
                            )
                            c = _sample_caption(c, weights, rng, clean=True)
                        elif isinstance(c, list) and len(c) == 1:
                            c = text_preprocessing(c[0], True)[0]
                        captions_to_encode.append(c)

                    prompt_embeds = None
                    if args.text_encoder:
                        prompt_embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
                            text_encoder=text_encoder,
                            tokenizer=tokenizer,
                            prompt=captions_to_encode,
                            device=device,
                            dtype=DATA_TYPES[args.model_dtype],
                            max_sequence_length=args.max_sequence_length,
                            hidden_states_layers=tuple(args.text_encoder_out_layers),
                        )
                        prompt_embeds = (
                            prompt_embeds.to(DATA_TYPES[args.save_dtype])
                            .detach()
                            .cpu()
                            .numpy()
                        )

                for i in range(batch_size):
                    mds_sample = {"caption": captions_to_encode[i]}
                    if args.text_encoder:
                        mds_sample["text_embeds"] = prompt_embeds[i].tobytes()
                    if args.vae:
                        mds_sample[f"latents_{args.resolution}"] = latents_dict[
                            args.resolution
                        ][i].tobytes()
                    if args.save_images:
                        mds_sample["image"] = batch["sample"][i][image_key]
                    writer.write(mds_sample)
        except RuntimeError as e:
            print(f"Worker {worker_idx} runtime error, skipping batch: {e}")

    writer.finish()

    if torch.cuda.is_available():
        del text_encoder, vae
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    print(f"Worker {worker_idx} finished")


def _partition(shards: List[str], n: int) -> List[List[str]]:
    """Split shards into n chunks (may have empty chunks if n > len(shards))."""
    if n >= len(shards):
        return [[s] for s in shards] + [[] for _ in range(n - len(shards))]
    size = (len(shards) + n - 1) // n
    return [
        shards[i * size : (i + 1) * size]
        for i in range(n)
        if i * size < len(shards)
    ]


def _run_group(outdir: str, shards: List[str], num_workers: int, args: object) -> None:
    """
    Precompute one group: partition shards across workers, spawn subprocesses,
    merge worker outputs (outdir/0, 1, ...) into outdir.
    """
    num_workers = min(num_workers, len(shards)) or 1
    partitions = _partition(shards, num_workers)
    partitions = [p for p in partitions if p]
    num_workers = len(partitions)

    os.makedirs(outdir, exist_ok=True)
    script = os.path.abspath(__file__)

    with tempfile.TemporaryDirectory() as tmpdir:
        procs = []
        for i in range(num_workers):
            data = {
                k: v
                for k, v in vars(args).items()
                if k not in ("args_file", "worker_idx", "subfolder_paths")
            }
            data["savedir"] = outdir
            data["worker_idx"] = i
            data["subfolder_paths"] = partitions[i]
            args_path = os.path.join(tmpdir, f"args_{i}.json")
            with open(args_path, "w") as f:
                json.dump(data, f)
            proc = subprocess.Popen(
                [sys.executable, script, "--args_file", args_path],
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            procs.append(proc)
        for proc in procs:
            proc.wait()
            if proc.returncode != 0:
                raise SystemExit(proc.returncode)

    shards_meta = [
        os.path.join(outdir, str(i), "index.json") for i in range(num_workers)
    ]
    merge_index(shards_meta, out=outdir, keep_local=True)
    print("Merged into", outdir)


def main(args: object) -> None:
    # Worker mode: invoked as subprocess with shard paths and output dir
    if args.worker_idx is not None and args.subfolder_paths is not None:
        paths = (
            args.subfolder_paths
            if isinstance(args.subfolder_paths, list)
            else [p.strip() for p in args.subfolder_paths.split(",") if p.strip()]
        )
        _precompute_worker((paths, args.worker_idx, args))
        return

    if args.datadir is None:
        raise ValueError("--datadir is required")
    # Discover groups: each (outdir, shards) maps input layout to output location
    groups = discover_groups(args.datadir, args.savedir)
    num_proc = args.num_proc or 1

    # Iterate over groups sequentially; each group uses num_proc workers
    for outdir, shards in groups:
        print(f"Precomputing -> {outdir}")
        _run_group(outdir, shards, num_proc, args)


if __name__ == "__main__":
    main(parse_args())
