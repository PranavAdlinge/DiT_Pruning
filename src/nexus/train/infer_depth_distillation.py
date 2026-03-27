"""
Run prompt inference for depth-distilled Flux checkpoints.

Usage:
    python -m nexus.train.infer_depth_distillation \
        --config configs/klein4b-base/t2i_pruning_depth_distillation.yaml \
        --checkpoint /path/to/checkpoint-1000 \
        --prompt "a cinematic portrait"
"""

from __future__ import annotations

import argparse
import inspect
from contextlib import nullcontext
from pathlib import Path

import torch

from nexus.train.config import load_config, ns_to_kwargs


def _resolve_weight_dtype(cfg) -> torch.dtype:
    mp = getattr(cfg, "mixed_precision", None)
    if mp == "fp16":
        return torch.float16
    if mp == "bf16":
        return torch.bfloat16
    return torch.float32


def _resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type not in {"cuda", "cpu"}:
        return nullcontext()
    if dtype not in {torch.float16, torch.bfloat16}:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _build_depth_distillation_loss(cfg, *, device: torch.device, weight_dtype: torch.dtype):
    model_cfg = cfg.model
    dit_cfg = getattr(model_cfg, "dit", model_cfg.transformer)
    loss_kwargs = ns_to_kwargs(getattr(cfg.loss, "kwargs", None))
    loss_kwargs.update(
        transformer_cls=dit_cfg._class,
        transformer_subfolder=getattr(dit_cfg, "subfolder", "transformer"),
        revision=getattr(model_cfg, "revision", None),
        variant=getattr(model_cfg, "variant", None),
        device=device,
        weight_dtype=weight_dtype,
    )
    ctor = cfg.loss._class.__init__ if inspect.isclass(cfg.loss._class) else cfg.loss._class
    signature = inspect.signature(ctor)
    accepts_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
    )
    if not accepts_var_kwargs:
        supported = {
            name
            for name, param in signature.parameters.items()
            if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        loss_kwargs = {key: value for key, value in loss_kwargs.items() if key in supported}
    loss = cfg.loss._class(**loss_kwargs)
    if not hasattr(loss, "load_checkpoint_artifacts") or not hasattr(loss, "apply_students_to_transformer"):
        raise ValueError(
            f"{cfg.loss.class_name} does not expose depth-distillation inference helpers."
        )
    return loss


def _resolve_pretrained_path(cfg) -> str:
    pipeline_cfg = getattr(cfg, "pipeline", None) or getattr(cfg.model, "pipeline", None)
    if not pipeline_cfg:
        raise ValueError("pipeline config is required")
    pretrained_path = getattr(pipeline_cfg, "pretrained_model_name_or_path", None) or getattr(
        cfg.model, "pretrained_model_name_or_path", None
    )
    if not pretrained_path:
        raise ValueError("pipeline.pretrained_model_name_or_path is required")
    return pretrained_path


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Depth-distillation inference for Flux checkpoints.")
    parser.add_argument("--config", type=str, required=True, help="Path to the training YAML config.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint directory or final run directory containing depth-distillation student weights.",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to generate.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save generated images.")
    parser.add_argument("--num_images", type=int, default=None, help="Number of images to generate.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--resolution", type=int, default=None, help="Square output resolution.")
    parser.add_argument("--inference_steps", type=int, default=None, help="Number of denoising steps.")
    parser.add_argument("--guidance_scale", type=float, default=None, help="CFG scale.")
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cuda, cuda:1, cpu.")
    return parser.parse_args(input_args) if input_args is not None else parser.parse_args()


def main(args=None):
    args = parse_args(args)
    cfg = load_config(args.config)

    device = _resolve_device(args.device)
    weight_dtype = _resolve_weight_dtype(cfg)
    pretrained_path = _resolve_pretrained_path(cfg)
    model_cfg = cfg.model
    pipeline_cfg = getattr(cfg, "pipeline", None) or getattr(model_cfg, "pipeline", None)
    dit_cfg = getattr(model_cfg, "dit", model_cfg.transformer)

    checkpoint_dir = Path(args.checkpoint)
    if checkpoint_dir.is_file():
        checkpoint_dir = checkpoint_dir.parent

    transformer = dit_cfg._class.from_pretrained(
        pretrained_path,
        subfolder=dit_cfg.subfolder,
        revision=getattr(model_cfg, "revision", None),
        variant=getattr(model_cfg, "variant", None),
        torch_dtype=weight_dtype,
    )
    transformer.requires_grad_(False)

    loss = _build_depth_distillation_loss(cfg, device=device, weight_dtype=weight_dtype)
    loss.prepare_for_training(transformer)
    loss.load_checkpoint_artifacts(checkpoint_dir)
    transformer = loss.apply_students_to_transformer(transformer, merge_lora=True)
    transformer.to(device=device, dtype=weight_dtype)
    transformer.eval()

    pipeline = pipeline_cfg._class.from_pretrained(
        pretrained_path,
        transformer=transformer,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(device=device, dtype=weight_dtype)
    pipeline.set_progress_bar_config(disable=False)

    val_cfg = getattr(cfg, "validation", None)
    output_dir = Path(args.output_dir or (checkpoint_dir / "inference"))
    output_dir.mkdir(parents=True, exist_ok=True)
    num_images = args.num_images if args.num_images is not None else getattr(val_cfg, "num_images", 1)
    seed = args.seed if args.seed is not None else getattr(val_cfg, "seed", 42)
    resolution = args.resolution if args.resolution is not None else getattr(val_cfg, "resolution", 512)
    inference_steps = (
        args.inference_steps
        if args.inference_steps is not None
        else getattr(val_cfg, "inference_steps", 4)
    )
    guidance_scale = (
        args.guidance_scale
        if args.guidance_scale is not None
        else getattr(val_cfg, "guidance_scale", 1.0)
    )

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    saved_paths: list[Path] = []
    for index in range(num_images):
        image_generator = generator
        if generator is not None and device.type == "cpu":
            image_generator = torch.Generator(device="cpu").manual_seed(seed + index)
        elif generator is not None:
            image_generator = torch.Generator(device=device).manual_seed(seed + index)

        with _autocast_context(device, weight_dtype):
            result = pipeline(
                prompt=args.prompt,
                height=resolution,
                width=resolution,
                generator=image_generator,
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
            )
        image = result.images[0]
        image_path = output_dir / f"sample_{index:02d}.png"
        image.save(image_path)
        saved_paths.append(image_path)

    print("Saved images:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
