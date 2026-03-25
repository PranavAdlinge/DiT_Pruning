"""
Checkpoint utilities: state dict load/save, pruning, Klein save/load hooks.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Callable

import torch
from diffusers.training_utils import _collate_lora_metadata, cast_training_params
from diffusers.utils import convert_unet_state_dict_to_peft
from peft import get_peft_model_state_dict, set_peft_model_state_dict

TRANSFORMER_SAFE = "transformer.safetensors"
TRANSFORMER_PT = "transformer.pt"


def load_transformer_state(path: Path) -> dict:
    """Load state dict from safetensors or .pt."""
    if path.suffix == ".safetensors":
        try:
            import safetensors.torch
            return dict(safetensors.torch.load_file(str(path)))
        except ImportError:
            return dict(torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=True))
    return dict(torch.load(path, map_location="cpu", weights_only=True))


def save_transformer_state(state: dict, path: Path) -> None:
    """Save state dict to safetensors or .pt."""
    if path.suffix == ".safetensors":
        try:
            import safetensors.torch
            safetensors.torch.save_file(state, str(path))
        except ImportError:
            torch.save(state, path.with_suffix(".pt"))
    else:
        torch.save(state, path)


def prune_old_checkpoints(output_dir: str, limit: int) -> None:
    """Remove oldest checkpoints if count exceeds limit."""
    dirs = sorted(
        [d for d in os.listdir(output_dir) if d.startswith("checkpoint")],
        key=lambda x: int(x.split("-")[1]),
    )
    if len(dirs) > limit:
        for d in dirs[: len(dirs) - limit]:
            shutil.rmtree(os.path.join(output_dir, d))


def make_klein_save_hook(
    accelerator: Any,
    trans_cls: type,
    train_mode: str,
    pipeline_cls: type | None,
    unwrap_fn: Callable,
    is_fsdp: bool,
) -> Callable:
    """Return save hook for Klein DiT (LoRA or full)."""

    def save_hook(models, weights, output_dir):
        trans = next((m for m in models if isinstance(unwrap_fn(m), trans_cls)), None)
        if trans is None:
            raise ValueError(f"No {trans_cls} in save models")
        if not accelerator.is_main_process:
            return
        unwrapped = unwrap_fn(trans)
        if weights:
            weights.pop()
        if train_mode == "lora":
            lora_sd = get_peft_model_state_dict(
                unwrapped,
                state_dict=accelerator.get_state_dict(trans) if is_fsdp else None,
            )
            if is_fsdp:
                from diffusers.training_utils import _to_cpu_contiguous
                lora_sd = _to_cpu_contiguous(lora_sd)
            pipeline_cls.save_lora_weights(
                output_dir,
                transformer_lora_layers=lora_sd,
                **_collate_lora_metadata({"transformer": trans}),
            )
        else:
            state = (
                accelerator.get_state_dict(trans) if is_fsdp else unwrapped.state_dict()
            )
            save_transformer_state(state, Path(output_dir) / TRANSFORMER_SAFE)

    return save_hook


def make_klein_load_hook(
    accelerator: Any,
    trans_cls: type,
    pretrained_path: str,
    subfolder: str,
    train_mode: str,
    pipeline_cls: type | None,
    lora_config: Any,
    unwrap_fn: Callable,
    mixed_precision: str | None,
) -> Callable:
    """Return load hook for Klein DiT (LoRA or full)."""

    def load_hook(models, input_dir):
        input_path = Path(input_dir)
        if train_mode == "lora" and pipeline_cls is None:
            raise ValueError("pipeline_cls required for LoRA load")
        is_fsdp = getattr(accelerator.state, "fsdp_plugin", None) is not None
        if is_fsdp:
            trans = trans_cls.from_pretrained(pretrained_path, subfolder=subfolder)
            if train_mode == "lora":
                trans.add_adapter(lora_config)
        else:
            trans = None
            while models:
                m = models.pop()
                if isinstance(unwrap_fn(m), trans_cls):
                    trans = unwrap_fn(m)
                    break
            if trans is None:
                raise ValueError("No transformer in load hook")
        if train_mode == "lora":
            lora_sd = pipeline_cls.lora_state_dict(input_dir)
            trans_sd = {
                k[len("transformer."):]: v
                for k, v in lora_sd.items()
                if k.startswith("transformer.")
            }
            trans_sd = convert_unet_state_dict_to_peft(trans_sd)
            set_peft_model_state_dict(trans, trans_sd, adapter_name="default")
        else:
            for name in (TRANSFORMER_SAFE, TRANSFORMER_PT):
                p = input_path / name
                if p.exists():
                    trans.load_state_dict(load_transformer_state(p), strict=False)
                    break
        if mixed_precision == "fp16":
            cast_training_params([trans])

    return load_hook


def save_final_klein(
    output_dir: Path,
    transformer: torch.nn.Module,
    train_mode: str,
    pipeline_cls: type | None,
    unwrap_fn: Callable,
    logger: Any = None,
) -> None:
    """Save final Klein DiT (LoRA or full) to output_dir."""
    trans = unwrap_fn(transformer)
    if train_mode == "lora":
        lora_sd = get_peft_model_state_dict(trans)
        pipeline_cls.save_lora_weights(
            str(output_dir),
            transformer_lora_layers=lora_sd,
            **_collate_lora_metadata({"transformer": trans}),
        )
    else:
        path = output_dir / TRANSFORMER_SAFE
        save_transformer_state(trans.state_dict(), path)
    if logger:
        logger.info("Saved %s weights to %s", "LoRA" if train_mode == "lora" else "full", output_dir)
