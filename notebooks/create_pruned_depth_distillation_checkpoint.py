"""
Create a standalone pruned checkpoint from a depth-distillation training run.

This script:
1. Loads a base Flux.2 Klein transformer.
2. Reads the depth-distillation YAML config.
3. Rebuilds the student LoRA blocks described by the config.
4. Loads the trained student LoRA weights from a checkpoint directory.
5. Merges the LoRA weights into the copied student blocks.
6. Physically removes the pruned blocks from the transformer.
7. Saves only the compact transformer weights and transformer config.

Example:
    python notebooks/create_pruned_depth_distillation_checkpoint.py \
        --base_checkpoint black-forest-labs/FLUX.2-klein-base-4B \
        --config configs/klein4b-base/t2i_pruning_depth_distillation.yaml \
        --lora_checkpoint /path/to/checkpoint-1000 \
        --output_dir /path/to/merged-transformer
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from peft import LoraConfig, get_peft_model
from peft.utils import set_peft_model_state_dict

DEPTH_DISTILLATION_STATE_SAFE = "depth_distillation_students.safetensors"
DEPTH_DISTILLATION_STATE_PT = "depth_distillation_students.pt"
DEPTH_DISTILLATION_META = "depth_distillation_config.json"


@dataclass(frozen=True)
class IntervalSpec:
    stream: str
    start: int
    end: int
    source_index: int
    student_index: int


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml_config(path: str | Path, *, _visited: set[Path] | None = None) -> dict:
    path = Path(path).resolve()
    visited = set() if _visited is None else set(_visited)
    if path in visited:
        raise ValueError(f"Config inheritance cycle detected at {path}")
    visited.add(path)

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    extends = raw.pop("extends", None)
    if not extends:
        return raw

    base_path = (path.parent / extends).resolve()
    if not base_path.exists():
        base_path = Path(extends).resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Could not resolve base config {extends!r} from {path}")

    base_raw = load_yaml_config(base_path, _visited=visited)
    return _deep_merge(base_raw, raw)


def resolve_class(class_path: str):
    if ":" not in class_path:
        raise ValueError(f"Class path must look like 'module:ClassName', got {class_path!r}")
    module_path, class_name = class_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def resolve_weight_dtype(config: dict, override: str | None) -> torch.dtype:
    if override == "fp16":
        return torch.float16
    if override == "bf16":
        return torch.bfloat16
    if override == "fp32":
        return torch.float32

    mixed_precision = config.get("mixed_precision")
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _parse_stream_index(value, expected_stream: str | None = None) -> tuple[str | None, int]:
    if isinstance(value, int):
        return expected_stream, int(value)

    text = str(value).strip().lower()
    if not text:
        raise ValueError("Empty pruning block entry is not allowed.")

    stream = None
    if text[0] in {"d", "s"}:
        stream = text[0]
        text = text[1:]

    if not text.isdigit():
        raise ValueError(f"Invalid pruning block entry: {value!r}")
    if expected_stream is not None and stream is not None and stream != expected_stream:
        raise ValueError(f"Expected stream '{expected_stream}' but got {value!r}")

    return stream or expected_stream, int(text)


def _parse_interval_entry(value, expected_stream: str | None = None) -> tuple[str | None, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            "Pruning interval entries must be 2-item lists or tuples, "
            f"got {value!r}."
        )

    start_stream, start = _parse_stream_index(value[0], expected_stream=expected_stream)
    end_stream, end = _parse_stream_index(value[1], expected_stream=expected_stream)

    stream = start_stream or end_stream or expected_stream
    if stream is None:
        raise ValueError(
            "Pruning interval entries must include a stream prefix ('d' or 's') "
            "when using unified `pruned_blocks`."
        )
    if start_stream is not None and start_stream != stream:
        raise ValueError(f"Invalid interval start stream in {value!r}.")
    if end_stream is not None and end_stream != stream:
        raise ValueError(f"Invalid interval end stream in {value!r}.")
    if start > end:
        raise ValueError(f"Pruning interval start must be <= end, got {value!r}.")

    return stream, start, end


def _coalesce_indices(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []

    merged: list[tuple[int, int]] = []
    start = prev = indices[0]
    for index in indices[1:]:
        if index == prev + 1:
            prev = index
            continue
        merged.append((start, prev))
        start = prev = index
    merged.append((start, prev))
    return merged


def _parse_interval_collection(values, *, expected_stream: str | None = None) -> list[tuple[int, int]]:
    if values is None:
        return []

    explicit_intervals: list[tuple[int, int]] = []
    block_indices: list[int] = []

    for value in values:
        if isinstance(value, (list, tuple)):
            _, start, end = _parse_interval_entry(value, expected_stream=expected_stream)
            explicit_intervals.append((start, end))
        else:
            _, index = _parse_stream_index(value, expected_stream=expected_stream)
            block_indices.append(index)

    merged = explicit_intervals + _coalesce_indices(sorted(set(block_indices)))
    return sorted(set(merged))


def parse_pruning_intervals(
    *,
    pruned_blocks=None,
    double_stream_pruned_blocks=None,
    single_stream_pruned_blocks=None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if pruned_blocks is not None and (
        double_stream_pruned_blocks is not None or single_stream_pruned_blocks is not None
    ):
        raise ValueError(
            "Use either `pruned_blocks` or the per-stream pruning lists, not both."
        )

    double_intervals: list[tuple[int, int]] = []
    single_intervals: list[tuple[int, int]] = []

    if pruned_blocks is not None:
        double_block_indices: list[int] = []
        single_block_indices: list[int] = []
        for value in pruned_blocks:
            if isinstance(value, (list, tuple)):
                stream, start, end = _parse_interval_entry(value)
                if stream == "d":
                    double_intervals.append((start, end))
                else:
                    single_intervals.append((start, end))
            else:
                stream, index = _parse_stream_index(value)
                if stream == "d":
                    double_block_indices.append(index)
                else:
                    single_block_indices.append(index)
        double_intervals.extend(_coalesce_indices(sorted(set(double_block_indices))))
        single_intervals.extend(_coalesce_indices(sorted(set(single_block_indices))))
    else:
        double_intervals = _parse_interval_collection(
            double_stream_pruned_blocks,
            expected_stream="d",
        )
        single_intervals = _parse_interval_collection(
            single_stream_pruned_blocks,
            expected_stream="s",
        )

    return sorted(set(double_intervals)), sorted(set(single_intervals))


def resolve_student_source_index(start: int, end: int, mode: str) -> int:
    mode = str(mode).strip().lower()
    if mode == "start":
        return start
    if mode == "midpoint":
        return (start + end) // 2
    raise ValueError(
        "student_block_init must be one of {'start', 'midpoint'}, "
        f"got {mode!r}."
    )


def module_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


def module_dtype(module: nn.Module) -> torch.dtype:
    return next(module.parameters()).dtype


def find_lora_target_modules(module: nn.Module) -> list[str]:
    target_modules: list[str] = []
    for name, child in module.named_modules():
        if not name:
            continue
        if isinstance(child, (nn.Linear, nn.Embedding, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            target_modules.append(name)
    if not target_modules:
        raise ValueError(
            f"Could not find any LoRA-compatible submodules inside {module.__class__.__name__}."
        )
    return sorted(set(target_modules))


def make_lora_student_block(
    block: nn.Module,
    *,
    rank: int,
    alpha: int | None,
    dropout: float,
    target_modules: list[str] | None,
) -> nn.Module:
    resolved_targets = (
        sorted(set(target_modules))
        if target_modules is not None
        else find_lora_target_modules(block)
    )
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank if alpha is None else alpha,
        lora_dropout=dropout,
        init_lora_weights="gaussian",
        target_modules=resolved_targets,
    )
    return get_peft_model(block, lora_config)


def resolve_state_path(checkpoint_dir: Path) -> Path:
    for filename in (DEPTH_DISTILLATION_STATE_SAFE, DEPTH_DISTILLATION_STATE_PT):
        path = checkpoint_dir / filename
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No depth-distillation student checkpoint found in {checkpoint_dir} "
        f"(expected {DEPTH_DISTILLATION_STATE_SAFE} or {DEPTH_DISTILLATION_STATE_PT})."
    )


def load_checkpoint_state(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    state_path = resolve_state_path(checkpoint_dir)
    if state_path.suffix == ".safetensors":
        try:
            import safetensors.torch
        except ImportError:
            fallback_path = state_path.with_suffix(".pt")
            if fallback_path.exists():
                return dict(torch.load(fallback_path, map_location="cpu", weights_only=True))
            raise ImportError(
                "safetensors is required to load a .safetensors depth-distillation checkpoint."
            )
        return dict(safetensors.torch.load_file(str(state_path)))
    return dict(torch.load(state_path, map_location="cpu", weights_only=True))


def load_checkpoint_metadata(checkpoint_dir: Path) -> dict | None:
    meta_path = checkpoint_dir / DEPTH_DISTILLATION_META
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def validate_checkpoint_metadata(
    meta: dict,
    *,
    double_specs: list[IntervalSpec],
    single_specs: list[IntervalSpec],
    student_block_init: str,
) -> None:
    current_double = [[spec.start, spec.end] for spec in double_specs]
    current_single = [[spec.start, spec.end] for spec in single_specs]
    current_double_sources = [spec.source_index for spec in double_specs]
    current_single_sources = [spec.source_index for spec in single_specs]

    if "double_stream_intervals" in meta and meta["double_stream_intervals"] != current_double:
        raise ValueError(
            "Checkpoint double-stream intervals do not match the config: "
            f"{meta['double_stream_intervals']} vs {current_double}"
        )
    if "single_stream_intervals" in meta and meta["single_stream_intervals"] != current_single:
        raise ValueError(
            "Checkpoint single-stream intervals do not match the config: "
            f"{meta['single_stream_intervals']} vs {current_single}"
        )
    if (
        "double_stream_student_sources" in meta
        and meta["double_stream_student_sources"] != current_double_sources
    ):
        raise ValueError(
            "Checkpoint double-stream source blocks do not match the config: "
            f"{meta['double_stream_student_sources']} vs {current_double_sources}"
        )
    if (
        "single_stream_student_sources" in meta
        and meta["single_stream_student_sources"] != current_single_sources
    ):
        raise ValueError(
            "Checkpoint single-stream source blocks do not match the config: "
            f"{meta['single_stream_student_sources']} vs {current_single_sources}"
        )
    if "student_block_init" in meta and meta["student_block_init"] != student_block_init:
        raise ValueError(
            "Checkpoint student_block_init does not match the config: "
            f"{meta['student_block_init']!r} vs {student_block_init!r}"
        )


def load_student_state_into_blocks(
    state: dict[str, torch.Tensor],
    *,
    prefix: str,
    blocks: list[nn.Module],
) -> None:
    for index, block in enumerate(blocks):
        block_prefix = f"{prefix}.{index}."
        block_state = {
            key[len(block_prefix):]: value
            for key, value in state.items()
            if key.startswith(block_prefix)
        }
        if not block_state:
            raise ValueError(f"Missing checkpoint state for {block_prefix.rstrip('.')}")
        set_peft_model_state_dict(block, block_state, adapter_name="default")


def validate_non_overlapping_specs(specs: list[IntervalSpec], *, stream_name: str) -> list[IntervalSpec]:
    ordered = sorted(specs, key=lambda spec: (spec.start, spec.end))
    previous_end = -1
    for spec in ordered:
        if spec.start <= previous_end:
            raise ValueError(
                f"Overlapping {stream_name} pruning intervals are not supported: "
                f"[{spec.start}, {spec.end}] overlaps a previous interval."
            )
        previous_end = spec.end
    return ordered


def build_compact_block_list(
    *,
    blocks: nn.ModuleList,
    specs: list[IntervalSpec],
    student_blocks: list[nn.Module],
) -> nn.ModuleList:
    compact_blocks: list[nn.Module] = []
    cursor = 0
    for spec in specs:
        compact_blocks.extend(blocks[cursor:spec.start])
        compact_blocks.append(student_blocks[spec.student_index].merge_and_unload())
        cursor = spec.end + 1
    compact_blocks.extend(blocks[cursor:])
    return nn.ModuleList(compact_blocks)


def update_transformer_config_value(transformer: nn.Module, key: str, value: int) -> None:
    if hasattr(transformer, "register_to_config"):
        transformer.register_to_config(**{key: value})
        return
    if getattr(transformer, "config", None) is not None:
        setattr(transformer.config, key, value)


def assert_no_lora_keys(transformer: nn.Module) -> None:
    lora_keys = [key for key in transformer.state_dict().keys() if "lora_" in key.lower()]
    if lora_keys:
        preview = ", ".join(lora_keys[:5])
        raise RuntimeError(
            "LoRA keys are still present after merge. Example keys: "
            f"{preview}"
        )


def build_student_blocks(transformer: nn.Module, loss_kwargs: dict) -> tuple[list[IntervalSpec], list[nn.Module], list[IntervalSpec], list[nn.Module]]:
    double_intervals, single_intervals = parse_pruning_intervals(
        pruned_blocks=loss_kwargs.get("pruned_blocks"),
        double_stream_pruned_blocks=loss_kwargs.get("double_stream_pruned_blocks"),
        single_stream_pruned_blocks=loss_kwargs.get("single_stream_pruned_blocks"),
    )
    if not double_intervals and not single_intervals:
        raise ValueError("At least one pruning interval is required.")

    student_block_init = loss_kwargs.get("student_block_init", "midpoint")
    lora_rank = int(loss_kwargs.get("lora_rank", 128))
    lora_alpha = loss_kwargs.get("lora_alpha")
    lora_dropout = float(loss_kwargs.get("lora_dropout", 0.0))
    lora_target_modules = loss_kwargs.get("lora_target_modules")
    if isinstance(lora_target_modules, str):
        lora_target_modules = [item.strip() for item in lora_target_modules.split(",") if item.strip()]

    double_specs: list[IntervalSpec] = []
    double_blocks: list[nn.Module] = []
    total_double = len(transformer.transformer_blocks)

    for student_index, (start, end) in enumerate(double_intervals):
        if start < 0 or end >= total_double:
            raise ValueError(
                f"Double-stream pruning interval [{start}, {end}] is outside "
                f"the available range [0, {total_double - 1}]."
            )
        source_index = resolve_student_source_index(start, end, student_block_init)
        source_block = transformer.transformer_blocks[source_index]
        student_block = make_lora_student_block(
            copy.deepcopy(source_block),
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=lora_target_modules,
        )
        student_block.to(device=module_device(source_block), dtype=module_dtype(source_block))
        double_blocks.append(student_block)
        double_specs.append(
            IntervalSpec(
                stream="double",
                start=start,
                end=end,
                source_index=source_index,
                student_index=student_index,
            )
        )

    single_specs: list[IntervalSpec] = []
    single_blocks: list[nn.Module] = []
    total_single = len(getattr(transformer, "single_transformer_blocks", []))

    for student_index, (start, end) in enumerate(single_intervals):
        if not hasattr(transformer, "single_transformer_blocks"):
            raise ValueError(
                "Single-stream pruning was requested but the transformer has no "
                "`single_transformer_blocks`."
            )
        if start < 0 or end >= total_single:
            raise ValueError(
                f"Single-stream pruning interval [{start}, {end}] is outside "
                f"the available range [0, {total_single - 1}]."
            )
        source_index = resolve_student_source_index(start, end, student_block_init)
        source_block = transformer.single_transformer_blocks[source_index]
        student_block = make_lora_student_block(
            copy.deepcopy(source_block),
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=lora_target_modules,
        )
        student_block.to(device=module_device(source_block), dtype=module_dtype(source_block))
        single_blocks.append(student_block)
        single_specs.append(
            IntervalSpec(
                stream="single",
                start=start,
                end=end,
                source_index=source_index,
                student_index=student_index,
            )
        )

    return double_specs, double_blocks, single_specs, single_blocks


def compact_transformer(
    transformer: nn.Module,
    *,
    double_specs: list[IntervalSpec],
    double_blocks: list[nn.Module],
    single_specs: list[IntervalSpec],
    single_blocks: list[nn.Module],
) -> nn.Module:
    ordered_double_specs = validate_non_overlapping_specs(
        double_specs,
        stream_name="double-stream",
    )
    transformer.transformer_blocks = build_compact_block_list(
        blocks=transformer.transformer_blocks,
        specs=ordered_double_specs,
        student_blocks=double_blocks,
    )
    update_transformer_config_value(
        transformer,
        "num_layers",
        len(transformer.transformer_blocks),
    )

    if single_specs:
        ordered_single_specs = validate_non_overlapping_specs(
            single_specs,
            stream_name="single-stream",
        )
        transformer.single_transformer_blocks = build_compact_block_list(
            blocks=transformer.single_transformer_blocks,
            specs=ordered_single_specs,
            student_blocks=single_blocks,
        )
        update_transformer_config_value(
            transformer,
            "num_single_layers",
            len(transformer.single_transformer_blocks),
        )

    return transformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a standalone pruned checkpoint from a depth-distillation LoRA checkpoint."
    )
    parser.add_argument("--base_checkpoint", type=str, required=True, help="Base Flux.2 Klein checkpoint or model repo.")
    parser.add_argument("--config", type=str, required=True, help="Depth-distillation YAML config.")
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        required=True,
        help="Checkpoint directory containing depth_distillation_students.* artifacts.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the merged transformer checkpoint and config.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=("fp16", "bf16", "fp32"),
        default=None,
        help="Override output/model dtype. Defaults to mixed_precision from the YAML config.",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cuda, cuda:1, cpu.")
    parser.add_argument(
        "--disable_safe_serialization",
        action="store_true",
        help="Save .bin files instead of safetensors where supported.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_config = load_yaml_config(args.config)
    model_cfg = raw_config.get("model", {})
    dit_cfg = model_cfg.get("dit") or model_cfg.get("transformer")
    loss_cfg = raw_config.get("loss") or {}
    loss_kwargs = dict(loss_cfg.get("kwargs") or {})

    if dit_cfg is None:
        raise ValueError("The YAML config must define `model.dit` or `model.transformer`.")

    transformer_cls = resolve_class(dit_cfg["class_name"])
    revision = model_cfg.get("revision")
    variant = model_cfg.get("variant")
    subfolder = dit_cfg.get("subfolder", "transformer")

    device = resolve_device(args.device)
    weight_dtype = resolve_weight_dtype(raw_config, args.dtype)

    checkpoint_dir = Path(args.lora_checkpoint)
    if checkpoint_dir.is_file():
        checkpoint_dir = checkpoint_dir.parent
    checkpoint_dir = checkpoint_dir.resolve()

    print(f"Loading base transformer from: {args.base_checkpoint}")
    transformer = transformer_cls.from_pretrained(
        args.base_checkpoint,
        subfolder=subfolder,
        revision=revision,
        variant=variant,
        torch_dtype=weight_dtype,
    )
    transformer.requires_grad_(False)
    transformer.to(device=device, dtype=weight_dtype)
    transformer.eval()

    print("Rebuilding student LoRA blocks from config...")
    double_specs, double_blocks, single_specs, single_blocks = build_student_blocks(
        transformer,
        loss_kwargs,
    )

    checkpoint_meta = load_checkpoint_metadata(checkpoint_dir)
    if checkpoint_meta is not None:
        validate_checkpoint_metadata(
            checkpoint_meta,
            double_specs=double_specs,
            single_specs=single_specs,
            student_block_init=loss_kwargs.get("student_block_init", "midpoint"),
        )

    print(f"Loading trained student weights from: {checkpoint_dir}")
    checkpoint_state = load_checkpoint_state(checkpoint_dir)
    load_student_state_into_blocks(
        checkpoint_state,
        prefix="student_double_blocks",
        blocks=double_blocks,
    )
    load_student_state_into_blocks(
        checkpoint_state,
        prefix="student_single_blocks",
        blocks=single_blocks,
    )

    print("Merging LoRA weights and compacting transformer block lists...")
    transformer = compact_transformer(
        transformer,
        double_specs=double_specs,
        double_blocks=double_blocks,
        single_specs=single_specs,
        single_blocks=single_blocks,
    )
    transformer.to(device="cpu")
    assert_no_lora_keys(transformer)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving standalone transformer checkpoint to: {output_dir}")
    transformer.save_pretrained(
        output_dir,
        safe_serialization=not args.disable_safe_serialization,
    )

    print("Done.")
    print(f"Saved transformer config: {output_dir / 'config.json'}")
    print(f"Final double-stream blocks: {len(transformer.transformer_blocks)}")
    print(f"Final single-stream blocks: {len(getattr(transformer, 'single_transformer_blocks', []))}")


if __name__ == "__main__":
    main()
