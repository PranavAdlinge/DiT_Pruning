"""
Non-sequential depth distillation for pruned Flux.2 transformer intervals.

This loss trains standalone student blocks cloned from each pruned interval,
with trainable LoRA adapters attached to those cloned blocks. By default the
student is initialized from the interval midpoint, matching the Qwen pruning
distillation setup that compresses an interval into one replacement block. The
student output is matched against the frozen teacher representation at the end
of the interval.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

from .context import LossContext

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
        raise ValueError(
            f"Pruning interval start must be <= end, got {value!r}."
        )

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


def _parse_interval_collection(
    values,
    *,
    expected_stream: str | None = None,
) -> list[tuple[int, int]]:
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

    return sorted(set(explicit_intervals + _coalesce_indices(sorted(set(block_indices)))))


def _resolve_student_source_index(start: int, end: int, mode: str) -> int:
    mode = str(mode).strip().lower()
    if mode == "start":
        return start
    if mode == "midpoint":
        return (start + end) // 2
    raise ValueError(
        "student_block_init must be one of {'start', 'midpoint'}, "
        f"got {mode!r}."
    )


def parse_pruning_intervals(
    *,
    pruned_blocks=None,
    double_stream_pruned_blocks=None,
    single_stream_pruned_blocks=None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Parse pruning specs into contiguous intervals.

    Accepted formats:
    - `pruned_blocks: [["d1", "d2"], ["s2", "s4"]]`
    - `pruned_blocks: ["d1", "d2", "s2", "s3", "s4"]`
    - `double_stream_pruned_blocks: [[1, 2]]`
    - `double_stream_pruned_blocks: [1, 2]`
    - `single_stream_pruned_blocks: [["s2", "s4"]]`
    - `single_stream_pruned_blocks: ["s2", "s3", "s4"]`
    """

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


def _l2_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    denom = torch.linalg.norm(x.float(), ord=2, dim=-1, keepdim=True).clamp_min(eps)
    return x.float() / denom


def _normalized_mse(student: torch.Tensor, teacher: torch.Tensor, eps: float) -> torch.Tensor:
    student_norm = _l2_normalize(student, eps)
    teacher_norm = _l2_normalize(teacher, eps)
    return torch.nn.functional.mse_loss(student_norm, teacher_norm, reduction="none").sum(dim=-1).mean()


def _module_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


def _find_lora_target_modules(module: nn.Module) -> list[str]:
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
    rank: int = 128,
    alpha: int | None = None,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
) -> nn.Module:
    target_modules = (
        sorted(set(target_modules))
        if target_modules is not None
        else _find_lora_target_modules(block)
    )
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank if alpha is None else alpha,
        lora_dropout=dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    return get_peft_model(block, lora_config)


class _DoubleStreamPassthroughBlock(nn.Module):
    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return encoder_hidden_states, hidden_states


class _SingleStreamPassthroughBlock(nn.Module):
    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        **kwargs,
    ) -> torch.Tensor:
        return hidden_states


class PruningDepthDistillationLoss(nn.Module):
    """
    Interval-wise depth distillation for Flux.2 pruning.

    The student consists of frozen cloned blocks with trainable LoRA adapters
    attached to a per-interval source block. By default this source block is
    the interval midpoint, matching the Qwen-style interval compression flow.
    Each student block receives the frozen teacher input at the start of the
    interval and matches the teacher output at the end of the interval.
    """

    manages_model_forward = True
    train_base_model = False
    supports_validation = False
    save_transformer = False

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        pruned_blocks=None,
        double_stream_pruned_blocks=None,
        single_stream_pruned_blocks=None,
        lora_rank: int = 128,
        lora_alpha: int | None = None,
        lora_dropout: float = 0.0,
        lora_target_modules: list[str] | None = None,
        student_block_init: str = "midpoint",
        depth_weight: float = 1.0,
        normalize_eps: float = 1.0e-6,
        transformer_cls: type | None = None,
        transformer_subfolder: str = "transformer",
        revision: str | None = None,
        variant: str | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        model_cfg=None,
        accelerator=None,
        weight_dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.teacher_path = pretrained_model_name_or_path
        self.depth_weight = depth_weight
        self.normalize_eps = normalize_eps
        self.lora_rank = lora_rank
        self.lora_alpha = lora_rank if lora_alpha is None else lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.student_block_init = student_block_init

        self.double_intervals, self.single_intervals = parse_pruning_intervals(
            pruned_blocks=pruned_blocks,
            double_stream_pruned_blocks=double_stream_pruned_blocks,
            single_stream_pruned_blocks=single_stream_pruned_blocks,
        )
        if not self.double_intervals and not self.single_intervals:
            raise ValueError("PruningDepthDistillationLoss requires at least one pruning interval.")

        self.double_interval_specs: list[IntervalSpec] = []
        self.single_interval_specs: list[IntervalSpec] = []
        self.student_double_blocks = nn.ModuleList()
        self.student_single_blocks = nn.ModuleList()

        self.transformer_cls = transformer_cls
        self.transformer_subfolder = transformer_subfolder
        self.revision = revision
        self.variant = variant
        self.device = device
        self.dtype = dtype if weight_dtype is None else weight_dtype
        self.__dict__["_teacher"] = None
        self._prepared = False

        if self.transformer_cls is None or self.device is None:
            self._resolve_from_context(model_cfg, accelerator, weight_dtype)

    @staticmethod
    def _artifact_paths(output_dir: str | Path) -> tuple[Path, Path]:
        output_dir = Path(output_dir)
        return output_dir / DEPTH_DISTILLATION_STATE_SAFE, output_dir / DEPTH_DISTILLATION_META

    @staticmethod
    def _resolve_state_path(output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        for name in (DEPTH_DISTILLATION_STATE_SAFE, DEPTH_DISTILLATION_STATE_PT):
            path = output_dir / name
            if path.exists():
                return path
        raise FileNotFoundError(
            f"No depth-distillation student checkpoint found in {output_dir} "
            f"(expected {DEPTH_DISTILLATION_STATE_SAFE} or {DEPTH_DISTILLATION_STATE_PT})."
        )

    def _student_state_dict(self) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for index, block in enumerate(self.student_double_blocks):
            for key, value in get_peft_model_state_dict(block).items():
                state[f"student_double_blocks.{index}.{key}"] = value.detach().cpu()
        for index, block in enumerate(self.student_single_blocks):
            for key, value in get_peft_model_state_dict(block).items():
                state[f"student_single_blocks.{index}.{key}"] = value.detach().cpu()
        return state

    def _student_metadata(self) -> dict:
        return {
            "teacher_path": self.teacher_path,
            "depth_weight": self.depth_weight,
            "normalize_eps": self.normalize_eps,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "student_block_init": self.student_block_init,
            "double_stream_intervals": [[spec.start, spec.end] for spec in self.double_interval_specs],
            "double_stream_student_sources": [spec.source_index for spec in self.double_interval_specs],
            "single_stream_intervals": [[spec.start, spec.end] for spec in self.single_interval_specs],
            "single_stream_student_sources": [spec.source_index for spec in self.single_interval_specs],
        }

    def _write_student_artifacts(self, output_dir: str | Path) -> list[Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        state_path, meta_path = self._artifact_paths(output_dir)
        state = self._student_state_dict()
        try:
            import safetensors.torch

            safetensors.torch.save_file(state, str(state_path))
        except ImportError:
            state_path = state_path.with_suffix(".pt")
            torch.save(state, state_path)

        meta_path.write_text(json.dumps(self._student_metadata(), indent=2), encoding="utf-8")
        return [state_path, meta_path]

    def _load_student_state(self, state: dict[str, torch.Tensor]) -> None:
        for index, block in enumerate(self.student_double_blocks):
            prefix = f"student_double_blocks.{index}."
            block_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            if block_state:
                set_peft_model_state_dict(block, block_state, adapter_name="default")

        for index, block in enumerate(self.student_single_blocks):
            prefix = f"student_single_blocks.{index}."
            block_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            if block_state:
                set_peft_model_state_dict(block, block_state, adapter_name="default")

    def _validate_checkpoint_metadata(self, meta: dict) -> None:
        current_double = [[spec.start, spec.end] for spec in self.double_interval_specs]
        current_single = [[spec.start, spec.end] for spec in self.single_interval_specs]
        if "double_stream_intervals" in meta and meta["double_stream_intervals"] != current_double:
            raise ValueError(
                "Checkpoint double-stream intervals do not match the current config: "
                f"{meta['double_stream_intervals']} vs {current_double}"
            )
        if "single_stream_intervals" in meta and meta["single_stream_intervals"] != current_single:
            raise ValueError(
                "Checkpoint single-stream intervals do not match the current config: "
                f"{meta['single_stream_intervals']} vs {current_single}"
            )

    def _load_checkpoint_metadata(self, input_dir: str | Path) -> dict | None:
        _, meta_path = self._artifact_paths(input_dir)
        if not meta_path.exists():
            return None
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def _resolve_from_context(self, model_cfg, accelerator, weight_dtype) -> None:
        if model_cfg is None or accelerator is None or weight_dtype is None:
            raise ValueError(
                "PruningDepthDistillationLoss requires model_cfg, accelerator, and "
                "weight_dtype from build_loss_fn."
            )

        dit = getattr(model_cfg, "dit", None) or getattr(model_cfg, "transformer", None)
        if dit is None:
            raise ValueError("model.dit or model.transformer must be configured.")

        self.transformer_cls = getattr(dit, "_class", None)
        if self.transformer_cls is None:
            raise ValueError("model.dit.class_name must be resolved for pruning distillation.")

        self.transformer_subfolder = getattr(dit, "subfolder", "transformer")
        self.revision = getattr(model_cfg, "revision", None)
        self.variant = getattr(model_cfg, "variant", None)
        self.device = accelerator.device
        self.dtype = weight_dtype

    def prepare_for_training(self, transformer: torch.nn.Module) -> None:
        if self._prepared:
            return

        if not hasattr(transformer, "transformer_blocks"):
            raise ValueError(
                "PruningDepthDistillationLoss expects a transformer with `transformer_blocks`."
            )

        total_double = len(transformer.transformer_blocks)
        total_single = len(getattr(transformer, "single_transformer_blocks", []))

        for idx, (start, end) in enumerate(self.double_intervals):
            if start < 0 or end >= total_double:
                raise ValueError(
                    f"Double-stream pruning interval [{start}, {end}] is outside "
                    f"the available range [0, {total_double - 1}]."
                )
            source_index = _resolve_student_source_index(start, end, self.student_block_init)
            student_block = make_lora_student_block(
                copy.deepcopy(transformer.transformer_blocks[source_index]),
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
                target_modules=self.lora_target_modules,
            )
            self.student_double_blocks.append(student_block)
            self.double_interval_specs.append(
                IntervalSpec(
                    stream="double",
                    start=start,
                    end=end,
                    source_index=source_index,
                    student_index=idx,
                )
            )

        for idx, (start, end) in enumerate(self.single_intervals):
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
            source_index = _resolve_student_source_index(start, end, self.student_block_init)
            student_block = make_lora_student_block(
                copy.deepcopy(transformer.single_transformer_blocks[source_index]),
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
                target_modules=self.lora_target_modules,
            )
            self.student_single_blocks.append(student_block)
            self.single_interval_specs.append(
                IntervalSpec(
                    stream="single",
                    start=start,
                    end=end,
                    source_index=source_index,
                    student_index=idx,
                )
            )

        self._prepared = True

    def _ensure_teacher(self) -> torch.nn.Module:
        if self._teacher is not None:
            return self._teacher

        teacher = self.transformer_cls.from_pretrained(
            self.teacher_path,
            subfolder=self.transformer_subfolder,
            revision=self.revision,
            variant=self.variant,
            torch_dtype=self.dtype,
        )
        teacher.requires_grad_(False)
        teacher.eval()
        teacher.to(device=self.device, dtype=self.dtype)
        self.__dict__["_teacher"] = teacher
        return teacher

    def save_checkpoint_artifacts(self, output_dir: str | Path) -> list[Path]:
        return self._write_student_artifacts(output_dir)

    def load_checkpoint_artifacts(self, input_dir: str | Path) -> list[Path]:
        if not self._prepared:
            raise RuntimeError(
                "PruningDepthDistillationLoss.prepare_for_training(transformer) must be "
                "called before loading checkpoint artifacts."
            )

        input_dir = Path(input_dir)
        meta = self._load_checkpoint_metadata(input_dir)
        if meta is not None:
            self._validate_checkpoint_metadata(meta)

        state_path = self._resolve_state_path(input_dir)
        if state_path.suffix == ".safetensors":
            try:
                import safetensors.torch

                state = dict(safetensors.torch.load_file(str(state_path)))
            except ImportError:
                fallback_path = state_path.with_suffix(".pt")
                state = dict(torch.load(fallback_path, map_location="cpu", weights_only=True))
                state_path = fallback_path
        else:
            state = dict(torch.load(state_path, map_location="cpu", weights_only=True))

        self._load_student_state(state)
        loaded_paths = [state_path]
        _, meta_path = self._artifact_paths(input_dir)
        if meta_path.exists():
            loaded_paths.append(meta_path)
        return loaded_paths

    def apply_students_to_transformer(
        self,
        transformer: torch.nn.Module,
        *,
        merge_lora: bool = True,
    ) -> torch.nn.Module:
        if not self._prepared:
            self.prepare_for_training(transformer)

        for spec in self.double_interval_specs:
            student_block = self.student_double_blocks[spec.student_index]
            replacement = student_block.merge_and_unload() if merge_lora else student_block
            transformer.transformer_blocks[spec.start] = replacement
            for index in range(spec.start + 1, spec.end + 1):
                transformer.transformer_blocks[index] = _DoubleStreamPassthroughBlock()

        for spec in self.single_interval_specs:
            student_block = self.student_single_blocks[spec.student_index]
            replacement = student_block.merge_and_unload() if merge_lora else student_block
            transformer.single_transformer_blocks[spec.start] = replacement
            for index in range(spec.start + 1, spec.end + 1):
                transformer.single_transformer_blocks[index] = _SingleStreamPassthroughBlock()

        return transformer

    def _prepare_flux2_inputs(self, teacher: torch.nn.Module, ctx: LossContext) -> dict[str, torch.Tensor | None]:
        hidden_states = ctx.packed_noisy
        encoder_hidden_states = ctx.text_embeds

        timestep = ctx.timesteps.to(hidden_states.dtype) * 1000
        guidance = None
        if ctx.guidance is not None:
            guidance = ctx.guidance.to(hidden_states.dtype) * 1000

        temb = teacher.time_guidance_embed(timestep, guidance)
        double_stream_mod_img = teacher.double_stream_modulation_img(temb)
        double_stream_mod_txt = teacher.double_stream_modulation_txt(temb)
        single_stream_mod = teacher.single_stream_modulation(temb)

        hidden_states = teacher.x_embedder(hidden_states)
        encoder_hidden_states = teacher.context_embedder(encoder_hidden_states)

        img_ids = ctx.model_input_ids
        txt_ids = ctx.text_ids
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        image_rotary_emb = teacher.pos_embed(img_ids)
        text_rotary_emb = teacher.pos_embed(txt_ids)
        concat_rotary_emb = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )

        return {
            "encoder_hidden_states": encoder_hidden_states,
            "hidden_states": hidden_states,
            "double_stream_mod_img": double_stream_mod_img,
            "double_stream_mod_txt": double_stream_mod_txt,
            "single_stream_mod": single_stream_mod,
            "concat_rotary_emb": concat_rotary_emb,
        }

    def _compute_flux2_depth_loss(self, teacher: torch.nn.Module, ctx: LossContext) -> tuple[torch.Tensor, dict[str, float]]:
        teacher_inputs = self._prepare_flux2_inputs(teacher, ctx)
        encoder_hidden_states = teacher_inputs["encoder_hidden_states"]
        hidden_states = teacher_inputs["hidden_states"]
        double_stream_mod_img = teacher_inputs["double_stream_mod_img"]
        double_stream_mod_txt = teacher_inputs["double_stream_mod_txt"]
        single_stream_mod = teacher_inputs["single_stream_mod"]
        concat_rotary_emb = teacher_inputs["concat_rotary_emb"]

        double_start_states: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        double_end_states: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        double_starts = {spec.start for spec in self.double_interval_specs}
        double_ends = {spec.end for spec in self.double_interval_specs}

        with torch.no_grad():
            for index_block, block in enumerate(teacher.transformer_blocks):
                if index_block in double_starts:
                    double_start_states[index_block] = (encoder_hidden_states, hidden_states)

                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb_mod_img=double_stream_mod_img,
                    temb_mod_txt=double_stream_mod_txt,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=None,
                )

                if index_block in double_ends:
                    double_end_states[index_block] = (encoder_hidden_states, hidden_states)

            single_hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

            single_start_states: dict[int, torch.Tensor] = {}
            single_end_states: dict[int, torch.Tensor] = {}
            single_starts = {spec.start for spec in self.single_interval_specs}
            single_ends = {spec.end for spec in self.single_interval_specs}

            if 0 in single_starts:
                single_start_states[0] = single_hidden_states

            for index_block, block in enumerate(teacher.single_transformer_blocks):
                single_hidden_states = block(
                    hidden_states=single_hidden_states,
                    encoder_hidden_states=None,
                    temb_mod=single_stream_mod,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=None,
                )

                if index_block in single_ends:
                    single_end_states[index_block] = single_hidden_states
                if index_block + 1 in single_starts:
                    single_start_states[index_block + 1] = single_hidden_states

        if self.student_double_blocks:
            device = _module_device(self.student_double_blocks[0])
        else:
            device = _module_device(self.student_single_blocks[0])
        total_loss = torch.zeros((), device=device)
        double_loss = torch.zeros((), device=device)
        single_loss = torch.zeros((), device=device)

        for spec in self.double_interval_specs:
            start_encoder_hidden_states, start_hidden_states = double_start_states[spec.start]
            end_encoder_hidden_states, end_hidden_states = double_end_states[spec.end]
            student_encoder_hidden_states, student_hidden_states = self.student_double_blocks[spec.student_index](
                hidden_states=start_hidden_states,
                encoder_hidden_states=start_encoder_hidden_states,
                temb_mod_img=double_stream_mod_img,
                temb_mod_txt=double_stream_mod_txt,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=None,
            )
            interval_loss = _normalized_mse(
                student_encoder_hidden_states,
                end_encoder_hidden_states,
                self.normalize_eps,
            ) + _normalized_mse(
                student_hidden_states,
                end_hidden_states,
                self.normalize_eps,
            )
            double_loss = double_loss + interval_loss

        for spec in self.single_interval_specs:
            start_hidden_states = single_start_states[spec.start]
            end_hidden_states = single_end_states[spec.end]
            student_hidden_states = self.student_single_blocks[spec.student_index](
                hidden_states=start_hidden_states,
                encoder_hidden_states=None,
                temb_mod=single_stream_mod,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=None,
            )
            interval_loss = _normalized_mse(
                student_hidden_states,
                end_hidden_states,
                self.normalize_eps,
            )
            single_loss = single_loss + interval_loss

        total_loss = self.depth_weight * (double_loss + single_loss)
        return total_loss, {
            "loss": total_loss.detach().item(),
            "loss_depth": total_loss.detach().item(),
            "loss_depth_double": double_loss.detach().item(),
            "loss_depth_single": single_loss.detach().item(),
        }

    def forward(self, ctx: LossContext) -> tuple[torch.Tensor, dict[str, float]]:
        if not self._prepared:
            raise RuntimeError(
                "PruningDepthDistillationLoss.prepare_for_training(transformer) must be "
                "called before training."
            )

        teacher = self._ensure_teacher()
        if not hasattr(teacher, "time_guidance_embed"):
            raise ValueError(
                "PruningDepthDistillationLoss currently supports Flux.2 style transformers "
                "with `time_guidance_embed`."
            )
        return self._compute_flux2_depth_loss(teacher, ctx)

    def save_training_artifacts(self, output_dir: str | Path) -> list[Path]:
        return self._write_student_artifacts(output_dir)
