"""
Distillation loss: flow loss + distillation loss. Teacher created inside.
"""

import torch

from .context import LossContext
from .flow_matching import FlowMatchingLoss


class DistillationLoss:
    """
    Combines flow-matching loss and distillation loss. Teacher created inside on first call.

    Config (via YAML loss.kwargs):
      base: mse | l1 | huber | logcosh (for both flow and distillation terms)
      huber_delta: float (when base=huber)
      flow_weight: weight for flow loss (default 0.5)
      distillation_weight: weight for distillation loss (default 0.5)
      pretrained_model_name_or_path: teacher model path (required for distillation)

    When pretrained_model_name_or_path is set, also provide model_cfg, accelerator, weight_dtype
    (passed by build_loss_fn from main). DistillationLoss resolves transformer_cls, device, etc.
    """

    def __init__(
        self,
        base: str = "mse",
        huber_delta: float = 1.0,
        flow_weight: float = 0.5,
        distillation_weight: float = 0.5,
        pretrained_model_name_or_path: str | None = None,
        transformer_cls: type | None = None,
        transformer_subfolder: str = "transformer",
        revision: str | None = None,
        variant: str | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        model_cfg=None,
        accelerator=None,
        weight_dtype=None,
    ):
        self.flow_loss = FlowMatchingLoss(base=base, huber_delta=huber_delta)
        self.flow_weight = flow_weight
        self.distillation_weight = distillation_weight
        self.teacher_path = pretrained_model_name_or_path
        self._teacher: torch.nn.Module | None = None

        if not pretrained_model_name_or_path:
            self.transformer_cls = None
            self.transformer_subfolder = "transformer"
            self.revision = None
            self.variant = None
            self.device = None
            self.dtype = torch.float32
            return

        if transformer_cls is not None and device is not None:
            self.transformer_cls = transformer_cls
            self.transformer_subfolder = transformer_subfolder
            self.revision = revision
            self.variant = variant
            self.device = device
            self.dtype = dtype
        else:
            self._resolve_from_context(model_cfg, accelerator, weight_dtype)

    def _resolve_from_context(self, model_cfg, accelerator, weight_dtype) -> None:
        """Resolve transformer_cls, device, dtype from training context."""
        if model_cfg is None or accelerator is None or weight_dtype is None:
            raise ValueError(
                "DistillationLoss with pretrained_model_name_or_path requires "
                "model_cfg, accelerator, and weight_dtype (passed by build_loss_fn)."
            )
        dit = getattr(model_cfg, "dit", None) or getattr(model_cfg, "transformer", None)
        if dit is None:
            raise ValueError("model.dit or model.transformer required for DistillationLoss")
        self.transformer_cls = getattr(dit, "_class", None)
        if self.transformer_cls is None:
            raise ValueError("model.dit.class_name must be resolved for DistillationLoss")
        self.transformer_subfolder = getattr(dit, "subfolder", "transformer")
        self.revision = getattr(model_cfg, "revision", None)
        self.variant = getattr(model_cfg, "variant", None)
        self.device = accelerator.device
        self.dtype = weight_dtype

    def _ensure_teacher(self, ctx: LossContext) -> torch.nn.Module | None:
        if self._teacher is not None:
            return self._teacher
        if not self.teacher_path or self.transformer_cls is None or self.device is None:
            return None
        self._teacher = self.transformer_cls.from_pretrained(
            self.teacher_path,
            subfolder=self.transformer_subfolder,
            revision=self.revision,
            variant=self.variant,
            torch_dtype=self.dtype,
        )
        self._teacher.requires_grad_(False)
        self._teacher.eval()
        self._teacher.to(device=self.device, dtype=self.dtype)
        return self._teacher

    def _compute_teacher_pred(self, ctx: LossContext) -> torch.Tensor | None:
        teacher = self._ensure_teacher(ctx)
        if teacher is None:
            return None
        from diffusers import Flux2KleinPipeline

        with torch.no_grad():
            out = teacher(
                hidden_states=ctx.packed_noisy,
                timestep=ctx.timesteps / 1000,
                guidance=ctx.guidance,
                encoder_hidden_states=ctx.text_embeds,
                txt_ids=ctx.text_ids,
                img_ids=ctx.model_input_ids,
                return_dict=False,
            )[0]
        out = out[:, : ctx.packed_noisy.size(1) :]
        return Flux2KleinPipeline._unpack_latents_with_ids(out, ctx.model_input_ids)

    def _compute_flow_loss(self, ctx: LossContext) -> torch.Tensor:
        target = ctx.noise - ctx.model_input
        return self.flow_loss._base_loss(
            ctx.model_output, target, ctx.weighting
        )

    def _compute_distillation_loss(
        self, pred: torch.Tensor, teacher_pred: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        return self.flow_loss._base_loss(pred, teacher_pred, w)

    def __call__(self, ctx: LossContext) -> tuple[torch.Tensor, dict[str, float]]:
        flow_loss = self._compute_flow_loss(ctx)
        teacher_pred = self._compute_teacher_pred(ctx)

        if teacher_pred is not None:
            distill_loss = self._compute_distillation_loss(
                ctx.model_output, teacher_pred, ctx.weighting
            )
            total = self.flow_weight * flow_loss + self.distillation_weight * distill_loss
            logs = {
                "loss": total.detach().item(),
                "loss_flow": flow_loss.detach().item(),
                "loss_distillation": distill_loss.detach().item(),
            }
        else:
            total = flow_loss
            logs = {"loss": total.detach().item(), "loss_flow": flow_loss.detach().item()}

        return total, logs
