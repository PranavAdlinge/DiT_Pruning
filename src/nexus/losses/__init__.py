"""
Losses: FlowMatchingLoss, FlowMatchingWithPriorPreservation, DistillationLoss.

Each receives LossContext and returns (scalar, log_dict).

Configure in YAML via loss.class_name and loss.kwargs.
"""

from nexus.train.config import ns_to_kwargs

from .context import LossContext
from .distillation import DistillationLoss
from .flow_matching import BASE_LOSSES, FlowMatchingLoss
from .prior_preservation import FlowMatchingWithPriorPreservation


def build_loss_fn(cfg, *, model_cfg=None, accelerator=None, weight_dtype=None):
    """Build loss from config: instantiate cfg.loss._class(**kwargs)."""
    loss_cls = getattr(cfg.loss, "_class", None)
    if loss_cls is None:
        raise ValueError("Config must define loss.class_name (e.g. nexus.losses:FlowMatchingLoss)")
    loss_kwargs = ns_to_kwargs(getattr(cfg.loss, "kwargs", None))
    if model_cfg is not None:
        loss_kwargs["model_cfg"] = model_cfg
    if accelerator is not None:
        loss_kwargs["accelerator"] = accelerator
    if weight_dtype is not None:
        loss_kwargs["weight_dtype"] = weight_dtype
    return loss_cls(**loss_kwargs)


__all__ = [
    "LossContext",
    "FlowMatchingLoss",
    "FlowMatchingWithPriorPreservation",
    "DistillationLoss",
    "BASE_LOSSES",
    "build_loss_fn",
]
