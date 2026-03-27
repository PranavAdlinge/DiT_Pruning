"""
Losses: FlowMatchingLoss, FlowMatchingWithPriorPreservation, DistillationLoss.

Each receives LossContext and returns (scalar, log_dict).

Configure in YAML via loss.class_name and loss.kwargs.
"""

import inspect

from nexus.train.config import ns_to_kwargs

from .context import LossContext
from .depth_distillation import PruningDepthDistillationLoss
from .distillation import DistillationLoss
from .flow_matching import BASE_LOSSES, FlowMatchingLoss
from .prior_preservation import FlowMatchingWithPriorPreservation


def _filter_supported_kwargs(callable_obj, kwargs: dict) -> dict:
    signature = inspect.signature(callable_obj)
    accepts_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return kwargs

    supported = {
        name
        for name, param in signature.parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {key: value for key, value in kwargs.items() if key in supported}


def build_loss_fn(cfg, *, model_cfg=None, accelerator=None, weight_dtype=None, **extra_context):
    """Build loss from config: instantiate cfg.loss._class(**kwargs)."""
    loss_cls = getattr(cfg.loss, "_class", None)
    if loss_cls is None:
        raise ValueError("Config must define loss.class_name (e.g. nexus.losses:FlowMatchingLoss)")
    loss_kwargs = ns_to_kwargs(getattr(cfg.loss, "kwargs", None))
    context_kwargs = dict(extra_context)
    if model_cfg is not None:
        context_kwargs["model_cfg"] = model_cfg
    if accelerator is not None:
        context_kwargs["accelerator"] = accelerator
    if weight_dtype is not None:
        context_kwargs["weight_dtype"] = weight_dtype

    ctor = loss_cls.__init__ if inspect.isclass(loss_cls) else loss_cls
    loss_kwargs.update(_filter_supported_kwargs(ctor, context_kwargs))
    return loss_cls(**loss_kwargs)


__all__ = [
    "LossContext",
    "FlowMatchingLoss",
    "FlowMatchingWithPriorPreservation",
    "DistillationLoss",
    "PruningDepthDistillationLoss",
    "BASE_LOSSES",
    "build_loss_fn",
]
