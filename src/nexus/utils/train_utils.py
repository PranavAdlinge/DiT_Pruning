"""
Training utilities: model unwrapping.
"""

import torch
from diffusers.utils.torch_utils import is_compiled_module


def unwrap_model(accelerator, m: torch.nn.Module) -> torch.nn.Module:
    """Unwrap DDP/FSDP/compile wrapper to get the raw model."""
    m = accelerator.unwrap_model(m)
    return m._orig_mod if is_compiled_module(m) else m
