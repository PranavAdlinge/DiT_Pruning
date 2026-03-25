"""
Loss context: full information passed to loss. Loss has access to everything.
"""

from dataclasses import dataclass

import torch


@dataclass
class LossContext:
    """
    Full context for loss computation. Passed to every loss; loss has access to everything.

    Attributes:
        batch: dict with latents, text_embeds, text_ids, captions
        noise: Random noise used for flow-matching (same shape as model_input)
        model_output: Student model prediction (unpacked latents)
        model_input: Patchified, normalized latents
        weighting: Per-element loss weight from sigma schedule
        packed_noisy: Packed noisy input for transformer forward
        model_input_ids: Latent position IDs for transformer
        timesteps: Diffusion timestep per sample
        sigmas: Noise schedule sigma per sample
        guidance: Guidance scale tensor or None
        text_embeds: Text encoder outputs
        text_ids: Text position IDs for transformer
        model: Student transformer (for EMA or model-dependent logic)
        step: Current training step

    Note:
        Target for flow-matching is computed in the loss: target = noise - model_input
    """

    # --- Batch and tensors from forward pass ---
    batch: dict
    noise: torch.Tensor
    model_output: torch.Tensor
    model_input: torch.Tensor
    weighting: torch.Tensor
    packed_noisy: torch.Tensor
    model_input_ids: torch.Tensor
    timesteps: torch.Tensor
    sigmas: torch.Tensor
    guidance: torch.Tensor | None
    text_embeds: torch.Tensor
    text_ids: torch.Tensor

    # --- Optional: model, step ---
    model: torch.nn.Module | None = None
    step: int = 0
