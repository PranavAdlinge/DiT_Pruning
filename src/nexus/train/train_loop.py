"""
Training step: flow-matching loss on precomputed latents + text embeddings.

Prepares context (batch, noise, model_output, etc.) and calls loss with full info.
Loss handles implementation (MSE, distillation, prior preservation).
"""

from typing import Any

import torch
from diffusers import Flux2KleinPipeline
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

from nexus.losses.context import LossContext


def get_sigmas(
    timesteps: torch.Tensor,
    noise_scheduler: Any,
    device: torch.device,
    n_dim: int = 4,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Get sigma values for flow-matching noise schedule."""
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def training_step_precomputed(
    batch: dict,
    transformer: torch.nn.Module,
    latents_bn_mean: torch.Tensor,
    latents_bn_std: torch.Tensor,
    noise_scheduler_copy: Any,
    weighting_scheme: str,
    logit_mean: float,
    logit_std: float,
    guidance_scale: float,
    accelerator: Any,
    loss_fn: Any,
    step: int = 0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    One training step: patchify, add noise, run model, build context, call loss.

    Loss receives full context and returns (scalar, log_dict). Loss handles
    distillation (teacher inside), prior preservation, etc.
    """
    latents = batch["latents"]
    text_embeds = batch["text_embeds"]
    text_ids = batch["text_ids"]

    model_input = Flux2KleinPipeline._patchify_latents(latents)
    model_input = (model_input - latents_bn_mean) / latents_bn_std

    model_input_ids = Flux2KleinPipeline._prepare_latent_ids(model_input).to(
        device=model_input.device
    )
    noise = torch.randn_like(model_input)
    bsz = model_input.shape[0]

    u = compute_density_for_timestep_sampling(
        weighting_scheme=weighting_scheme,
        batch_size=bsz,
        logit_mean=logit_mean,
        logit_std=logit_std,
    )
    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
    timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

    sigmas = get_sigmas(
        timesteps,
        noise_scheduler_copy,
        accelerator.device,
        n_dim=model_input.ndim,
        dtype=model_input.dtype,
    )
    noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

    packed_noisy = Flux2KleinPipeline._pack_latents(noisy_model_input)

    guidance = (
        torch.full([1], guidance_scale, device=accelerator.device).expand(bsz)
        if transformer.config.guidance_embeds
        else None
    )

    model_pred = transformer(
        hidden_states=packed_noisy,
        timestep=timesteps / 1000,
        guidance=guidance,
        encoder_hidden_states=text_embeds,
        txt_ids=text_ids,
        img_ids=model_input_ids,
        return_dict=False,
    )[0]
    model_pred = model_pred[:, : packed_noisy.size(1) :]
    model_pred = Flux2KleinPipeline._unpack_latents_with_ids(model_pred, model_input_ids)

    weighting = compute_loss_weighting_for_sd3(weighting_scheme=weighting_scheme, sigmas=sigmas)

    ctx = LossContext(
        batch=batch,
        noise=noise,
        model_output=model_pred,
        model_input=model_input,
        weighting=weighting,
        packed_noisy=packed_noisy,
        model_input_ids=model_input_ids,
        timesteps=timesteps,
        sigmas=sigmas,
        guidance=guidance,
        text_embeds=text_embeds,
        text_ids=text_ids,
        model=transformer,
        step=step,
    )

    return loss_fn(ctx)
