"""
Validation during training: build pipeline with trained transformer, generate sample
images, and log to MLflow.
"""

import logging
from pathlib import Path
from typing import Any

import torch
from diffusers.training_utils import free_memory
from tqdm.auto import tqdm

from nexus.utils.log_utils import log_validation_images_to_mlflow

logger = logging.getLogger(__name__)


def run_validation(
    pipeline_cls: type,
    transformer: torch.nn.Module,
    validation_prompt: str,
    accelerator: Any,
    step: int,
    output_dir: str | Path,
    num_images: int = 4,
    seed: int | None = 42,
    resolution: int = 512,
    weight_dtype: torch.dtype = torch.float16,
    pretrained_path: str | None = None,
    inference_steps: int = 4,
    guidance_scale: float = 1.0,
) -> None:
    """
    Build a pipeline with the trained transformer, run inference, and log images
    to MLflow.
    """
    pipeline = pipeline_cls.from_pretrained(
        pretrained_path,
        transformer=transformer,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(dtype=weight_dtype)
    pipeline.enable_model_cpu_offload()
    pipeline.set_progress_bar_config(disable=True)

    generator = (
        torch.Generator(device=accelerator.device).manual_seed(seed) if seed is not None else None
    )

    images = []
    for _ in tqdm(range(num_images), desc="Validation images", leave=False):
        with torch.autocast(device_type=accelerator.device.type, dtype=weight_dtype):
            out = pipeline(
                prompt=validation_prompt,
                height=resolution,
                width=resolution,
                generator=generator,
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
            )
        images.append(out.images[0])

    for tracker in accelerator.trackers:
        if tracker.name == "mlflow":
            log_validation_images_to_mlflow(images, step, output_dir)

    del pipeline
    free_memory()
    logger.info(f"Validation at step {step}: generated {num_images} images")
