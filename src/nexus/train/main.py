"""
Flux.2 Klein training on precomputed SSTK MDS data.

Config-driven via YAML. Entry point for LoRA or full fine-tuning of Flux.2
transformers on precomputed VAE latents and text embeddings.

Usage:
    accelerate launch -m nexus.train --config configs/klein4b/t2i_finetune.yaml
    accelerate launch -m nexus.train --config configs/klein4b/t2i_finetune.yaml \\
        --precomputed_data_dir /path/to/mds --output_dir ./out
"""

import warnings

warnings.filterwarnings("ignore", module="distutils")

import copy
import logging
import math
import os
import shutil
from pathlib import Path

import diffusers
import torch
import torch.nn as nn
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import check_min_version
from peft import LoraConfig
from tqdm.auto import tqdm

from nexus.utils.checkpoint_utils import (
    make_klein_load_hook,
    make_klein_save_hook,
    prune_old_checkpoints,
    save_final_klein,
)
from nexus.utils.log_utils import init_trackers, log_dataset_input, setup_tracking
from nexus.utils.train_utils import unwrap_model

from .config import ns_to_kwargs, parse_args
from nexus.losses import build_loss_fn
from .train_loop import training_step_precomputed
from .validation import run_validation

check_min_version("0.37.0.dev0")

# Pipeline class -> (vae_cls, vae_subfolder), (scheduler_cls, scheduler_subfolder)
# Used when loading everything from pipeline except DiT.
_PIPELINE_COMPONENTS = {}
try:
    from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, FlowMatchEulerDiscreteScheduler

    _PIPELINE_COMPONENTS[Flux2KleinPipeline] = {
        "vae": (AutoencoderKLFlux2, "vae"),
        "scheduler": (FlowMatchEulerDiscreteScheduler, "scheduler"),
    }
except ImportError:
    pass
logger = get_logger(__name__)


def _get_obj_attr(obj, name: str, default=None):
    if hasattr(obj, name):
        return getattr(obj, name)
    module = getattr(obj, "module", None)
    if module is not None and hasattr(module, name):
        return getattr(module, name)
    return default


def _collect_trainable_params(*modules: nn.Module) -> list[torch.nn.Parameter]:
    seen: set[int] = set()
    params: list[torch.nn.Parameter] = []
    for module in modules:
        if module is None:
            continue
        for param in module.parameters():
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            params.append(param)
    return params


def _make_skip_transformer_checkpoint_hooks(accelerator: Accelerator, trans_cls: type, unwrap_fn):
    """Skip saving/loading the frozen base transformer for loss-managed training."""

    def save_hook(models, weights, output_dir):
        custom_modules: list[nn.Module] = []

        trans_idx = next(
            (idx for idx, model in enumerate(models) if isinstance(unwrap_fn(model), trans_cls)),
            None,
        )

        for idx, model in enumerate(models):
            unwrapped = unwrap_fn(model)
            if hasattr(unwrapped, "save_checkpoint_artifacts"):
                custom_modules.append(unwrapped)

        if trans_idx is not None and weights and trans_idx < len(weights):
            weights.pop(trans_idx)

        if accelerator.is_main_process:
            for module in custom_modules:
                module.save_checkpoint_artifacts(output_dir)

    def load_hook(models, input_dir):
        trans_idx = next(
            (idx for idx, model in enumerate(models) if isinstance(unwrap_fn(model), trans_cls)),
            None,
        )
        if trans_idx is not None:
            models.pop(trans_idx)

    return save_hook, load_hook


def main(args=None):
    """Run Flux.2 Klein LoRA/full training on precomputed SSTK data."""
    cfg = parse_args(args)

    # --- Config & logging ---
    config_path = getattr(cfg, "_config_path", None)

    # MPS (Apple Silicon) does not support bf16
    if torch.backends.mps.is_available() and getattr(cfg, "mixed_precision", None) == "bf16":
        raise ValueError("bf16 not supported on MPS. Use fp16 or fp32.")

    train_cfg = cfg.train
    model_cfg = cfg.model
    lora_cfg = getattr(cfg, "lora", None)
    train_mode = getattr(cfg, "train_mode", "lora")
    pipeline_cfg = getattr(cfg, "pipeline", None) or getattr(model_cfg, "pipeline", None)
    if not pipeline_cfg:
        raise ValueError("pipeline config is required")
    pretrained_path = getattr(pipeline_cfg, "pretrained_model_name_or_path", None) or getattr(
        model_cfg, "pretrained_model_name_or_path", None
    )
    if not pretrained_path:
        raise ValueError("pipeline.pretrained_model_name_or_path is required")

    # --- Accelerator & trackers ---
    output_dir, log_with = setup_tracking(cfg)

    proj_config = ProjectConfiguration(project_dir=output_dir, logging_dir=output_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        mixed_precision=getattr(cfg, "mixed_precision", None),
        log_with=log_with,
        project_config=proj_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if accelerator.is_main_process:
        logger.info("Config: %s", config_path or "(unknown)")
        logger.info(str(accelerator.state))
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if getattr(cfg, "seed", None) is not None:
        set_seed(cfg.seed)

    # --- Model loading ---
    weight_dtype = torch.float32
    mp = getattr(cfg, "mixed_precision", None)
    if mp == "fp16":
        weight_dtype = torch.float16
    elif mp == "bf16":
        weight_dtype = torch.bfloat16

    revision = getattr(model_cfg, "revision", None)
    variant = getattr(model_cfg, "variant", None)
    pipeline_cls = pipeline_cfg._class
    components = _PIPELINE_COMPONENTS.get(pipeline_cls)
    if not components:
        raise ValueError(
            f"No pipeline component registry for {pipeline_cls}. "
            "Add vae/scheduler to model config or extend _PIPELINE_COMPONENTS."
        )

    # Load vae, scheduler from pipeline path (everything except DiT)
    vae_cls, vae_subfolder = components["vae"]
    vae = vae_cls.from_pretrained(
        pretrained_path,
        subfolder=vae_subfolder,
        revision=revision,
        variant=variant,
    )
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(accelerator.device)
    latents_bn_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
    ).to(accelerator.device)
    del vae

    sched_cls, sched_subfolder = components["scheduler"]
    noise_scheduler = sched_cls.from_pretrained(
        pretrained_path,
        subfolder=sched_subfolder,
        revision=revision,
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    # DiT: load separately
    dit_cfg = getattr(model_cfg, "dit", model_cfg.transformer)
    trans_cls = dit_cfg._class
    subfolder = dit_cfg.subfolder

    transformer = trans_cls.from_pretrained(
        pretrained_path,
        subfolder=subfolder,
        revision=revision,
        variant=variant,
        torch_dtype=weight_dtype,
    )
    transformer.requires_grad_(False)
    transformer.to(device=accelerator.device, dtype=weight_dtype)

    loss_cfg = cfg.loss
    loss_fn = build_loss_fn(cfg, model_cfg=model_cfg, accelerator=accelerator, weight_dtype=weight_dtype)
    if hasattr(loss_fn, "prepare_for_training"):
        loss_fn.prepare_for_training(transformer)

    train_base_model = bool(_get_obj_attr(loss_fn, "train_base_model", True))
    checkpoint_train_mode = train_mode if train_base_model else None
    lora_config = None
    if train_base_model and train_mode == "lora" and lora_cfg:
        target_modules = (
            [m.strip() for m in lora_cfg.target_modules]
            if isinstance(lora_cfg.target_modules, list)
            else [s.strip() for s in str(lora_cfg.target_modules).split(",")]
        )
        lora_config = LoraConfig(
            r=lora_cfg.rank,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        transformer.add_adapter(lora_config)
    elif train_base_model and train_mode == "full":
        transformer.requires_grad_(True)
    elif not train_base_model and accelerator.is_main_process:
        logger.info(
            "Loss-managed training detected: top-level train_mode/lora config is ignored; "
            "trainable student adapters should be configured under loss.kwargs."
        )

    pipeline_cls = pipeline_cfg._class if checkpoint_train_mode == "lora" else None

    if train_base_model and train_cfg.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    os.makedirs(cfg.output_dir, exist_ok=True)
    if config_path and accelerator.is_local_main_process:
        dest = Path(cfg.output_dir) / "config.yaml"
        shutil.copy2(config_path, dest)
        logger.info("Config copied to %s", dest)

    is_fsdp = getattr(accelerator.state, "fsdp_plugin", None) is not None
    unwrap = lambda m: unwrap_model(accelerator, m)

    if train_base_model:
        save_hook = make_klein_save_hook(
            accelerator=accelerator,
            trans_cls=trans_cls,
            train_mode=checkpoint_train_mode,
            pipeline_cls=pipeline_cls,
            unwrap_fn=unwrap,
            is_fsdp=is_fsdp,
        )
        load_hook = make_klein_load_hook(
            accelerator=accelerator,
            trans_cls=trans_cls,
            pretrained_path=pretrained_path,
            subfolder=subfolder,
            train_mode=checkpoint_train_mode,
            pipeline_cls=pipeline_cls,
            lora_config=lora_config,
            unwrap_fn=unwrap,
            mixed_precision=mp,
        )
    else:
        save_hook, load_hook = _make_skip_transformer_checkpoint_hooks(accelerator, trans_cls, unwrap)
    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)

    if getattr(cfg, "allow_tf32", False) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    learning_rate = train_cfg.learning_rate
    if train_cfg.scale_lr:
        learning_rate *= (
            train_cfg.gradient_accumulation_steps * train_cfg.batch_size * accelerator.num_processes
        )

    loss_module = loss_fn if isinstance(loss_fn, nn.Module) else None

    if mp == "fp16":
        cast_training_params([transformer], dtype=torch.float32)
        if loss_module is not None:
            cast_training_params([loss_module], dtype=torch.float32)

    opt_cls = cfg.optimizer._class
    opt_kwargs = ns_to_kwargs(getattr(cfg.optimizer, "kwargs", None))
    trainable_params = _collect_trainable_params(
        transformer if train_base_model else None,
        loss_module,
    )
    if not trainable_params:
        raise ValueError(
            "No trainable parameters were found. Check train_mode and loss configuration."
        )
    optimizer = opt_cls(
        trainable_params,
        lr=learning_rate,
        **opt_kwargs,
    )

    # --- Dataset & dataloader ---
    ds_kwargs = ns_to_kwargs(
        cfg.dataset.kwargs,
        batch_size=train_cfg.batch_size,
        latent_dtype=weight_dtype,
    )
    if ds_kwargs.get("local") is None:
        raise ValueError("dataset.kwargs.local (or --precomputed_data_dir) is required")
    train_dataset = cfg.dataset._class(**ds_kwargs)

    collate_fn = cfg.collate._fn if hasattr(cfg.collate, "_fn") else None
    if collate_fn is None:
        from ..data.precomputed_sstk_dataset import collate_precomputed

        collate_fn = collate_precomputed

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=train_cfg.dataloader_num_workers,
        drop_last=True,
    )

    num_warmup = train_cfg.lr_warmup_steps * accelerator.num_processes
    len_dl = math.ceil(len(train_dataloader) / accelerator.num_processes)
    num_updates_per_epoch = math.ceil(len_dl / train_cfg.gradient_accumulation_steps)
    max_steps_cfg = getattr(train_cfg, "max_steps", None)
    # When max_steps not set: default to 1 epoch
    num_training_steps = (
        num_updates_per_epoch if max_steps_cfg is None else max_steps_cfg
    )

    lr_scheduler = get_scheduler(
        train_cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup,
        num_training_steps=num_training_steps,
        num_cycles=train_cfg.lr_num_cycles,
        power=train_cfg.lr_power,
    )

    if loss_module is not None:
        transformer, loss_module, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, loss_module, optimizer, train_dataloader, lr_scheduler
        )
        loss_fn = loss_module
    else:
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )

    trainable_params = _collect_trainable_params(
        transformer if train_base_model else None,
        loss_fn if isinstance(loss_fn, nn.Module) else None,
    )

    len_dl_per_process = len(train_dataloader)
    max_steps = (
        num_updates_per_epoch if max_steps_cfg is None else max_steps_cfg
    )
    num_epochs = math.ceil(max_steps / num_updates_per_epoch)

    # --- Training loop math (printed once, main process only) ---
    total_bs = (
        train_cfg.batch_size
        * accelerator.num_processes
        * train_cfg.gradient_accumulation_steps
    )
    if accelerator.is_main_process:
        logger.info("***** Training loop *****")
        logger.info(
            "  Dataset: len=%s (per process) | batch_size=%s | accumulation=%s | num_proc=%s",
            len_dl_per_process,
            train_cfg.batch_size,
            train_cfg.gradient_accumulation_steps,
            accelerator.num_processes,
        )
        logger.info(
            "  Steps: per_epoch=%s | max_steps=%s | num_epochs=%s | effective_batch_size=%s",
            num_updates_per_epoch,
            max_steps,
            num_epochs,
            total_bs,
        )
        if max_steps_cfg is not None:
            logger.info("  (max_steps=%s set)", max_steps_cfg)

    if accelerator.is_main_process:
        init_trackers(accelerator, cfg)
        log_dataset_input(
            class_name=getattr(cfg.dataset, "class_name", None),
            kwargs=ns_to_kwargs(cfg.dataset.kwargs) if hasattr(cfg.dataset, "kwargs") and cfg.dataset.kwargs else None,
            name=getattr(cfg.dataset, "name", None),
            source_path=ds_kwargs.get("local"),
            context="training",
        )

    if accelerator.is_main_process:
        logger.info("***** Running training *****")
        logger.info("  Data = %s", ds_kwargs.get("local"))
        if not train_base_model:
            logger.info(
                "  Using loss-managed LoRA student blocks; base transformer remains frozen and is excluded from checkpoints."
            )

    global_step = 0
    first_epoch = 0

    resume = getattr(cfg, "resume_from_checkpoint", None)
    if resume:
        resume_path = None
        if resume == "latest":
            dirs = [d for d in os.listdir(cfg.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            if dirs:
                resume_path = Path(cfg.output_dir) / dirs[-1]
        else:
            candidate = Path(resume)
            if candidate.exists():
                resume_path = candidate
            else:
                candidate = Path(cfg.output_dir) / os.path.basename(str(resume))
                if candidate.exists():
                    resume_path = candidate
        if resume_path is not None:
            accelerator.print(f"Resuming from {resume_path}")
            accelerator.load_state(str(resume_path))
            global_step = int(resume_path.name.split("-")[1])
            first_epoch = global_step // num_updates_per_epoch
        else:
            resume = None

    progress_bar = tqdm(
        range(max_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for _epoch in range(first_epoch, num_epochs):
        if train_base_model:
            transformer.train()
        else:
            transformer.eval()
        if loss_module is not None:
            loss_fn.train()
        for _step, batch in enumerate(train_dataloader):
            batch = {
                "latents": batch["latents"].to(accelerator.device, dtype=weight_dtype),
                "text_embeds": batch["text_embeds"].to(accelerator.device, dtype=weight_dtype),
                "text_ids": batch["text_ids"].to(accelerator.device),
            }

            accumulate_modules = [transformer] if train_base_model or loss_module is None else []
            if loss_module is not None:
                accumulate_modules.append(loss_fn)

            with accelerator.accumulate(accumulate_modules):
                loss, loss_breakdown = training_step_precomputed(
                    batch=batch,
                    transformer=transformer,
                    latents_bn_mean=latents_bn_mean,
                    latents_bn_std=latents_bn_std,
                    noise_scheduler_copy=noise_scheduler_copy,
                    weighting_scheme=loss_cfg.weighting_scheme,
                    logit_mean=loss_cfg.logit_mean,
                    logit_std=loss_cfg.logit_std,
                    guidance_scale=train_cfg.guidance_scale,
                    accelerator=accelerator,
                    loss_fn=loss_fn,
                    step=global_step,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                for k, v in loss_breakdown.items():
                    if k != "loss":
                        logs[k] = v
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                # Periodic validation: generate images and log to trackers
                val_cfg = getattr(cfg, "validation", None)
                if (
                    accelerator.is_main_process
                    and _get_obj_attr(loss_fn, "supports_validation", True)
                    and val_cfg
                    and getattr(val_cfg, "prompt", None)
                    and global_step % getattr(val_cfg, "steps", 500) == 0
                ):
                    run_validation(
                        pipeline_cls=pipeline_cfg._class,
                        transformer=unwrap_model(accelerator, transformer),
                        validation_prompt=val_cfg.prompt,
                        accelerator=accelerator,
                        step=global_step,
                        output_dir=cfg.output_dir,
                        num_images=getattr(val_cfg, "num_images", 4),
                        seed=getattr(val_cfg, "seed", 42),
                        resolution=getattr(val_cfg, "resolution", 512),
                        weight_dtype=weight_dtype,
                        pretrained_path=pretrained_path,
                        inference_steps=getattr(val_cfg, "inference_steps", 4),
                        guidance_scale=getattr(val_cfg, "guidance_scale", 1.0),
                    )

                if (
                    accelerator.is_main_process or is_fsdp
                ) and global_step % cfg.checkpointing_steps == 0:
                    limit = getattr(cfg, "checkpoints_total_limit", None)
                    if limit is not None:
                        prune_old_checkpoints(cfg.output_dir, limit)
                    save_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

            if global_step >= max_steps:
                break

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unwrapped_loss = unwrap(loss_fn) if isinstance(loss_fn, nn.Module) else loss_fn
        if _get_obj_attr(unwrapped_loss, "save_transformer", True):
            save_final_klein(
                output_dir=Path(cfg.output_dir),
                transformer=transformer,
                train_mode=checkpoint_train_mode,
                pipeline_cls=pipeline_cls,
                unwrap_fn=unwrap,
                logger=logger,
            )
        if hasattr(unwrapped_loss, "save_training_artifacts"):
            artifact_paths = unwrapped_loss.save_training_artifacts(Path(cfg.output_dir))
            if artifact_paths:
                logger.info("Saved custom training artifacts to %s", ", ".join(str(p) for p in artifact_paths))

    accelerator.end_training()


if __name__ == "__main__":
    main()
