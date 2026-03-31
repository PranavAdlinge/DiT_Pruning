import argparse
import os

import numpy as np
import torch

from lib.prune import check_size, check_sparsity, prune_OBS_Diff, prune_OBS_Diff_Structured
from lib.prune_flux2 import check_flux2_size, prune_OBS_Diff_Structured_Flux2


def build_timestep_weight(args):
    if args.timestep_weight_strategy == "linear_increase":
        return np.linspace(args.timestep_min_weight, args.timestep_max_weight, args.num_inference_steps)
    if args.timestep_weight_strategy == "linear_decrease":
        return np.linspace(args.timestep_max_weight, args.timestep_min_weight, args.num_inference_steps)
    if args.timestep_weight_strategy == "uniform":
        return np.ones(args.num_inference_steps)
    if args.timestep_weight_strategy == "log_increase":
        linear_space = np.arange(0, args.num_inference_steps)
        return args.timestep_min_weight + (
            (args.timestep_max_weight - args.timestep_min_weight) / np.log(args.num_inference_steps) * np.log(1 + linear_space)
        )

    linear_space = np.arange(0, args.num_inference_steps)
    timestep_weight = args.timestep_min_weight + (
        (args.timestep_max_weight - args.timestep_min_weight) / np.log(args.num_inference_steps) * np.log(1 + linear_space)
    )
    return timestep_weight[::-1]


def resolve_range(min_layer, max_layer, total_layers, default_min=0, default_max=None):
    if default_max is None:
        default_max = total_layers

    start = default_min if min_layer is None else max(min_layer, 0)
    end = default_max if max_layer is None else min(max_layer, total_layers)
    if end < start:
        raise ValueError(f"Invalid layer range: [{start}, {end})")
    return start, end


def get_diffusers_classes(model_family):
    if model_family == "sd3":
        try:
            from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
        except ImportError as exc:
            raise ImportError("Stable Diffusion 3 support requires diffusers with SD3 pipeline classes installed.") from exc
        return StableDiffusion3Pipeline, SD3Transformer2DModel

    try:
        from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
    except ImportError as exc:
        raise ImportError(
            "Flux2 support requires a newer diffusers build that includes Flux2KleinPipeline and Flux2Transformer2DModel."
        ) from exc
    return Flux2KleinPipeline, Flux2Transformer2DModel


def load_transformer_checkpoint(transformer_cls, transformer_path, transformer_subfolder, torch_dtype, model_family):
    if os.path.isfile(transformer_path) and transformer_path.endswith((".pt", ".pth")):
        if model_family == "flux2":
            raise ValueError(
                "For Flux2, --transformer_path must point to a diffusers transformer checkpoint that can be loaded with "
                "Flux2Transformer2DModel.from_pretrained(...), not a .pt or .pth file."
            )
        transformer = torch.load(transformer_path, map_location="cpu")
        if isinstance(transformer, dict):
            raise ValueError(
                "State-dict checkpoints are not loaded automatically. Pass a torch-saved transformer module or a diffusers transformer directory."
            )
        return transformer.to(dtype=torch_dtype)

    from_pretrained_kwargs = {"torch_dtype": torch_dtype}
    normalized_path = os.path.normpath(transformer_path)
    is_transformer_dir = os.path.isdir(normalized_path) and os.path.isfile(os.path.join(normalized_path, "config.json"))

    if not is_transformer_dir:
        from_pretrained_kwargs["subfolder"] = transformer_subfolder

    return transformer_cls.from_pretrained(transformer_path, **from_pretrained_kwargs)


def load_pipeline(args, torch_dtype):
    pipeline_cls, transformer_cls = get_diffusers_classes(args.model_family)
    pipeline_source = args.pipeline_path or args.model_path

    if args.transformer_path:
        if pipeline_source is None:
            raise ValueError("--pipeline_path is required when --transformer_path is provided.")

        transformer = load_transformer_checkpoint(
            transformer_cls=transformer_cls,
            transformer_path=args.transformer_path,
            transformer_subfolder=args.transformer_subfolder,
            torch_dtype=torch_dtype,
            model_family=args.model_family,
        )

        pipe = pipeline_cls.from_pretrained(
            pipeline_source,
            transformer=transformer,
            torch_dtype=torch_dtype,
        )
    else:
        if pipeline_source is None:
            raise ValueError("Provide either --transformer_path or --pipeline_path (or legacy --model_path).")

        pipe = pipeline_cls.from_pretrained(
            pipeline_source,
            torch_dtype=torch_dtype,
        )

    return pipe.to("cuda")


def resolve_sd3_layers(args, transformer):
    if args.minlayer is not None and args.maxlayer is not None:
        args.minlayer = max(args.minlayer, 0)
        args.maxlayer = min(args.maxlayer, transformer.config.num_layers)
    elif args.minlayer is not None:
        args.minlayer = max(args.minlayer, 0)
        args.maxlayer = transformer.config.num_layers
    elif args.maxlayer is not None:
        args.maxlayer = min(args.maxlayer, transformer.config.num_layers)
        args.minlayer = 0
    else:
        args.minlayer = 0
        args.maxlayer = transformer.config.num_layers

    if args.sparsity_type == "structured" and args.maxlayer == transformer.config.num_layers:
        args.maxlayer = transformer.config.num_layers - 1

    print(f"pruning from layer {args.minlayer} to {args.maxlayer}")


def resolve_flux2_layers(args, transformer):
    num_double_layers = transformer.config.num_layers
    num_single_layers = transformer.config.num_single_layers

    default_single_max = num_single_layers - 1 if num_single_layers > 0 else 0
    args.double_minlayer, args.double_maxlayer = resolve_range(
        args.double_minlayer,
        args.double_maxlayer,
        num_double_layers,
        default_min=1,
        default_max=num_double_layers,
    )
    args.single_minlayer, args.single_maxlayer = resolve_range(
        args.single_minlayer,
        args.single_maxlayer,
        num_single_layers,
        default_min=0,
        default_max=default_single_max,
    )

    print(f"pruning double blocks in [{args.double_minlayer}, {args.double_maxlayer}) out of {num_double_layers}")
    print(f"pruning single blocks in [{args.single_minlayer}, {args.single_maxlayer}) out of {num_single_layers}")


def resolve_calibration_source(args):
    if args.dataset is None:
        args.dataset = "prompt_file" if args.model_family == "flux2" else "gcc3m"

    if args.model_family == "flux2" and args.dataset == "prompt_file":
        print(f"using Flux2 prompt file calibration: {args.prompt_file}")
    else:
        print(f"using calibration dataset: {args.dataset}")


def save_transformer_checkpoint(transformer, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "pruned_transformer.pt")
    torch.save(transformer, save_path)
    print(f"saved transformer-only checkpoint to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_family", type=str, default="sd3", choices=["sd3", "flux2"], help="Model family to prune.")
    parser.add_argument("--model_path", type=str, default=None, help="Legacy alias for --pipeline_path.")
    parser.add_argument(
        "--pipeline_path",
        type=str,
        default=None,
        help="Path or repo id for the whole diffusion pipeline. Flux2 uses this as the base pipeline checkpoint.",
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help="Optional alternate transformer checkpoint. For Flux2 this should be a diffusers transformer checkpoint loaded via from_pretrained and injected into the base pipeline.",
    )
    parser.add_argument(
        "--transformer_subfolder",
        type=str,
        default="transformer",
        help="Subfolder to load from when --transformer_path points at a diffusers repo or directory root.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampling the calibration data.")
    parser.add_argument("--sparsity_ratio", type=float, default=0, help="Sparsity level")
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4", "structured"])
    parser.add_argument(
        "--prune_method",
        type=str,
        choices=["magnitude", "wanda", "OBS-Diff", "OBS-Diff-Structured", "dsnot", "magnitude_structured"],
    )
    parser.add_argument("--save_model", type=str, default=None, help="Path to save the pruned transformer-only checkpoint.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["gcc3m", "prompt_file"],
        help="Calibration source. Defaults to gcc3m for SD3 and prompt_file for Flux2.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="./data/flux2_prompts_100.json",
        help="Local JSON prompt file used for prompt_file calibration. Flux2 uses this by default.",
    )
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to use for calibration.")
    parser.add_argument("--minlayer", type=int, default=None, help="Minimum SD3 layer to prune.")
    parser.add_argument("--maxlayer", type=int, default=None, help="Maximum SD3 layer to prune.")
    parser.add_argument("--double_minlayer", type=int, default=None, help="Minimum Flux2 double-stream block to prune.")
    parser.add_argument("--double_maxlayer", type=int, default=None, help="Exclusive upper bound for Flux2 double-stream pruning.")
    parser.add_argument("--single_minlayer", type=int, default=None, help="Minimum Flux2 single-stream block to prune.")
    parser.add_argument("--single_maxlayer", type=int, default=None, help="Exclusive upper bound for Flux2 single-stream pruning.")
    parser.add_argument("--demo_evaluate", action="store_true", help="A single image evaluation by the pruned model")
    parser.add_argument("--demo_dir", type=str, default="eval_output.png", help="Path to save the demo images.")
    parser.add_argument("--num_pruned_groups", type=int, default=4, help="Number of pruned groups.")
    parser.add_argument(
        "--timestep_weight_strategy",
        type=str,
        default="uniform",
        choices=["uniform", "linear_increase", "linear_decrease", "log_increase", "log_decrease"],
        help="Timestep weight strategy for Hessian update",
    )
    parser.add_argument("--timestep_min_weight", type=float, default=0.8, help="Min weight for timestep-aware weighting")
    parser.add_argument("--timestep_max_weight", type=float, default=1.2, help="Max weight for timestep-aware weighting")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--height", type=int, default=512, help="Height of the image")
    parser.add_argument("--width", type=int, default=512, help="Width of the image")
    parser.add_argument("--guidance_scale", type=float, default=7.0, help="Guidance scale")
    parser.add_argument("--no_compensate", action="store_true", help="Skip error compensation in OBS-Diff")
    parser.add_argument("--percdamp", type=float, default=0.01, help="Hessian dampening factor")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured" and args.sparsity_type != "structured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    if args.model_family == "flux2" and (args.prune_method != "OBS-Diff-Structured" or args.sparsity_type != "structured"):
        raise NotImplementedError("Flux2 support is currently implemented for structured OBS-Diff pruning only.")
    if args.model_family == "flux2" and (args.pipeline_path is None and args.model_path is None):
        raise ValueError("For Flux2, --pipeline_path is required.")

    device = torch.device("cuda:0")
    torch_dtype = torch.float16

    print(f"loading {args.model_family} model")
    if args.transformer_path:
        print(f"  transformer source: {args.transformer_path}")
    if args.pipeline_path or args.model_path:
        print(f"  pipeline source: {args.pipeline_path or args.model_path}")

    pipe = load_pipeline(args, torch_dtype=torch_dtype)
    pipe.transformer.eval()

    if args.model_family == "sd3":
        resolve_sd3_layers(args, pipe.transformer)
    else:
        resolve_flux2_layers(args, pipe.transformer)

    resolve_calibration_source(args)
    print(f"use device {device}")

    if args.model_family == "sd3":
        target_modules = [
            "ff.net.2",
            "ff_context.net.2",
            "ff_context.net.0.proj",
            "ff.net.0.proj",
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
        ]

        if args.sparsity_type == "structured":
            target_modules = [
                "ff.net.2",
                "ff_context.net.2",
                "attn.to_out.0",
            ]
    else:
        target_modules = []

    if args.sparsity_ratio != 0:
        print("pruning starts")
        timestep_weight = build_timestep_weight(args)
        print(f"timestep_weight: {timestep_weight}")

        if args.model_family == "sd3":
            if args.prune_method == "OBS-Diff":
                prune_OBS_Diff(
                    args,
                    pipe,
                    target_modules,
                    device,
                    prune_n=prune_n,
                    prune_m=prune_m,
                    timestep_weight=timestep_weight,
                )
            elif args.prune_method == "OBS-Diff-Structured":
                prune_OBS_Diff_Structured(args, pipe, target_modules, device, timestep_weight=timestep_weight)
        else:
            prune_OBS_Diff_Structured_Flux2(args, pipe, device, timestep_weight=timestep_weight)

    if args.model_family == "sd3":
        if args.sparsity_type != "structured":
            sparsity_ratio = check_sparsity(pipe.transformer, target_modules)
            print(f"sparsity sanity check {sparsity_ratio:.4f}")
        else:
            check_size(pipe.transformer, target_modules)
    else:
        check_flux2_size(pipe.transformer, args)

    if args.demo_evaluate:
        image = pipe(
            prompt="A cat holding a sign that says hello world",
            height=1024,
            width=1024,
            num_inference_steps=25,
            guidance_scale=7.0,
            generator=torch.Generator("cuda").manual_seed(0),
        ).images[0]
        os.makedirs("./eval_output", exist_ok=True)
        image.save(f"./eval_output/{args.demo_dir}")
        print(f"save image to ./eval_output/{args.demo_dir}")

    if args.save_model:
        save_transformer_checkpoint(pipe.transformer, args.save_model)


if __name__ == "__main__":
    main()
