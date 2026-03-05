import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

try:
    from diffusers import Flux2KleinPipeline
except ImportError:
    # Fallback for diffusers versions exposing this pipeline under Flux2Pipeline.
    from diffusers import Flux2Pipeline as Flux2KleinPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run FLUX.2-klein pipeline over a prompt set, hook transformer layers, and measure "
            "mean(abs(output-input)) per layer per timestep for hidden_states and encoder_hidden_states."
        )
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="black-forest-labs/FLUX.2-klein-4B",
        help="Hugging Face model ID for Flux2KleinPipeline.",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=Path("flux2_klein_prompts.json"),
        help="JSON file containing a list of prompts or {'prompts': [...]}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("activation_plots"),
        help="Directory where activation CSV/plots (and optional images) are saved.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=500,
        help="Maximum number of prompts to process from the prompts file.",
    )
    parser.add_argument("--height", type=int, default=1024, help="Generated image height.")
    parser.add_argument("--width", type=int, default=1024, help="Generated image width.")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="Guidance scale for the pipeline.")
    parser.add_argument("--num-inference-steps", type=int, default=4, help="Number of denoising steps.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype for pipeline loading.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for generation (default: cuda if available else cpu).",
    )
    parser.add_argument(
        "--no-cpu-offload",
        action="store_true",
        help="Disable enable_model_cpu_offload(). By default this is enabled on CUDA.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed. Each prompt uses seed + prompt_index.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save generated images to output_dir/generated_images.",
    )
    parser.add_argument(
        "--save-images-limit",
        type=int,
        default=500,
        help="Maximum number of images to save when --save-images is set.",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def load_prompts(path: Path, max_prompts: int | None) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        prompts = data
    elif isinstance(data, dict) and isinstance(data.get("prompts"), list):
        prompts = data["prompts"]
    else:
        raise ValueError(f"Prompt JSON must be a list or {{'prompts': [...]}}: {path}")

    cleaned = [str(p).strip() for p in prompts if str(p).strip()]
    if max_prompts is not None:
        cleaned = cleaned[:max_prompts]
    if not cleaned:
        raise ValueError(f"No prompts found in {path}.")
    return cleaned


def to_float_timestep(t: Any) -> float | None:
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        if t.numel() == 0:
            return None
        return float(t.detach().reshape(-1)[0].to(torch.float32).cpu().item())
    return float(t)


def layer_sort_key(layer_name: str) -> tuple[int, int]:
    prefix, idx_str = layer_name.split("_")
    group = 0 if prefix == "double" else 1
    return (group, int(idx_str))


def register_activation_hooks(pipe, metrics):
    handles = []
    run_context = {"text_seq_len": None}

    def get_current_raw_timestep() -> float | None:
        t = getattr(pipe, "current_timestep", None)
        if t is None:
            t = getattr(pipe, "_current_timestep", None)
        value = to_float_timestep(t)
        if value is None:
            return None
        return round(value, 6)

    def record(layer_name: str, stream_name: str, timestep_raw: float, value: float) -> None:
        metrics[layer_name][stream_name][timestep_raw].append(float(value))

    def make_double_hook(layer_name: str):
        def hook(_module, inputs, output):
            timestep_raw = get_current_raw_timestep()
            if timestep_raw is None:
                return

            hidden_in = inputs[0]
            encoder_in = inputs[1]
            encoder_out, hidden_out = output

            run_context["text_seq_len"] = int(encoder_in.shape[1])

            hidden_delta = (hidden_out - hidden_in).abs().mean().item()
            encoder_delta = (encoder_out - encoder_in).abs().mean().item()

            record(layer_name, "hidden_states", timestep_raw, hidden_delta)
            record(layer_name, "encoder_hidden_states", timestep_raw, encoder_delta)

        return hook

    def make_single_hook(layer_name: str):
        def hook(_module, inputs, output):
            timestep_raw = get_current_raw_timestep()
            if timestep_raw is None:
                return

            concat_in = inputs[0]
            concat_out = output

            text_len = run_context["text_seq_len"]
            if text_len is None:
                raise RuntimeError(
                    "text_seq_len not set before single-stream hook. "
                    "Expected double-stream hooks to run first in this forward pass."
                )

            if isinstance(concat_out, tuple):
                encoder_out, hidden_out = concat_out
                encoder_in = concat_in[:, :text_len, ...]
                hidden_in = concat_in[:, text_len:, ...]
            else:
                encoder_in = concat_in[:, :text_len, ...]
                hidden_in = concat_in[:, text_len:, ...]
                encoder_out = concat_out[:, :text_len, ...]
                hidden_out = concat_out[:, text_len:, ...]

            hidden_delta = (hidden_out - hidden_in).abs().mean().item()
            encoder_delta = (encoder_out - encoder_in).abs().mean().item()

            record(layer_name, "hidden_states", timestep_raw, hidden_delta)
            record(layer_name, "encoder_hidden_states", timestep_raw, encoder_delta)

        return hook

    for i, block in enumerate(pipe.transformer.transformer_blocks):
        layer_name = f"double_{i:02d}"
        handles.append(block.register_forward_hook(make_double_hook(layer_name)))

    for i, block in enumerate(pipe.transformer.single_transformer_blocks):
        layer_name = f"single_{i:02d}"
        handles.append(block.register_forward_hook(make_single_hook(layer_name)))

    return handles


def mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def build_timestep_order(
    timestep_order_from_runs: list[float] | None,
    metrics,
) -> list[float]:
    if timestep_order_from_runs:
        return timestep_order_from_runs

    all_timesteps = set()
    for layer_data in metrics.values():
        all_timesteps.update(layer_data["hidden_states"].keys())
        all_timesteps.update(layer_data["encoder_hidden_states"].keys())
    return sorted(all_timesteps, reverse=True)


def save_csv(output_dir: Path, timestep_order: list[float], metrics) -> None:
    path = output_dir / "layer_activation_deltas_pipeline.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "layer",
                "raw_timestep",
                "hidden_states_abs_delta_mean",
                "encoder_hidden_states_abs_delta_mean",
                "num_hidden_samples",
                "num_encoder_samples",
            ]
        )

        for layer_name in sorted(metrics.keys(), key=layer_sort_key):
            for timestep_raw in timestep_order:
                hidden_vals = metrics[layer_name]["hidden_states"].get(timestep_raw, [])
                encoder_vals = metrics[layer_name]["encoder_hidden_states"].get(timestep_raw, [])
                writer.writerow(
                    [
                        layer_name,
                        timestep_raw,
                        mean_or_nan(hidden_vals),
                        mean_or_nan(encoder_vals),
                        len(hidden_vals),
                        len(encoder_vals),
                    ]
                )


def save_plots(output_dir: Path, timestep_order: list[float], metrics) -> None:
    for layer_name in sorted(metrics.keys(), key=layer_sort_key):
        hidden_values = [mean_or_nan(metrics[layer_name]["hidden_states"].get(t, [])) for t in timestep_order]
        encoder_values = [mean_or_nan(metrics[layer_name]["encoder_hidden_states"].get(t, [])) for t in timestep_order]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(timestep_order, hidden_values, marker="o", label="hidden_states")
        ax.plot(timestep_order, encoder_values, marker="o", label="encoder_hidden_states")
        ax.set_title(f"{layer_name}: mean(|output - input|) across prompts")
        ax.set_xlabel("raw timestep")
        ax.set_ylabel("mean absolute delta")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{layer_name}_abs_delta_pipeline.png", dpi=180)
        plt.close(fig)

    fig_h, ax_h = plt.subplots(figsize=(11, 6))
    for layer_name in sorted(metrics.keys(), key=layer_sort_key):
        hidden_values = [mean_or_nan(metrics[layer_name]["hidden_states"].get(t, [])) for t in timestep_order]
        ax_h.plot(timestep_order, hidden_values, linewidth=1.0, label=layer_name)
    ax_h.set_title("hidden_states mean(|output - input|) by layer across prompts")
    ax_h.set_xlabel("raw timestep")
    ax_h.set_ylabel("mean absolute delta")
    ax_h.grid(alpha=0.3)
    ax_h.legend(ncol=3, fontsize=8)
    fig_h.tight_layout()
    fig_h.savefig(output_dir / "all_layers_hidden_states_pipeline.png", dpi=180)
    plt.close(fig_h)

    fig_e, ax_e = plt.subplots(figsize=(11, 6))
    for layer_name in sorted(metrics.keys(), key=layer_sort_key):
        encoder_values = [mean_or_nan(metrics[layer_name]["encoder_hidden_states"].get(t, [])) for t in timestep_order]
        ax_e.plot(timestep_order, encoder_values, linewidth=1.0, label=layer_name)
    ax_e.set_title("encoder_hidden_states mean(|output - input|) by layer across prompts")
    ax_e.set_xlabel("raw timestep")
    ax_e.set_ylabel("mean absolute delta")
    ax_e.grid(alpha=0.3)
    ax_e.legend(ncol=3, fontsize=8)
    fig_e.tight_layout()
    fig_e.savefig(output_dir / "all_layers_encoder_hidden_states_pipeline.png", dpi=180)
    plt.close(fig_e)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompts_file, args.max_prompts)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype)

    pipe = Flux2KleinPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    use_cpu_offload = (device.startswith("cuda") and not args.no_cpu_offload)
    if use_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    metrics = defaultdict(
        lambda: {
            "hidden_states": defaultdict(list),
            "encoder_hidden_states": defaultdict(list),
        }
    )
    handles = register_activation_hooks(pipe, metrics)

    images_dir = output_dir / "generated_images"
    if args.save_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    timestep_order = None

    try:
        with torch.no_grad():
            for prompt_idx, prompt in enumerate(prompts):
                step_timesteps = []

                def callback_on_step_end(_pipe, _step, timestep, callback_kwargs):
                    t_raw = to_float_timestep(timestep)
                    if t_raw is not None:
                        step_timesteps.append(round(t_raw, 6))
                    return callback_kwargs

                generator_device = "cuda" if device.startswith("cuda") else "cpu"
                generator = torch.Generator(device=generator_device).manual_seed(args.seed + prompt_idx)

                save_this_image = args.save_images and prompt_idx < args.save_images_limit
                output_type = "pil" if save_this_image else "latent"

                result = pipe(
                    prompt=prompt,
                    height=args.height,
                    width=args.width,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=["latents"],
                    output_type=output_type,
                )

                if save_this_image:
                    image_path = images_dir / f"prompt_{prompt_idx:03d}.png"
                    result.images[0].save(image_path)

                if timestep_order is None:
                    timestep_order = step_timesteps

                del result

    finally:
        for handle in handles:
            handle.remove()

    timestep_order = build_timestep_order(timestep_order, metrics)

    save_csv(output_dir, timestep_order, metrics)
    save_plots(output_dir, timestep_order, metrics)

    metadata = {
        "model_id": args.model_id,
        "prompts_file": str(args.prompts_file),
        "num_prompts_run": len(prompts),
        "height": args.height,
        "width": args.width,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "dtype": args.dtype,
        "device": device,
        "cpu_offload_enabled": use_cpu_offload,
        "timestep_order": timestep_order,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Prompts processed: {len(prompts)}")
    print(f"Saved outputs to: {output_dir.resolve()}")
    print(f"CSV: {(output_dir / 'layer_activation_deltas_pipeline.csv').resolve()}")


if __name__ == "__main__":
    main()
