import argparse
import os

import torch


def resolve_transformer_checkpoint(path):
    normalized_path = os.path.normpath(path)
    if os.path.isdir(normalized_path):
        candidate = os.path.join(normalized_path, "pruned_transformer.pt")
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(
            f"Could not find pruned transformer checkpoint at {candidate}. "
            "Pass either the checkpoint file directly or a directory containing pruned_transformer.pt."
        )

    if os.path.isfile(normalized_path):
        return normalized_path

    raise FileNotFoundError(f"Transformer checkpoint not found: {path}")


def load_pruned_transformer(path):
    checkpoint_path = resolve_transformer_checkpoint(path)
    try:
        transformer = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        transformer = torch.load(checkpoint_path, map_location="cpu")
    return transformer, checkpoint_path


def count_parameters(module):
    return sum(param.numel() for param in module.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline_path",
        type=str,
        default="black-forest-labs/FLUX.2-klein-base-4B",
        help="Base Flux2 pipeline path or repo id.",
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        required=True,
        help="Path to a saved pruned transformer checkpoint file or a directory containing pruned_transformer.pt.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cinematic portrait of an astronaut standing in a sunflower field at golden hour, ultra detailed, natural light.",
        help="Sample prompt for inference.",
    )
    parser.add_argument("--negative_prompt", type=str, default=None, help="Optional negative prompt.")
    parser.add_argument("--height", type=int, default=1024, help="Output height.")
    parser.add_argument("--width", type=int, default=1024, help="Output width.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps.")
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="Guidance scale.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--output",
        type=str,
        default="./eval_output/flux2_pruned_sample.png",
        help="Output image path.",
    )
    args = parser.parse_args()

    try:
        from diffusers import Flux2KleinPipeline
    except ImportError as exc:
        raise ImportError("This script requires diffusers with Flux2KleinPipeline support installed.") from exc

    torch_dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"loading pruned transformer from {args.transformer_path}")
    transformer, checkpoint_path = load_pruned_transformer(args.transformer_path)
    transformer = transformer.to(dtype=torch_dtype)

    print(f"loading base pipeline from {args.pipeline_path}")
    pipe = Flux2KleinPipeline.from_pretrained(
        args.pipeline_path,
        transformer=transformer,
        torch_dtype=torch_dtype,
    ).to(device)
    pipe.transformer.eval()

    param_count = count_parameters(pipe.transformer)
    print(f"pruned transformer parameters: {param_count}")

    generator = torch.Generator(device).manual_seed(args.seed)
    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output.images[0].save(args.output)
    print(f"loaded checkpoint: {checkpoint_path}")
    print(f"saved image to {args.output}")


if __name__ == "__main__":
    main()
