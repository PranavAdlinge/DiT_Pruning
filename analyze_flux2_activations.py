import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from transformer_flux2 import Flux2Transformer2DModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure per-layer mean absolute activation deltas |output - input| across timesteps "
            "for hidden_states and encoder_hidden_states in Flux2Transformer2DModel."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("flux2_klein_transformer_config.json"),
        help="Path to Flux2 transformer config JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("activation_plots"),
        help="Directory where plots and CSV will be saved.",
    )
    parser.add_argument(
        "--timesteps",
        type=str,
        default=None,
        help=(
            "Comma-separated raw scheduler timesteps, e.g. '1000,900,800,700'. "
            "If omitted, values are generated from --timestep-start/end/num-timesteps."
        ),
    )
    parser.add_argument("--timestep-start", type=int, default=1000, help="Start raw timestep (inclusive).")
    parser.add_argument("--timestep-end", type=int, default=1, help="End raw timestep (inclusive).")
    parser.add_argument("--num-timesteps", type=int, default=20, help="Number of generated timesteps.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for synthetic input.")
    parser.add_argument("--image-seq-len", type=int, default=32, help="Image token sequence length.")
    parser.add_argument("--text-seq-len", type=int, default=32, help="Text token sequence length.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (e.g. 'cpu', 'cuda'). Defaults to cuda if available else cpu.",
    )
    return parser.parse_args()


def load_model_from_config(config_path: Path, device: torch.device) -> Flux2Transformer2DModel:
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    model_kwargs = {k: v for k, v in config.items() if not k.startswith("_")}
    model = Flux2Transformer2DModel(**model_kwargs).to(device)
    model.eval()
    return model


def parse_raw_timesteps(args: argparse.Namespace) -> list[int]:
    if args.timesteps:
        values = [int(v.strip()) for v in args.timesteps.split(",") if v.strip()]
        if not values:
            raise ValueError("Parsed empty timestep list from --timesteps.")
        return values

    if args.num_timesteps < 1:
        raise ValueError("--num-timesteps must be >= 1.")
    if args.num_timesteps == 1:
        return [int(args.timestep_start)]

    grid = torch.linspace(args.timestep_start, args.timestep_end, steps=args.num_timesteps)
    return [int(round(v.item())) for v in grid]


def make_position_ids(seq_len: int, num_axes: int, device: torch.device) -> torch.Tensor:
    base = torch.arange(seq_len, device=device, dtype=torch.long)
    return base.unsqueeze(-1).repeat(1, num_axes)


def register_delta_hooks(
    model: Flux2Transformer2DModel,
    text_seq_len: int,
) -> tuple[OrderedDict[str, dict[str, list[float]]], list[torch.utils.hooks.RemovableHandle]]:
    layer_metrics: OrderedDict[str, dict[str, list[float]]] = OrderedDict()
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def get_layer_store(layer_name: str) -> dict[str, list[float]]:
        if layer_name not in layer_metrics:
            layer_metrics[layer_name] = {"hidden_states": [], "encoder_hidden_states": []}
        return layer_metrics[layer_name]

    def make_double_hook(layer_name: str):
        def hook(_module, inputs, output):
            hidden_in = inputs[0]
            encoder_in = inputs[1]
            encoder_out, hidden_out = output

            store = get_layer_store(layer_name)
            store["hidden_states"].append((hidden_out - hidden_in).abs().mean().item())
            store["encoder_hidden_states"].append((encoder_out - encoder_in).abs().mean().item())

        return hook

    def make_single_hook(layer_name: str):
        def hook(_module, inputs, output):
            concat_in = inputs[0]
            concat_out = output

            if isinstance(concat_out, tuple):
                encoder_out, hidden_out = concat_out
                encoder_in = inputs[1]
                hidden_in = inputs[0]
            else:
                encoder_in = concat_in[:, :text_seq_len, ...]
                hidden_in = concat_in[:, text_seq_len:, ...]
                encoder_out = concat_out[:, :text_seq_len, ...]
                hidden_out = concat_out[:, text_seq_len:, ...]

            store = get_layer_store(layer_name)
            store["hidden_states"].append((hidden_out - hidden_in).abs().mean().item())
            store["encoder_hidden_states"].append((encoder_out - encoder_in).abs().mean().item())

        return hook

    for i, block in enumerate(model.transformer_blocks):
        layer_name = f"double_{i:02d}"
        handles.append(block.register_forward_hook(make_double_hook(layer_name)))

    for i, block in enumerate(model.single_transformer_blocks):
        layer_name = f"single_{i:02d}"
        handles.append(block.register_forward_hook(make_single_hook(layer_name)))

    return layer_metrics, handles


def save_csv(
    output_dir: Path,
    raw_timesteps: list[int],
    layer_metrics: OrderedDict[str, dict[str, list[float]]],
) -> None:
    csv_path = output_dir / "layer_activation_deltas.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "raw_timestep", "hidden_states_abs_delta_mean", "encoder_hidden_states_abs_delta_mean"])
        for layer_name, streams in layer_metrics.items():
            for t, hidden_delta, encoder_delta in zip(
                raw_timesteps, streams["hidden_states"], streams["encoder_hidden_states"]
            ):
                writer.writerow([layer_name, t, hidden_delta, encoder_delta])


def save_plots(
    output_dir: Path,
    raw_timesteps: list[int],
    layer_metrics: OrderedDict[str, dict[str, list[float]]],
) -> None:
    for layer_name, streams in layer_metrics.items():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(raw_timesteps, streams["hidden_states"], marker="o", label="hidden_states")
        ax.plot(raw_timesteps, streams["encoder_hidden_states"], marker="o", label="encoder_hidden_states")
        ax.set_title(f"{layer_name}: mean(|output - input|)")
        ax.set_xlabel("raw timestep")
        ax.set_ylabel("mean absolute delta")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{layer_name}_abs_delta.png", dpi=180)
        plt.close(fig)

    fig_h, ax_h = plt.subplots(figsize=(10, 5))
    for layer_name, streams in layer_metrics.items():
        ax_h.plot(raw_timesteps, streams["hidden_states"], linewidth=1.0, label=layer_name)
    ax_h.set_title("hidden_states mean(|output - input|) by layer")
    ax_h.set_xlabel("raw timestep")
    ax_h.set_ylabel("mean absolute delta")
    ax_h.grid(alpha=0.3)
    ax_h.legend(ncol=3, fontsize=8)
    fig_h.tight_layout()
    fig_h.savefig(output_dir / "all_layers_hidden_states.png", dpi=180)
    plt.close(fig_h)

    fig_e, ax_e = plt.subplots(figsize=(10, 5))
    for layer_name, streams in layer_metrics.items():
        ax_e.plot(raw_timesteps, streams["encoder_hidden_states"], linewidth=1.0, label=layer_name)
    ax_e.set_title("encoder_hidden_states mean(|output - input|) by layer")
    ax_e.set_xlabel("raw timestep")
    ax_e.set_ylabel("mean absolute delta")
    ax_e.grid(alpha=0.3)
    ax_e.legend(ncol=3, fontsize=8)
    fig_e.tight_layout()
    fig_e.savefig(output_dir / "all_layers_encoder_hidden_states.png", dpi=180)
    plt.close(fig_e)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model = load_model_from_config(args.config, device)
    config = model.config

    raw_timesteps = parse_raw_timesteps(args)
    normalized_timesteps = [t / 1000.0 for t in raw_timesteps]

    hidden_states = torch.randn(
        args.batch_size,
        args.image_seq_len,
        config.in_channels,
        device=device,
        dtype=torch.float32,
    )
    encoder_hidden_states = torch.randn(
        args.batch_size,
        args.text_seq_len,
        config.joint_attention_dim,
        device=device,
        dtype=torch.float32,
    )
    img_ids = make_position_ids(args.image_seq_len, len(config.axes_dims_rope), device)
    txt_ids = make_position_ids(args.text_seq_len, len(config.axes_dims_rope), device)

    layer_metrics, handles = register_delta_hooks(model, text_seq_len=args.text_seq_len)
    try:
        with torch.no_grad():
            for normalized_t in normalized_timesteps:
                timestep_tensor = torch.full(
                    (args.batch_size,),
                    normalized_t,
                    device=device,
                    dtype=torch.float32,
                )
                _ = model(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep_tensor,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=None,
                    return_dict=True,
                )
    finally:
        for handle in handles:
            handle.remove()

    save_csv(output_dir, raw_timesteps, layer_metrics)
    save_plots(output_dir, raw_timesteps, layer_metrics)

    print(f"Saved activation analysis to: {output_dir.resolve()}")
    print(f"CSV: {(output_dir / 'layer_activation_deltas.csv').resolve()}")
    print("Generated per-layer and all-layer plots for hidden_states and encoder_hidden_states.")


if __name__ == "__main__":
    main()
