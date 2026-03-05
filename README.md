# Flux2 Layer Activation Delta Analysis

This repository contains local Flux2 model files plus a utility script to measure how much each transformer layer changes activations across denoising timesteps.

The main analysis entrypoint is:
- `analyze_flux2_activations.py`

It uses:
- `transformer_flux2.py` (local `Flux2Transformer2DModel`)
- `flux2_klein_transformer_config.json` (model architecture config)

## What This Analysis Measures

For each timestep and for each transformer layer, the script computes:

- `mean(abs(output - input))` for `hidden_states` (image stream)
- `mean(abs(output - input))` for `encoder_hidden_states` (text stream)

This is done separately for:
- Double-stream blocks (`transformer_blocks`)
- Single-stream blocks (`single_transformer_blocks`) by splitting concatenated text/image tokens

So for every layer you get a curve over timesteps showing how strongly that layer updates each stream.

## Detailed Flow

1. Load model config from `flux2_klein_transformer_config.json`.
1. Build `Flux2Transformer2DModel` from the config.
1. Generate or parse raw timesteps (for example `1000 -> 1`).
1. Convert raw timesteps to normalized values expected by the model (`t / 1000`).
1. Create synthetic input tensors:
   - `hidden_states`: shape `[batch, image_seq_len, in_channels]`
   - `encoder_hidden_states`: shape `[batch, text_seq_len, joint_attention_dim]`
   - `img_ids` and `txt_ids` for RoPE positions
1. Register forward hooks on each layer:
   - `double_XX` for each double-stream block
   - `single_XX` for each single-stream block
1. For each timestep, run one model forward pass.
1. Inside hooks, compute:
   - `abs_delta_hidden = mean(abs(hidden_out - hidden_in))`
   - `abs_delta_encoder = mean(abs(encoder_out - encoder_in))`
1. Save results to:
   - CSV table
   - Per-layer line plots
   - Combined all-layer plots for each stream

## Outputs

By default, outputs are written to `activation_plots/`.

- `layer_activation_deltas.csv`
  - Columns:
    - `layer`
    - `raw_timestep`
    - `hidden_states_abs_delta_mean`
    - `encoder_hidden_states_abs_delta_mean`
- `double_XX_abs_delta.png` and `single_XX_abs_delta.png`
  - One figure per layer with two lines (`hidden_states`, `encoder_hidden_states`)
- `all_layers_hidden_states.png`
  - All layers on one figure for hidden stream
- `all_layers_encoder_hidden_states.png`
  - All layers on one figure for encoder stream

## Setup

Use a Python environment with at least:

- `torch`
- `matplotlib`
- `diffusers`
- `transformers`

Example install command:

```bash
pip install torch matplotlib diffusers transformers
```

## Run Instructions

### 1) Default run (20 timesteps from 1000 to 1)

```bash
python analyze_flux2_activations.py \
  --config flux2_klein_transformer_config.json \
  --output-dir activation_plots \
  --num-timesteps 20 \
  --timestep-start 1000 \
  --timestep-end 1
```

### 2) Explicit custom timesteps

```bash
python analyze_flux2_activations.py \
  --config flux2_klein_transformer_config.json \
  --timesteps "1000,900,800,700,600,500,400,300,200,100,1"
```

### 3) Control sequence lengths, batch, and device

```bash
python analyze_flux2_activations.py \
  --config flux2_klein_transformer_config.json \
  --image-seq-len 64 \
  --text-seq-len 64 \
  --batch-size 2 \
  --device cuda \
  --output-dir activation_plots_b2
```

## CLI Arguments

- `--config`: path to model config JSON
- `--output-dir`: directory for CSV and plots
- `--timesteps`: comma-separated raw timesteps; overrides generated schedule
- `--timestep-start`: start raw timestep for generated schedule
- `--timestep-end`: end raw timestep for generated schedule
- `--num-timesteps`: number of points in generated schedule
- `--batch-size`: synthetic batch size
- `--image-seq-len`: image token count
- `--text-seq-len`: text token count
- `--seed`: RNG seed
- `--device`: device override (`cpu` or `cuda`)

## Notes and Interpretation

- The script measures internal layer update magnitude, not final image quality.
- Inputs are synthetic random tensors by default; this is useful for structural/profiling comparisons.
- For data-dependent behavior, replace synthetic tensors with real latents/text embeddings from your pipeline and keep the same hook logic.
- Higher `mean(abs(output - input))` means that layer is making a stronger update at that timestep.
- Comparing `hidden_states` vs `encoder_hidden_states` shows whether image stream or text stream is being changed more strongly over time.

## Related Files

- `transformer_flux2.py`
- `pipeline_flux2.py`
- `flux2_klein_transformer_config.json`
- `analyze_flux2_activations.py`
