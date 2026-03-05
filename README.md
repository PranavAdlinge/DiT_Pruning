# FLUX.2-Klein Pruning Workflow

This GitHub repo covers three pruning tasks for `FLUX.2-klein-4B`:
1. Transformer layer pruning
2. MLP channel pruning
3. Attention head pruning

## Pruning TODO

- [x] Transformer layer pruning: remove low-impact transformer blocks (double/single stream) using timestep-aware activation-delta ranking.
- [ ] MLP channel pruning: remove low-importance feed-forward channels inside transformer blocks to shrink MLP width with minimal quality loss.
- [ ] Attention head pruning: remove low-importance attention heads to reduce attention compute and memory while preserving generation quality.

The implemented workflow in this repository currently focuses on transformer layer pruning end-to-end.

## Repository Files

- `analyze_flux2_activations.py`: Pipeline-based activation collection.
- `rank_flux2_redundant_layers.py`: Redundancy ranking from generated CSV files.
- `flux2_klein_prompts.json`: Prompt set file (currently contains 500 prompts).
- `pipeline_flux2.py`: Local FLUX2 pipeline source.
- `transformer_flux2.py`: Local FLUX2 transformer source.

## Unified Workflow

### Step 0: Environment

Install dependencies:

```bash
pip install torch matplotlib diffusers transformers accelerate
```

Recommended:
- CUDA GPU
- `torch.bfloat16` or `torch.float16`

### Step 1: Run Pipeline Activation Analysis

This step uses real pipeline inference (not synthetic direct-transformer calls).

Core behavior:
- Loads `Flux2KleinPipeline` from `black-forest-labs/FLUX.2-klein-4B`.
- Runs prompts one by one.
- Hooks:
  - `pipe.transformer.transformer_blocks` (`double_XX`)
  - `pipe.transformer.single_transformer_blocks` (`single_XX`)
- At each denoising step, records:
  - `mean(abs(hidden_out - hidden_in))`
  - `mean(abs(encoder_out - encoder_in))`
- Aggregates by `(layer, raw_timestep)` across prompts.

Run command:

```bash
python analyze_flux2_activations.py \
  --model-id black-forest-labs/FLUX.2-klein-4B \
  --prompts-file flux2_klein_prompts.json \
  --output-dir activation_plots \
  --num-inference-steps 4 \
  --height 1024 \
  --width 1024 \
  --guidance-scale 1.0 \
  --dtype bfloat16
```

Optional:
- Save generated images:

```bash
python analyze_flux2_activations.py \
  --prompts-file flux2_klein_prompts.json \
  --save-images \
  --save-images-limit 500
```

Performance note:
- If `--save-images` is not set, the script uses `output_type="latent"` to skip VAE decode for faster profiling.

### Step 2: Rank Redundant Layers

This step reads activation CSV output and produces a removal order heuristic.

Default ranking command:

```bash
python rank_flux2_redundant_layers.py \
  --csv activation_plots/layer_activation_deltas_pipeline.csv \
  --output-dir activation_plots/redundancy_report \
  --top-k 20
```

Aggregate across multiple CSV runs:

```bash
python rank_flux2_redundant_layers.py \
  --csv-glob "activation_plots/**/layer_activation_deltas_pipeline.csv" \
  --output-dir activation_plots/redundancy_report_multi \
  --top-k 30
```

Protect specific/final layers:

```bash
python rank_flux2_redundant_layers.py \
  --csv activation_plots/layer_activation_deltas_pipeline.csv \
  --protect-final-double 1 \
  --protect-final-single 2 \
  --protect-layers "double_04,single_19"
```

### Step 3: Validate Pruning Decisions

Use ranking output as candidate order only. Then validate by:
- Removing candidate layers incrementally.
- Re-running generation on held-out prompts.
- Measuring visual quality and latency trade-off.

## Outputs

### From `analyze_flux2_activations.py`

- `activation_plots/layer_activation_deltas_pipeline.csv`
- `activation_plots/double_XX_abs_delta_pipeline.png`
- `activation_plots/single_XX_abs_delta_pipeline.png`
- `activation_plots/all_layers_hidden_states_pipeline.png`
- `activation_plots/all_layers_encoder_hidden_states_pipeline.png`
- `activation_plots/run_metadata.json`
- `activation_plots/generated_images/prompt_XXX.png` (if `--save-images`)

### From `rank_flux2_redundant_layers.py`

- `activation_plots/redundancy_report/layer_redundancy_ranking.csv`
- `activation_plots/redundancy_report/suggested_removal_order.txt`
- `activation_plots/redundancy_report/redundancy_metadata.json`

## Scoring Logic for Redundancy

Per layer, the ranking script computes:
- `combined_mean_delta`
- `combined_peak_delta`
- `combined_std_delta`

Then:
- Min-max normalize each metric across layers.
- Compute:
  - `importance_score = 0.6*mean + 0.3*peak + 0.1*std` (defaults)
- Compute:
  - `redundancy_score = 1 - importance_score`

Higher `redundancy_score` means earlier suggestion for removal.

## Prompt Set Note

`flux2_klein_prompts.json` contains **500** prompts.

## What I Could Not Execute in This Sandbox

The code and scripts were implemented and wired together, but the following could not be executed in this Codex sandbox session:

- Python runtime execution (`python` launcher not runnable in this shell context).
- Full FLUX.2-klein inference runs (requires working Python + model download + GPU runtime).
- End-to-end generation of activation CSV/plots from this environment.
- Empirical verification of ranked layer removals on output quality.

Because of that, run the commands above in your local training/inference environment to generate real outputs and validate pruning decisions.

## Recommended Execution Order (Local)

1. Run `analyze_flux2_activations.py` on all 500 prompts.
2. Inspect activation plots for obvious low-impact layers.
3. Run `rank_flux2_redundant_layers.py` to get removal order.
4. Prune in small batches and validate image quality and speed.
