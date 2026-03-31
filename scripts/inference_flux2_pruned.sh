python scripts/inference_flux2_pruned.py \
    --pipeline_path black-forest-labs/FLUX.2-klein-base-4B \
    --transformer_path /your/path/to/save/model/pruned_transformer.pt \
    --prompt "A cinematic portrait of an astronaut standing in a sunflower field at golden hour, ultra detailed, natural light." \
    --height 1024 \
    --width 1024 \
    --num_inference_steps 50 \
    --guidance_scale 4.0 \
    --seed 0 \
    --output ./eval_output/flux2_pruned_sample.png

# To run a pruned FLUX.2-klein-4B transformer, change --pipeline_path to:
# black-forest-labs/FLUX.2-klein-4B
