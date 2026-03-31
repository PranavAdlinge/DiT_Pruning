python main.py \
    --model_family flux2 \
    --pipeline_path black-forest-labs/FLUX.2-klein-base-4B \
    --prune_method OBS-Diff-Structured \
    --seed 24432 \
    --sparsity_ratio 0.20 \
    --sparsity_type structured \
    --timestep_weight_strategy log_decrease \
    --timestep_min_weight 0.8 \
    --timestep_max_weight 1.2 \
    --num_samples 100 \
    --num_inference_steps 25 \
    --batch_size 4 \
    --height 512 \
    --width 512 \
    --guidance_scale 3.5 \
    --num_pruned_groups 4 \
    --save_model /your/path/to/save/model \
    --demo_evaluate \
    --demo_dir OBS_Diff_Structured_Flux2_test.png

# To prune FLUX.2-klein-4B instead, change --pipeline_path to:
# black-forest-labs/FLUX.2-klein-4B
#
# To prune a different trained transformer on top of the same base pipeline, add:
# --transformer_path /your/path/to/custom/transformer
