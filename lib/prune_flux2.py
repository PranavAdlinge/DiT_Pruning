from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch_pruning as tp

from .OBS_Diff_Structured_Flux2 import Flux2StructuredJointAttentionPruner, Flux2StructuredLinearPruner
from .dataloader import get_loaders


DOUBLE_ATTN_TARGET = "attn.to_out.0"
DOUBLE_FF_TARGETS = ("ff.linear_out", "ff_context.linear_out")
SINGLE_ATTN_TARGET = "attn.to_out::__heads__"
SINGLE_FFN_TARGET = "attn.to_out::__ffn__"
SINGLE_SHARED_KEY = "attn.to_out"


def get_module_by_name(layer, name):
    module = layer
    for attr in name.split("."):
        module = getattr(module, attr)
    return module


def find_layers(module, layers=(nn.Linear,), name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + "." + name1 if name else name1))
    return res


def _normalize_chunked_groups(groupable_items, num_groups):
    num_items = len(groupable_items)
    if num_items == 0:
        return []

    group_size = num_items // num_groups
    remainder = num_items % num_groups
    if group_size == 0:
        group_size = 1
        num_groups = num_items
        remainder = 0

    final_groups = []
    start_index = 0
    for group_idx in range(num_groups):
        end_index = start_index + group_size + (1 if group_idx < remainder else 0)
        current_chunk = groupable_items[start_index:end_index]
        final_groups.append([module for unit in current_chunk for module in unit])
        start_index = end_index

    return final_groups


def group_flux2_modules_with_parallelism(target_pruned_modules, num_groups):
    modules_by_block = defaultdict(list)
    for stream_name, block_idx, module_name in target_pruned_modules:
        modules_by_block[(stream_name, block_idx)].append(module_name)

    groupable_items = []
    for (stream_name, block_idx), module_names in sorted(modules_by_block.items(), key=lambda item: (item[0][0], item[0][1])):
        module_set = set(module_names)
        if stream_name == "double":
            if DOUBLE_ATTN_TARGET in module_set:
                groupable_items.append([(stream_name, block_idx, DOUBLE_ATTN_TARGET)])
            ff_targets = [(stream_name, block_idx, name) for name in DOUBLE_FF_TARGETS if name in module_set]
            if ff_targets:
                groupable_items.append(ff_targets)
        else:
            single_targets = []
            if SINGLE_ATTN_TARGET in module_set:
                single_targets.append((stream_name, block_idx, SINGLE_ATTN_TARGET))
            if SINGLE_FFN_TARGET in module_set:
                single_targets.append((stream_name, block_idx, SINGLE_FFN_TARGET))
            if single_targets:
                groupable_items.append(single_targets)

    return _normalize_chunked_groups(groupable_items, num_groups)


def create_hook_fn(block_key, pruner_dict, timestep_weight):
    def hook_fn(module, input, output):
        step = step_info["current"]
        pruner = pruner_dict[block_key]
        input_data = input[0].data

        current_weight = timestep_weight[step]
        num_samples = input_data.shape[0]
        weight_new = current_weight * num_samples
        input_data = input_data * np.sqrt(current_weight)
        pruner.add_batch(input_data, output.data, weight_new)

    return hook_fn


def create_hook_fn_joint_attn(block_key, layer_name, pruner_dict, timestep_weight):
    def hook_fn(module, input, output):
        step = step_info["current"]
        pruner = pruner_dict[block_key]
        input_data = input[0].data

        current_weight = timestep_weight[step]
        num_samples = input_data.shape[0]
        weight_new = current_weight * num_samples
        input_data = input_data * np.sqrt(current_weight)
        pruner.add_batch(input_data, output.data, layer_name, weight_new)

    return hook_fn


step_info = {"current": 0}


def callback_on_step_end(pipeline, step, timestep, callback_kwargs):
    step_info["current"] += 1
    return callback_kwargs


def _collect_flux2_targets(args, pipe):
    target_pruned_modules = []

    for block_idx in range(args.double_minlayer, args.double_maxlayer):
        block = pipe.transformer.transformer_blocks[block_idx]
        block_layers = find_layers(block)
        if DOUBLE_ATTN_TARGET in block_layers:
            target_pruned_modules.append(("double", block_idx, DOUBLE_ATTN_TARGET))
        for ff_target in DOUBLE_FF_TARGETS:
            if ff_target in block_layers:
                target_pruned_modules.append(("double", block_idx, ff_target))

    for block_idx in range(args.single_minlayer, args.single_maxlayer):
        block = pipe.transformer.single_transformer_blocks[block_idx]
        block_layers = find_layers(block)
        if "attn.to_out" in block_layers:
            target_pruned_modules.append(("single", block_idx, SINGLE_ATTN_TARGET))
            target_pruned_modules.append(("single", block_idx, SINGLE_FFN_TARGET))

    return target_pruned_modules


def check_flux2_size(transformer_model, args):
    print("\n" + "=" * 50)
    print("Checking Flux2 Structured Module Sizes...")
    print("=" * 50)

    module_shapes = OrderedDict()
    for block_idx in range(args.double_minlayer, args.double_maxlayer):
        block = transformer_model.transformer_blocks[block_idx]
        module_shapes[f"double.{block_idx}.attn.to_out.0"] = block.attn.to_out[0].weight.shape
        module_shapes[f"double.{block_idx}.attn.to_add_out"] = block.attn.to_add_out.weight.shape
        module_shapes[f"double.{block_idx}.ff.linear_out"] = block.ff.linear_out.weight.shape
        module_shapes[f"double.{block_idx}.ff.linear_in"] = block.ff.linear_in.weight.shape
        module_shapes[f"double.{block_idx}.ff_context.linear_out"] = block.ff_context.linear_out.weight.shape
        module_shapes[f"double.{block_idx}.ff_context.linear_in"] = block.ff_context.linear_in.weight.shape

    for block_idx in range(args.single_minlayer, args.single_maxlayer):
        block = transformer_model.single_transformer_blocks[block_idx]
        module_shapes[f"single.{block_idx}.attn.to_out"] = block.attn.to_out.weight.shape
        module_shapes[f"single.{block_idx}.attn.to_qkv_mlp_proj"] = block.attn.to_qkv_mlp_proj.weight.shape
        module_shapes[f"single.{block_idx}.attn.heads"] = torch.Size([block.attn.heads, block.attn.head_dim])
        module_shapes[f"single.{block_idx}.attn.mlp_hidden_dim"] = torch.Size([block.attn.mlp_hidden_dim])

    for module_name, shape in module_shapes.items():
        print(f"{module_name:<48s} | {shape}")

    print("\n" + "=" * 50)
    print("Flux2 size check finished.")
    print("=" * 50)


def _validate_flux2_double_block(block, block_idx):
    inner_dim = block.attn.inner_dim
    head_dim = block.attn.head_dim
    heads = block.attn.heads
    if inner_dim != heads * head_dim:
        raise ValueError(f"Double block {block_idx}: inner_dim {inner_dim} does not match heads * head_dim = {heads * head_dim}.")

    if block.attn.to_out[0].in_features != inner_dim:
        raise ValueError(
            f"Double block {block_idx}: attn.to_out.0 in_features {block.attn.to_out[0].in_features} does not match inner_dim {inner_dim}."
        )
    if block.attn.to_add_out.in_features != inner_dim:
        raise ValueError(
            f"Double block {block_idx}: attn.to_add_out in_features {block.attn.to_add_out.in_features} does not match inner_dim {inner_dim}."
        )

    for layer_name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"):
        layer = getattr(block.attn, layer_name)
        if layer.out_features != inner_dim:
            raise ValueError(
                f"Double block {block_idx}: attn.{layer_name} out_features {layer.out_features} does not match inner_dim {inner_dim}."
            )

    for ff_name in ("ff", "ff_context"):
        ff = getattr(block, ff_name)
        hidden_dim = ff.linear_out.in_features
        if ff.linear_in.out_features != hidden_dim * 2:
            raise ValueError(
                f"Double block {block_idx}: {ff_name}.linear_in out_features {ff.linear_in.out_features} "
                f"does not match SwiGLU 2x hidden dim {hidden_dim * 2}."
            )
        if ff.linear_out.out_features != ff.linear_in.in_features:
            raise ValueError(
                f"Double block {block_idx}: {ff_name}.linear_out out_features {ff.linear_out.out_features} "
                f"does not match model dim {ff.linear_in.in_features}."
            )


def _validate_flux2_single_block(block, block_idx):
    attn = block.attn
    expected_inner_dim = attn.heads * attn.head_dim
    if attn.inner_dim != expected_inner_dim:
        raise ValueError(
            f"Single block {block_idx}: inner_dim {attn.inner_dim} does not match heads * head_dim = {expected_inner_dim}."
        )

    expected_mlp_hidden_dim = int(attn.query_dim * attn.mlp_ratio)
    if attn.mlp_hidden_dim != expected_mlp_hidden_dim:
        raise ValueError(
            f"Single block {block_idx}: mlp_hidden_dim {attn.mlp_hidden_dim} does not match query_dim * mlp_ratio = {expected_mlp_hidden_dim}."
        )
    if attn.mlp_mult_factor != 2:
        raise ValueError(f"Single block {block_idx}: expected mlp_mult_factor 2, got {attn.mlp_mult_factor}.")

    expected_qkv_mlp_shape = (3 * attn.inner_dim + attn.mlp_hidden_dim * attn.mlp_mult_factor, attn.query_dim)
    expected_to_out_shape = (attn.out_dim, attn.inner_dim + attn.mlp_hidden_dim)
    if tuple(attn.to_qkv_mlp_proj.weight.shape) != expected_qkv_mlp_shape:
        raise ValueError(
            f"Single block {block_idx}: to_qkv_mlp_proj weight shape {tuple(attn.to_qkv_mlp_proj.weight.shape)} "
            f"does not match expected {expected_qkv_mlp_shape}."
        )
    if tuple(attn.to_out.weight.shape) != expected_to_out_shape:
        raise ValueError(
            f"Single block {block_idx}: to_out weight shape {tuple(attn.to_out.weight.shape)} does not match expected {expected_to_out_shape}."
        )


def validate_flux2_transformer_shapes(transformer_model, args):
    for block_idx in range(args.double_minlayer, args.double_maxlayer):
        _validate_flux2_double_block(transformer_model.transformer_blocks[block_idx], block_idx)

    for block_idx in range(args.single_minlayer, args.single_maxlayer):
        _validate_flux2_single_block(transformer_model.single_transformer_blocks[block_idx], block_idx)


def _expand_swiglu_indices(pruned_idx, hidden_dim):
    expanded_idx = torch.cat([pruned_idx, pruned_idx + hidden_dim])
    return torch.sort(expanded_idx).values.tolist()


def _apply_double_ffn_pruning(block, module_name, pruned_idx):
    target_layer = get_module_by_name(block, module_name)
    input_layer_name = module_name.replace("linear_out", "linear_in")
    input_layer = get_module_by_name(block, input_layer_name)

    hidden_dim = target_layer.in_features
    if input_layer.out_features != hidden_dim * 2:
        raise ValueError(f"Expected SwiGLU expansion of 2x for {module_name}, got {input_layer.out_features} vs {hidden_dim}.")

    pruned_idx = torch.sort(pruned_idx).values
    tp.prune_linear_in_channels(target_layer, pruned_idx.tolist())
    tp.prune_linear_out_channels(input_layer, _expand_swiglu_indices(pruned_idx, hidden_dim))


def _apply_double_attention_pruning(block, pruned_idx):
    pruned_idx = torch.sort(pruned_idx).values

    tp.prune_linear_in_channels(block.attn.to_out[0], pruned_idx.tolist())
    tp.prune_linear_in_channels(block.attn.to_add_out, pruned_idx.tolist())
    tp.prune_linear_out_channels(block.attn.to_q, pruned_idx.tolist())
    tp.prune_linear_out_channels(block.attn.to_k, pruned_idx.tolist())
    tp.prune_linear_out_channels(block.attn.to_v, pruned_idx.tolist())
    tp.prune_linear_out_channels(block.attn.add_q_proj, pruned_idx.tolist())
    tp.prune_linear_out_channels(block.attn.add_k_proj, pruned_idx.tolist())
    tp.prune_linear_out_channels(block.attn.add_v_proj, pruned_idx.tolist())

    block.attn.inner_dim = block.attn.to_q.out_features
    block.attn.heads = block.attn.inner_dim // block.attn.head_dim


def _apply_single_attention_and_ffn_pruning(block, pruner, sparsity, percdamp):
    attn = block.attn
    old_inner_dim = attn.inner_dim
    old_mlp_hidden_dim = attn.mlp_hidden_dim
    old_mlp_mult_factor = attn.mlp_mult_factor
    if old_mlp_mult_factor != 2:
        raise ValueError(f"Expected Flux2 single-stream SwiGLU multiplier of 2, got {old_mlp_mult_factor}.")
    expected_qkv_mlp_out = old_inner_dim * 3 + old_mlp_hidden_dim * old_mlp_mult_factor
    expected_to_out_in = old_inner_dim + old_mlp_hidden_dim
    if attn.to_qkv_mlp_proj.out_features != expected_qkv_mlp_out:
        raise ValueError(
            f"Unexpected Flux2 single-stream fused projection size: {attn.to_qkv_mlp_proj.out_features} vs {expected_qkv_mlp_out}."
        )
    if attn.to_out.in_features != expected_to_out_in:
        raise ValueError(f"Unexpected Flux2 single-stream output size: {attn.to_out.in_features} vs {expected_to_out_in}.")

    attn_idx = pruner.struct_prune(
        sparsity=sparsity,
        percdamp=percdamp,
        group_size=attn.head_dim,
        candidate_idx=torch.arange(old_inner_dim, device=pruner.dev),
    )
    ffn_global_idx = pruner.struct_prune(
        sparsity=sparsity,
        percdamp=percdamp,
        candidate_idx=torch.arange(old_inner_dim, old_inner_dim + old_mlp_hidden_dim, device=pruner.dev),
    )

    attn_idx = torch.sort(attn_idx).values
    ffn_global_idx = torch.sort(ffn_global_idx).values
    ffn_local_idx = ffn_global_idx - old_inner_dim
    combined_to_out_idx = torch.sort(torch.cat([attn_idx, ffn_global_idx])).values

    if combined_to_out_idx.numel() > 0:
        tp.prune_linear_in_channels(attn.to_out, combined_to_out_idx.tolist())

    qkv_attn_idx = torch.cat([attn_idx, attn_idx + old_inner_dim, attn_idx + 2 * old_inner_dim])
    mlp_proj_base = 3 * old_inner_dim
    mlp_proj_idx = torch.cat(
        [ffn_local_idx + mlp_proj_base, ffn_local_idx + mlp_proj_base + old_mlp_hidden_dim]
    )
    qkv_mlp_pruned_idx = torch.sort(torch.cat([qkv_attn_idx, mlp_proj_idx])).values
    if qkv_mlp_pruned_idx.numel() > 0:
        tp.prune_linear_out_channels(attn.to_qkv_mlp_proj, qkv_mlp_pruned_idx.tolist())

    attn.inner_dim = old_inner_dim - attn_idx.numel()
    attn.heads = attn.inner_dim // attn.head_dim
    attn.mlp_hidden_dim = old_mlp_hidden_dim - ffn_local_idx.numel()
    attn.mlp_mult_factor = old_mlp_mult_factor


@torch.no_grad()
def prune_OBS_Diff_Structured_Flux2(args, pipe, dev, timestep_weight=None):
    print("Starting Flux2 structured pruning...")
    dataloader = get_loaders(args.dataset, num_samples=args.num_samples, prompt_file=args.prompt_file)
    if len(dataloader) == 0:
        raise ValueError("Calibration prompts are empty. Check --dataset or --prompt_file.")
    validate_flux2_transformer_shapes(pipe.transformer, args)
    target_pruned_modules = _collect_flux2_targets(args, pipe)
    modules_groups = group_flux2_modules_with_parallelism(target_pruned_modules, args.num_pruned_groups)

    print(f"\nintelligently divided {len(target_pruned_modules)} modules into {len(modules_groups)} groups:")
    for group_idx, group in enumerate(modules_groups):
        print(f"Group {group_idx + 1}: {group}")

    for group_idx, group_modules in enumerate(modules_groups):
        print(f"\nProcessing Flux2 group {group_idx + 1}/{len(modules_groups)}...")

        pruner_dict = {}
        hooks = []
        shared_single_hooks = set()

        for stream_name, block_idx, module_name in group_modules:
            if stream_name == "double":
                block = pipe.transformer.transformer_blocks[block_idx]
                if module_name == DOUBLE_ATTN_TARGET:
                    block_key = (stream_name, block_idx, module_name)
                    pruner_dict[block_key] = Flux2StructuredJointAttentionPruner(block.attn.to_out[0], block.attn.to_add_out, args)
                    hooks.append(block.attn.to_out[0].register_forward_hook(create_hook_fn_joint_attn(block_key, "attn.to_out.0", pruner_dict, timestep_weight)))
                    hooks.append(block.attn.to_add_out.register_forward_hook(create_hook_fn_joint_attn(block_key, "attn.to_add_out", pruner_dict, timestep_weight)))
                else:
                    block_key = (stream_name, block_idx, module_name)
                    module = get_module_by_name(block, module_name)
                    pruner_dict[block_key] = Flux2StructuredLinearPruner(module, args)
                    hooks.append(module.register_forward_hook(create_hook_fn(block_key, pruner_dict, timestep_weight)))
            else:
                block_key = (stream_name, block_idx, SINGLE_SHARED_KEY)
                if block_key in shared_single_hooks:
                    continue
                block = pipe.transformer.single_transformer_blocks[block_idx]
                pruner_dict[block_key] = Flux2StructuredLinearPruner(block.attn.to_out, args)
                hooks.append(block.attn.to_out.register_forward_hook(create_hook_fn(block_key, pruner_dict, timestep_weight)))
                shared_single_hooks.add(block_key)

        print(f"Running diffusion for Flux2 group {group_idx + 1} to collect activations...")
        batch_size = args.batch_size
        num_batches = (len(dataloader) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            prompts = dataloader[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            print(f"  Prompts {batch_idx}: {prompts}")
            step_info["current"] = 0
            pipe(
                prompt=prompts,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=["latents"],
                generator=torch.Generator("cuda").manual_seed(args.seed),
            )

        for hook in hooks:
            hook.remove()

        print(f"Pruning Flux2 group {group_idx + 1}...")
        processed_single_blocks = set()
        for stream_name, block_idx, module_name in group_modules:
            sparsity = args.sparsity_ratio

            if stream_name == "double":
                block = pipe.transformer.transformer_blocks[block_idx]
                block_key = (stream_name, block_idx, module_name)

                if module_name == DOUBLE_ATTN_TARGET:
                    print(f"Pruning double block {block_idx}: attention heads")
                    pruned_idx = pruner_dict[block_key].struct_prune(
                        sparsity=sparsity,
                        percdamp=args.percdamp,
                        headsize=pipe.transformer.config.attention_head_dim,
                    )
                    _apply_double_attention_pruning(block, pruned_idx)
                else:
                    print(f"Pruning double block {block_idx}: {module_name}")
                    pruned_idx = pruner_dict[block_key].struct_prune(
                        sparsity=sparsity,
                        percdamp=args.percdamp,
                    )
                    _apply_double_ffn_pruning(block, module_name, pruned_idx)

                _validate_flux2_double_block(block, block_idx)
                pruner_dict[block_key].free()
            else:
                if block_idx in processed_single_blocks:
                    continue

                print(f"Pruning single block {block_idx}: attention heads + FFN")
                block = pipe.transformer.single_transformer_blocks[block_idx]
                block_key = (stream_name, block_idx, SINGLE_SHARED_KEY)
                _apply_single_attention_and_ffn_pruning(
                    block=block,
                    pruner=pruner_dict[block_key],
                    sparsity=sparsity,
                    percdamp=args.percdamp,
                )
                _validate_flux2_single_block(block, block_idx)
                pruner_dict[block_key].free()
                processed_single_blocks.add(block_idx)

        torch.cuda.empty_cache()
        print(f"Flux2 group {group_idx + 1} pruning completed.")
