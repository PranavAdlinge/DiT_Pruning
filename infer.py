"""
Corrected BAGEL-7B-MoT parallel inference with HuggingFace Accelerate.

Each rank owns exactly one GPU.  The model is loaded with BAGEL's required
device_map ceremony, NOT with .to(device), because load_checkpoint_and_dispatch
already places every layer.

Run:
    accelerate launch --num_processes 3 infer.py

Or with explicit GPU pinning (recommended):
    CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch \
        --num_processes 3 \
        --multi_gpu \
        infer.py
"""

import os
import sys
import torch
from accelerate import Accelerator, infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
BAGEL_REPO  = "./BAGEL"          # cloned BAGEL source repo
MODEL_PATH  = "./BAGEL-7B-MoT"  # model weights directory
MAX_TOKENS  = 512

IMAGES = [
    "img1.jpg",
    "img2.jpg",
    "img3.jpg",
]

PROMPTS = [
    "Describe this image in detail.",
    "List every object you can see.",
    "Summarise the scene in one sentence.",
]
# ─────────────────────────────────────────────────────────────────────────────


def load_bagel_for_rank(rank: int):
    """
    Load BAGEL onto the single GPU that Accelerate assigned to this rank.

    KEY FIX: do NOT call model.to(device) after this.
    load_checkpoint_and_dispatch owns all device placement.
    """
    if BAGEL_REPO not in sys.path:
        sys.path.insert(0, BAGEL_REPO)

    from modeling.bagel import (
        BagelConfig, Bagel,
        Qwen2MoTConfig, Qwen2MoTForCausalLM,
        SiglipVisionConfig, SiglipVisionModel,
    )
    from modeling.qwen2_navit import NaViTConfig
    from modeling.autoencoder import load_ae
    from data.data_utils import add_special_tokens
    from transformers import AutoTokenizer

    # ── Tokenizer ─────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # ── VAE ───────────────────────────────────────────────────────────
    vae = load_ae(
        local_path=os.path.join(MODEL_PATH, "vae"),
        device="cpu",
        max_length=256,
    )

    # ── Empty model skeleton ──────────────────────────────────────────
    llm_config   = Qwen2MoTConfig.from_pretrained(MODEL_PATH)
    vit_config   = SiglipVisionConfig.from_pretrained(MODEL_PATH)
    navit_config = NaViTConfig.from_pretrained(MODEL_PATH)

    bagel_config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        navit_config=navit_config,
    )

    with init_empty_weights():
        language_model = Qwen2MoTForCausalLM(llm_config)
        vit_model      = SiglipVisionModel(vit_config)
        model          = Bagel(language_model, vit_model, bagel_config)

    # ── Device map: only expose the one GPU for this rank ─────────────
    # Accelerate sets CUDA_VISIBLE_DEVICES per rank when launched with
    # --multi_gpu, so cuda:0 here IS the correct physical card.
    max_memory = {0: "75GiB", "cpu": "48GiB"}

    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    # Modules that must share the same device (BAGEL requirement)
    same_device_mods = [
        "language_model.model.embed_tokens", "time_embedder",
        "latent_pos_embed", "vae2llm", "llm2vae",
        "connector", "vit_pos_embed",
    ]
    anchor = device_map.get(same_device_mods[0], "cuda:0")
    for k in same_device_mods:
        if k in device_map:
            device_map[k] = anchor

    # ── Dispatch — this handles ALL device placement ──────────────────
    # FIX: never call .to(device) after this line
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(MODEL_PATH, "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
    )
    model.eval()

    return model, tokenizer, vae


def build_inputs(image_path: str, prompt: str, tokenizer, device):
    """
    Build the input dict BAGEL's InterleaveInferencer expects.
    FIX: generate_inputs() was undefined in the original — this replaces it.
    """
    if BAGEL_REPO not in sys.path:
        sys.path.insert(0, BAGEL_REPO)

    from data.transforms import ImageTransform          # BAGEL image pre-proc
    from data.data_utils import prepare_vqa_inputs      # text+image packing

    transform = ImageTransform()
    image     = Image.open(image_path).convert("RGB")

    inputs = prepare_vqa_inputs(
        tokenizer=tokenizer,
        image=transform(image),
        question=prompt,
    )
    # Move every tensor to this rank's device
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }


def main():
    # ── One Accelerator per process — handles rank, device, comms ─────
    accelerator = Accelerator()
    rank        = accelerator.process_index
    device      = accelerator.device          # cuda:0 in every rank's view

    accelerator.print(f"Launched {accelerator.num_processes} processes")

    # ── Each rank picks its own image + prompt ────────────────────────
    if rank >= len(IMAGES):
        raise ValueError(
            f"Rank {rank} has no image assigned — "
            f"add more entries to IMAGES/PROMPTS."
        )

    my_image  = IMAGES[rank]
    my_prompt = PROMPTS[rank]

    # ── Load model (each rank loads independently onto its own GPU) ───
    accelerator.print(f"[Rank {rank}] Loading BAGEL onto {device} …")
    model, tokenizer, vae = load_bagel_for_rank(rank)

    # FIX: do NOT do model.to(device) here — load_checkpoint_and_dispatch
    # already placed the model. Calling .to() again corrupts the device_map.

    # ── Import inferencer after BAGEL repo is in sys.path ─────────────
    from inferencer import InterleaveInferencer
    inferencer = InterleaveInferencer(
        model=model,
        vae=vae,
        tokenizer=tokenizer,
        dtype=torch.bfloat16,
    )

    # ── Run inference ─────────────────────────────────────────────────
    accelerator.print(f"[Rank {rank}] Inferring on {my_image} …")

    with torch.inference_mode():
        # FIX: call through InterleaveInferencer, not raw model()
        inputs = build_inputs(my_image, my_prompt, tokenizer, device)
        result = inferencer(
            image=Image.open(my_image).convert("RGB"),
            text=my_prompt,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
        )

    # ── Gather results from all ranks onto rank 0 ─────────────────────
    # FIX: original just printed locally — this collects across all ranks
    all_results = [None] * accelerator.num_processes
    torch.distributed.all_gather_object(all_results, result)

    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("ALL RESULTS")
        print("=" * 60)
        for i, (img, prompt, res) in enumerate(
            zip(IMAGES, PROMPTS, all_results)
        ):
            print(f"\n[GPU {i}] Image : {img}")
            print(f"        Prompt: {prompt}")
            print(f"        Output: {res}")
        print("=" * 60)


if __name__ == "__main__":
    # FIX: required guard — spawn-based multiprocessing will re-import
    # this module in each worker; without this, main() runs recursively.
    main()
