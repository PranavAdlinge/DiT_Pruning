"""
Precomputed DreamBooth dataset: instance + class MDS streams for prior preservation.

When with_prior_preservation=True, each __getitem__ returns one instance and one
class sample (paired). Collate stacks pairs: first half of batch = instance,
second half = class. Loss splits accordingly.
"""

import torch
from streaming import StreamingDataset

from .precomputed_sstk_dataset import _bytes_to_latent, _bytes_to_text_embeds


def _decode_sample(
    sample: dict,
    latent_key: str,
    resolution: int,
    latent_channels: int,
    text_embed_hidden: int,
    latent_dtype: torch.dtype,
) -> dict:
    """Decode one MDS sample to latents, text_embeds, text_ids."""
    from diffusers import Flux2KleinPipeline

    latent_bytes = sample[latent_key]
    latents = _bytes_to_latent(
        latent_bytes,
        resolution,
        latent_channels,
        latent_dtype,
    )

    text_embeds = _bytes_to_text_embeds(
        sample["text_embeds"],
        text_embed_hidden,
        latent_dtype,
    )

    # text_ids from text_embeds, same as Flux2KleinPipeline.encode_prompt
    text_embeds_batched = text_embeds.unsqueeze(0)
    text_ids = Flux2KleinPipeline._prepare_text_ids(text_embeds_batched)
    text_ids = text_ids.squeeze(0)

    return {
        "latents": latents,
        "text_embeds": text_embeds,
        "text_ids": text_ids,
        "caption": sample.get("caption", ""),
    }


class PrecomputedDreamBoothDataset(torch.utils.data.Dataset):
    """
    Dataset for DreamBooth training from precomputed instance and class MDS.

    Always returns instance_latents (+ instance_text_embeds, instance_text_ids).
    When with_prior_preservation=True, also returns class_latents. Collate
    merges when class_latents present: first half = instance, second half = class.
    """

    def __init__(
        self,
        local: str,
        instance_subdir: str = "instance",
        class_subdir: str = "class",
        with_prior_preservation: bool = True,
        resolution: int = 512,
        latent_channels: int = 32,
        text_embed_hidden: int = 0,
        batch_size: int = 4,
        shuffle: bool = True,
        latent_dtype: torch.dtype = torch.float32,
    ):
        self.with_prior_preservation = with_prior_preservation
        self.resolution = resolution
        self.latent_channels = latent_channels
        self.text_embed_hidden = text_embed_hidden
        self.latent_dtype = latent_dtype
        self.latent_key = f"latents_{resolution}"

        instance_local = f"{local.rstrip('/')}/{instance_subdir}"
        self.instance_ds = StreamingDataset(
            local=instance_local,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        if with_prior_preservation:
            class_local = f"{local.rstrip('/')}/{class_subdir}"
            self.class_ds = StreamingDataset(
                local=class_local,
                batch_size=batch_size,
                shuffle=shuffle,
            )
            self._len = len(self.instance_ds)
        else:
            self.class_ds = None
            self._len = len(self.instance_ds)

    def __len__(self) -> int:
        return self._len

    def _decode_one(self, sample: dict) -> dict:
        return _decode_sample(
            sample,
            self.latent_key,
            self.resolution,
            self.latent_channels,
            self.text_embed_hidden,
            self.latent_dtype,
        )

    def __getitem__(self, index: int) -> dict:
        inst_idx = index % len(self.instance_ds)
        inst_sample = self.instance_ds[inst_idx]
        inst_decoded = self._decode_one(inst_sample)

        out = {
            "instance_latents": inst_decoded["latents"],
            "instance_text_embeds": inst_decoded["text_embeds"],
            "instance_text_ids": inst_decoded["text_ids"],
            "caption": inst_decoded["caption"],
        }
        if self.with_prior_preservation:
            class_idx = index % len(self.class_ds)
            class_sample = self.class_ds[class_idx]
            class_decoded = self._decode_one(class_sample)
            out["class_latents"] = class_decoded["latents"]
            out["class_text_embeds"] = class_decoded["text_embeds"]
            out["class_text_ids"] = class_decoded["text_ids"]
        return out


def collate_precomputed_dreambooth(batch: list[dict]) -> dict:
    """Collate DreamBooth batch. Instance always present; if class_latents in batch, merge."""
    if not batch:
        return {}
    instance_latents = torch.stack([b["instance_latents"] for b in batch])
    instance_text_embeds = torch.stack([b["instance_text_embeds"] for b in batch])
    instance_text_ids = torch.stack([b["instance_text_ids"] for b in batch])
    captions = [b["caption"] for b in batch]

    if "class_latents" in batch[0]:
        class_latents = torch.stack([b["class_latents"] for b in batch])
        class_text_embeds = torch.stack([b["class_text_embeds"] for b in batch])
        class_text_ids = torch.stack([b["class_text_ids"] for b in batch])
        latents = torch.cat([instance_latents, class_latents], dim=0)
        text_embeds = torch.cat([instance_text_embeds, class_text_embeds], dim=0)
        text_ids = torch.cat([instance_text_ids, class_text_ids], dim=0)
        captions = captions * 2  # instance + class per sample
    else:
        latents = instance_latents
        text_embeds = instance_text_embeds
        text_ids = instance_text_ids

    return {
        "latents": latents,
        "text_embeds": text_embeds,
        "text_ids": text_ids,
        "captions": captions,
    }
