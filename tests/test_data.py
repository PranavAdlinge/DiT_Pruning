"""Tests for nexus.data.precomputed_sstk_dataset"""

import numpy as np
import pytest
import torch

from nexus.data.precomputed_sstk_dataset import collate_precomputed


class TestCollatePrecomputed:
    """Flux.2 Klein: latent_channels=32, resolution 512 -> 64x64, text_embed_hidden=7680."""

    def test_collate_stacks_tensors(self):
        batch = [
            {
                "latents": torch.randn(32, 64, 64),
                "text_embeds": torch.randn(512, 7680),
                "text_ids": torch.zeros(512, 4, dtype=torch.int64),
                "caption": "cap1",
            },
            {
                "latents": torch.randn(32, 64, 64),
                "text_embeds": torch.randn(512, 7680),
                "text_ids": torch.zeros(512, 4, dtype=torch.int64),
                "caption": "cap2",
            },
        ]
        out = collate_precomputed(batch)
        assert out["latents"].shape == (2, 32, 64, 64)
        assert out["text_embeds"].shape == (2, 512, 7680)
        assert out["text_ids"].shape == (2, 512, 4)
        assert out["captions"] == ["cap1", "cap2"]

    def test_collate_single_sample(self):
        batch = [
            {
                "latents": torch.randn(32, 64, 64),
                "text_embeds": torch.randn(512, 7680),
                "text_ids": torch.zeros(512, 4, dtype=torch.int64),
                "caption": "",
            },
        ]
        out = collate_precomputed(batch)
        assert out["latents"].shape == (1, 32, 64, 64)
        assert len(out["captions"]) == 1
