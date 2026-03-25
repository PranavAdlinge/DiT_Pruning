"""Tests for nexus.utils.checkpoint_utils"""

from pathlib import Path

import pytest
import torch

from nexus.utils.checkpoint_utils import (
    load_transformer_state,
    prune_old_checkpoints,
    save_transformer_state,
)


class TestTransformerStateIO:
    def test_save_and_load_pt(self, tmp_path):
        state = {"layer.weight": torch.randn(2, 3)}
        path = tmp_path / "model.pt"
        save_transformer_state(state, path)
        loaded = load_transformer_state(path)
        assert loaded.keys() == state.keys()
        assert torch.allclose(loaded["layer.weight"], state["layer.weight"])

    def test_save_and_load_safetensors(self, tmp_path):
        pytest.importorskip("safetensors")
        state = {"layer.weight": torch.randn(2, 3)}
        path = tmp_path / "model.safetensors"
        save_transformer_state(state, path)
        loaded = load_transformer_state(path)
        assert loaded.keys() == state.keys()
        assert torch.allclose(loaded["layer.weight"], state["layer.weight"])


class TestPruneOldCheckpoints:
    def test_prunes_when_over_limit(self, tmp_path):
        for i in (100, 200, 300):
            (tmp_path / f"checkpoint-{i}").mkdir()
        prune_old_checkpoints(str(tmp_path), limit=2)
        remaining = sorted(d.name for d in tmp_path.iterdir() if d.is_dir())
        assert remaining == ["checkpoint-200", "checkpoint-300"]

    def test_keeps_all_under_limit(self, tmp_path):
        (tmp_path / "checkpoint-100").mkdir()
        prune_old_checkpoints(str(tmp_path), limit=3)
        assert (tmp_path / "checkpoint-100").exists()
