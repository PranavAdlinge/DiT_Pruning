"""Tests for nexus.losses"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from nexus.losses import (
    BASE_LOSSES,
    DistillationLoss,
    FlowMatchingLoss,
    FlowMatchingWithPriorPreservation,
    LossContext,
    build_loss_fn,
)


def _make_ctx(
    b=2, c=4, h=4, w=4, has_model_output=True
) -> tuple[LossContext, dict]:
    """Build minimal LossContext for testing."""
    batch = {
        "latents": torch.randn(b, 32, h, w),
        "text_embeds": torch.randn(b, 512, 64),
        "text_ids": torch.randn(b, 512, 4),
    }
    noise = torch.randn(b, c, h, w)
    model_input = torch.randn(b, c, h, w)
    target = noise - model_input  # loss computes this from ctx
    weighting = torch.ones(b, c, h, w)
    if has_model_output:
        model_output = torch.randn(b, c, h, w)
    else:
        model_output = target.clone()  # perfect pred

    packed_noisy = torch.randn(b, c * 2, h, w)  # placeholder
    model_input_ids = torch.zeros(b, c, h, w, 4, dtype=torch.long)
    timesteps = torch.tensor([500.0] * b)
    sigmas = torch.ones(b, 1, 1, 1)

    ctx = LossContext(
        batch=batch,
        noise=noise,
        model_output=model_output,
        model_input=model_input,
        weighting=weighting,
        packed_noisy=packed_noisy,
        model_input_ids=model_input_ids,
        timesteps=timesteps,
        sigmas=sigmas,
        guidance=None,
        text_embeds=batch["text_embeds"],
        text_ids=batch["text_ids"],
    )
    return ctx, batch


class TestFlowMatchingLoss:
    def test_base_mse_returns_scalar_and_log_dict(self):
        ctx, _ = _make_ctx()
        loss_fn = FlowMatchingLoss(base="mse")
        total, logs = loss_fn(ctx)
        assert total.dim() == 0
        assert total.item() >= 0
        assert "loss" in logs
        assert "loss_flow" in logs

    def test_perfect_pred_zero_loss(self):
        ctx, _ = _make_ctx(has_model_output=False)
        loss_fn = FlowMatchingLoss(base="mse")
        total, logs = loss_fn(ctx)
        assert total.item() == pytest.approx(0.0)
        assert logs["loss"] == pytest.approx(0.0)

    def test_base_types(self):
        ctx, _ = _make_ctx()
        for base in BASE_LOSSES:
            loss_fn = FlowMatchingLoss(base=base, huber_delta=1.0)
            total, logs = loss_fn(ctx)
            assert total.dim() == 0
            assert total.item() >= 0

    def test_invalid_base_raises(self):
        with pytest.raises(ValueError, match="base must be one of"):
            FlowMatchingLoss(base="invalid")


class TestFlowMatchingWithPriorPreservation:
    def test_splits_batch(self):
        ctx, _ = _make_ctx(b=4)
        loss_fn = FlowMatchingWithPriorPreservation(base="mse", weight=1.0)
        total, logs = loss_fn(ctx)
        assert total.dim() == 0
        assert "loss_instance" in logs
        assert "loss_prior" in logs
        assert "loss_flow" in logs
        assert "loss" in logs

    def test_odd_batch_raises(self):
        ctx, _ = _make_ctx(b=3)
        loss_fn = FlowMatchingWithPriorPreservation(base="mse", weight=1.0)
        with pytest.raises(AssertionError, match="even batch size"):
            loss_fn(ctx)


class TestDistillationLoss:
    def test_instantiates_without_teacher(self):
        """Without pretrained_model_name_or_path, falls back to flow loss only."""
        loss_fn = DistillationLoss(base="mse", flow_weight=0.5, distillation_weight=0.5)
        ctx, _ = _make_ctx()
        total, logs = loss_fn(ctx)
        assert total.dim() == 0
        assert "loss" in logs
        assert "loss_flow" in logs


class TestBuildLossFn:
    def test_build_flow_matching_from_config(self, tmp_path):
        cfg = _minimal_config(tmp_path, "nexus.losses:FlowMatchingLoss", {"base": "mse"})
        loss_fn = build_loss_fn(cfg)
        assert isinstance(loss_fn, FlowMatchingLoss)
        ctx, _ = _make_ctx()
        total, _ = loss_fn(ctx)
        assert total.dim() == 0

    def test_missing_loss_class_raises(self, tmp_path):
        cfg_path = tmp_path / "bad.yaml"
        cfg_path.write_text("loss:\n  kwargs:\n    base: mse\n")
        from nexus.train.config import load_config
        cfg = load_config(cfg_path)
        with pytest.raises(ValueError, match="loss.class_name"):
            build_loss_fn(cfg)

    def test_passes_extra_kwargs_to_loss(self, tmp_path):
        cfg = _minimal_config(tmp_path, "nexus.losses:FlowMatchingLoss", {"base": "mse"})
        loss_fn = build_loss_fn(cfg, model_cfg=SimpleNamespace(), accelerator=MagicMock(), weight_dtype=torch.float32)
        assert isinstance(loss_fn, FlowMatchingLoss)


def _minimal_config(tmp_path, class_name: str, kwargs: dict):
    from nexus.train.config import load_config
    import yaml
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump({
        "loss": {"class_name": class_name, "kwargs": kwargs},
    }))
    return load_config(cfg_path)
