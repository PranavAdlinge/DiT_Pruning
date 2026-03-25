"""
Flow-matching with prior preservation (DreamBooth).

First half of batch = instance, second half = class.
L = instance_loss + prior_weight * prior_loss.
"""

import torch

from .context import LossContext
from .flow_matching import BASE_LOSSES, FlowMatchingLoss


class FlowMatchingWithPriorPreservation:
    """
    DreamBooth prior preservation: split batch into instance and class.

    Config:
      base: mse | l1 | huber | logcosh (for both instance and prior)
      huber_delta: float (when base=huber)
      weight: prior loss weight (default 1.0)
    """

    def __init__(self, base: str = "mse", huber_delta: float = 1.0, weight: float = 1.0, **kwargs):
        self.base_loss = FlowMatchingLoss(base=base, huber_delta=huber_delta)
        self.prior_weight = weight

    def __call__(self, ctx: LossContext) -> tuple[torch.Tensor, dict[str, float]]:
        pred = ctx.model_output
        target = ctx.noise - ctx.model_input
        w = ctx.weighting
        assert pred.shape[0] % 2 == 0, "Prior preservation requires even batch size (instance + class)"
        mid = pred.shape[0] // 2

        instance_loss = self.base_loss._base_loss(pred[:mid], target[:mid], w[:mid])
        prior_loss = self.base_loss._base_loss(pred[mid:], target[mid:], w[mid:])
        total = instance_loss + self.prior_weight * prior_loss

        logs = {
            "loss": total.detach().item(),
            "loss_flow": total.detach().item(),
            "loss_instance": instance_loss.detach().item(),
            "loss_prior": prior_loss.detach().item(),
        }
        return total, logs
