"""
Flow-matching loss: pred vs target with configurable base (MSE, L1, Huber, LogCosh).
"""

import torch

from .context import LossContext


def _weighted_mse(pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    err = (pred.float() - target.float()) ** 2
    return (w.float() * err).reshape(target.shape[0], -1).mean(1).mean()


def _weighted_l1(pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    err = (pred.float() - target.float()).abs()
    return (w.float() * err).reshape(target.shape[0], -1).mean(1).mean()


def _weighted_huber(
    pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor, delta: float = 1.0
) -> torch.Tensor:
    err = pred.float() - target.float()
    abs_err = err.abs()
    quad = torch.where(
        abs_err <= delta, 0.5 * err**2, delta * (abs_err - 0.5 * delta)
    )
    return (w.float() * quad).reshape(target.shape[0], -1).mean(1).mean()


def _weighted_logcosh(pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    err = pred.float() - target.float()
    logcosh = torch.log(torch.cosh(err.clamp(min=-50, max=50)))
    return (w.float() * logcosh).reshape(target.shape[0], -1).mean(1).mean()


BASE_LOSSES = {
    "mse": _weighted_mse,
    "l1": _weighted_l1,
    "huber": _weighted_huber,
    "logcosh": _weighted_logcosh,
}


class FlowMatchingLoss:
    """
    Flow-matching loss: L = weighted_base(pred, target).

    Config:
      base: mse | l1 | huber | logcosh
      huber_delta: float (when base=huber)
    """

    def __init__(self, base: str = "mse", huber_delta: float = 1.0, **kwargs):
        self.base = base.lower()
        if self.base not in BASE_LOSSES:
            raise ValueError(f"base must be one of {list(BASE_LOSSES.keys())}, got {base}")
        self.huber_delta = huber_delta

    def _base_loss(self, pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        fn = BASE_LOSSES[self.base]
        if self.base == "huber":
            return fn(pred, target, w, self.huber_delta)
        return fn(pred, target, w)

    def __call__(self, ctx: LossContext) -> tuple[torch.Tensor, dict[str, float]]:
        pred = ctx.model_output
        target = ctx.noise - ctx.model_input
        w = ctx.weighting
        total = self._base_loss(pred, target, w)
        return total, {"loss": total.detach().item(), "loss_flow": total.detach().item()}
