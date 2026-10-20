"""Learning-rate schedules determined by epoch, independent of held-out scores."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CosineEpochSchedule:
    """Smoothly reduce LR from the initial rate to the minimum at the last epoch.

    Pass this callable to Keras LearningRateScheduler. Absolute epoch indexes
    make uninterrupted and resumed runs follow the same schedule.
    """

    initial_learning_rate: float
    minimum_learning_rate: float
    epochs: int

    def __post_init__(self) -> None:
        """Reject invalid rates and budgets before training or writing provenance."""
        if type(self.epochs) is not int or self.epochs < 1:
            raise ValueError("Schedule epochs must be a positive integer")
        rates = (self.initial_learning_rate, self.minimum_learning_rate)
        if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
            raise ValueError("Schedule learning rates must be finite and positive")
        if self.minimum_learning_rate > self.initial_learning_rate:
            raise ValueError("Minimum learning rate must not exceed the initial rate")

    def __call__(self, epoch: int, learning_rate: float | None = None) -> float:
        """Return the predeclared rate without inspecting losses or predictions."""
        last_epoch = max(1, self.epochs - 1)
        progress = min(max(epoch, 0), self.epochs - 1) / last_epoch
        cosine_weight = 0.5 * (1.0 + math.cos(math.pi * progress))
        rate_range = self.initial_learning_rate - self.minimum_learning_rate
        return self.minimum_learning_rate + rate_range * cosine_weight

    def get_config(self) -> dict[str, float | int]:
        """Persist all schedule parameters for callback and resume contracts."""
        return asdict(self)
