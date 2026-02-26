"""
Learning rate scheduler helpers for Audio-to-Score training.
"""

import math
from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR


def build_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    total_steps: int,
    warmup_steps: int = 0,
    base_lr: Optional[float] = None,
):
    scheduler_type = (scheduler_type or "cosine").lower()

    if scheduler_type in {"cosine", "cosine_with_restarts"}:
        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(max(warmup_steps, 1))
            progress = (step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        if scheduler_type == "cosine_with_restarts":
            print("Note: cosine_with_restarts uses a cosine decay with warmup (no hard restarts).")
        return LambdaLR(optimizer, lr_lambda)

    if scheduler_type == "onecycle":
        if base_lr is None:
            base_lr = optimizer.param_groups[0]["lr"]
        return OneCycleLR(optimizer, max_lr=base_lr, total_steps=total_steps)

    print(f"Unknown scheduler '{scheduler_type}', using constant LR.")
    return LambdaLR(optimizer, lambda _: 1.0)
