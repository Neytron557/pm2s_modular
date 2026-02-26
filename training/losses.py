"""
Loss Functions for Audio-to-Score Training

Includes:
- Focal Loss for handling class imbalance
- Onset Regression Loss for precise timing
- Label Smoothing Cross Entropy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Reference: "Focal Loss for Dense Object Detection"
    https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits (batch, seq_len, vocab_size)
            targets: Target token IDs (batch, seq_len)

        Returns:
            Focal loss value
        """
        # Flatten
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.reshape(-1, vocab_size)
        targets = targets.reshape(-1)

        # Create mask for valid positions
        mask = targets != self.ignore_index

        # Compute cross-entropy
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none", ignore_index=self.ignore_index
        )

        # Compute focal weight
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha if specified
        if self.alpha is not None:
            focal_weight = self.alpha * focal_weight

        # Compute focal loss
        focal_loss = focal_weight * ce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss[mask].mean() if mask.any() else focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss[mask].sum()
        else:
            return focal_loss.view(batch_size, seq_len)


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing."""

    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits (batch, seq_len, vocab_size)
            targets: Target token IDs (batch, seq_len)

        Returns:
            Loss value
        """
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.reshape(-1, vocab_size)
        targets = targets.reshape(-1)

        # Create mask
        mask = targets != self.ignore_index

        # Log softmax
        log_probs = F.log_softmax(logits, dim=-1)

        # Smooth labels
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (vocab_size - 1))
        safe_targets = targets.clone()
        safe_targets[~mask] = 0
        smooth_targets.scatter_(1, safe_targets.unsqueeze(1), 1.0 - self.smoothing)

        # Mask invalid positions
        smooth_targets[~mask] = 0

        # Compute loss
        loss = -(smooth_targets * log_probs).sum(dim=-1)

        if self.reduction == "mean":
            return loss[mask].mean() if mask.any() else loss.mean()
        elif self.reduction == "sum":
            return loss[mask].sum()
        else:
            return loss.view(batch_size, seq_len)


class OnsetRegressionLoss(nn.Module):
    """
    Auxiliary loss for precise onset prediction.

    Predicts the exact onset time offset within each frame.
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
        self.mse = nn.MSELoss(reduction="none")

    def forward(
        self,
        predicted_offsets: torch.Tensor,
        target_offsets: torch.Tensor,
        onset_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predicted_offsets: Predicted onset offsets (batch, seq_len)
            target_offsets: Target onset offsets in [0, 1] (batch, seq_len)
            onset_mask: Mask for onset positions (batch, seq_len)

        Returns:
            Weighted MSE loss
        """
        loss = self.mse(predicted_offsets, target_offsets)
        loss = loss * onset_mask.float()

        if onset_mask.any():
            return self.weight * loss.sum() / onset_mask.sum()
        return torch.tensor(0.0, device=loss.device)


class TranscriptionLoss(nn.Module):
    """
    Combined loss for music transcription training.

    Combines:
    - Focal loss for token prediction
    - Optional onset regression loss
    - MoE auxiliary loss
    """

    def __init__(
        self,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        onset_regression_weight: float = 0.1,
        moe_aux_weight: float = 0.01,
        ignore_index: int = 0,  # PAD token
    ):
        super().__init__()

        self.focal_loss = FocalLoss(
            gamma=focal_gamma,
            ignore_index=ignore_index,
        )
        self.label_smoothing = LabelSmoothingCrossEntropy(
            smoothing=label_smoothing,
            ignore_index=ignore_index,
        )
        self.onset_regression = OnsetRegressionLoss(weight=onset_regression_weight)
        self.moe_aux_weight = moe_aux_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        moe_aux_loss: float = 0.0,
        onset_predictions: Optional[torch.Tensor] = None,
        onset_targets: Optional[torch.Tensor] = None,
        onset_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute combined loss.

        Args:
            logits: Model output logits (batch, seq_len, vocab_size)
            targets: Target token IDs (batch, seq_len)
            moe_aux_loss: Auxiliary MoE load balancing loss
            onset_predictions: Predicted onset offsets (optional)
            onset_targets: Target onset offsets (optional)
            onset_mask: Mask for onset positions (optional)

        Returns:
            Dictionary with loss components and total loss
        """
        # Main transcription loss (focal + label smoothing)
        focal = self.focal_loss(logits, targets)
        smoothing = self.label_smoothing(logits, targets)
        main_loss = 0.5 * focal + 0.5 * smoothing

        # Onset regression loss
        onset_loss = torch.tensor(0.0, device=logits.device)
        if onset_predictions is not None and onset_targets is not None:
            onset_loss = self.onset_regression(
                onset_predictions, onset_targets, onset_mask
            )

        # Total loss
        total_loss = main_loss + onset_loss + self.moe_aux_weight * moe_aux_loss

        return {
            "loss": total_loss,
            "focal_loss": focal,
            "smoothing_loss": smoothing,
            "onset_loss": onset_loss,
            "moe_aux_loss": moe_aux_loss,
        }
