"""
Conformer Encoder for Audio-to-Score

Implements the Conformer architecture (Convolution-augmented Transformer)
which is state-of-the-art for audio processing tasks.

Reference: "Conformer: Convolution-augmented Transformer for Speech Recognition"
https://arxiv.org/abs/2005.08100
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange

from utils.serialization import safe_torch_load

class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding for self-attention."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Create relative position embedding table
        pe = torch.zeros(2 * max_len - 1, d_model)
        position = torch.arange(-(max_len - 1), max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, length: int) -> torch.Tensor:
        """Get relative position embeddings for given length."""
        center = self.max_len - 1
        start = center - length + 1
        end = center + length
        return self.pe[start:end]


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with relative positional encoding."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # Relative positional encoding
        self.pos_enc = RelativePositionalEncoding(d_model)
        self.w_pos = nn.Linear(d_model, d_model, bias=False)

        # Learnable biases for relative attention
        self.u = nn.Parameter(torch.Tensor(num_heads, self.d_head))
        self.v = nn.Parameter(torch.Tensor(num_heads, self.d_head))
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Attention mask (batch, seq_len)

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections
        q = self.w_q(x)  # (batch, seq, d_model)
        k = self.w_k(x)
        v = self.w_v(x)

        # Get relative position embeddings
        pos = self.pos_enc(seq_len)  # (2*seq-1, d_model)
        pos = self.w_pos(pos)  # (2*seq-1, d_model)

        # Reshape for multi-head attention
        q = rearrange(q, "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.num_heads)
        pos = rearrange(pos, "t (h d) -> h t d", h=self.num_heads)

        # Compute content-based attention
        # (q + u) @ k^T
        content_score = torch.einsum("bhid,bhjd->bhij", q + self.u.unsqueeze(0).unsqueeze(2), k)

        # Compute position-based attention
        # (q + v) @ pos^T with relative position shift
        pos_score = torch.einsum("bhid,hjd->bhij", q + self.v.unsqueeze(0).unsqueeze(2), pos)
        pos_score = self._relative_shift(pos_score)

        # Combined attention scores
        scores = (content_score + pos_score) / math.sqrt(self.d_head)

        # Apply mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h t d -> b t (h d)")

        return self.w_o(out)

    def _relative_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Shift relative position scores to align with keys."""
        batch, heads, seq_len, rel_len = x.shape

        # Pad and reshape
        x = F.pad(x, (1, 0))
        x = x.view(batch, heads, rel_len + 1, seq_len)
        x = x[:, :, 1:, :]

        return x[:, :, :seq_len, :]


class ConvolutionModule(nn.Module):
    """
    Convolution module in Conformer.

    Structure: LayerNorm -> Pointwise Conv -> GLU -> Depthwise Conv ->
               BatchNorm -> Swish -> Pointwise Conv -> Dropout
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd"

        self.layer_norm = nn.LayerNorm(d_model)

        # Pointwise expansion (2x for GLU)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)

        # Depthwise convolution
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )

        self.batch_norm = nn.BatchNorm1d(d_model)

        # Pointwise projection
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        x = self.layer_norm(x)

        # Transpose for conv1d: (batch, d_model, seq_len)
        x = x.transpose(1, 2)

        # Pointwise conv with GLU
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)

        # Depthwise conv
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.silu(x)  # Swish activation

        # Pointwise conv
        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        # Transpose back: (batch, seq_len, d_model)
        return x.transpose(1, 2)


class FeedForwardModule(nn.Module):
    """Feed-forward module with expansion and dropout."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = F.silu(x)  # Swish activation
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class ConformerBlock(nn.Module):
    """
    Single Conformer block.

    Structure: FFN -> MHSA -> Conv -> FFN (with residual connections)
    Uses "Macaron-Net" style with half-step FFN before and after.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        conv_dropout: float = 0.1,
    ):
        super().__init__()

        # First half-step feed-forward
        self.ff1 = FeedForwardModule(d_model, d_ff, dropout)

        # Multi-head self-attention
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, attention_dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)

        # Convolution module
        self.conv = ConvolutionModule(d_model, conv_kernel_size, conv_dropout)

        # Second half-step feed-forward
        self.ff2 = FeedForwardModule(d_model, d_ff, dropout)

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Attention mask

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # First FFN (half-step)
        x = x + 0.5 * self.ff1(x)

        # Self-attention
        residual = x
        x = self.attn_norm(x)
        x = self.self_attn(x, mask)
        x = self.attn_dropout(x)
        x = residual + x

        # Convolution
        x = x + self.conv(x)

        # Second FFN (half-step)
        x = x + 0.5 * self.ff2(x)

        # Final normalization
        x = self.final_norm(x)

        return x


class ConformerEncoder(nn.Module):
    """
    Full Conformer Encoder.

    Consists of:
    1. Optional subsampling layer
    2. Linear projection
    3. Stack of Conformer blocks
    """

    def __init__(
        self,
        input_dim: int = 128,
        d_model: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        d_ff: int = 2048,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        conv_dropout: float = 0.1,
        subsampling: bool = True,
        subsampling_factor: int = 4,
        max_seq_len: int = 20000,
    ):
        super().__init__()

        self.d_model = d_model
        self.subsampling = subsampling
        self.max_seq_len = max_seq_len

        # Input projection / subsampling
        if subsampling:
            from .feature_extractor import ConvSubsampling
            self.input_proj = ConvSubsampling(input_dim, d_model, subsampling_factor)
        else:
            self.input_proj = nn.Linear(input_dim, d_model)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # Conformer blocks
        self.layers = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
                attention_dropout=attention_dropout,
                conv_dropout=conv_dropout,
            )
            for _ in range(num_layers)
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input features (batch, seq_len, input_dim)
            lengths: Sequence lengths

        Returns:
            encoder_output: Encoded features (batch, seq_len', d_model)
            output_lengths: Output lengths after subsampling
        """
        # Input projection / subsampling
        if self.subsampling:
            x, lengths = self.input_proj(x, lengths)
        else:
            x = self.input_proj(x)

        # Add positional embedding
        seq_len = x.size(1)
        if seq_len > self.pos_embed.size(1):
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=seq_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        else:
            pos_embed = self.pos_embed[:, :seq_len, :]
        x = x + pos_embed
        x = self.dropout(x)

        # Create attention mask from lengths
        mask = None
        if lengths is not None:
            max_len = x.size(1)
            mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)

        # Pass through Conformer blocks
        for layer in self.layers:
            x = layer(x, mask)

        return x, lengths

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name: str,
        **kwargs,
    ) -> "ConformerEncoder":
        """
        Load pretrained weights.

        Args:
            pretrained_name: Name of pretrained model ("musicfm", "mert", etc.)
            **kwargs: Override config parameters

        Returns:
            ConformerEncoder with loaded weights
        """
        # Default configs for known pretrained models
        pretrained_configs = {
            "musicfm": {
                "d_model": 768,
                "num_layers": 12,
                "num_heads": 12,
                "d_ff": 3072,
            },
            "mert": {
                "d_model": 768,
                "num_layers": 12,
                "num_heads": 12,
                "d_ff": 3072,
            },
        }

        pretrained_path = kwargs.pop("pretrained_path", None)

        if pretrained_name in pretrained_configs:
            config = {**pretrained_configs[pretrained_name], **kwargs}
        else:
            config = kwargs

        model = cls(**config)

        if pretrained_path:
            state = safe_torch_load(pretrained_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(
                    f"Loaded pretrained weights with missing={len(missing)} "
                    f"and unexpected={len(unexpected)} keys"
                )
        else:
            print(
                f"Note: Pretrained weights for '{pretrained_name}' must be provided "
                "via pretrained_path in the checkpoint format used by this encoder."
            )

        return model
