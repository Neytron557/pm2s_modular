"""
Perceiver IO Bottleneck for Audio-to-Score

Compresses variable-length encoder outputs into fixed-length latent representations.
This allows efficient processing of long audio sequences.

Reference: "Perceiver IO: A General Architecture for Structured Inputs & Outputs"
https://arxiv.org/abs/2107.14795
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange
import math


class CrossAttention(nn.Module):
    """Cross-attention layer for Perceiver."""

    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        inner_dim = num_heads * head_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(kv_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: Query tensor (batch, query_len, query_dim)
            context: Context tensor (batch, context_len, kv_dim)
            mask: Optional mask for context

        Returns:
            Output tensor (batch, query_len, query_dim)
        """
        batch, query_len, _ = query.shape
        context_len = context.shape[1]

        # Project
        q = self.to_q(query)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        # Reshape for multi-head attention
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        # Attention scores
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * self.scale

        # Apply mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, context_len)
            scores = scores.masked_fill(~mask, float("-inf"))

        # Softmax and apply to values
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bhkd->bhqd", attn, v)

        # Reshape and project
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class SelfAttention(nn.Module):
    """Self-attention layer for latent processing."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        inner_dim = num_heads * head_dim

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)

        Returns:
            Output tensor (batch, seq_len, dim)
        """
        batch, seq_len, _ = x.shape

        # Project to q, k, v
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
            qkv,
        )

        # Attention
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * self.scale
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bhkd->bhqd", attn, v)

        # Reshape and project
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class FeedForward(nn.Module):
    """Feed-forward network."""

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        hidden_dim = hidden_dim or dim * 4

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerceiverLayer(nn.Module):
    """Single Perceiver layer with cross-attention and self-attention."""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        ff_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_cross_attention: bool = True,
    ):
        super().__init__()

        self.use_cross_attention = use_cross_attention

        if use_cross_attention:
            self.cross_attn_norm_latent = nn.LayerNorm(latent_dim)
            self.cross_attn_norm_context = nn.LayerNorm(context_dim)
            self.cross_attn = CrossAttention(
                query_dim=latent_dim,
                kv_dim=context_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
            )

        self.self_attn_norm = nn.LayerNorm(latent_dim)
        self.self_attn = SelfAttention(
            dim=latent_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

        self.ff_norm = nn.LayerNorm(latent_dim)
        self.ff = FeedForward(latent_dim, ff_hidden_dim, dropout)

    def forward(
        self,
        latents: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            latents: Latent tensor (batch, num_latents, latent_dim)
            context: Context tensor for cross-attention (batch, context_len, context_dim)
            context_mask: Mask for context

        Returns:
            Updated latents (batch, num_latents, latent_dim)
        """
        # Cross-attention (if enabled and context provided)
        if self.use_cross_attention and context is not None:
            latents = latents + self.cross_attn(
                self.cross_attn_norm_latent(latents),
                self.cross_attn_norm_context(context),
                context_mask,
            )

        # Self-attention
        latents = latents + self.self_attn(self.self_attn_norm(latents))

        # Feed-forward
        latents = latents + self.ff(self.ff_norm(latents))

        return latents


class PerceiverIO(nn.Module):
    """
    Perceiver IO module for compressing encoder outputs.

    Uses learnable latent queries to cross-attend to variable-length inputs
    and produces fixed-length outputs.
    """

    def __init__(
        self,
        input_dim: int = 512,
        latent_dim: int = 512,
        num_latents: int = 512,
        num_cross_attention_layers: int = 2,
        num_self_attention_layers: int = 6,
        num_heads: int = 8,
        head_dim: int = 64,
        ff_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_latents = num_latents
        self.latent_dim = latent_dim

        # Learnable latent queries
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)

        # Cross-attention layers (attend to input)
        self.cross_attention_layers = nn.ModuleList([
            PerceiverLayer(
                latent_dim=latent_dim,
                context_dim=input_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                ff_hidden_dim=ff_hidden_dim,
                dropout=dropout,
                use_cross_attention=True,
            )
            for _ in range(num_cross_attention_layers)
        ])

        # Self-attention layers (process latents)
        self.self_attention_layers = nn.ModuleList([
            PerceiverLayer(
                latent_dim=latent_dim,
                context_dim=input_dim,  # Not used
                num_heads=num_heads,
                head_dim=head_dim,
                ff_hidden_dim=ff_hidden_dim,
                dropout=dropout,
                use_cross_attention=False,
            )
            for _ in range(num_self_attention_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            encoder_output: Encoder output (batch, seq_len, input_dim)
            encoder_mask: Mask for encoder output (batch, seq_len)

        Returns:
            Latent representation (batch, num_latents, latent_dim)
        """
        batch_size = encoder_output.shape[0]

        # Expand latents for batch
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)

        # Cross-attention to input
        for layer in self.cross_attention_layers:
            latents = layer(latents, encoder_output, encoder_mask)

        # Self-attention processing
        for layer in self.self_attention_layers:
            latents = layer(latents)

        # Final normalization
        latents = self.final_norm(latents)

        return latents


class PerceiverDecoder(nn.Module):
    """
    Perceiver-style decoder that queries latents to produce output.

    Used to decode latent representation back to sequence for autoregressive decoding.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        output_dim: int = 512,
        num_output_queries: int = 256,
        num_heads: int = 8,
        head_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_output_queries = num_output_queries

        # Learnable output queries
        self.output_queries = nn.Parameter(
            torch.randn(num_output_queries, output_dim) * 0.02
        )

        # Cross-attention layers
        self.layers = nn.ModuleList([
            PerceiverLayer(
                latent_dim=output_dim,
                context_dim=latent_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
                use_cross_attention=True,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(output_dim)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: Latent representation (batch, num_latents, latent_dim)

        Returns:
            Output queries (batch, num_output_queries, output_dim)
        """
        batch_size = latents.shape[0]

        # Expand output queries for batch
        queries = self.output_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Cross-attention to latents
        for layer in self.layers:
            queries = layer(queries, latents)

        return self.final_norm(queries)
