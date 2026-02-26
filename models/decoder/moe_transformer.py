"""
Mixture of Experts (MoE) Transformer Decoder for Audio-to-Score

Implements an autoregressive transformer decoder with MoE layers
for multi-instrument music transcription.

Reference: "YourMT3+: Multi-instrument Music Transcription with Enhanced
Transformer Architectures and Cross-dataset Stem Augmentation"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from einops import rearrange
import math


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 8192):
        super().__init__()

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute sin/cos
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos[:seq_len], self.sin[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to query and key."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Expert(nn.Module):
    """Single expert (feed-forward network)."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoELayer(nn.Module):
    """
    Mixture of Experts layer with Top-K routing.

    Each token is routed to top-k experts based on a learned gating network.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        load_balancing_loss_weight: float = 0.01,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balancing_loss_weight = load_balancing_loss_weight

        # Gating network
        self.gate = nn.Linear(dim, num_experts, bias=False)

        # Expert networks
        self.experts = nn.ModuleList([
            Expert(dim, hidden_dim, dropout)
            for _ in range(num_experts)
        ])

        self._aux_loss = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)

        Returns:
            Output tensor (batch, seq_len, dim)
        """
        batch, seq_len, dim = x.shape
        num_tokens = batch * seq_len

        # Compute gating scores
        gate_logits = self.gate(x)  # (batch, seq_len, num_experts)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Top-k gating
        top_k_probs, top_k_indices = gate_probs.topk(self.top_k, dim=-1)

        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Compute load balancing loss
        self._compute_load_balancing_loss(gate_probs, top_k_indices)

        # Flatten for efficient processing
        x_flat = x.reshape(num_tokens, dim)  # (num_tokens, dim)
        top_k_indices_flat = top_k_indices.reshape(num_tokens, self.top_k)  # (num_tokens, top_k)
        top_k_probs_flat = top_k_probs.reshape(num_tokens, self.top_k)  # (num_tokens, top_k)

        # Apply experts efficiently using batched processing
        output_flat = torch.zeros_like(x_flat)

        for expert_idx in range(self.num_experts):
            # Find all tokens routed to this expert (across all top_k positions)
            expert_mask = (top_k_indices_flat == expert_idx)  # (num_tokens, top_k)

            if expert_mask.any():
                # Get token indices that route to this expert
                token_indices = expert_mask.any(dim=1).nonzero(as_tuple=True)[0]

                if len(token_indices) > 0:
                    # Get inputs for this expert
                    expert_input = x_flat[token_indices]  # (num_expert_tokens, dim)

                    # Process through expert
                    expert_output = self.experts[expert_idx](expert_input)  # (num_expert_tokens, dim)

                    # Get routing weights for these tokens
                    token_expert_mask = expert_mask[token_indices]  # (num_expert_tokens, top_k)
                    token_probs = top_k_probs_flat[token_indices]  # (num_expert_tokens, top_k)

                    # Sum weights across top_k positions where this expert was selected
                    routing_weights = (token_probs * token_expert_mask.float()).sum(dim=1, keepdim=True)

                    # Add weighted output
                    output_flat[token_indices] += expert_output * routing_weights

        return output_flat.reshape(batch, seq_len, dim)

    def _compute_load_balancing_loss(
        self,
        gate_probs: torch.Tensor,
        top_k_indices: torch.Tensor,
    ):
        """Compute auxiliary load balancing loss to ensure even expert usage."""
        # Fraction of tokens routed to each expert
        num_tokens = gate_probs.shape[0] * gate_probs.shape[1]
        expert_counts = torch.zeros(
            self.num_experts, device=gate_probs.device, dtype=gate_probs.dtype
        )
        for k in range(self.top_k):
            indices = top_k_indices[:, :, k].flatten()
            expert_counts.scatter_add_(0, indices, torch.ones_like(indices, dtype=gate_probs.dtype))
        expert_fraction = expert_counts / (num_tokens * self.top_k)

        # Mean routing probability to each expert
        mean_gate_prob = gate_probs.mean(dim=(0, 1))

        # Load balancing loss (variance of expert usage)
        self._aux_loss = self.load_balancing_loss_weight * self.num_experts * (
            expert_fraction * mean_gate_prob
        ).sum()

    def get_aux_loss(self) -> float:
        """Get the auxiliary load balancing loss."""
        return self._aux_loss


class CausalSelfAttention(nn.Module):
    """Causal self-attention with RoPE."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
    ):
        super().__init__()

        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # Rotary positional embedding
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).bool(),
        )

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            past_kv: Cached key-value pairs for incremental decoding
            use_cache: Whether to return updated cache

        Returns:
            output: Output tensor (batch, seq_len, dim)
            present_kv: Updated key-value cache
        """
        batch, seq_len, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = rearrange(q, "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.num_heads)

        # Apply rotary position embedding and handle caching
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.shape[2]
            total_len = past_len + seq_len

            cos, sin = self.rope(total_len)
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)

            cos_slice = cos[:, :, past_len:past_len + seq_len, :]
            sin_slice = sin[:, :, past_len:past_len + seq_len, :]
            q, k = apply_rotary_pos_emb(q, k, cos_slice, sin_slice)

            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        else:
            cos, sin = self.rope(seq_len)
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        present_kv = (k, v) if use_cache else None

        # Attention scores
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * self.scale

        # Apply causal mask
        if past_kv is None:
            causal_mask = self.causal_mask[:seq_len, :seq_len]
            scores = scores.masked_fill(~causal_mask, float("-inf"))

        # Softmax and apply to values
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.einsum("bhqk,bhkd->bhqd", attn, v)
        out = rearrange(out, "b h t d -> b t (h d)")

        return self.out_dropout(self.out_proj(out)), present_kv


class CrossAttention(nn.Module):
    """Cross-attention to encoder outputs."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Decoder input (batch, seq_len, dim)
            encoder_output: Encoder output (batch, enc_len, dim)
            encoder_mask: Mask for encoder output

        Returns:
            Output tensor (batch, seq_len, dim)
        """
        # Project query from decoder, key-value from encoder
        q = self.q_proj(x)
        k, v = self.kv_proj(encoder_output).chunk(2, dim=-1)

        # Reshape for multi-head attention
        q = rearrange(q, "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.num_heads)

        # Attention scores
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * self.scale

        # Apply encoder mask
        if encoder_mask is not None:
            mask = encoder_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, float("-inf"))

        # Softmax and apply to values
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.einsum("bhqk,bhkd->bhqd", attn, v)
        out = rearrange(out, "b h t d -> b t (h d)")

        return self.out_dropout(self.out_proj(out))


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with optional MoE."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        use_moe: bool = False,
        moe_config: Optional[Dict[str, Any]] = None,
        max_seq_len: int = 8192,
    ):
        super().__init__()

        # Self-attention
        self.self_attn_norm = nn.LayerNorm(dim)
        self.self_attn = CausalSelfAttention(dim, num_heads, dropout, max_seq_len)

        # Cross-attention
        self.cross_attn_norm = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads, dropout)

        # Feed-forward (or MoE)
        self.ff_norm = nn.LayerNorm(dim)
        if use_moe:
            moe_config = moe_config or {}
            self.ff = MoELayer(
                dim=dim,
                hidden_dim=moe_config.get("hidden_dim", ff_dim),
                num_experts=moe_config.get("num_experts", 8),
                top_k=moe_config.get("top_k", 2),
                dropout=dropout,
                load_balancing_loss_weight=moe_config.get(
                    "load_balancing_loss_weight", 0.01
                ),
            )
        else:
            self.ff = nn.Sequential(
                nn.Linear(dim, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, dim),
                nn.Dropout(dropout),
            )

        self.use_moe = use_moe

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Decoder input (batch, seq_len, dim)
            encoder_output: Encoder output (batch, enc_len, dim)
            encoder_mask: Mask for encoder output
            past_kv: Cached key-value pairs
            use_cache: Whether to return updated cache

        Returns:
            output: Output tensor
            present_kv: Updated key-value cache
        """
        # Self-attention
        residual = x
        x = self.self_attn_norm(x)
        x, present_kv = self.self_attn(x, past_kv, use_cache)
        x = residual + x

        # Cross-attention
        residual = x
        x = self.cross_attn_norm(x)
        x = self.cross_attn(x, encoder_output, encoder_mask)
        x = residual + x

        # Feed-forward (or MoE)
        residual = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = residual + x

        return x, present_kv

    def get_aux_loss(self) -> float:
        """Get auxiliary loss from MoE layer."""
        if self.use_moe:
            return self.ff.get_aux_loss()
        return 0.0


class MoETransformerDecoder(nn.Module):
    """
    Full MoE Transformer Decoder.

    Features:
    - Autoregressive decoding with causal self-attention
    - Cross-attention to encoder outputs (or Perceiver latents)
    - Mixture of Experts in selected layers
    - Key-value caching for efficient inference
    """

    def __init__(
        self,
        vocab_size: int = 519,
        dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
        moe_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, dim)

        # MoE configuration
        moe_config = moe_config or {
            "enabled": True,
            "num_experts": 8,
            "top_k": 2,
            "moe_layer_freq": 2,
            "hidden_dim": ff_dim,
        }
        moe_layer_freq = moe_config.get("moe_layer_freq", 2)

        # Decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                dim=dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                use_moe=moe_config.get("enabled", True) and (i % moe_layer_freq == moe_layer_freq - 1),
                moe_config=moe_config,
                max_seq_len=max_seq_len,
            )
            for i in range(num_layers)
        ])

        # Output projection
        self.final_norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, vocab_size, bias=False)

        # Tie weights with token embedding
        self.output_proj.weight = self.token_embed.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list], float]:
        """
        Args:
            input_ids: Token IDs (batch, seq_len)
            encoder_output: Encoder output (batch, enc_len, dim)
            encoder_mask: Mask for encoder output
            past_key_values: List of cached key-value pairs
            use_cache: Whether to return updated cache

        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
            present_key_values: Updated key-value cache
            aux_loss: Auxiliary MoE loss
        """
        # Token embeddings
        x = self.token_embed(input_ids)

        # Process through layers
        present_key_values = [] if use_cache else None
        aux_loss = 0.0

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            x, present_kv = layer(x, encoder_output, encoder_mask, past_kv, use_cache)

            if use_cache:
                present_key_values.append(present_kv)

            aux_loss += layer.get_aux_loss()

        # Output projection
        x = self.final_norm(x)
        logits = self.output_proj(x)

        return logits, present_key_values, aux_loss

    def generate(
        self,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        max_length: int = 1024,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        sos_token_id: int = 1,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Args:
            encoder_output: Encoder output
            encoder_mask: Encoder mask
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling (0 = greedy)
            top_p: Nucleus sampling threshold
            sos_token_id: Start of sequence token
            eos_token_id: End of sequence token

        Returns:
            Generated token IDs (batch, seq_len)
        """
        batch_size = encoder_output.shape[0]
        device = encoder_output.device

        # Start with SOS token
        input_ids = torch.full(
            (batch_size, 1), sos_token_id, dtype=torch.long, device=device
        )

        past_key_values = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length - 1):
            # Get next token logits
            logits, past_key_values, _ = self.forward(
                input_ids if past_key_values is None else input_ids[:, -1:],
                encoder_output,
                encoder_mask,
                past_key_values,
                use_cache=True,
            )

            # Get logits for last position
            if temperature <= 0:
                next_logits = logits[:, -1, :]
            else:
                next_logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if temperature > 0 and top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float("-inf")

            # Top-p (nucleus) filtering
            if temperature > 0 and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = float("-inf")

            # Sample or argmax
            probs = F.softmax(next_logits, dim=-1)
            if temperature <= 0 or (top_k == 0 and top_p == 1.0):
                next_token = probs.argmax(dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            # Update finished sequences
            finished = finished | (next_token.squeeze(-1) == eos_token_id)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if finished.all():
                break

        return input_ids
