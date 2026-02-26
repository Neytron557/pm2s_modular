"""
Voice Separation Module for PM2S (Stage 2)

Separates polyphonic piano music into distinct voices
and assigns notes to appropriate staves (treble/bass).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class VoiceSeparator(nn.Module):
    """
    Voice Separation Model

    Uses a transformer to predict:
    1. Voice ID for each note (0 to max_voices-1)
    2. Staff assignment (treble/bass)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        max_voices: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_voices = max_voices

        # Note feature embedding - compute dims to sum exactly to hidden_dim
        pitch_dim = hidden_dim // 4
        pos_dim = hidden_dim // 4
        dur_dim = hidden_dim - pitch_dim - pos_dim  # remainder goes here

        self.pitch_embed = nn.Embedding(128, pitch_dim)
        self.position_embed = nn.Linear(2, pos_dim)  # onset, offset
        self.duration_embed = nn.Linear(1, dur_dim)

        self.input_proj = nn.Linear(hidden_dim, hidden_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)

        # Normalize token embeddings before the transformer for mixed-precision stability.
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Pairwise attention for voice linking
        self.pair_query = nn.Linear(hidden_dim, hidden_dim)
        self.pair_key = nn.Linear(hidden_dim, hidden_dim)

        # Output heads
        self.voice_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_voices),
        )

        self.staff_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),  # 0: bass, 1: treble
        )

        # Voice continuity head (same voice as previous note?)
        self.continuity_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        pitches: torch.Tensor,
        onsets: torch.Tensor,
        offsets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_pair_scores: bool = False,
    ) -> dict:
        """
        Forward pass.

        Args:
            pitches: MIDI pitches (batch, seq_len)
            onsets: Onset times (batch, seq_len)
            offsets: Offset times (batch, seq_len)
            mask: Padding mask (batch, seq_len)

        Returns:
            Dictionary with voice and staff predictions
        """
        batch_size, seq_len = pitches.shape

        # Rebase times per sequence to reduce magnitude (chunked inputs can retain
        # absolute MIDI times).
        if seq_len > 0:
            shift = onsets[:, :1]
            onsets = onsets - shift
            offsets = offsets - shift
        offsets = torch.maximum(offsets, onsets)

        # Encode note features
        pitch_emb = self.pitch_embed(pitches)
        pos_features = torch.stack([onsets, offsets], dim=-1)
        pos_emb = self.position_embed(pos_features)
        dur_emb = self.duration_embed((offsets - onsets).unsqueeze(-1))

        x = torch.cat([pitch_emb, pos_emb, dur_emb], dim=-1)
        x = self.input_proj(x)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len]
        if x.dtype in (torch.float16, torch.bfloat16):
            x = self.input_norm(x.float()).to(x.dtype)
        else:
            x = self.input_norm(x)

        # Attention mask
        attn_mask = ~mask if mask is not None else None

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # Voice and staff predictions
        voice_logits = self.voice_head(x)
        staff_logits = self.staff_head(x)

        return {
            "voice_logits": voice_logits,
            "staff_logits": staff_logits,
            "pair_scores": (
                torch.einsum("bih,bjh->bij", self.pair_query(x), self.pair_key(x)) / math.sqrt(self.hidden_dim)
                if return_pair_scores
                else None
            ),
        }

    def separate_voices(
        self,
        pitches: torch.Tensor,
        onsets: torch.Tensor,
        offsets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Separate notes into voices.

        Args:
            pitches: MIDI pitches (batch, seq_len)
            onsets: Onset times (batch, seq_len)
            offsets: Offset times (batch, seq_len)

        Returns:
            voice_ids: Voice assignment for each note (batch, seq_len)
            staff_ids: Staff assignment (0=bass, 1=treble) (batch, seq_len)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(pitches, onsets, offsets)

        voice_ids = outputs["voice_logits"].argmax(dim=-1)
        staff_ids = outputs["staff_logits"].argmax(dim=-1)

        # Apply heuristic refinement
        voice_ids, staff_ids = self._refine_assignments(
            pitches, onsets, offsets, voice_ids, staff_ids
        )

        return voice_ids, staff_ids

    def _refine_assignments(
        self,
        pitches: torch.Tensor,
        onsets: torch.Tensor,
        offsets: torch.Tensor,
        voice_ids: torch.Tensor,
        staff_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rule-based refinement to voice assignments."""
        batch_size, seq_len = pitches.shape

        # Heuristic: split point for staff assignment
        # Middle C (MIDI 60) is the typical split point
        split_point = 60

        for b in range(batch_size):
            for i in range(seq_len):
                pitch = pitches[b, i].item()

                # Override staff based on pitch if far from split point
                if pitch < split_point - 12:  # More than octave below
                    staff_ids[b, i] = 0  # Bass
                elif pitch > split_point + 12:  # More than octave above
                    staff_ids[b, i] = 1  # Treble

        return voice_ids, staff_ids


class StreamSeparator(nn.Module):
    """
    Melodic Stream Separation

    Groups notes into melodic streams based on pitch proximity and voice leading.
    Uses a graph-based approach with learned edge weights.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        max_pitch_gap: int = 12,
        max_time_gap: float = 0.5,
    ):
        super().__init__()

        self.max_pitch_gap = max_pitch_gap
        self.max_time_gap = max_time_gap

        # Edge weight prediction
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # pitch_diff, time_diff, dur1, dur2
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        pitches: torch.Tensor,
        onsets: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute edge weights between consecutive notes.

        Args:
            pitches: MIDI pitches (batch, seq_len)
            onsets: Onset times (batch, seq_len)
            offsets: Offset times (batch, seq_len)

        Returns:
            Edge weights (batch, seq_len-1)
        """
        batch_size, seq_len = pitches.shape

        if seq_len < 2:
            return torch.ones(batch_size, 1, device=pitches.device)

        # Compute pairwise features for consecutive notes
        pitch_diff = (pitches[:, 1:] - pitches[:, :-1]).float().abs()
        time_diff = onsets[:, 1:] - offsets[:, :-1]  # Gap between notes
        dur1 = offsets[:, :-1] - onsets[:, :-1]
        dur2 = offsets[:, 1:] - onsets[:, 1:]

        features = torch.stack([pitch_diff, time_diff, dur1, dur2], dim=-1)
        edge_weights = self.edge_encoder(features).squeeze(-1)

        # Apply hard constraints
        too_far_pitch = pitch_diff > self.max_pitch_gap
        too_far_time = time_diff > self.max_time_gap
        edge_weights = edge_weights * (~too_far_pitch).float() * (~too_far_time).float()

        return edge_weights

    def segment_streams(
        self,
        pitches: torch.Tensor,
        onsets: torch.Tensor,
        offsets: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Segment notes into streams by breaking low-weight edges.

        Returns:
            Stream IDs for each note (batch, seq_len)
        """
        edge_weights = self.forward(pitches, onsets, offsets)

        batch_size, seq_len = pitches.shape
        stream_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=pitches.device)

        for b in range(batch_size):
            current_stream = 0
            for i in range(seq_len):
                if i > 0 and edge_weights[b, i - 1] < threshold:
                    current_stream += 1
                stream_ids[b, i] = current_stream

        return stream_ids
