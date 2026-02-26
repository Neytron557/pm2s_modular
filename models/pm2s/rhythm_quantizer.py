"""
Rhythm Quantization Module for PM2S (Stage 2)

Converts floating-point MIDI timestamps to beat-grid positions.
Predicts time signature and beat subdivisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass
import math


@dataclass
class QuantizationConfig:
    """Configuration for rhythm quantization."""
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 4
    ff_dim: int = 1024
    dropout: float = 0.1
    max_subdivisions: int = 16
    triplet_support: bool = True
    max_seq_len: int = 4096


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class NoteEncoder(nn.Module):
    """Encodes note features for rhythm quantization."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()

        # Input features: pitch, onset_time, duration, velocity, ioi
        self.pitch_embed = nn.Embedding(128, hidden_dim // 4)
        self.time_proj = nn.Linear(3, hidden_dim // 2)  # onset, duration, ioi
        self.velocity_proj = nn.Linear(1, hidden_dim // 4)

        self.combine = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        pitches: torch.Tensor,
        onsets: torch.Tensor,
        durations: torch.Tensor,
        velocities: torch.Tensor,
        iois: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode note features.

        Args:
            pitches: MIDI pitches (batch, seq_len)
            onsets: Onset times in seconds (batch, seq_len)
            durations: Durations in seconds (batch, seq_len)
            velocities: Velocities normalized to [0, 1] (batch, seq_len)
            iois: Inter-onset intervals (batch, seq_len)

        Returns:
            Note embeddings (batch, seq_len, hidden_dim)
        """
        pitch_emb = self.pitch_embed(pitches)

        time_features = torch.stack([onsets, durations, iois], dim=-1)
        time_emb = self.time_proj(time_features)

        velocity_emb = self.velocity_proj(velocities.unsqueeze(-1))

        combined = torch.cat([pitch_emb, time_emb, velocity_emb], dim=-1)
        return self.combine(combined)


class RhythmQuantizer(nn.Module):
    """
    Rhythm Quantization Model

    Uses a bidirectional transformer to predict:
    1. Beat position for each note
    2. Subdivision level (1, 2, 4, 8, 16 or triplets)
    3. Time signature
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        super().__init__()

        self.config = config or QuantizationConfig()

        # Note encoder
        self.note_encoder = NoteEncoder(self.config.hidden_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            self.config.hidden_dim,
            self.config.max_seq_len,
        )

        # Normalize inputs before the transformer for mixed-precision stability.
        self.input_norm = nn.LayerNorm(self.config.hidden_dim)

        # Bidirectional transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.ff_dim,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.num_layers,
        )

        # Output heads

        # Beat position head: predicts beat phase on the unit circle.
        # Representing beat_fraction as (sin, cos) avoids the 0/1 wrap-around
        # pathology of direct regression with MSE.
        self.beat_phase_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 2),  # (sin, cos) logits
        )

        # Beat number head: which beat within the measure
        self.beat_num_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 16),  # Up to 16 beats per measure
        )

        # Subdivision classification
        # Classes: 1, 2, 4, 8, 16, 3 (triplet), 6 (triplet 8th), 12 (triplet 16th)
        num_subdivisions = 8 if self.config.triplet_support else 5
        self.subdivision_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, num_subdivisions),
        )

        # Time signature prediction (global)
        # Common time signatures: 2/4, 3/4, 4/4, 6/8, 9/8, 12/8
        self.time_sig_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 6),
        )

        # Tempo estimation (global): predict log-tempo, then exponentiate.
        # This avoids hard floors (e.g., +60 BPM) and is stable with a log loss.
        self.tempo_mlp = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
        )
        self.log_tempo_out = nn.Linear(self.config.hidden_dim, 1)
        with torch.no_grad():
            # Initialize near a reasonable prior to avoid extreme early decoding.
            self.log_tempo_out.bias.fill_(math.log(120.0))

        self.subdivision_map = [1, 2, 4, 8, 16, 3, 6, 12]
        self.time_sig_map = [(2, 4), (3, 4), (4, 4), (6, 8), (9, 8), (12, 8)]

    def forward(
        self,
        pitches: torch.Tensor,
        onsets: torch.Tensor,
        durations: torch.Tensor,
        velocities: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass.

        Args:
            pitches: MIDI pitches (batch, seq_len)
            onsets: Onset times in seconds (batch, seq_len)
            durations: Durations in seconds (batch, seq_len)
            velocities: Normalized velocities (batch, seq_len)
            mask: Padding mask (batch, seq_len), True for valid positions

        Returns:
            Dictionary with predictions
        """
        batch_size, seq_len = pitches.shape

        # Rebase times per sequence (helps with stability when chunking keeps absolute
        # MIDI times and avoids very large onset values).
        if seq_len > 0:
            onsets = onsets - onsets[:, :1]

        # Compute inter-onset intervals
        iois = torch.zeros_like(onsets)
        iois[:, 1:] = onsets[:, 1:] - onsets[:, :-1]

        # Encode notes
        x = self.note_encoder(pitches, onsets, durations, velocities, iois)
        x = self.pos_encoding(x)
        if x.dtype in (torch.float16, torch.bfloat16):
            x = self.input_norm(x.float()).to(x.dtype)
        else:
            x = self.input_norm(x)

        # Create attention mask for padding
        attn_mask = None
        if mask is not None:
            attn_mask = ~mask  # PyTorch uses True for masked positions

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # Note-level predictions
        beat_phase = self.beat_phase_head(x)  # (batch, seq, 2)
        beat_phase = F.normalize(beat_phase.float(), dim=-1, eps=1e-8).to(x.dtype)
        sin_phase = beat_phase[..., 0].float()
        cos_phase = beat_phase[..., 1].float()
        angle = torch.atan2(sin_phase, cos_phase)  # [-pi, pi]
        beat_frac = torch.remainder(angle / (2.0 * math.pi), 1.0).to(x.dtype)  # [0, 1)
        beat_num = self.beat_num_head(x)  # (batch, seq, 16)
        subdivision = self.subdivision_head(x)  # (batch, seq, num_subdivisions)

        # Global predictions (mean pool over valid positions)
        if mask is not None:
            x_masked = x * mask.unsqueeze(-1).float()
            x_pooled = x_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            x_pooled = x.mean(dim=1)

        time_sig = self.time_sig_head(x_pooled)  # (batch, 6)
        log_tempo = self.log_tempo_out(self.tempo_mlp(x_pooled)).squeeze(-1).float()
        tempo = torch.exp(log_tempo).clamp(min=1e-3).to(x.dtype)

        return {
            "beat_fraction": beat_frac,
            "beat_phase": beat_phase,
            "beat_number": beat_num,
            "subdivision": subdivision,
            "time_signature": time_sig,
            "tempo": tempo,
            "log_tempo": log_tempo,
        }

    def quantize(
        self,
        pitches: torch.Tensor,
        onsets: torch.Tensor,
        durations: torch.Tensor,
        velocities: torch.Tensor,
        *,
        time_scale: float = 1.0,
        tempo_bpm_override: Optional[float] = None,
        time_signature_override: Optional[Tuple[int, int]] = None,
        mask: Optional[torch.Tensor] = None,
        align_to_measure: bool = True,
    ) -> dict:
        """
        Quantize notes to beat grid.

        Returns:
            Dictionary with quantized values
        """
        self.eval()
        with torch.no_grad():
            ts = float(time_scale)
            if not ts or ts <= 0:
                ts = 1.0
            if mask is None:
                mask = torch.ones_like(pitches, dtype=torch.bool, device=pitches.device)
            outputs = self.forward(pitches, onsets / ts, durations / ts, velocities, mask)

        # Get predictions
        beat_frac = outputs["beat_fraction"]
        beat_num = outputs["beat_number"].argmax(dim=-1)
        subdivision = outputs["subdivision"].argmax(dim=-1)
        time_sig_idx = outputs["time_signature"].argmax(dim=-1)
        tempo = outputs["tempo"]
        model_tempo = tempo
        if tempo_bpm_override is not None:
            try:
                override = float(tempo_bpm_override)
            except Exception:
                override = None
            if override is not None and math.isfinite(override) and override > 0:
                tempo = torch.full_like(tempo, override)

        # Convert to beat positions
        subdivisions = torch.tensor(
            [self.subdivision_map[s] for s in subdivision.flatten()],
            device=beat_frac.device,
        ).view_as(subdivision)

        # Quantize to grid
        quantized_frac = torch.round(beat_frac * subdivisions) / subdivisions
        # Beat phase is circular; rounding can produce exactly 1.0 which should wrap to 0.0.
        quantized_frac = torch.remainder(quantized_frac, 1.0)

        # Structured decoding:
        # Infer global beat indices from (onsets, beat_fraction, tempo) and optionally
        # align the grid phase to the beat_number logits.
        if mask is None:
            mask = torch.ones_like(pitches, dtype=torch.bool, device=pitches.device)

        tempo_safe = tempo.float().clamp(min=1e-3)
        beat_period = 60.0 / tempo_safe  # seconds per beat (in beat units)
        beat_starts = onsets - quantized_frac * beat_period.unsqueeze(-1)
        beat_starts_masked = beat_starts.masked_fill(~mask, float("inf"))
        base = beat_starts_masked.min(dim=1, keepdim=True).values
        base = torch.where(torch.isfinite(base), base, torch.zeros_like(base))

        beat_index = torch.round((beat_starts - base) / beat_period.unsqueeze(-1)).to(torch.long)
        beat_index = torch.clamp(beat_index, min=0)
        if beat_index.size(1) > 0:
            beat_index = torch.cummax(beat_index, dim=1).values

        # Get time signatures and beat-unit scaling in quarterLength units.
        model_time_sigs = [self.time_sig_map[int(idx)] for idx in time_sig_idx.tolist()]
        time_sigs = list(model_time_sigs)
        if time_signature_override is not None:
            try:
                num, den = int(time_signature_override[0]), int(time_signature_override[1])
            except Exception:
                num, den = None, None
            if num is not None and den is not None and num > 0 and den > 0:
                time_sigs = [(num, den) for _ in time_sigs]
        beat_unit_ql: List[float] = []
        beats_per_measure: List[int] = []
        for tsig in time_sigs:
            num, den = int(tsig[0]), int(tsig[1])
            if den == 8 and num % 3 == 0 and num > 3:
                beat_unit_ql.append(1.5)  # dotted-quarter
                beats_per_measure.append(max(1, num // 3))
            else:
                beat_unit_ql.append(4.0 / float(den))
                beats_per_measure.append(max(1, num))

        phase_shift = torch.zeros(beat_index.size(0), device=beat_index.device, dtype=torch.long)
        if align_to_measure and beat_index.numel() > 0:
            logits = outputs["beat_number"].float()  # (batch, seq, 16)
            for b in range(beat_index.size(0)):
                bpm = int(beats_per_measure[b])
                if bpm <= 1:
                    continue
                idx_mod = (beat_index[b] % bpm).to(torch.long)
                valid = mask[b]
                if valid.any():
                    idx_mod = idx_mod[valid]
                    l = logits[b][valid]
                else:
                    continue
                best_shift = 0
                best_score = None
                ar = torch.arange(idx_mod.numel(), device=beat_index.device)
                for shift in range(bpm):
                    targets = (idx_mod + shift) % bpm
                    score = l[ar, targets].sum().item()
                    if best_score is None or score > best_score:
                        best_score = score
                        best_shift = shift
                phase_shift[b] = int(best_shift)

        beat_unit_tensor = torch.tensor(beat_unit_ql, device=beat_index.device, dtype=torch.float32).unsqueeze(-1)
        quantized_beat_beats = beat_index.to(torch.float32) + phase_shift.unsqueeze(-1).to(torch.float32) + quantized_frac
        quantized_beat = quantized_beat_beats * beat_unit_tensor

        return {
            "quantized_beats": quantized_beat,
            "beat_index": beat_index,
            "beat_unit_quarter_length": beat_unit_tensor.squeeze(-1),
            "beats_per_measure": beats_per_measure,
            "beat_number": beat_num,
            "beat_fraction": quantized_frac,
            "subdivision": subdivisions,
            "time_signature": time_sigs,
            "model_time_signature": model_time_sigs,
            "tempo": tempo,
            "model_tempo": model_tempo,
        }


class DurationQuantizer(nn.Module):
    """
    Quantizes note durations to musical note values.

    Predicts note type (whole, half, quarter, etc.) and dot status.
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()

        # Note types: whole, half, quarter, 8th, 16th, 32nd
        # Plus dotted versions
        self.num_duration_types = 12

        self.encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # beat_pos, duration, tempo, time_sig_denom
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.duration_head = nn.Linear(hidden_dim, self.num_duration_types)

        # Duration types in beats
        self.duration_map = {
            0: 4.0,    # whole
            1: 6.0,    # dotted whole
            2: 2.0,    # half
            3: 3.0,    # dotted half
            4: 1.0,    # quarter
            5: 1.5,    # dotted quarter
            6: 0.5,    # 8th
            7: 0.75,   # dotted 8th
            8: 0.25,   # 16th
            9: 0.375,  # dotted 16th
            10: 0.125, # 32nd
            11: 0.1875,# dotted 32nd
        }

    def forward(
        self,
        beat_positions: torch.Tensor,
        durations: torch.Tensor,
        tempo: torch.Tensor,
        time_sig_denom: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict duration types.

        Args:
            beat_positions: Beat positions (batch, seq_len)
            durations: Durations in seconds (batch, seq_len)
            tempo: Tempo in BPM (batch,)
            time_sig_denom: Time signature denominator (batch,)

        Returns:
            Duration type logits (batch, seq_len, num_duration_types)
        """
        batch_size, seq_len = beat_positions.shape

        # Convert duration to beats
        beat_duration = durations * tempo.unsqueeze(-1) / 60

        # Create input features
        tempo_exp = tempo.unsqueeze(-1).expand(-1, seq_len)
        denom_exp = time_sig_denom.unsqueeze(-1).expand(-1, seq_len).float()

        features = torch.stack([
            beat_positions,
            beat_duration,
            tempo_exp / 120,  # Normalize tempo
            denom_exp / 4,    # Normalize denominator
        ], dim=-1)

        x = self.encoder(features)
        return self.duration_head(x)
