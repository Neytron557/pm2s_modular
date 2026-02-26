"""
PM2S inference helpers (checkpoint loading + model construction).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import math
import torch
import yaml

from models.pm2s.key_estimator import KeyEstimator
from models.pm2s.rhythm_quantizer import QuantizationConfig, RhythmQuantizer
from models.pm2s.voice_separator import VoiceSeparator
from utils.serialization import safe_torch_load


PathLike = Union[str, Path]


@dataclass
class PM2SComponents:
    rhythm_quantizer: RhythmQuantizer
    voice_separator: VoiceSeparator
    key_estimator: KeyEstimator


def load_yaml_config(path: PathLike) -> dict:
    with Path(path).open("r") as f:
        data = yaml.safe_load(f)
    return data or {}


def resolve_device(device: str = "auto", gpu_id: Optional[int] = None) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    if device.startswith("cuda") and gpu_id is not None:
        return torch.device(f"cuda:{gpu_id}")
    return torch.device(device)


def build_pm2s_components(model_config_path: PathLike) -> PM2SComponents:
    cfg = load_yaml_config(model_config_path)
    pm2s_cfg = cfg.get("pm2s_model", {})

    rq_cfg = pm2s_cfg.get("rhythm_quantizer", {})
    rhythm_quantizer = RhythmQuantizer(config=QuantizationConfig(
        hidden_dim=rq_cfg.get("hidden_dim", 256),
        num_layers=rq_cfg.get("num_layers", 6),
        num_heads=rq_cfg.get("num_heads", 4),
        ff_dim=rq_cfg.get("ffn_dim", 1024),
        dropout=rq_cfg.get("dropout", 0.1),
        max_subdivisions=rq_cfg.get("max_subdivisions", 16),
        triplet_support=rq_cfg.get("triplet_support", True),
        max_seq_len=rq_cfg.get("max_seq_len", 4096),
    ))

    vs_cfg = pm2s_cfg.get("voice_separator", {})
    voice_separator = VoiceSeparator(
        hidden_dim=vs_cfg.get("hidden_dim", 256),
        num_layers=vs_cfg.get("num_layers", 4),
        num_heads=vs_cfg.get("num_heads", 4),
        max_voices=vs_cfg.get("max_voices", 4),
        dropout=vs_cfg.get("dropout", 0.1),
        max_seq_len=vs_cfg.get("max_seq_len", 4096),
    )

    ke_cfg = pm2s_cfg.get("key_estimator", {})
    key_estimator = KeyEstimator(
        hidden_dim=ke_cfg.get("hidden_dim", 256),
        num_keys=ke_cfg.get("num_keys", 30),
    )

    return PM2SComponents(
        rhythm_quantizer=rhythm_quantizer,
        voice_separator=voice_separator,
        key_estimator=key_estimator,
    )


def load_checkpoint_into_model(
    model: torch.nn.Module,
    checkpoint_path: PathLike,
    *,
    device: torch.device,
    strict: bool = True,
) -> None:
    ckpt = safe_torch_load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict") if isinstance(ckpt, dict) else None
    if state_dict is None:
        raise ValueError(f"Checkpoint missing model_state_dict: {checkpoint_path}")

    load_result = model.load_state_dict(state_dict, strict=strict)
    if strict and (load_result.missing_keys or load_result.unexpected_keys):
        raise RuntimeError(
            f"Checkpoint mismatch for {checkpoint_path}.\n"
            f"Missing keys: {load_result.missing_keys}\n"
            f"Unexpected keys: {load_result.unexpected_keys}"
        )

    model.to(device)
    model.eval()


def reconstruct_global_beats(
    beat_number: torch.Tensor,
    beat_fraction: torch.Tensor,
    *,
    beats_per_measure: float,
) -> torch.Tensor:
    """
    Convert per-note (beat_number within measure, beat_fraction within beat) to
    monotonically increasing beat positions by counting wraparounds.

    This assumes notes are ordered in time.

    Args:
        beat_number: (seq,) or (batch, seq) integer beat index within measure
        beat_fraction: (seq,) or (batch, seq) fractional position within beat [0, 1)
        beats_per_measure: number of beats per measure in the same beat units as
            beat_number/beat_fraction (e.g. 4 for 4/4 quarter-note beats; 2 for
            6/8 dotted-quarter beats in pretty_midi)
    """
    if beat_number.ndim == 1:
        beat_number = beat_number.unsqueeze(0)
    if beat_fraction.ndim == 1:
        beat_fraction = beat_fraction.unsqueeze(0)
    if beat_number.shape != beat_fraction.shape:
        raise ValueError(f"beat_number and beat_fraction must have same shape; got {beat_number.shape} vs {beat_fraction.shape}")

    batch, seq_len = beat_number.shape
    out = torch.empty_like(beat_fraction, dtype=torch.float32)
    for b in range(batch):
        bn = beat_number[b].tolist()
        bf = beat_fraction[b].tolist()
        if not bn:
            continue
        measure = 0
        prev_bn = int(bn[0])
        vals = []
        for i, (curr_bn, curr_bf) in enumerate(zip(bn, bf)):
            curr_bn = int(curr_bn)
            if i > 0 and curr_bn < prev_bn:
                measure += 1
            vals.append(measure * float(beats_per_measure) + float(curr_bn) + float(curr_bf))
            prev_bn = curr_bn
        out[b] = torch.tensor(vals, device=beat_fraction.device, dtype=torch.float32)

    return out.squeeze(0) if out.shape[0] == 1 else out


def time_signature_to_beat_params(time_sig: tuple[int, int]) -> tuple[float, int]:
    """
    Return (quarterLength_per_beat, beats_per_measure) matching pretty_midi's beat definition.

    - For simple meters, pretty_midi beats follow the denominator note.
    - For compound meters like 6/8, 9/8, 12/8, pretty_midi beats follow dotted-quarter groupings.
    """
    num, den = int(time_sig[0]), int(time_sig[1])
    if den == 8 and num % 3 == 0 and num > 3:
        return 1.5, max(1, num // 3)  # dotted-quarter beat, e.g. 6/8 -> 2 beats
    return 4.0 / float(den), max(1, num)


def compute_tempo_from_beats(beat_times: list[float]) -> Optional[float]:
    """
    Robust tempo estimate (BPM) from a list of beat times in seconds.

    Uses the median of per-beat instantaneous tempos to reduce sensitivity to
    anacrusis/trailing beats and small jitter.
    """
    if not beat_times or len(beat_times) < 2:
        return None
    intervals: list[float] = []
    for t0, t1 in zip(beat_times[:-1], beat_times[1:]):
        dt = float(t1) - float(t0)
        if dt > 0 and math.isfinite(dt):
            intervals.append(dt)
    if not intervals:
        return None
    intervals.sort()
    dt = intervals[len(intervals) // 2]
    if dt <= 0:
        return None
    return float(60.0 / dt)


def infer_beat_indices_from_onsets(
    onsets_seconds: torch.Tensor,
    beat_fraction: torch.Tensor,
    *,
    tempo_bpm: float,
    enforce_monotonic: bool = True,
) -> torch.Tensor:
    """
    Infer global beat indices (integer) from onsets + predicted beat_fraction.

    This is a structured decoder that does not rely on the (often hard) beat_number
    head to recover absolute position; it instead reconstructs a beat grid from
    tempo and aligns notes to it.

    Returns:
        beat_index: (seq,) or (batch, seq) integer beat index starting at 0.
    """
    if onsets_seconds.ndim == 1:
        onsets_seconds = onsets_seconds.unsqueeze(0)
    if beat_fraction.ndim == 1:
        beat_fraction = beat_fraction.unsqueeze(0)
    if onsets_seconds.shape != beat_fraction.shape:
        raise ValueError(f"onsets_seconds and beat_fraction must have same shape; got {onsets_seconds.shape} vs {beat_fraction.shape}")

    tempo = float(tempo_bpm)
    if not tempo or tempo <= 0 or not math.isfinite(tempo):
        tempo = 120.0
    beat_period = 60.0 / tempo  # seconds per beat (in the model's beat units)

    beat_starts = onsets_seconds - beat_fraction * beat_period
    base = beat_starts.min(dim=1, keepdim=True).values
    beat_index = torch.round((beat_starts - base) / beat_period).to(torch.long)
    beat_index = torch.clamp(beat_index, min=0)
    if enforce_monotonic and beat_index.size(1) > 0:
        beat_index = torch.cummax(beat_index, dim=1).values
    return beat_index.squeeze(0) if beat_index.shape[0] == 1 else beat_index


def choose_phase_shift_from_beat_number_logits(
    beat_index: torch.Tensor,
    beat_number_logits: torch.Tensor,
    *,
    beats_per_measure: int,
) -> int:
    """
    Choose a constant beat-phase shift (in beats) that best matches beat_number logits.

    This aligns the reconstructed global beat grid to measure boundaries without
    affecting relative timing (a constant shift is removed by re-basing metrics).
    """
    bpm = int(beats_per_measure)
    if bpm <= 1:
        return 0

    if beat_index.ndim != 1:
        raise ValueError("beat_index must be 1D for phase selection")
    if beat_number_logits.ndim != 2:
        raise ValueError("beat_number_logits must be (seq, classes)")

    seq_len = beat_index.numel()
    if seq_len == 0:
        return 0

    idx_mod = (beat_index % bpm).to(torch.long)
    logits = beat_number_logits.float()
    best_shift = 0
    best_score = None
    for shift in range(bpm):
        targets = (idx_mod + shift) % bpm
        score = logits[torch.arange(seq_len, device=logits.device), targets].sum().item()
        if best_score is None or score > best_score:
            best_score = score
            best_shift = shift
    return int(best_shift)
