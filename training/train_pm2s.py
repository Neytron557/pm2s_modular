"""
Training script for PM2S components.

Requires a manifest (.pt or .jsonl) containing note sequences and targets.
"""

import argparse
import json
import math
import os
from collections import Counter
from functools import partial
from pathlib import Path
from contextlib import nullcontext
from typing import Dict, List, Optional, Callable, Any, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, random_split, Sampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import yaml

from models.pm2s.rhythm_quantizer import RhythmQuantizer, QuantizationConfig
from models.pm2s.voice_separator import VoiceSeparator
from models.pm2s.key_estimator import KeyEstimator
from utils.serialization import safe_torch_load


class PM2SDataset(Dataset):
    def __init__(self, items: List[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        return {
            k: torch.tensor(v) if isinstance(v, (list, tuple)) else v
            for k, v in item.items()
        }


def load_manifest(path: Path) -> List[dict]:
    if path.suffix == ".pt":
        data = safe_torch_load(str(path), map_location="cpu", weights_only=False)
        if isinstance(data, dict) and "items" in data:
            return data["items"]
        if isinstance(data, list):
            return data
        raise ValueError("Unsupported .pt manifest format")

    if path.suffix == ".jsonl":
        items = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    raise ValueError("Manifest must be .pt or .jsonl")


def load_yaml_config(path: Optional[str]) -> dict:
    if not path:
        return {}
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}


def resolve_value(cli_value, cfg_value, default):
    if cli_value is not None:
        return cli_value
    if cfg_value is not None:
        return cfg_value
    return default


def parse_gpu_ids(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = [p.strip() for p in text.split(",")] if "," in text else text.split()
    gpu_ids: List[int] = []
    for part in parts:
        if not part:
            continue
        try:
            gpu_ids.append(int(part))
        except ValueError as exc:
            raise ValueError(f"Invalid --gpu_ids value '{part}'. Use comma-separated integers.") from exc
    return gpu_ids or None


def ddp_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> LambdaLR:
    num_training_steps = max(1, int(num_training_steps))
    warmup_steps = max(0, int(warmup_steps))
    warmup_steps = min(warmup_steps, num_training_steps)
    min_lr_ratio = float(min_lr_ratio)
    min_lr_ratio = max(0.0, min(min_lr_ratio, 1.0))

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        if num_training_steps <= warmup_steps:
            return 1.0
        progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


class WeightedDistributedSampler(Sampler):
    """Distributed sampler that performs weighted sampling with replacement."""

    def __init__(
        self,
        weights: torch.Tensor,
        num_replicas: int,
        rank: int,
        seed: int = 42,
    ):
        if num_replicas <= 0:
            raise ValueError("num_replicas must be > 0")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.weights = weights.detach().double().cpu()
        if self.weights.numel() == 0:
            raise ValueError("weights must be non-empty")
        if torch.any(self.weights < 0):
            raise ValueError("weights must be non-negative")
        if float(self.weights.sum()) <= 0:
            raise ValueError("weights must sum to > 0")

        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0

        dataset_len = int(self.weights.numel())
        self.num_samples = int(math.ceil(dataset_len / self.num_replicas))
        self.total_size = int(self.num_samples * self.num_replicas)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        sampled = torch.multinomial(self.weights, self.total_size, replacement=True, generator=g).tolist()
        indices = sampled[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


DATASET_NAME_ALIASES = {
    "pdmx_piano_synth": "pdmx",
    "atepp_piano_synth": "atepp",
    "msmd_piano_synth": "msmd",
    "maestro_midi_piano": "maestro",
    "lakh_midi_piano": "lakh",
    "musicnet_piano": "musicnet",
}


def canonicalize_dataset_name(name: str) -> str:
    name = str(name).strip().lower()
    if not name:
        return name
    return DATASET_NAME_ALIASES.get(name, name)


MANIFEST_STATE_TASK_KEYS = {
    "rhythm_quantizer": "rhythm",
    "voice_separator": "voice",
    "key_estimator": "key",
}


def backfill_manifest_sources_from_state(
    items: List[dict],
    manifest_path: Optional[Path],
    task: str,
    is_main_process: bool,
) -> bool:
    if not isinstance(items, list) or not items:
        return False
    if manifest_path is None:
        return False

    state_task = MANIFEST_STATE_TASK_KEYS.get(task)
    if state_task is None:
        return False

    state_path = manifest_path.parent / "manifest_state.json"
    if not state_path.exists():
        return False

    try:
        with state_path.open("r") as f:
            state = json.load(f)
    except Exception:
        return False

    task_state = state.get("tasks", {}).get(state_task)
    if not isinstance(task_state, dict):
        return False

    segments = []
    total_items = 0
    for dataset_name, dataset_state in task_state.items():
        if not isinstance(dataset_state, dict):
            continue
        try:
            count = int(dataset_state.get("items", 0))
        except (TypeError, ValueError):
            continue
        if count <= 0:
            continue
        segments.append((canonicalize_dataset_name(dataset_name), count))
        total_items += count

    if total_items != len(items):
        return False

    idx = 0
    for dataset_name, count in segments:
        end = idx + count
        for j in range(idx, end):
            item = items[j]
            if isinstance(item, dict):
                item.setdefault("source", dataset_name)
        idx = end

    if is_main_process:
        print(f"Backfilled missing manifest 'source' field using {state_path}.")
    return True


def parse_dataset_weights(datasets_cfg: object) -> Dict[str, float]:
    if not isinstance(datasets_cfg, list):
        return {}
    weights: Dict[str, float] = {}
    for entry in datasets_cfg:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not name:
            continue
        weight = entry.get("weight", 0.0)
        try:
            weight_value = float(weight)
        except (TypeError, ValueError):
            continue
        if weight_value <= 0:
            continue
        canonical_name = canonicalize_dataset_name(name)
        weights[canonical_name] = weights.get(canonical_name, 0.0) + weight_value
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in weights.items()}


def build_adamw_optimizer(model: nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    decay: List[torch.Tensor] = []
    no_decay: List[torch.Tensor] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name_lower = name.lower()
        if param.ndim < 2 or name_lower.endswith(".bias") or "norm" in name_lower:
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups = []
    if decay:
        param_groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if no_decay:
        param_groups.append({"params": no_decay, "weight_decay": 0.0})

    return torch.optim.AdamW(param_groups, lr=float(learning_rate))


def apply_timing_jitter(
    item: dict,
    onset_jitter_std: float = 0.05,
    duration_jitter_std: float = 0.03,
    jitter_sources: Optional[List[str]] = None,
) -> dict:
    """
    Apply timing jitter augmentation to a training sample.

    Perturbs onsets, durations, and velocities with Gaussian noise while
    keeping targets unchanged. Re-sorts by onset to maintain monotonicity.

    Args:
        item: A single manifest item dict with 'onsets', 'durations', 'velocities', etc.
        onset_jitter_std: Std dev of onset jitter as fraction of beat duration.
            Applied as: onset += N(0, sigma) where sigma = onset_jitter_std * mean_ioi.
        duration_jitter_std: Std dev of multiplicative duration jitter.
            Applied as: duration *= (1 + N(0, duration_jitter_std)).
        jitter_sources: List of dataset source names to apply jitter to.
            None or ["all"] applies to all sources.

    Returns:
        Modified item dict (new object, original not mutated).
    """
    import random as _rand

    # Check if this source should be jittered
    if jitter_sources is not None:
        source = canonicalize_dataset_name(str(item.get("source", "")))
        if "all" not in jitter_sources and source not in jitter_sources:
            return item

    onsets = item.get("onsets")
    durations = item.get("durations")
    velocities = item.get("velocities")

    if onsets is None or not len(onsets):
        return item

    # Convert to lists if tensors
    if torch.is_tensor(onsets):
        onsets = onsets.tolist()
    else:
        onsets = list(onsets)
    if durations is not None:
        if torch.is_tensor(durations):
            durations = durations.tolist()
        else:
            durations = list(durations)
    if velocities is not None:
        if torch.is_tensor(velocities):
            velocities = velocities.tolist()
        else:
            velocities = list(velocities)

    n = len(onsets)

    # Compute onset jitter scale from mean inter-onset interval
    if n > 1 and onset_jitter_std > 0:
        iois = [onsets[i + 1] - onsets[i] for i in range(n - 1) if onsets[i + 1] > onsets[i]]
        mean_ioi = sum(iois) / len(iois) if iois else 0.5
        sigma = onset_jitter_std * mean_ioi
        for i in range(n):
            onsets[i] += _rand.gauss(0, sigma)
        # Ensure non-negative
        onsets = [max(0.0, o) for o in onsets]

    # Duration jitter
    if durations is not None and duration_jitter_std > 0:
        for i in range(n):
            factor = 1.0 + _rand.gauss(0, duration_jitter_std)
            durations[i] = max(0.01, durations[i] * factor)

    # Velocity jitter (small)
    if velocities is not None:
        vel_std = 0.05
        for i in range(n):
            velocities[i] = max(0.0, min(1.0, velocities[i] + _rand.gauss(0, vel_std)))

    # Re-sort by onset to maintain monotonicity (carry all per-note fields)
    seq_fields = ["pitches", "onsets", "durations", "velocities", "offsets",
                  "beat_fraction", "beat_number", "subdivision",
                  "voice_ids", "staff_ids"]

    indices = sorted(range(n), key=lambda i: onsets[i])

    new_item = dict(item)
    new_item["onsets"] = [onsets[indices[i]] for i in range(n)]
    if durations is not None:
        new_item["durations"] = [durations[indices[i]] for i in range(n)]
    if velocities is not None:
        new_item["velocities"] = [velocities[indices[i]] for i in range(n)]

    # Reorder offsets
    if "offsets" in item:
        offs = item["offsets"]
        if torch.is_tensor(offs):
            offs = offs.tolist()
        else:
            offs = list(offs)
        # Recompute offsets from jittered onsets + durations if both available
        if durations is not None:
            new_item["offsets"] = [new_item["onsets"][i] + new_item["durations"][i] for i in range(n)]
        else:
            new_item["offsets"] = [offs[indices[i]] for i in range(n)]

    # Reorder all other per-note fields
    for field in seq_fields:
        if field in ("onsets", "durations", "velocities", "offsets"):
            continue
        if field in item:
            vals = item[field]
            if torch.is_tensor(vals):
                vals = vals.tolist()
            else:
                vals = list(vals)
            if len(vals) == n:
                new_item[field] = [vals[indices[i]] for i in range(n)]

    return new_item


def collate_pm2s(
    batch: List[dict],
    max_seq_len: Optional[int] = None,
    random_crop: bool = False,
    onset_jitter_std: float = 0.0,
    duration_jitter_std: float = 0.0,
    jitter_sources: Optional[List[str]] = None,
) -> dict:
    if not batch:
        return {}

    # Apply timing jitter augmentation if enabled
    if onset_jitter_std > 0 or duration_jitter_std > 0:
        batch = [
            apply_timing_jitter(item, onset_jitter_std, duration_jitter_std, jitter_sources)
            for item in batch
        ]

    seq_fields = ["pitches", "onsets", "durations", "velocities", "offsets",
                  "beat_fraction", "beat_number", "subdivision",
                  "voice_ids", "staff_ids"]
    if max_seq_len is not None:
        max_seq_len = int(max_seq_len)
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0")
        cropped_batch: List[dict] = []
        for item in batch:
            seq_len = len(item.get("pitches", []))
            if seq_len > max_seq_len:
                if random_crop:
                    start = int(torch.randint(0, seq_len - max_seq_len + 1, (1,)).item())
                else:
                    start = 0
                end = start + max_seq_len
                cropped = dict(item)
                for field in seq_fields:
                    if field in cropped:
                        values = cropped[field]
                        cropped[field] = values[start:end] if not torch.is_tensor(values) else values[start:end]
                cropped_batch.append(cropped)
            else:
                cropped_batch.append(item)
        batch = cropped_batch

    max_len = max(len(b.get("pitches", [])) for b in batch)

    out = {}
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, item in enumerate(batch):
        seq_len = len(item.get("pitches", []))
        mask[i, :seq_len] = True
        for field in seq_fields:
            if field in item:
                values = item[field]
                if not torch.is_tensor(values):
                    values = torch.tensor(values)
                padded = torch.zeros(max_len, dtype=values.dtype)
                padded[:seq_len] = values
                out.setdefault(field, []).append(padded)

    for field, values in out.items():
        out[field] = torch.stack(values, dim=0)
    out["mask"] = mask

    for field in ["tempo", "time_signature", "key_index"]:
        if field in batch[0]:
            out[field] = torch.tensor([b[field] for b in batch])

    return out


class PM2STrainer:
    """Generic trainer for PM2S components."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        loss_fn: Callable,
        device: torch.device,
        learning_rate: float = 1e-4,
        epochs: int = 50,
        checkpoint_dir: str = "checkpoints/pm2s",
        task_name: str = "pm2s",
        log_every_n_steps: int = 100,
        is_main_process: bool = True,
        ddp_enabled: bool = False,
        grad_accum_steps: int = 1,
        amp_enabled: bool = False,
        amp_dtype: Optional[torch.dtype] = None,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.01,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
        save_every_epochs: int = 10,
        wandb_run=None,
        best_swd_eval: Optional[Dict[str, Any]] = None,
        early_stopping_patience: int = 0,
    ):
        if isinstance(model, DDP):
            self.model = model
        else:
            self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.device = device
        self.epochs = epochs
        self.task_name = task_name
        self.log_every = log_every_n_steps
        self.is_main_process = is_main_process
        self.ddp_enabled = ddp_enabled
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.amp_enabled = amp_enabled and device.type == "cuda"
        self.amp_dtype = amp_dtype if self.amp_enabled else None
        self.use_scaler = self.amp_enabled and self.amp_dtype == torch.float16
        self.save_every_epochs = max(1, save_every_epochs)
        self.wandb_run = wandb_run
        self.max_grad_norm = float(max_grad_norm)
        self.best_swd_eval = best_swd_eval
        self.early_stopping_patience = max(0, int(early_stopping_patience))
        self._best_swd_cache: Optional[Dict[str, Any]] = None

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = build_adamw_optimizer(self.model, learning_rate=learning_rate, weight_decay=weight_decay)
        steps_per_epoch = max(1, math.ceil(len(train_loader) / self.grad_accum_steps))
        t_max = max(1, epochs * steps_per_epoch)
        self.scheduler = build_warmup_cosine_scheduler(
            self.optimizer,
            num_training_steps=t_max,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
        )
        if self.use_scaler:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        self.best_val_loss = float("inf")
        self.global_step = 0

    def _barrier_if_needed(self) -> None:
        if not self.ddp_enabled or not ddp_is_initialized():
            return
        if self.device.type == "cuda":
            dist.barrier(device_ids=[self.device.index])
        else:
            dist.barrier()

    def _predominant_time_signature(self, midi: Any) -> Tuple[int, int]:
        changes = [
            (float(ts.time), (int(ts.numerator), int(ts.denominator)))
            for ts in getattr(midi, "time_signature_changes", [])
        ]
        if not changes:
            return (4, 4)
        changes.sort(key=lambda x: x[0])
        end_time = float(getattr(midi, "get_end_time")() or 0.0)
        if end_time <= 0:
            return tuple(changes[0][1])
        acc: Dict[Tuple[int, int], float] = {}
        for i, (t0, sig) in enumerate(changes):
            t1 = float(changes[i + 1][0]) if i + 1 < len(changes) else end_time
            dur = max(0.0, float(t1) - float(t0))
            acc[tuple(sig)] = acc.get(tuple(sig), 0.0) + dur
        return max(acc.items(), key=lambda kv: kv[1])[0]

    def _load_best_swd_cache(self, midi_path: Path, xml_path: Path) -> Dict[str, Any]:
        if self._best_swd_cache is not None:
            return self._best_swd_cache

        # Lazily import heavy deps so training doesn't require them unless enabled.
        try:
            import pretty_midi  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("pretty_midi is required for --eval-best-swd09") from exc

        try:
            from music21 import converter  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("music21 is required for --eval-best-swd09") from exc

        from evaluation.metrics import NoteEvent as EvalNoteEvent  # noqa: N811

        def score_to_eval_notes(score_obj: Any) -> List[EvalNoteEvent]:
            try:
                score_obj = score_obj.stripTies(inPlace=False)
            except Exception:
                pass
            flat = score_obj.flatten()
            notes: List[EvalNoteEvent] = []
            for el in flat.recurse().notes:
                onset = float(el.offset)
                dur = float(getattr(el.duration, "quarterLength", 0.0) or 0.0)
                if dur <= 0:
                    continue
                offset = onset + dur
                if el.isChord:
                    for p in el.pitches:
                        notes.append(EvalNoteEvent(onset=onset, offset=offset, pitch=int(p.midi), velocity=64))
                else:
                    notes.append(EvalNoteEvent(onset=onset, offset=offset, pitch=int(el.pitch.midi), velocity=64))
            return notes

        def rebase_notes(notes: List[EvalNoteEvent]) -> List[EvalNoteEvent]:
            if not notes:
                return []
            t0 = min(n.onset for n in notes)
            if t0 == 0:
                return notes
            return [
                EvalNoteEvent(onset=n.onset - t0, offset=n.offset - t0, pitch=n.pitch, velocity=n.velocity)
                for n in notes
            ]

        if not midi_path.exists():
            raise FileNotFoundError(f"SWD MIDI not found: {midi_path}")
        if not xml_path.exists():
            raise FileNotFoundError(f"SWD MusicXML not found: {xml_path}")

        midi = pretty_midi.PrettyMIDI(str(midi_path))
        notes_raw = []
        for inst in midi.instruments:
            if inst.is_drum:
                continue
            for n in inst.notes:
                notes_raw.append((int(n.pitch), float(n.start), float(n.end - n.start), float(n.velocity) / 127.0))
        if not notes_raw:
            raise ValueError(f"No notes found in MIDI: {midi_path}")
        notes_raw.sort(key=lambda x: (x[1], x[0]))
        pitches, onsets, durations, velocities = zip(*notes_raw)

        # Tempo in "beat BPM" (pretty_midi beat unit). This matches existing inference code.
        from utils.pm2s_inference import compute_tempo_from_beats

        tempo_bpm = compute_tempo_from_beats(midi.get_beats().tolist()) or 120.0
        time_sig = self._predominant_time_signature(midi)

        ref_score = converter.parse(str(xml_path))
        ref_notes = rebase_notes(score_to_eval_notes(ref_score))

        base_score = converter.parse(str(midi_path)).makeNotation(inPlace=False)
        base_notes = rebase_notes(score_to_eval_notes(base_score))

        self._best_swd_cache = {
            "midi_item": {
                "pitches": list(pitches),
                "onsets": list(onsets),
                "durations": list(durations),
                "velocities": list(velocities),
                "midi_tempo_bpm": float(tempo_bpm),
                "midi_time_signature": tuple(time_sig),
            },
            "ref_notes": ref_notes,
            "baseline_notes": base_notes,
        }
        return self._best_swd_cache

    def _evaluate_best_swd09(self, ckpt_path: Path) -> None:
        cfg = self.best_swd_eval or {}
        if not cfg.get("enabled"):
            return
        if self.task_name != "rhythm_quantizer":
            if self.is_main_process:
                print("[warn] --eval-best-swd09 is only supported for --task rhythm_quantizer (skipping).")
            return
        if not self.is_main_process:
            return

        midi_path = Path(str(cfg.get("midi_path", "")))
        xml_path = Path(str(cfg.get("xml_path", "")))
        tempo_source = str(cfg.get("tempo_source", "midi")).strip().lower()
        time_signature_source = str(cfg.get("time_signature_source", "midi")).strip().lower()
        time_scale = float(cfg.get("time_scale", 100.0))

        try:
            cache = self._load_best_swd_cache(midi_path, xml_path)
        except Exception as exc:
            print(f"[warn] best@SWD09 cache init failed: {type(exc).__name__}: {exc}")
            return

        from evaluation.metrics import frame_f1, note_f1
        from evaluation.metrics import NoteEvent as EvalNoteEvent  # noqa: N811
        from utils.pm2s_inference import (
            choose_phase_shift_from_beat_number_logits,
            infer_beat_indices_from_onsets,
            time_signature_to_beat_params,
        )

        item = cache["midi_item"]
        ref: List[EvalNoteEvent] = cache["ref_notes"]
        base_notes: List[EvalNoteEvent] = cache["baseline_notes"]

        pitches = torch.tensor(item["pitches"], dtype=torch.long, device=self.device).unsqueeze(0)
        onsets = torch.tensor(item["onsets"], dtype=torch.float32, device=self.device).unsqueeze(0)
        durations = torch.tensor(item["durations"], dtype=torch.float32, device=self.device).unsqueeze(0)
        velocities = torch.tensor(item["velocities"], dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = torch.ones_like(pitches, dtype=torch.bool, device=self.device)

        model = self.model.module if isinstance(self.model, DDP) else self.model
        model.eval()
        with torch.inference_mode():
            amp_ctx = (
                torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype)
                if self.amp_enabled
                else nullcontext()
            )
            with amp_ctx:
                rq_out = model(pitches, onsets / float(time_scale), durations / float(time_scale), velocities, mask)

        beat_frac = rq_out["beat_fraction"][0]
        beat_num_logits = rq_out["beat_number"][0]
        subdiv_idx = rq_out["subdivision"][0].argmax(dim=-1)
        time_sig_idx = int(rq_out["time_signature"][0].argmax(dim=-1).item())

        if tempo_source not in {"model", "midi"}:
            tempo_source = "midi"
        if time_signature_source not in {"model", "midi"}:
            time_signature_source = "midi"

        model_tempo = float(rq_out["tempo"][0].item())
        tempo = float(item.get("midi_tempo_bpm", model_tempo)) if tempo_source == "midi" else model_tempo
        tempo = float(max(1e-3, tempo))

        model_time_sig = model.time_sig_map[time_sig_idx]
        time_sig = tuple(item.get("midi_time_signature", model_time_sig)) if time_signature_source == "midi" else model_time_sig
        beat_unit_ql, beats_per_measure = time_signature_to_beat_params(time_sig)

        subdivs = torch.tensor(
            [model.subdivision_map[int(i)] for i in subdiv_idx.tolist()],
            device=self.device,
            dtype=torch.float32,
        )
        quantized_frac = torch.round(beat_frac * subdivs) / subdivs
        quantized_frac = torch.remainder(quantized_frac, 1.0)

        beat_index = infer_beat_indices_from_onsets(onsets[0], quantized_frac, tempo_bpm=float(tempo))
        phase_shift = choose_phase_shift_from_beat_number_logits(
            beat_index,
            beat_num_logits,
            beats_per_measure=beats_per_measure,
        )
        pred_onsets_ql = (beat_index.to(torch.float32) + float(phase_shift) + quantized_frac) * float(beat_unit_ql)
        pred_durations_ql = durations[0] * (float(tempo) / 60.0) * float(beat_unit_ql)

        pred: List[EvalNoteEvent] = []
        for p, o, d in zip(pitches[0].tolist(), pred_onsets_ql.tolist(), pred_durations_ql.tolist()):
            onset = float(o)
            dur = float(d)
            if dur <= 0:
                continue
            pred.append(EvalNoteEvent(onset=onset, offset=onset + dur, pitch=int(p), velocity=64))

        # Rebase to t=0 for fair comparison.
        if pred:
            t0 = min(n.onset for n in pred)
            if t0 != 0:
                pred = [EvalNoteEvent(onset=n.onset - t0, offset=n.offset - t0, pitch=n.pitch, velocity=n.velocity) for n in pred]

        base_metrics = {}
        pm_metrics = {}
        if ref and base_notes:
            base_metrics.update(note_f1(ref, base_notes, onset_tolerance=0.05, offset_ratio=0.2))
            base_metrics.update(frame_f1(ref, base_notes, frame_rate=100))
        if ref and pred:
            pm_metrics.update(note_f1(ref, pred, onset_tolerance=0.05, offset_ratio=0.2))
            pm_metrics.update(frame_f1(ref, pred, frame_rate=100))

        if base_metrics:
            print(
                f"[best@SWD09] baseline: onset+offset_f1={base_metrics.get('onset_offset_f1', 0.0):.4f}, frame_f1={base_metrics.get('frame_f1', 0.0):.4f}"
            )
        if pm_metrics:
            print(
                f"[best@SWD09] pm2s({tempo_source}/{time_signature_source}): onset+offset_f1={pm_metrics.get('onset_offset_f1', 0.0):.4f}, frame_f1={pm_metrics.get('frame_f1', 0.0):.4f}"
            )

        if self.wandb_run and pm_metrics:
            self.wandb_run.log(
                {
                    "swd09/pm2s_onset_offset_f1": float(pm_metrics.get("onset_offset_f1", 0.0)),
                    "swd09/pm2s_frame_f1": float(pm_metrics.get("frame_f1", 0.0)),
                    "swd09/tempo_source": tempo_source,
                    "swd09/time_signature_source": time_signature_source,
                },
                step=self.global_step,
            )

    def train(self):
        """Main training loop with epochs, validation, and checkpointing."""
        if self.is_main_process:
            print(f"Starting {self.task_name} training on {self.device}")
            steps_per_epoch = max(1, math.ceil(len(self.train_loader) / self.grad_accum_steps))
            print(f"Epochs: {self.epochs}, Steps per epoch: {steps_per_epoch} (accum={self.grad_accum_steps})")
            if self.early_stopping_patience > 0:
                print(f"Early stopping enabled: patience={self.early_stopping_patience}")

        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            # Training
            train_loss = self._train_epoch(epoch)

            # Validation
            val_loss = None
            if self.val_loader is not None:
                val_loss = self._validate()
                if self.is_main_process:
                    print(f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    epochs_without_improvement = 0
                    self.save_checkpoint("best.pt")
                    best_path = self.checkpoint_dir / self.task_name / "best.pt"
                    self._evaluate_best_swd09(best_path)
                    self._barrier_if_needed()
                else:
                    epochs_without_improvement += 1
            else:
                if self.is_main_process:
                    print(f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {train_loss:.4f}")

            if self.wandb_run:
                metrics = {
                    "train/epoch_loss": train_loss,
                    "train/epoch": epoch + 1,
                }
                if val_loss is not None:
                    metrics["val/loss"] = val_loss
                self.wandb_run.log(metrics, step=self.global_step)

            # Save periodic checkpoint
            if (epoch + 1) % self.save_every_epochs == 0 or epoch == self.epochs - 1:
                self.save_checkpoint(f"epoch_{epoch + 1}.pt")

            # Early stopping check
            if self.early_stopping_patience > 0 and epochs_without_improvement >= self.early_stopping_patience:
                if self.is_main_process:
                    print(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement (best val_loss={self.best_val_loss:.4f}).")
                break

        if self.is_main_process:
            print("Training complete!")
        self.save_checkpoint("final.pt")

    def _train_epoch(self, epoch: int) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        if self.is_main_process:
            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.epochs}",
            )
        else:
            progress_bar = self.train_loader

        self.optimizer.zero_grad(set_to_none=True)
        pending_steps = 0
        total_batches = len(self.train_loader)
        for batch_idx, batch in enumerate(progress_bar):
            is_last_batch = (batch_idx + 1) == total_batches
            will_sync = (pending_steps + 1) == self.grad_accum_steps or is_last_batch
            sync_context = (
                self.model.no_sync()
                if self.ddp_enabled and isinstance(self.model, DDP) and not will_sync
                else nullcontext()
            )

            with sync_context:
                amp_ctx = (
                    torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype)
                    if self.amp_enabled
                    else nullcontext()
                )
                with amp_ctx:
                    loss = self.loss_fn(self.model, batch, self.device)
                loss_value = loss.detach().item()
                loss = loss / self.grad_accum_steps

                loss_finite = torch.isfinite(loss.detach()).all()
                if self.ddp_enabled and ddp_is_initialized():
                    flag = torch.tensor(0 if loss_finite else 1, device=self.device)
                    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                    loss_finite = flag.item() == 0
                if not loss_finite:
                    if self.is_main_process:
                        print("Non-finite loss detected. Stopping training.")
                    raise RuntimeError("Non-finite loss detected.")

                if self.use_scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

            total_loss += loss_value
            num_batches += 1
            pending_steps += 1

            if will_sync:
                optimizer_step_taken = True
                if self.use_scaler:
                    self.scaler.unscale_(self.optimizer)
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                if self.use_scaler:
                    scale_before = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    optimizer_step_taken = self.scaler.get_scale() >= scale_before
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if optimizer_step_taken:
                    self.scheduler.step()
                    self.global_step += 1
                pending_steps = 0

                if self.wandb_run and self.global_step % self.log_every == 0:
                    self.wandb_run.log({
                        "train/loss": loss_value,
                        "train/lr": self.scheduler.get_last_lr()[0],
                    }, step=self.global_step)

            if self.is_main_process:
                progress_bar.set_postfix({
                    "loss": f"{loss_value:.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                })

        return self._reduce_epoch_loss(total_loss, num_batches)

    @torch.no_grad()
    def _validate(self) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            amp_ctx = (
                torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype)
                if self.amp_enabled
                else nullcontext()
            )
            with amp_ctx:
                loss = self.loss_fn(self.model, batch, self.device)
            loss_finite = torch.isfinite(loss.detach()).all()
            if self.ddp_enabled and ddp_is_initialized():
                flag = torch.tensor(0 if loss_finite else 1, device=self.device)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                loss_finite = flag.item() == 0
            if not loss_finite:
                if self.is_main_process:
                    print("Non-finite validation loss detected. Stopping training.")
                raise RuntimeError("Non-finite validation loss detected.")
            total_loss += loss.item()
            num_batches += 1

        return self._reduce_epoch_loss(total_loss, num_batches)

    def _reduce_epoch_loss(self, total_loss: float, num_batches: int) -> float:
        if not self.ddp_enabled or not ddp_is_initialized():
            return total_loss / max(1, num_batches)
        stats = torch.tensor([total_loss, float(num_batches)], device=self.device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total, count = stats.tolist()
        return total / max(1.0, count)

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        if not self.is_main_process:
            return
        model = self.model.module if isinstance(self.model, DDP) else self.model
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        path = self.checkpoint_dir / self.task_name / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = safe_torch_load(path, map_location=self.device, weights_only=False)
        model = self.model.module if isinstance(self.model, DDP) else self.model
        model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as exc:
                if self.is_main_process:
                    print(f"Warning: could not load scheduler state ({type(exc).__name__}: {exc}). Continuing with fresh scheduler.")
        if "scaler_state_dict" in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Loaded checkpoint from step {self.global_step}")


def rhythm_quantizer_loss(
    model,
    batch,
    device,
    time_scale: float = 1.0,
    beat_loss_weight: float = 1.0,
    beat_num_loss_weight: float = 1.0,
    subdivision_loss_weight: float = 1.0,
    time_sig_loss_weight: float = 1.0,
    tempo_loss_mode: str = "log",
    tempo_loss_weight: float = 1.0,
):
    """Compute loss for rhythm quantizer."""
    mse = nn.MSELoss(reduction="none")
    ce_seq = nn.CrossEntropyLoss(ignore_index=-100)
    ce_global = nn.CrossEntropyLoss()

    pitches = batch["pitches"].to(device, dtype=torch.long)
    onsets = batch["onsets"].to(device, dtype=torch.float32) / float(time_scale)
    durations = batch["durations"].to(device, dtype=torch.float32) / float(time_scale)
    velocities = batch["velocities"].to(device, dtype=torch.float32)
    mask = batch["mask"].to(device)

    outputs = model(pitches, onsets, durations, velocities, mask)

    beat_fraction_tgt = batch["beat_fraction"].to(device, dtype=torch.float32)
    beat_number_tgt = batch["beat_number"].to(device, dtype=torch.long)
    subdivision_tgt = batch["subdivision"].to(device, dtype=torch.long)
    time_sig_tgt = batch["time_signature"].to(device, dtype=torch.long)
    tempo_tgt = batch["tempo"].to(device, dtype=torch.float32)

    mask_float = mask.float()
    denom = mask_float.sum().clamp(min=1.0)

    # Circular beat phase loss (wrap-safe): target = (sin, cos) of 2πf.
    target_phase = torch.stack(
        [
            torch.sin(2.0 * math.pi * beat_fraction_tgt),
            torch.cos(2.0 * math.pi * beat_fraction_tgt),
        ],
        dim=-1,
    ).float()
    pred_phase = outputs.get("beat_phase")
    if pred_phase is None:
        # Backward-compat for older checkpoints/outputs (direct beat_fraction regression).
        pred_phase = torch.stack(
            [
                torch.sin(2.0 * math.pi * outputs["beat_fraction"].float()),
                torch.cos(2.0 * math.pi * outputs["beat_fraction"].float()),
            ],
            dim=-1,
        )
    pred_phase = pred_phase.float()
    pred_norm = torch.linalg.vector_norm(pred_phase, dim=-1, keepdim=True).clamp(min=1e-8)
    pred_phase = pred_phase / pred_norm
    cos_sim = (pred_phase * target_phase).sum(dim=-1)  # (batch, seq)
    phase_loss = 1.0 - cos_sim
    beat_loss = (phase_loss * mask_float).sum() / denom

    beat_number_masked = beat_number_tgt.masked_fill(~mask, -100)
    subdivision_masked = subdivision_tgt.masked_fill(~mask, -100)

    beat_num_loss = ce_seq(outputs["beat_number"].permute(0, 2, 1).float(), beat_number_masked)
    subdivision_loss = ce_seq(outputs["subdivision"].permute(0, 2, 1).float(), subdivision_masked)
    time_sig_loss = ce_global(outputs["time_signature"].float(), time_sig_tgt)
    if tempo_loss_mode == "log":
        tempo_pred = outputs.get("log_tempo")
        if tempo_pred is None:
            tempo_pred = torch.log(outputs["tempo"].float().clamp(min=1e-3))
        else:
            tempo_pred = tempo_pred.float()
        tempo_target = torch.log(tempo_tgt.clamp(min=1e-3))
        tempo_loss = mse(tempo_pred, tempo_target).mean()
    elif tempo_loss_mode == "mse":
        tempo_loss = mse(outputs["tempo"].float(), tempo_tgt).mean()
    else:
        raise ValueError(f"Unsupported tempo_loss_mode: {tempo_loss_mode}")

    beat_loss_weight = float(beat_loss_weight)
    beat_num_loss_weight = float(beat_num_loss_weight)
    subdivision_loss_weight = float(subdivision_loss_weight)
    time_sig_loss_weight = float(time_sig_loss_weight)
    tempo_loss_weight = float(tempo_loss_weight)

    return (
        beat_loss * beat_loss_weight
        + beat_num_loss * beat_num_loss_weight
        + subdivision_loss * subdivision_loss_weight
        + time_sig_loss * time_sig_loss_weight
        + tempo_loss * tempo_loss_weight
    )


def voice_separator_loss(model, batch, device, time_scale: float = 1.0):
    """Compute loss for voice separator."""
    ce = nn.CrossEntropyLoss(ignore_index=-100)

    pitches = batch["pitches"].to(device, dtype=torch.long)
    onsets = batch["onsets"].to(device, dtype=torch.float32) / float(time_scale)
    offsets = batch["offsets"].to(device, dtype=torch.float32) / float(time_scale)
    mask = batch["mask"].to(device)

    outputs = model(pitches, onsets, offsets, mask)
    voice_tgt = batch["voice_ids"].to(device, dtype=torch.long).masked_fill(~mask, -100)
    staff_tgt = batch["staff_ids"].to(device, dtype=torch.long).masked_fill(~mask, -100)

    voice_loss = ce(outputs["voice_logits"].permute(0, 2, 1).float(), voice_tgt)
    staff_loss = ce(outputs["staff_logits"].permute(0, 2, 1).float(), staff_tgt)

    return voice_loss + staff_loss


def key_estimator_loss(model, batch, device, time_scale: float = 1.0):
    """Compute loss for key estimator."""
    ce = nn.CrossEntropyLoss()

    pitches = batch["pitches"].to(device, dtype=torch.long)
    durations = batch["durations"].to(device, dtype=torch.float32) / float(time_scale)
    mask = batch["mask"].to(device)
    key_index = batch["key_index"].to(device, dtype=torch.long)

    outputs = model(pitches, durations, mask)
    return ce(outputs["key_logits"].float(), key_index)


def generate_dummy_data(task: str, num_samples: int = 100) -> List[dict]:
    """Generate dummy data for testing training pipeline."""
    import random
    items = []

    for _ in range(num_samples):
        seq_len = random.randint(20, 100)

        # Common fields
        pitches = [random.randint(21, 108) for _ in range(seq_len)]
        onsets = sorted([random.random() * 10 for _ in range(seq_len)])
        durations = [random.random() * 0.5 + 0.1 for _ in range(seq_len)]
        velocities = [random.random() for _ in range(seq_len)]
        offsets = [onsets[i] + durations[i] for i in range(seq_len)]

        item = {
            "pitches": pitches,
            "onsets": onsets,
            "durations": durations,
            "velocities": velocities,
            "offsets": offsets,
        }

        if task == "rhythm_quantizer":
            item.update({
                "beat_fraction": [random.random() for _ in range(seq_len)],
                "beat_number": [random.randint(0, 15) for _ in range(seq_len)],
                "subdivision": [random.randint(0, 7) for _ in range(seq_len)],
                "time_signature": random.randint(0, 5),
                "tempo": random.random() * 60 + 60,  # 60-120 BPM
            })
        elif task == "voice_separator":
            item.update({
                "voice_ids": [random.randint(0, 3) for _ in range(seq_len)],
                "staff_ids": [0 if p < 60 else 1 for p in pitches],
            })
        elif task == "key_estimator":
            item.update({
                "key_index": random.randint(0, 29),
            })

        items.append(item)

    return items


def main():
    parser = argparse.ArgumentParser(description="Train PM2S components")
    parser.add_argument("--config", type=str, default=None, help="Path to training config YAML")
    parser.add_argument("--model-config", type=str, default=None, help="Path to model config YAML")
    parser.add_argument("--task", choices=["rhythm_quantizer", "voice_separator", "key_estimator"], required=True)
    parser.add_argument("--manifest", type=str, default=None, help="Path to .pt or .jsonl manifest")
    parser.add_argument("--use-dummy", action="store_true", help="Use dummy data for testing")
    parser.add_argument("--dummy-samples", type=int, default=100, help="Number of dummy samples")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (model init + data split)")
    parser.add_argument("--warmup-steps", "--warmup_steps", type=int, default=None,
                        help="Linear warmup steps for LR schedule (optimizer steps)")
    parser.add_argument("--min-lr-ratio", "--min_lr_ratio", type=float, default=None,
                        help="Final LR as a ratio of base LR (cosine decay), e.g. 0.01")
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=None,
                        help="AdamW weight decay (0 disables; applied to non-bias/non-norm params).")
    parser.add_argument("--max-grad-norm", "--max_grad_norm", type=float, default=None,
                        help="Clip grad norm (0 disables).")
    parser.add_argument("--train-max-seq-len", "--train_max_seq_len", type=int, default=None,
                        help="Optional max note sequence length for training batches (cropping).")
    parser.add_argument("--eval-max-seq-len", "--eval_max_seq_len", type=int, default=None,
                        help="Optional max note sequence length for validation batches (cropping).")
    parser.add_argument("--random-crop", dest="random_crop", action="store_true",
                        help="When cropping, randomly select a window per sample (train only).")
    parser.add_argument("--no-random-crop", dest="random_crop", action="store_false",
                        help="When cropping, always take the first window (train only).")
    parser.set_defaults(random_crop=None)
    parser.add_argument("--onset-jitter-std", type=float, default=None,
                        help="Onset jitter std as fraction of mean IOI (0 disables; default 0). Applied to training data only.")
    parser.add_argument("--duration-jitter-std", type=float, default=None,
                        help="Duration jitter std (multiplicative; default 0). Applied to training data only.")
    parser.add_argument("--jitter-sources", type=str, default=None,
                        help="Comma-separated dataset sources to apply jitter to (default 'pdmx'; 'all' for all).")
    parser.add_argument("--sample-by-dataset-weights", "--sample_by_dataset_weights",
                        dest="sample_by_dataset_weights", action="store_true",
                        help="Sample training items by pm2s_training.<task>.datasets weights (requires manifests with 'source').")
    parser.add_argument("--no-sample-by-dataset-weights", "--no_sample_by_dataset_weights",
                        dest="sample_by_dataset_weights", action="store_false")
    parser.set_defaults(sample_by_dataset_weights=None)
    parser.add_argument("--time-scale", type=float, default=None, help="Scale note times by this factor")
    parser.add_argument("--val-split", type=float, default=None, help="Validation split ratio")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--tempo-loss", type=str, default=None, choices=["log", "mse"],
                        help="Tempo loss type for rhythm quantizer")
    parser.add_argument("--tempo-loss-weight", type=float, default=None,
                        help="Weight for tempo loss term")
    parser.add_argument("--beat-loss-weight", type=float, default=None,
                        help="Weight for beat phase loss (rhythm quantizer only)")
    parser.add_argument("--beat-num-loss-weight", type=float, default=None,
                        help="Weight for beat-number classification loss (rhythm quantizer only)")
    parser.add_argument("--subdivision-loss-weight", type=float, default=None,
                        help="Weight for subdivision classification loss (rhythm quantizer only)")
    parser.add_argument("--time-sig-loss-weight", type=float, default=None,
                        help="Weight for time-signature classification loss (rhythm quantizer only)")
    parser.add_argument("--early-stopping-patience", type=int, default=None,
                        help="Stop training after N epochs without val loss improvement (0 disables; default 0).")
    parser.add_argument("--save-every", type=int, default=None, help="Save checkpoint every N epochs")
    parser.add_argument("--log-every", type=int, default=None, help="Log metrics every N steps")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gpu-ids", "--gpu_ids", type=str, default=None,
                        help="Comma-separated GPU ids (e.g. 6,7 or [6,7])")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=None)
    parser.add_argument("--wandb", dest="wandb", action="store_true")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.set_defaults(wandb=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default=None, help="Comma-separated tags")
    parser.add_argument("--amp", dest="amp", action="store_true", help="Enable mixed precision (CUDA only)")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=None)
    parser.add_argument("--amp-dtype", type=str, default=None, choices=["fp16", "bf16"],
                        help="AMP dtype (bf16 recommended if supported)")
    parser.add_argument(
        "--eval-best-swd09",
        action="store_true",
        help="After saving a new best checkpoint, evaluate rhythm quantization on SWD Schubert_D911-09.",
    )
    parser.add_argument(
        "--eval-best-swd-midi",
        type=str,
        default="datasets/SWD/01_RawData/score_midi/Schubert_D911-09.mid",
        help="Score MIDI path for --eval-best-swd09.",
    )
    parser.add_argument(
        "--eval-best-swd-xml",
        type=str,
        default="datasets/SWD/01_RawData/score_musicxml/Schubert_D911-09.xml",
        help="Ground-truth MusicXML path for --eval-best-swd09.",
    )
    parser.add_argument(
        "--eval-best-tempo-source",
        type=str,
        default="midi",
        choices=["model", "midi"],
        help="Tempo source for --eval-best-swd09 (model prediction vs MIDI tempo map).",
    )
    parser.add_argument(
        "--eval-best-time-signature-source",
        type=str,
        default="midi",
        choices=["model", "midi"],
        help="Time signature source for --eval-best-swd09 (model prediction vs MIDI metadata).",
    )
    parser.add_argument("--ddp", dest="ddp", action="store_true", help="Enable DDP (requires torchrun)")
    parser.add_argument("--no-ddp", dest="ddp", action="store_false")
    parser.set_defaults(ddp=None)
    args = parser.parse_args()

    train_cfg = load_yaml_config(args.config)
    model_cfg = load_yaml_config(args.model_config)
    pm2s_global_cfg = train_cfg.get("pm2s_training", {})
    pm2s_cfg = pm2s_global_cfg.get(args.task, {})
    hardware_cfg = train_cfg.get("hardware", {})
    wandb_cfg = train_cfg.get("wandb", {})

    batch_size = resolve_value(args.batch_size, pm2s_cfg.get("batch_size"), 32)
    grad_accum_steps = resolve_value(
        args.grad_accum_steps,
        pm2s_cfg.get("gradient_accumulation_steps", pm2s_global_cfg.get("gradient_accumulation_steps")),
        1,
    )
    time_scale = resolve_value(
        args.time_scale,
        pm2s_cfg.get("time_scale", pm2s_global_cfg.get("time_scale")),
        1.0,
    )
    tempo_loss_mode = resolve_value(
        args.tempo_loss,
        pm2s_cfg.get("tempo_loss", pm2s_global_cfg.get("tempo_loss")),
        "log",
    )
    tempo_loss_weight = resolve_value(
        args.tempo_loss_weight,
        pm2s_cfg.get("tempo_loss_weight", pm2s_global_cfg.get("tempo_loss_weight")),
        1.0,
    )
    beat_loss_weight = resolve_value(
        args.beat_loss_weight,
        pm2s_cfg.get("beat_loss_weight", pm2s_global_cfg.get("beat_loss_weight")),
        1.0,
    )
    beat_num_loss_weight = resolve_value(
        args.beat_num_loss_weight,
        pm2s_cfg.get("beat_num_loss_weight", pm2s_global_cfg.get("beat_num_loss_weight")),
        1.0,
    )
    subdivision_loss_weight = resolve_value(
        args.subdivision_loss_weight,
        pm2s_cfg.get("subdivision_loss_weight", pm2s_global_cfg.get("subdivision_loss_weight")),
        1.0,
    )
    time_sig_loss_weight = resolve_value(
        args.time_sig_loss_weight,
        pm2s_cfg.get("time_sig_loss_weight", pm2s_global_cfg.get("time_sig_loss_weight")),
        1.0,
    )
    epochs = resolve_value(args.epochs, pm2s_cfg.get("epochs"), 50)
    learning_rate = resolve_value(args.lr, pm2s_cfg.get("learning_rate"), 1e-4)
    seed = resolve_value(args.seed, pm2s_cfg.get("seed", pm2s_global_cfg.get("seed")), 42)
    warmup_steps = resolve_value(args.warmup_steps, pm2s_cfg.get("warmup_steps", pm2s_global_cfg.get("warmup_steps")), 0)
    min_lr_ratio = resolve_value(args.min_lr_ratio, pm2s_cfg.get("min_lr_ratio", pm2s_global_cfg.get("min_lr_ratio")), 0.01)
    weight_decay = resolve_value(args.weight_decay, pm2s_cfg.get("weight_decay", pm2s_global_cfg.get("weight_decay")), 0.0)
    max_grad_norm = resolve_value(args.max_grad_norm, pm2s_cfg.get("max_grad_norm", pm2s_global_cfg.get("max_grad_norm")), 1.0)
    train_max_seq_len = resolve_value(args.train_max_seq_len, pm2s_cfg.get("train_max_seq_len", pm2s_global_cfg.get("train_max_seq_len")), None)
    eval_max_seq_len = resolve_value(args.eval_max_seq_len, pm2s_cfg.get("eval_max_seq_len", pm2s_global_cfg.get("eval_max_seq_len")), None)
    random_crop = resolve_value(args.random_crop, pm2s_cfg.get("random_crop", pm2s_global_cfg.get("random_crop")), True)
    onset_jitter_std = resolve_value(
        args.onset_jitter_std,
        pm2s_cfg.get("onset_jitter_std", pm2s_global_cfg.get("onset_jitter_std")),
        0.0,
    )
    duration_jitter_std = resolve_value(
        args.duration_jitter_std,
        pm2s_cfg.get("duration_jitter_std", pm2s_global_cfg.get("duration_jitter_std")),
        0.0,
    )
    jitter_sources_raw = resolve_value(
        args.jitter_sources,
        pm2s_cfg.get("jitter_sources", pm2s_global_cfg.get("jitter_sources")),
        "pdmx",
    )
    jitter_sources: Optional[List[str]] = None
    if jitter_sources_raw:
        jitter_sources = [canonicalize_dataset_name(s.strip()) for s in str(jitter_sources_raw).split(",") if s.strip()]
    sample_by_dataset_weights = resolve_value(
        args.sample_by_dataset_weights,
        pm2s_cfg.get("sample_by_dataset_weights", pm2s_global_cfg.get("sample_by_dataset_weights")),
        False,
    )
    val_split = resolve_value(args.val_split, pm2s_cfg.get("val_split"), 0.1)
    checkpoint_dir = resolve_value(args.checkpoint_dir, pm2s_cfg.get("checkpoint_dir"), "checkpoints/pm2s")
    early_stopping_patience = resolve_value(
        args.early_stopping_patience,
        pm2s_cfg.get("early_stopping_patience", pm2s_global_cfg.get("early_stopping_patience")),
        0,
    )
    save_every_epochs = resolve_value(args.save_every, pm2s_cfg.get("save_every_epochs"), 10)
    log_every = resolve_value(args.log_every, pm2s_cfg.get("log_every_n_steps"), 100)
    amp_enabled = resolve_value(
        args.amp,
        pm2s_cfg.get("mixed_precision", pm2s_global_cfg.get("mixed_precision")),
        False,
    )
    amp_dtype_name = resolve_value(
        args.amp_dtype,
        pm2s_cfg.get("amp_dtype", pm2s_global_cfg.get("amp_dtype")),
        "bf16",
    )
    num_workers = resolve_value(args.num_workers, hardware_cfg.get("num_workers"), 4)
    pin_memory = resolve_value(args.pin_memory, hardware_cfg.get("pin_memory"), True)
    grad_accum_steps = max(1, int(grad_accum_steps))
    if not time_scale or time_scale <= 0:
        raise ValueError("--time-scale must be > 0")

    device_name = resolve_value(args.device, hardware_cfg.get("device"), None)
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp_enabled = (world_size > 1) if args.ddp is None else args.ddp
    if world_size > 1 and not ddp_enabled:
        raise RuntimeError("WORLD_SIZE>1 but DDP disabled. Remove --no-ddp or run without torchrun.")
    if ddp_enabled and world_size <= 1:
        raise RuntimeError("DDP requested but WORLD_SIZE=1. Use torchrun --nproc_per_node=2 ...")
    if ddp_enabled and not ddp_is_initialized():
        backend = "nccl" if device_name.startswith("cuda") and torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    global_rank = int(os.environ.get("RANK", "0"))
    is_main_process = global_rank == 0

    torch.manual_seed(int(seed))

    cli_gpu_ids = parse_gpu_ids(args.gpu_ids)
    gpu_ids = cli_gpu_ids if cli_gpu_ids is not None else (
        hardware_cfg.get("gpu_ids") if isinstance(hardware_cfg.get("gpu_ids"), list) else []
    )
    if ddp_enabled and is_main_process:
        if gpu_ids:
            print(f"DDP enabled (world_size={world_size}) using GPU ids {gpu_ids}.")
        else:
            print(f"DDP enabled (world_size={world_size}) using local rank GPU mapping.")

    if ddp_enabled and device_name.startswith("cuda:"):
        if is_main_process:
            print("DDP enabled: ignoring explicit cuda:<id> device override.")
        device_name = "cuda"

    if device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            if is_main_process:
                print("CUDA not available, falling back to CPU.")
            device_name = "cpu"
        else:
            if ddp_enabled:
                if gpu_ids:
                    if local_rank >= len(gpu_ids):
                        raise RuntimeError(f"LOCAL_RANK {local_rank} out of range for --gpu_ids {gpu_ids}")
                    selected_gpu = gpu_ids[local_rank]
                else:
                    selected_gpu = local_rank
            else:
                if gpu_ids and not os.environ.get("CUDA_VISIBLE_DEVICES"):
                    if cli_gpu_ids and len(cli_gpu_ids) > 1:
                        print(f"Multiple --gpu_ids provided ({cli_gpu_ids}); using GPU {cli_gpu_ids[0]} (single-GPU training).")
                    selected_gpu = gpu_ids[0]
                else:
                    selected_gpu = int(device_name.split(":")[1]) if ":" in device_name else 0
            device_name = f"cuda:{selected_gpu}"
            torch.cuda.set_device(selected_gpu)

    amp_effective = amp_enabled and device_name.startswith("cuda")
    amp_dtype = None
    if amp_effective:
        if amp_dtype_name == "bf16":
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                if is_main_process:
                    print("BF16 not supported, falling back to FP16 AMP.")
                amp_dtype = torch.float16
        else:
            amp_dtype = torch.float16

    if not 0 <= val_split < 1:
        raise ValueError("--val-split must be in the range [0, 1).")

    # Load data from manifest or generate dummy data
    manifest_path_obj: Optional[Path] = None
    if args.use_dummy:
        print(f"Generating {args.dummy_samples} dummy samples for {args.task}...")
        items = generate_dummy_data(args.task, args.dummy_samples)
    else:
        manifest_path = args.manifest or pm2s_cfg.get("manifest")
        if manifest_path is None:
            raise ValueError("--manifest is required when not using --use-dummy")
        manifest_path_obj = Path(manifest_path)
        items = load_manifest(manifest_path_obj)

    if not items:
        raise ValueError("Data is empty")

    required_fields = {
        "rhythm_quantizer": ["pitches", "onsets", "durations", "velocities",
                             "beat_fraction", "beat_number", "subdivision",
                             "time_signature", "tempo"],
        "voice_separator": ["pitches", "onsets", "offsets", "voice_ids", "staff_ids"],
        "key_estimator": ["pitches", "durations", "key_index"],
    }
    missing = [f for f in required_fields[args.task] if f not in items[0]]
    if missing:
        raise ValueError(f"Manifest missing required fields for {args.task}: {missing}")

    # Create dataset and split
    full_dataset = PM2SDataset(items)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    if train_size == 0:
        raise ValueError("Training split is empty. Reduce --val-split or add more data.")

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(int(seed)),
    )

    train_sampler = None
    val_sampler = None
    if ddp_enabled:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
        if val_size > 0:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=global_rank,
                shuffle=False,
            )

    dataset_weights = parse_dataset_weights(pm2s_cfg.get("datasets"))
    if sample_by_dataset_weights and dataset_weights:
        base_dataset = getattr(train_dataset, "dataset", None)
        base_indices = getattr(train_dataset, "indices", None)
        if isinstance(base_dataset, PM2SDataset) and isinstance(base_indices, list):
            sources: List[str] = []
            counts: Counter = Counter()

            def collect_sources() -> bool:
                sources.clear()
                counts.clear()
                for idx in base_indices:
                    item = base_dataset.items[idx]
                    src = item.get("source")
                    if src is None:
                        return False
                    src = canonicalize_dataset_name(src)
                    sources.append(src)
                    counts[src] += 1
                return True

            have_sources = collect_sources()
            if not have_sources:
                backfilled = backfill_manifest_sources_from_state(
                    base_dataset.items,
                    manifest_path_obj,
                    args.task,
                    is_main_process=is_main_process,
                )
                if backfilled:
                    have_sources = collect_sources()

            if have_sources:
                sample_weights = []
                for src in sources:
                    if src in dataset_weights and counts[src] > 0:
                        sample_weights.append(float(dataset_weights[src]) / float(counts[src]))
                    else:
                        sample_weights.append(0.0)
                weights_tensor = torch.tensor(sample_weights, dtype=torch.double)
                if float(weights_tensor.sum()) > 0:
                    if ddp_enabled:
                        train_sampler = WeightedDistributedSampler(
                            weights_tensor,
                            num_replicas=world_size,
                            rank=global_rank,
                            seed=int(seed),
                        )
                        if is_main_process:
                            print(f"Using dataset-weighted sampling for training ({len(dataset_weights)} sources).")
                    else:
                        train_sampler = WeightedRandomSampler(
                            weights_tensor,
                            num_samples=len(train_dataset),
                            replacement=True,
                        )
                        if is_main_process:
                            print(f"Using dataset-weighted sampling for training ({len(dataset_weights)} sources).")
            else:
                if is_main_process:
                    print("Dataset-weighted sampling requested, but manifest items have no 'source' and no usable manifest_state.json was found.")

    if is_main_process and (onset_jitter_std > 0 or duration_jitter_std > 0):
        print(f"Timing jitter enabled: onset_std={onset_jitter_std}, duration_std={duration_jitter_std}, sources={jitter_sources}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=partial(
            collate_pm2s,
            max_seq_len=train_max_seq_len,
            random_crop=bool(random_crop),
            onset_jitter_std=float(onset_jitter_std),
            duration_jitter_std=float(duration_jitter_std),
            jitter_sources=jitter_sources,
        ),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=partial(collate_pm2s, max_seq_len=eval_max_seq_len, random_crop=False),
        num_workers=num_workers,
        pin_memory=pin_memory,
    ) if val_size > 0 else None

    device = torch.device(device_name)

    # Create model and loss function
    pm2s_model_cfg = model_cfg.get("pm2s_model", {})
    if args.task == "rhythm_quantizer":
        rq_cfg = pm2s_model_cfg.get("rhythm_quantizer", {})
        model = RhythmQuantizer(config=QuantizationConfig(
            hidden_dim=rq_cfg.get("hidden_dim", 256),
            num_layers=rq_cfg.get("num_layers", 6),
            num_heads=rq_cfg.get("num_heads", 4),
            ff_dim=rq_cfg.get("ffn_dim", 1024),
            dropout=rq_cfg.get("dropout", 0.1),
            max_subdivisions=rq_cfg.get("max_subdivisions", 16),
            triplet_support=rq_cfg.get("triplet_support", True),
            max_seq_len=rq_cfg.get("max_seq_len", 4096),
        ))
        loss_fn = lambda m, b, d: rhythm_quantizer_loss(
            m,
            b,
            d,
            time_scale=time_scale,
            beat_loss_weight=beat_loss_weight,
            beat_num_loss_weight=beat_num_loss_weight,
            subdivision_loss_weight=subdivision_loss_weight,
            time_sig_loss_weight=time_sig_loss_weight,
            tempo_loss_mode=tempo_loss_mode,
            tempo_loss_weight=tempo_loss_weight,
        )
    elif args.task == "voice_separator":
        vs_cfg = pm2s_model_cfg.get("voice_separator", {})
        model = VoiceSeparator(
            hidden_dim=vs_cfg.get("hidden_dim", 256),
            num_layers=vs_cfg.get("num_layers", 4),
            num_heads=vs_cfg.get("num_heads", 4),
            max_voices=vs_cfg.get("max_voices", 4),
            dropout=vs_cfg.get("dropout", 0.1),
            max_seq_len=vs_cfg.get("max_seq_len", 4096),
        )
        loss_fn = lambda m, b, d: voice_separator_loss(m, b, d, time_scale=time_scale)
    else:
        ke_cfg = pm2s_model_cfg.get("key_estimator", {})
        model = KeyEstimator(
            hidden_dim=ke_cfg.get("hidden_dim", 256),
            num_keys=ke_cfg.get("num_keys", 30),
        )
        loss_fn = lambda m, b, d: key_estimator_loss(m, b, d, time_scale=time_scale)

    wandb_run = None
    use_wandb = resolve_value(args.wandb, wandb_cfg.get("enabled"), False)
    if use_wandb and is_main_process:
        try:
            import wandb
            wandb_project = resolve_value(args.wandb_project, wandb_cfg.get("project"), "PM2S")
            wandb_entity = resolve_value(args.wandb_entity, wandb_cfg.get("entity"), None)
            wandb_name = resolve_value(args.wandb_name, wandb_cfg.get("name"), None)
            wandb_tags = None
            if args.wandb_tags:
                wandb_tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
            elif isinstance(wandb_cfg.get("tags"), list):
                wandb_tags = wandb_cfg.get("tags")

            if wandb_name and "{task}" in wandb_name:
                wandb_name = wandb_name.format(task=args.task)

            effective_world = world_size if ddp_enabled else 1
            wandb_config = {
                "task": args.task,
                "manifest": None if args.use_dummy else (args.manifest or pm2s_cfg.get("manifest")),
                "epochs": epochs,
                "batch_size": batch_size,
                "grad_accum_steps": grad_accum_steps,
                "effective_batch_size": batch_size * grad_accum_steps * effective_world,
                "seed": seed,
                "warmup_steps": warmup_steps,
                "min_lr_ratio": min_lr_ratio,
                "weight_decay": weight_decay,
                "max_grad_norm": max_grad_norm,
                "train_max_seq_len": train_max_seq_len,
                "eval_max_seq_len": eval_max_seq_len,
                "random_crop": bool(random_crop),
                "sample_by_dataset_weights": bool(sample_by_dataset_weights),
                "onset_jitter_std": onset_jitter_std,
                "duration_jitter_std": duration_jitter_std,
                "jitter_sources": jitter_sources,
                "early_stopping_patience": early_stopping_patience,
                "time_scale": time_scale,
                "tempo_loss": tempo_loss_mode,
                "tempo_loss_weight": tempo_loss_weight,
                "beat_loss_weight": beat_loss_weight if args.task == "rhythm_quantizer" else None,
                "beat_num_loss_weight": beat_num_loss_weight if args.task == "rhythm_quantizer" else None,
                "subdivision_loss_weight": subdivision_loss_weight if args.task == "rhythm_quantizer" else None,
                "time_sig_loss_weight": time_sig_loss_weight if args.task == "rhythm_quantizer" else None,
                "learning_rate": learning_rate,
                "val_split": val_split,
                "device": device_name,
                "amp": amp_effective,
                "amp_dtype": "bf16" if amp_dtype == torch.bfloat16 else ("fp16" if amp_dtype == torch.float16 else None),
                "model_config": pm2s_model_cfg.get(args.task, {}),
            }
            wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_name,
                tags=wandb_tags,
                config=wandb_config,
            )
        except ImportError:
            print("WandB not installed, disabling logging.")

    # Create trainer
    if ddp_enabled and device.type == "cuda":
        model = model.to(device)
        ddp_kwargs = {"device_ids": [device.index], "output_device": device.index}
        if args.task == "voice_separator":
            ddp_kwargs["find_unused_parameters"] = True
            if is_main_process:
                print("Voice separator: enabling DDP find_unused_parameters=True (unused auxiliary heads).")
        model = DDP(model, **ddp_kwargs)

    trainer = PM2STrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
        learning_rate=learning_rate,
        epochs=epochs,
        checkpoint_dir=checkpoint_dir,
        task_name=args.task,
        log_every_n_steps=log_every,
        is_main_process=is_main_process,
        ddp_enabled=ddp_enabled,
        grad_accum_steps=grad_accum_steps,
        amp_enabled=amp_effective,
        amp_dtype=amp_dtype,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        save_every_epochs=save_every_epochs,
        early_stopping_patience=int(early_stopping_patience),
        wandb_run=wandb_run,
        best_swd_eval={
            "enabled": bool(args.eval_best_swd09),
            "midi_path": Path(args.eval_best_swd_midi),
            "xml_path": Path(args.eval_best_swd_xml),
            "tempo_source": str(args.eval_best_tempo_source),
            "time_signature_source": str(args.eval_best_time_signature_source),
            "time_scale": float(time_scale),
        }
        if args.eval_best_swd09
        else None,
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()
    if wandb_run:
        wandb_run.finish()
    if ddp_is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
