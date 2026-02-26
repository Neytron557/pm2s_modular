"""
Lightweight evaluation for PM2S component checkpoints.

Evaluates a small random subset of each PM2S task manifest (default: 30 samples)
and reports basic accuracy/regression metrics + inference timing.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from contextlib import nullcontext
from typing import Dict, List, Optional

import torch

from utils.pm2s_inference import (
    build_pm2s_components,
    load_yaml_config,
    load_checkpoint_into_model,
    resolve_device,
)
from utils.serialization import safe_torch_load


def _cuda_sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_manifest_items(path: str) -> List[dict]:
    data = safe_torch_load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "items" in data:
        return data["items"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported manifest format: {path}")


def _accumulate_mean(acc: Dict[str, float], key: str, value: float) -> None:
    acc[key] = acc.get(key, 0.0) + float(value)


def _accumulate_int(acc: Dict[str, int], key: str, value: int) -> None:
    acc[key] = acc.get(key, 0) + int(value)


def evaluate_rhythm_quantizer(
    model,
    items: List[dict],
    *,
    device: torch.device,
    indices: List[int],
    time_scale: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    total_notes = 0

    _cuda_sync_if_needed(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype) if (use_amp and device.type == "cuda") else nullcontext()
        with amp_ctx:
            for idx in indices:
                item = items[idx]
                pitches = torch.tensor(item["pitches"], dtype=torch.long, device=device).unsqueeze(0)
                onsets = torch.tensor(item["onsets"], dtype=torch.float32, device=device).unsqueeze(0) / time_scale
                durations = torch.tensor(item["durations"], dtype=torch.float32, device=device).unsqueeze(0) / time_scale
                velocities = torch.tensor(item["velocities"], dtype=torch.float32, device=device).unsqueeze(0)
                mask = torch.ones_like(pitches, dtype=torch.bool, device=device)
                out = model(pitches, onsets, durations, velocities, mask)

                beat_frac_pred = out["beat_fraction"][0].float()
                beat_num_pred = out["beat_number"][0].argmax(dim=-1)
                subdiv_pred = out["subdivision"][0].argmax(dim=-1)
                time_sig_pred = int(out["time_signature"][0].argmax(dim=-1).item())
                tempo_pred = float(out["tempo"][0].float().item())

                beat_frac_tgt = torch.tensor(item["beat_fraction"], dtype=torch.float32, device=device)
                beat_num_tgt = torch.tensor(item["beat_number"], dtype=torch.long, device=device)
                subdiv_tgt = torch.tensor(item["subdivision"], dtype=torch.long, device=device)
                time_sig_tgt = int(item["time_signature"])
                tempo_tgt = float(item["tempo"])

                n = int(pitches.size(1))
                total_notes += n
                _accumulate_mean(sums, "beat_fraction_mse_sum", float(torch.mean((beat_frac_pred - beat_frac_tgt) ** 2).item()))
                _accumulate_int(counts, "beat_fraction_items", 1)

                _accumulate_int(counts, "beat_number_correct", int((beat_num_pred == beat_num_tgt).sum().item()))
                _accumulate_int(counts, "beat_number_total", n)
                _accumulate_int(counts, "subdivision_correct", int((subdiv_pred == subdiv_tgt).sum().item()))
                _accumulate_int(counts, "subdivision_total", n)
                _accumulate_int(counts, "time_signature_correct", int(time_sig_pred == time_sig_tgt))
                _accumulate_int(counts, "time_signature_total", 1)

                _accumulate_mean(sums, "tempo_mae_sum", abs(tempo_pred - tempo_tgt))
                _accumulate_mean(sums, "tempo_mape_sum", abs(tempo_pred - tempo_tgt) / max(1e-6, abs(tempo_tgt)))
                _accumulate_int(counts, "tempo_items", 1)
    _cuda_sync_if_needed(device)
    t1 = time.perf_counter()

    results: Dict[str, float] = {}
    results["beat_fraction_mse"] = sums.get("beat_fraction_mse_sum", 0.0) / max(1, counts.get("beat_fraction_items", 0))
    results["beat_number_acc"] = counts.get("beat_number_correct", 0) / max(1, counts.get("beat_number_total", 0))
    results["subdivision_acc"] = counts.get("subdivision_correct", 0) / max(1, counts.get("subdivision_total", 0))
    results["time_signature_acc"] = counts.get("time_signature_correct", 0) / max(1, counts.get("time_signature_total", 0))
    results["tempo_mae"] = sums.get("tempo_mae_sum", 0.0) / max(1, counts.get("tempo_items", 0))
    results["tempo_mape"] = sums.get("tempo_mape_sum", 0.0) / max(1, counts.get("tempo_items", 0))

    total_time = max(1e-9, t1 - t0)
    results["inference_ms_per_item"] = (total_time / max(1, len(indices))) * 1000.0
    results["notes_per_sec"] = total_notes / total_time
    return results


def evaluate_voice_separator(
    model,
    items: List[dict],
    *,
    device: torch.device,
    indices: List[int],
    time_scale: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    total_notes = 0

    _cuda_sync_if_needed(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype) if (use_amp and device.type == "cuda") else nullcontext()
        with amp_ctx:
            for idx in indices:
                item = items[idx]
                pitches = torch.tensor(item["pitches"], dtype=torch.long, device=device).unsqueeze(0)
                onsets = torch.tensor(item["onsets"], dtype=torch.float32, device=device).unsqueeze(0) / time_scale
                offsets = torch.tensor(item["offsets"], dtype=torch.float32, device=device).unsqueeze(0) / time_scale
                mask = torch.ones_like(pitches, dtype=torch.bool, device=device)
                out = model(pitches, onsets, offsets, mask)

                voice_pred = out["voice_logits"][0].argmax(dim=-1)
                staff_pred = out["staff_logits"][0].argmax(dim=-1)
                voice_tgt = torch.tensor(item["voice_ids"], dtype=torch.long, device=device)
                staff_tgt = torch.tensor(item["staff_ids"], dtype=torch.long, device=device)

                n = int(pitches.size(1))
                total_notes += n
                _accumulate_int(counts, "voice_correct", int((voice_pred == voice_tgt).sum().item()))
                _accumulate_int(counts, "voice_total", n)
                _accumulate_int(counts, "staff_correct", int((staff_pred == staff_tgt).sum().item()))
                _accumulate_int(counts, "staff_total", n)
    _cuda_sync_if_needed(device)
    t1 = time.perf_counter()

    results: Dict[str, float] = {}
    results["voice_acc"] = counts.get("voice_correct", 0) / max(1, counts.get("voice_total", 0))
    results["staff_acc"] = counts.get("staff_correct", 0) / max(1, counts.get("staff_total", 0))
    total_time = max(1e-9, t1 - t0)
    results["inference_ms_per_item"] = (total_time / max(1, len(indices))) * 1000.0
    results["notes_per_sec"] = total_notes / total_time
    return results


def evaluate_key_estimator(
    model,
    items: List[dict],
    *,
    device: torch.device,
    indices: List[int],
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    total_notes = 0

    _cuda_sync_if_needed(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype) if (use_amp and device.type == "cuda") else nullcontext()
        with amp_ctx:
            for idx in indices:
                item = items[idx]
                pitches = torch.tensor(item["pitches"], dtype=torch.long, device=device).unsqueeze(0)
                durations = torch.tensor(item["durations"], dtype=torch.float32, device=device).unsqueeze(0)
                mask = torch.ones_like(pitches, dtype=torch.bool, device=device)
                out = model(pitches, durations, mask)

                key_tgt = int(item["key_index"])
                key_neural = int(out["key_logits"][0].argmax(dim=-1).item())
                key_combined = int((out["key_logits"][0] + out["template_scores"][0]).argmax(dim=-1).item())

                n = int(pitches.size(1))
                total_notes += n
                _accumulate_int(counts, "key_neural_correct", int(key_neural == key_tgt))
                _accumulate_int(counts, "key_combined_correct", int(key_combined == key_tgt))
                _accumulate_int(counts, "key_total", 1)
    _cuda_sync_if_needed(device)
    t1 = time.perf_counter()

    results: Dict[str, float] = {}
    total = max(1, counts.get("key_total", 0))
    results["key_acc_neural"] = counts.get("key_neural_correct", 0) / total
    results["key_acc_combined"] = counts.get("key_combined_correct", 0) / total
    total_time = max(1e-9, t1 - t0)
    results["inference_ms_per_item"] = (total_time / max(1, len(indices))) * 1000.0
    results["notes_per_sec"] = total_notes / total_time
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PM2S checkpoints on a small random subset")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--model-config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints/pm2s")
    parser.add_argument("--rhythm-ckpt", type=str, default=None)
    parser.add_argument("--voice-ckpt", type=str, default=None)
    parser.add_argument("--key-ckpt", type=str, default=None)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--time-scale", type=float, default=100.0)
    parser.add_argument("--tasks", type=str, default="rhythm_quantizer,voice_separator,key_estimator")
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be > 0")
    if not args.time_scale or args.time_scale <= 0:
        raise ValueError("--time-scale must be > 0")

    device = resolve_device(args.device, gpu_id=args.gpu_id)
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    train_cfg = load_yaml_config(args.config)
    pm2s_global_cfg = train_cfg.get("pm2s_training", {})

    ckpt_root = Path(args.checkpoints_dir)
    rhythm_ckpt = Path(args.rhythm_ckpt) if args.rhythm_ckpt else (ckpt_root / "rhythm_quantizer" / "best.pt")
    voice_ckpt = Path(args.voice_ckpt) if args.voice_ckpt else (ckpt_root / "voice_separator" / "best.pt")
    key_ckpt = Path(args.key_ckpt) if args.key_ckpt else (ckpt_root / "key_estimator" / "best.pt")

    components = build_pm2s_components(args.model_config)
    load_checkpoint_into_model(components.rhythm_quantizer, rhythm_ckpt, device=device)
    load_checkpoint_into_model(components.voice_separator, voice_ckpt, device=device)
    load_checkpoint_into_model(components.key_estimator, key_ckpt, device=device)

    rng = random.Random(args.seed)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    print(f"Device: {device} (amp={use_amp}, amp_dtype={args.amp_dtype}, time_scale={args.time_scale})")
    print(f"Samples per task: {args.samples}")
    print("")

    if "rhythm_quantizer" in tasks:
        rhythm_cfg = pm2s_global_cfg.get("rhythm_quantizer", {})
        manifest_path = rhythm_cfg.get("manifest")
        if not manifest_path:
            raise ValueError("Missing pm2s_training.rhythm_quantizer.manifest in training config")
        items = _load_manifest_items(manifest_path)
        indices = rng.sample(range(len(items)), k=min(args.samples, len(items)))
        results = evaluate_rhythm_quantizer(
            components.rhythm_quantizer,
            items,
            device=device,
            indices=indices,
            time_scale=float(args.time_scale),
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        print("Rhythm Quantizer")
        for k, v in results.items():
            print(f"  {k}: {v:.6f}")
        print("")

    if "voice_separator" in tasks:
        voice_cfg = pm2s_global_cfg.get("voice_separator", {})
        manifest_path = voice_cfg.get("manifest")
        if not manifest_path:
            raise ValueError("Missing pm2s_training.voice_separator.manifest in training config")
        items = _load_manifest_items(manifest_path)
        indices = rng.sample(range(len(items)), k=min(args.samples, len(items)))
        results = evaluate_voice_separator(
            components.voice_separator,
            items,
            device=device,
            indices=indices,
            time_scale=float(args.time_scale),
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        print("Voice Separator")
        for k, v in results.items():
            print(f"  {k}: {v:.6f}")
        print("")

    if "key_estimator" in tasks:
        key_cfg = pm2s_global_cfg.get("key_estimator", {})
        manifest_path = key_cfg.get("manifest")
        if not manifest_path:
            raise ValueError("Missing pm2s_training.key_estimator.manifest in training config")
        items = _load_manifest_items(manifest_path)
        indices = rng.sample(range(len(items)), k=min(args.samples, len(items)))
        results = evaluate_key_estimator(
            components.key_estimator,
            items,
            device=device,
            indices=indices,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        print("Key Estimator")
        for k, v in results.items():
            print(f"  {k}: {v:.6f}")
        print("")


if __name__ == "__main__":
    main()
