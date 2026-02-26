"""
Stage 2 inference: MIDI to MusicXML.
"""

import argparse
from pathlib import Path

import torch

try:
    import pretty_midi
except ImportError:  # pragma: no cover
    pretty_midi = None

from models.pm2s.score_generator import PM2SModel, MUSIC21_AVAILABLE
from utils.pm2s_inference import (
    build_pm2s_components,
    compute_tempo_from_beats,
    load_checkpoint_into_model,
    resolve_device,
)


def predominant_time_signature(midi: "pretty_midi.PrettyMIDI") -> tuple[int, int]:
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

    acc: dict[tuple[int, int], float] = {}
    for i, (t0, sig) in enumerate(changes):
        t1 = float(changes[i + 1][0]) if i + 1 < len(changes) else end_time
        dur = max(0.0, float(t1) - float(t0))
        acc[tuple(sig)] = acc.get(tuple(sig), 0.0) + dur
    return max(acc.items(), key=lambda kv: kv[1])[0]


def extract_midi_data(midi: "pretty_midi.PrettyMIDI", *, device: torch.device) -> dict:
    if pretty_midi is None:
        raise ImportError("pretty_midi is required for MIDI decoding")
    notes = []
    for inst in midi.instruments:
        for note in inst.notes:
            notes.append(
                {
                    "pitch": note.pitch,
                    "onset": note.start,
                    "duration": note.end - note.start,
                    "velocity": note.velocity / 127.0,
                }
            )

    if not notes:
        raise ValueError("No notes found in MIDI file")

    notes.sort(key=lambda x: x["onset"])
    return {
        "pitches": torch.tensor([n["pitch"] for n in notes], device=device, dtype=torch.long),
        "onsets": torch.tensor([n["onset"] for n in notes], device=device, dtype=torch.float32),
        "durations": torch.tensor([n["duration"] for n in notes], device=device, dtype=torch.float32),
        "velocities": torch.tensor([n["velocity"] for n in notes], device=device, dtype=torch.float32),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert MIDI to MusicXML")
    parser.add_argument("input", type=str, help="Input MIDI file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output MusicXML path")
    parser.add_argument("--simple", action="store_true", help="Use direct music21 conversion")
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|cuda:<id>")
    parser.add_argument("--gpu-id", type=int, default=None, help="CUDA device id to use when --device=cuda/auto")
    parser.add_argument("--time-scale", type=float, default=100.0, help="Must match training time scaling")
    parser.add_argument(
        "--tempo-source",
        type=str,
        default="model",
        choices=["model", "midi"],
        help="Tempo used for seconds->quarterLength conversion (model prediction vs MIDI tempo map).",
    )
    parser.add_argument(
        "--time-signature-source",
        type=str,
        default="midi",
        choices=["model", "midi"],
        help="Time signature used for beat-unit conversion (model prediction vs MIDI metadata).",
    )
    parser.add_argument("--model-config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints/pm2s")
    parser.add_argument("--rhythm-ckpt", type=str, default=None)
    parser.add_argument("--voice-ckpt", type=str, default=None)
    parser.add_argument("--key-ckpt", type=str, default=None)
    args = parser.parse_args()

    midi_path = Path(args.input)
    output_path = Path(args.output) if args.output else midi_path.with_suffix(".musicxml")

    if args.simple:
        if not MUSIC21_AVAILABLE:
            raise ImportError("music21 is required for simple conversion")
        from music21 import converter
        score = converter.parse(str(midi_path))
        score = score.makeNotation(inPlace=False)
        score.write("musicxml", fp=str(output_path))
        return

    if pretty_midi is None:
        raise ImportError("pretty_midi is required for PM2S conversion")
    if not args.time_scale or args.time_scale <= 0:
        raise ValueError("--time-scale must be > 0")

    device = resolve_device(args.device, gpu_id=args.gpu_id)
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    midi_data = extract_midi_data(midi, device=device)

    tempo_override = None
    if args.tempo_source == "midi":
        tempo_override = compute_tempo_from_beats(midi.get_beats().tolist()) or 120.0

    time_sig_override = None
    if args.time_signature_source == "midi":
        time_sig_override = predominant_time_signature(midi)

    components = build_pm2s_components(args.model_config)
    ckpt_root = Path(args.checkpoints_dir)
    rhythm_ckpt = Path(args.rhythm_ckpt) if args.rhythm_ckpt else (ckpt_root / "rhythm_quantizer" / "best.pt")
    voice_ckpt = Path(args.voice_ckpt) if args.voice_ckpt else (ckpt_root / "voice_separator" / "best.pt")
    key_ckpt = Path(args.key_ckpt) if args.key_ckpt else (ckpt_root / "key_estimator" / "best.pt")
    for p in (rhythm_ckpt, voice_ckpt, key_ckpt):
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")

    load_checkpoint_into_model(components.rhythm_quantizer, rhythm_ckpt, device=device)
    load_checkpoint_into_model(components.voice_separator, voice_ckpt, device=device)
    load_checkpoint_into_model(components.key_estimator, key_ckpt, device=device)

    pm2s = PM2SModel(
        rhythm_quantizer=components.rhythm_quantizer,
        voice_separator=components.voice_separator,
        key_estimator=components.key_estimator,
    )
    result = pm2s.process(
        midi_data,
        title=midi_path.stem,
        time_scale=float(args.time_scale),
        tempo_bpm_override=tempo_override,
        time_signature_override=time_sig_override,
    )

    if hasattr(result, "write"):
        result.write("musicxml", fp=str(output_path))
    else:
        raise RuntimeError("MusicXML generation requires music21 to be installed")


if __name__ == "__main__":
    main()
