"""
Evaluation runner for Audio-to-Score.

Computes note and frame metrics for paired MIDI predictions and ground truth.
"""

import argparse
from pathlib import Path
from typing import List

try:
    import pretty_midi
except ImportError:  # pragma: no cover - optional dependency
    pretty_midi = None

from evaluation.metrics import (
    NoteEvent,
    note_f1,
    frame_f1,
    compute_mv2h,
)


def midi_to_notes(midi_path: Path) -> List[NoteEvent]:
    if pretty_midi is None:
        raise ImportError("pretty_midi is required for evaluation")

    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []
    for inst in midi.instruments:
        for n in inst.notes:
            notes.append(NoteEvent(onset=n.start, offset=n.end, pitch=n.pitch, velocity=n.velocity))
    return notes


def build_score_dict(notes: List[NoteEvent]) -> dict:
    return {
        "notes": notes,
        "durations": [n.offset - n.onset for n in notes],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate transcription outputs")
    parser.add_argument("--predictions", type=str, required=True, help="Predictions directory")
    parser.add_argument("--ground_truth", type=str, required=True, help="Ground truth directory")
    parser.add_argument(
        "--metrics",
        type=str,
        default="note_f1,frame_f1",
        help="Comma-separated metrics: note_f1,frame_f1,mv2h",
    )
    args = parser.parse_args()

    pred_dir = Path(args.predictions)
    gt_dir = Path(args.ground_truth)

    metrics = {m.strip() for m in args.metrics.split(",") if m.strip()}

    pred_files = sorted(pred_dir.glob("*.mid")) + sorted(pred_dir.glob("*.midi"))
    if not pred_files:
        raise FileNotFoundError(f"No MIDI predictions found in {pred_dir}")

    all_results = []

    for pred_path in pred_files:
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            print(f"Skipping {pred_path.name}: ground truth not found")
            continue

        ref_notes = midi_to_notes(gt_path)
        est_notes = midi_to_notes(pred_path)

        result = {}
        if "note_f1" in metrics:
            result.update(note_f1(ref_notes, est_notes))
        if "frame_f1" in metrics:
            result.update(frame_f1(ref_notes, est_notes))
        if "mv2h" in metrics:
            result.update(compute_mv2h(build_score_dict(ref_notes), build_score_dict(est_notes)))

        all_results.append(result)

    if not all_results:
        print("No matching prediction/ground-truth pairs found.")
        return

    avg = {}
    for key in all_results[0].keys():
        avg[key] = sum(r[key] for r in all_results) / len(all_results)

    print("\nAverage Metrics:")
    for key, value in avg.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
