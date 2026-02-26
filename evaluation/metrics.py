"""
Evaluation Metrics for Audio-to-Score

Implements:
- Note-level F1 (Onset, Onset+Offset)
- Frame-level F1
- MV2H Score (Multi-pitch Voice, Meter, Value, Harmony)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

try:
    import mir_eval
    MIR_EVAL_AVAILABLE = True
except ImportError:
    MIR_EVAL_AVAILABLE = False


@dataclass
class NoteEvent:
    """Represents a note for evaluation."""
    onset: float       # Onset time in seconds
    offset: float      # Offset time in seconds
    pitch: int         # MIDI pitch
    velocity: int = 64 # Optional velocity


def notes_to_intervals_pitches(
    notes: List[NoteEvent],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert note events to mir_eval format.

    Returns:
        intervals: (N, 2) array of onset/offset times
        pitches: (N,) array of pitches in Hz (for mir_eval)
    """
    if not notes:
        return np.zeros((0, 2)), np.zeros(0)

    intervals = np.array([[n.onset, n.offset] for n in notes])
    pitches_midi = np.array([n.pitch for n in notes], dtype=float)
    if MIR_EVAL_AVAILABLE:
        pitches = mir_eval.util.midi_to_hz(pitches_midi)
    else:
        # Fallback conversion (A4=440Hz) to keep a consistent representation.
        pitches = 440.0 * (2.0 ** ((pitches_midi - 69.0) / 12.0))

    return intervals, pitches


def note_f1(
    ref_notes: List[NoteEvent],
    est_notes: List[NoteEvent],
    onset_tolerance: float = 0.05,
    offset_ratio: Optional[float] = 0.2,
    offset_min_tolerance: float = 0.05,
    pitch_tolerance_cents: float = 50.0,
) -> Dict[str, float]:
    """
    Compute note-level F1 score (mir_eval transcription metrics).

    Args:
        ref_notes: Reference (ground truth) notes
        est_notes: Estimated (predicted) notes
        onset_tolerance: Onset tolerance in seconds (default 50ms)
        offset_ratio: Offset tolerance as ratio of note length (default 20%)
        offset_min_tolerance: Minimum offset tolerance in seconds
        pitch_tolerance_cents: Pitch tolerance in cents (default 50 cents = 1/2 semitone)

    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    if not MIR_EVAL_AVAILABLE:
        raise ImportError("mir_eval is required for evaluation")

    ref_intervals, ref_pitches = notes_to_intervals_pitches(ref_notes)
    est_intervals, est_pitches = notes_to_intervals_pitches(est_notes)

    # Onset-only evaluation
    onset_precision, onset_recall, onset_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches,
        est_intervals, est_pitches,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=float(pitch_tolerance_cents),
        offset_ratio=None,  # Onset only
    )

    # Onset + Offset evaluation
    if offset_ratio is not None:
        full_precision, full_recall, full_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, ref_pitches,
            est_intervals, est_pitches,
            onset_tolerance=onset_tolerance,
            pitch_tolerance=float(pitch_tolerance_cents),
            offset_ratio=offset_ratio,
            offset_min_tolerance=offset_min_tolerance,
        )
    else:
        full_precision, full_recall, full_f1 = onset_precision, onset_recall, onset_f1

    return {
        "onset_precision": onset_precision,
        "onset_recall": onset_recall,
        "onset_f1": onset_f1,
        "onset_offset_precision": full_precision,
        "onset_offset_recall": full_recall,
        "onset_offset_f1": full_f1,
    }


def frame_f1(
    ref_notes: List[NoteEvent],
    est_notes: List[NoteEvent],
    frame_rate: int = 100,
    duration: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute frame-level F1 score.

    Args:
        ref_notes: Reference notes
        est_notes: Estimated notes
        frame_rate: Frames per second
        duration: Total duration (if None, uses max offset)

    Returns:
        Dictionary with frame-level metrics
    """
    if not ref_notes and not est_notes:
        return {"frame_precision": 1.0, "frame_recall": 1.0, "frame_f1": 1.0}

    # Determine duration
    if duration is None:
        all_notes = ref_notes + est_notes
        duration = max(n.offset for n in all_notes) if all_notes else 0

    num_frames = int(duration * frame_rate) + 1

    # Create piano roll matrices
    ref_roll = np.zeros((128, num_frames), dtype=bool)
    est_roll = np.zeros((128, num_frames), dtype=bool)

    for note in ref_notes:
        start_frame = int(note.onset * frame_rate)
        end_frame = int(note.offset * frame_rate)
        ref_roll[note.pitch, start_frame:end_frame] = True

    for note in est_notes:
        start_frame = int(note.onset * frame_rate)
        end_frame = int(note.offset * frame_rate)
        est_roll[note.pitch, start_frame:end_frame] = True

    # Compute metrics
    true_positives = np.sum(ref_roll & est_roll)
    false_positives = np.sum(est_roll & ~ref_roll)
    false_negatives = np.sum(ref_roll & ~est_roll)

    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "frame_precision": precision,
        "frame_recall": recall,
        "frame_f1": f1,
    }


def compute_mv2h(
    ref_score: Dict,
    est_score: Dict,
) -> Dict[str, float]:
    """
    Compute MV2H score for score-level evaluation.

    MV2H evaluates:
    - Multi-pitch: Correct notes
    - Voice: Correct voice assignment
    - Meter: Correct beat/measure alignment
    - Value: Correct note durations
    - Harmony: Correct key/spelling

    Args:
        ref_score: Reference score data
        est_score: Estimated score data

    Returns:
        Dictionary with MV2H component scores
    """
    # Simplified MV2H implementation
    # Full implementation would require aligned score data

    scores = {}

    # Multi-pitch score (based on note F1)
    if "notes" in ref_score and "notes" in est_score:
        note_metrics = note_f1(ref_score["notes"], est_score["notes"])
        scores["multi_pitch"] = note_metrics["onset_offset_f1"]
    else:
        scores["multi_pitch"] = 0.0

    # Voice score (if voice assignments available)
    if "voices" in ref_score and "voices" in est_score:
        scores["voice"] = _compute_voice_score(
            ref_score["voices"], est_score["voices"]
        )
    else:
        scores["voice"] = 0.0

    # Meter score (beat alignment)
    if "beats" in ref_score and "beats" in est_score:
        scores["meter"] = _compute_meter_score(
            ref_score["beats"], est_score["beats"]
        )
    else:
        scores["meter"] = 0.0

    # Value score (note durations)
    if "durations" in ref_score and "durations" in est_score:
        scores["value"] = _compute_value_score(
            ref_score["durations"], est_score["durations"]
        )
    else:
        scores["value"] = 0.0

    # Harmony score (key/spelling)
    if "key" in ref_score and "key" in est_score:
        scores["harmony"] = 1.0 if ref_score["key"] == est_score["key"] else 0.0
    else:
        scores["harmony"] = 0.0

    # Overall MV2H score
    scores["mv2h"] = np.mean([
        scores["multi_pitch"],
        scores["voice"],
        scores["meter"],
        scores["value"],
        scores["harmony"],
    ])

    return scores


def _compute_voice_score(ref_voices: List[int], est_voices: List[int]) -> float:
    """Compute voice assignment accuracy."""
    if len(ref_voices) != len(est_voices):
        return 0.0

    correct = sum(1 for r, e in zip(ref_voices, est_voices) if r == e)
    return correct / len(ref_voices)


def _compute_meter_score(ref_beats: List[float], est_beats: List[float]) -> float:
    """Compute beat alignment score."""
    if not ref_beats or not est_beats:
        return 0.0

    tolerance = 0.07  # 70ms tolerance

    # Match beats
    matched = 0
    for ref_beat in ref_beats:
        for est_beat in est_beats:
            if abs(ref_beat - est_beat) < tolerance:
                matched += 1
                break

    precision = matched / len(est_beats)
    recall = matched / len(ref_beats)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def _compute_value_score(
    ref_durations: List[float],
    est_durations: List[float],
) -> float:
    """Compute note duration accuracy."""
    if len(ref_durations) != len(est_durations):
        return 0.0

    # Tolerance: 10% or 50ms, whichever is larger
    correct = 0
    for ref, est in zip(ref_durations, est_durations):
        tolerance = max(0.1 * ref, 0.05)
        if abs(ref - est) <= tolerance:
            correct += 1

    return correct / len(ref_durations)


def compute_mv2h_from_notes(
    ref_notes: List[NoteEvent],
    est_notes: List[NoteEvent],
    ref_voices: List[int],
    est_voices: List[int],
    ref_beats: List[float],
    est_beats: List[float],
    ref_key: str,
    est_key: str,
    onset_tolerance: float = 0.05,
    offset_ratio: float = 0.2,
) -> Dict[str, float]:
    """
    Compute MV2H score from paired note-level data extracted from music21 scores.

    Args:
        ref_notes: Reference notes (onset/offset/pitch)
        est_notes: Estimated notes
        ref_voices: Per-note voice IDs for reference (same order as ref_notes)
        est_voices: Per-note voice IDs for estimated (same order as est_notes)
        ref_beats: Beat onset positions (e.g. in quarterLength)
        est_beats: Beat onset positions for estimated score
        ref_key: Key signature string (e.g. "C major")
        est_key: Key signature string
        onset_tolerance: Onset tolerance for note matching
        offset_ratio: Offset tolerance as ratio of note length

    Returns:
        Dictionary with {multi_pitch, voice, meter, value, harmony, mv2h}
    """
    scores: Dict[str, float] = {}

    # --- Multi-pitch (M): onset+offset F1 ---
    if ref_notes and est_notes:
        nf1 = note_f1(ref_notes, est_notes, onset_tolerance=onset_tolerance, offset_ratio=offset_ratio)
        scores["multi_pitch"] = nf1["onset_offset_f1"]
    else:
        scores["multi_pitch"] = 0.0

    # --- Match ref/est notes by onset+pitch for Voice and Value ---
    matched_pairs = _match_notes_by_onset_pitch(ref_notes, est_notes, onset_tolerance)

    # --- Voice (V): voice assignment accuracy with permutation handling ---
    if matched_pairs and ref_voices and est_voices:
        ref_v = [ref_voices[ri] for ri, _ in matched_pairs]
        est_v = [est_voices[ei] for _, ei in matched_pairs]
        scores["voice"] = _permutation_voice_accuracy(ref_v, est_v)
    else:
        scores["voice"] = 0.0

    # --- Meter (Mt): beat alignment F1 ---
    scores["meter"] = _compute_meter_score(ref_beats, est_beats)

    # --- Value (Va): duration accuracy for matched notes ---
    if matched_pairs:
        correct = 0
        for ri, ei in matched_pairs:
            ref_dur = ref_notes[ri].offset - ref_notes[ri].onset
            est_dur = est_notes[ei].offset - est_notes[ei].onset
            tolerance = max(0.1 * abs(ref_dur), 0.05)
            if abs(ref_dur - est_dur) <= tolerance:
                correct += 1
        scores["value"] = correct / len(matched_pairs)
    else:
        scores["value"] = 0.0

    # --- Harmony (H): key signature match ---
    scores["harmony"] = 1.0 if ref_key.strip().lower() == est_key.strip().lower() else 0.0

    # --- Overall MV2H ---
    scores["mv2h"] = float(np.mean([
        scores["multi_pitch"],
        scores["voice"],
        scores["meter"],
        scores["value"],
        scores["harmony"],
    ]))

    return scores


def _match_notes_by_onset_pitch(
    ref_notes: List[NoteEvent],
    est_notes: List[NoteEvent],
    onset_tolerance: float,
) -> List[Tuple[int, int]]:
    """Match ref and est notes by onset+pitch within tolerance. Returns (ref_idx, est_idx) pairs."""
    if not ref_notes or not est_notes:
        return []

    matched: List[Tuple[int, int]] = []
    used_est = set()

    # Sort ref by onset for stable matching
    ref_order = sorted(range(len(ref_notes)), key=lambda i: (ref_notes[i].onset, ref_notes[i].pitch))

    for ri in ref_order:
        rn = ref_notes[ri]
        best_ei = -1
        best_dt = float("inf")
        for ei in range(len(est_notes)):
            if ei in used_est:
                continue
            en = est_notes[ei]
            if en.pitch != rn.pitch:
                continue
            dt = abs(rn.onset - en.onset)
            if dt <= onset_tolerance and dt < best_dt:
                best_dt = dt
                best_ei = ei
        if best_ei >= 0:
            matched.append((ri, best_ei))
            used_est.add(best_ei)

    return matched


def _permutation_voice_accuracy(ref_voices: List[int], est_voices: List[int]) -> float:
    """Compute voice accuracy with best permutation mapping (handles label mismatch)."""
    if not ref_voices:
        return 0.0

    # Count co-occurrences to find best mapping
    from collections import Counter
    cooccur: Dict[Tuple[int, int], int] = Counter()
    for rv, ev in zip(ref_voices, est_voices):
        cooccur[(rv, ev)] += 1

    # Greedy best mapping: for each ref voice, pick the est voice with most co-occurrences
    ref_labels = set(ref_voices)
    est_labels = set(est_voices)
    mapping: Dict[int, int] = {}
    used_est_labels = set()

    # Sort by frequency (descending) for greedy assignment
    sorted_pairs = sorted(cooccur.items(), key=lambda x: -x[1])
    for (rv, ev), _ in sorted_pairs:
        if rv not in mapping and ev not in used_est_labels:
            mapping[rv] = ev
            used_est_labels.add(ev)

    correct = sum(1 for rv, ev in zip(ref_voices, est_voices) if mapping.get(rv) == ev)
    return correct / len(ref_voices)


class TranscriptionEvaluator:
    """
    Evaluator for music transcription.

    Computes multiple metrics and aggregates results.
    """

    def __init__(
        self,
        onset_tolerance: float = 0.05,
        offset_ratio: float = 0.2,
        frame_rate: int = 100,
    ):
        self.onset_tolerance = onset_tolerance
        self.offset_ratio = offset_ratio
        self.frame_rate = frame_rate

    def evaluate(
        self,
        ref_notes: List[NoteEvent],
        est_notes: List[NoteEvent],
    ) -> Dict[str, float]:
        """
        Evaluate transcription.

        Args:
            ref_notes: Reference notes
            est_notes: Estimated notes

        Returns:
            Dictionary with all metrics
        """
        results = {}

        # Note-level metrics
        note_metrics = note_f1(
            ref_notes, est_notes,
            onset_tolerance=self.onset_tolerance,
            offset_ratio=self.offset_ratio,
        )
        results.update(note_metrics)

        # Frame-level metrics
        frame_metrics = frame_f1(
            ref_notes, est_notes,
            frame_rate=self.frame_rate,
        )
        results.update(frame_metrics)

        return results

    def evaluate_batch(
        self,
        references: List[List[NoteEvent]],
        estimates: List[List[NoteEvent]],
    ) -> Dict[str, float]:
        """
        Evaluate batch of transcriptions.

        Returns:
            Dictionary with average metrics
        """
        all_metrics = []

        for ref, est in zip(references, estimates):
            metrics = self.evaluate(ref, est)
            all_metrics.append(metrics)

        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0]:
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        return avg_metrics
