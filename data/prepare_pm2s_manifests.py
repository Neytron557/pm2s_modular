#!/usr/bin/env python3
"""
Prepare PM2S manifests (rhythm quantizer, voice separator, key estimator).

This script builds piano-only manifests by merging multiple datasets:
ASAP, ACPAS, PDMX, ATEPP, MSMD plus key-only pretrain sets.
"""

import argparse
import csv
import json
import math
import os
import time
from bisect import bisect_right
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import pretty_midi
except ImportError as exc:
    raise ImportError("pretty_midi is required for manifest preparation") from exc

try:
    from music21 import converter, note as m21_note, chord as m21_chord
except ImportError:
    converter = None
    m21_note = None
    m21_chord = None


PIANO_PROGRAMS = set(range(0, 8))
TIME_SIG_MAP = {
    (2, 4): 0,
    (3, 4): 1,
    (4, 4): 2,
    (6, 8): 3,
    (9, 8): 4,
    (12, 8): 5,
}
SUBDIVISION_MAP = [1, 2, 4, 8, 16, 3, 6, 12]


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "tasks": {}, "items": {}}
    try:
        with path.open("r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {"version": 1, "tasks": {}, "items": {}}


def save_state(path: Path, state: dict) -> None:
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w") as f:
        json.dump(state, f, indent=2)
    tmp_path.replace(path)


def ensure_dataset_state(state: dict, task: str, dataset: str) -> dict:
    tasks = state.setdefault("tasks", {})
    task_state = tasks.setdefault(task, {})
    dataset_state = task_state.setdefault(dataset, {"index": 0, "items": 0})
    state.setdefault("items", {}).setdefault(task, 0)
    return dataset_state


def reset_task_state(state: dict, task: str) -> None:
    if "tasks" in state and task in state["tasks"]:
        state["tasks"].pop(task)
    if "items" in state and task in state["items"]:
        state["items"].pop(task)


def format_eta(seconds: float) -> str:
    if seconds < 0 or math.isinf(seconds) or math.isnan(seconds):
        return "unknown"
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def open_jsonl_writer(path: Path, resume: bool) -> object:
    mode = "a" if resume and path.exists() else "w"
    return path.open(mode)


def append_jsonl(writer, items: List[dict]) -> int:
    for item in items:
        writer.write(json.dumps(item) + "\n")
    return len(items)


def log_error(error_log, context: str, exc: Exception) -> None:
    if error_log is None:
        return
    error_log.write(f"{context}\t{type(exc).__name__}\t{exc}\n")
    error_log.flush()


def iter_with_progress(iterable, total: int, desc: str, start: int):
    if tqdm is not None:
        return tqdm(iterable, total=total, initial=start, desc=desc)
    return iterable


def process_file_list(
    task: str,
    dataset: str,
    files: List,
    process_fn,
    state: dict,
    state_path: Path,
    writer,
    error_log,
    save_every: int,
    progress_every: int,
) -> None:
    dataset_state = ensure_dataset_state(state, task, dataset)
    start_idx = int(dataset_state.get("index", 0))
    total = len(files)
    if start_idx >= total:
        return

    start_time = time.time()
    iterable = iter_with_progress(range(start_idx, total), total=total, desc=f"{task}/{dataset}", start=start_idx)

    processed = 0
    for idx in iterable:
        try:
            items = process_fn(files[idx])
        except Exception as exc:
            log_error(error_log, f"{task}/{dataset}:{files[idx]}", exc)
            items = []

        if items:
            for item in items:
                if isinstance(item, dict):
                    item.setdefault("source", dataset)
            added = append_jsonl(writer, items)
            dataset_state["items"] = dataset_state.get("items", 0) + added
            state["items"][task] = state["items"].get(task, 0) + added

        processed += 1
        if (idx + 1) % save_every == 0 or (idx + 1) == total:
            dataset_state["index"] = idx + 1
            save_state(state_path, state)
            writer.flush()

        if tqdm is None and ((idx + 1) % progress_every == 0 or (idx + 1) == total):
            elapsed = max(time.time() - start_time, 1e-6)
            rate = processed / elapsed
            remaining = total - (idx + 1)
            eta = format_eta(remaining / rate) if rate > 0 else "unknown"
            print(f"[{task}/{dataset}] {idx + 1}/{total} files, {rate:.2f} files/s, ETA {eta}")

    dataset_state["index"] = total
    save_state(state_path, state)


def build_pt_from_jsonl(jsonl_path: Path, pt_path: Path) -> int:
    items = []
    if not jsonl_path.exists():
        return 0
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    torch.save({"items": items}, pt_path)
    return len(items)


def safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def normalize_velocity(velocity: float) -> float:
    return max(0.0, min(1.0, velocity / 127.0))


def is_piano_program(program: int) -> bool:
    return program in PIANO_PROGRAMS


def extract_midi_notes(path: Path, strict_piano: bool = True) -> Optional[List[pretty_midi.Note]]:
    try:
        midi = pretty_midi.PrettyMIDI(str(path))
    except Exception:
        return None

    notes: List[pretty_midi.Note] = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        if strict_piano and inst.notes and not is_piano_program(inst.program):
            return None
        if strict_piano and not is_piano_program(inst.program):
            continue
        notes.extend(inst.notes)

    if not notes:
        return None
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


def chunk_notes(
    notes: List[pretty_midi.Note],
    max_notes: int,
    min_notes: int,
) -> List[List[pretty_midi.Note]]:
    if len(notes) < min_notes:
        return []
    if len(notes) <= max_notes:
        return [notes]
    chunks = []
    for i in range(0, len(notes), max_notes):
        chunk = notes[i:i + max_notes]
        if len(chunk) >= min_notes:
            chunks.append(chunk)
    return chunks


def build_time_sig_changes_from_asap(ts_dict: Dict[str, List]) -> List[Tuple[float, Tuple[int, int]]]:
    entries = []
    for k, v in ts_dict.items():
        if not v:
            continue
        sig = v[0]
        parts = sig.split("/")
        if len(parts) != 2:
            continue
        num = int(parts[0])
        den = int(parts[1])
        entries.append((safe_float(k), (num, den)))
    return sorted(entries, key=lambda x: x[0])


def build_time_sig_changes_from_acpas(labels: List[Tuple[float, str]]) -> List[Tuple[float, Tuple[int, int]]]:
    entries = []
    for t, label in labels:
        tokens = [tok.strip() for tok in label.split(",") if tok.strip()]
        for tok in tokens:
            if "/" in tok:
                parts = tok.split("/")
                if len(parts) == 2:
                    try:
                        entries.append((t, (int(parts[0]), int(parts[1]))))
                    except ValueError:
                        pass
    return sorted(entries, key=lambda x: x[0])


def build_time_sig_changes_from_midi(midi: pretty_midi.PrettyMIDI) -> List[Tuple[float, Tuple[int, int]]]:
    entries = []
    for ts in midi.time_signature_changes:
        entries.append((ts.time, (ts.numerator, ts.denominator)))
    if not entries:
        entries.append((0.0, (4, 4)))
    return sorted(entries, key=lambda x: x[0])


def pick_time_signature_index(time_sigs: List[Tuple[float, Tuple[int, int]]], time_sec: float) -> Optional[int]:
    if not time_sigs:
        return TIME_SIG_MAP.get((4, 4))
    idx = bisect_right([t for t, _ in time_sigs], time_sec) - 1
    idx = max(idx, 0)
    sig = time_sigs[idx][1]
    return TIME_SIG_MAP.get(sig)


def build_beat_numbers(beat_times: List[float], downbeat_times: List[float]) -> List[int]:
    downbeat_set = {round(t, 6) for t in downbeat_times}
    beat_numbers: List[int] = []
    current = 0
    for t in beat_times:
        if round(t, 6) in downbeat_set:
            current = 0
        beat_numbers.append(current)
        current += 1
    return beat_numbers


def compute_tempo_from_beats(beat_times: List[float]) -> Optional[float]:
    if len(beat_times) < 2:
        return None
    intervals = []
    for i in range(len(beat_times) - 1):
        dt = beat_times[i + 1] - beat_times[i]
        if dt > 0:
            intervals.append(60.0 / dt)
    if not intervals:
        return None
    intervals.sort()
    return intervals[len(intervals) // 2]


def nearest_subdivision_index(beat_fraction: float) -> int:
    beat_fraction = max(0.0, min(beat_fraction, 0.999999))
    best_idx = 0
    best_err = 1e9
    for idx, denom in enumerate(SUBDIVISION_MAP):
        num = int(round(beat_fraction * denom))
        num = max(0, min(num, denom - 1))
        frac = num / float(denom)
        err = abs(frac - beat_fraction)
        if err < best_err:
            best_err = err
            best_idx = idx
    return best_idx


def build_rhythm_item(
    notes: List[pretty_midi.Note],
    beat_times: List[float],
    downbeat_times: List[float],
    time_sigs: List[Tuple[float, Tuple[int, int]]],
) -> Optional[dict]:
    if len(beat_times) < 2:
        return None
    beat_numbers = build_beat_numbers(beat_times, downbeat_times)
    tempo = compute_tempo_from_beats(beat_times)
    if tempo is None:
        return None

    pitches = []
    onsets = []
    durations = []
    velocities = []
    beat_fraction = []
    beat_number = []
    subdivision = []

    beat_times_list = beat_times

    for n in notes:
        t = n.start
        i = bisect_right(beat_times_list, t) - 1
        if i < 0 or i >= len(beat_times_list) - 1:
            continue
        beat_start = beat_times_list[i]
        beat_end = beat_times_list[i + 1]
        if beat_end <= beat_start:
            continue
        frac = (t - beat_start) / (beat_end - beat_start)

        pitches.append(int(n.pitch))
        onsets.append(float(n.start))
        durations.append(float(n.end - n.start))
        velocities.append(normalize_velocity(n.velocity))
        beat_fraction.append(float(frac))
        beat_number.append(int(min(beat_numbers[i], 15)))
        subdivision.append(int(nearest_subdivision_index(frac)))

    if not pitches:
        return None

    time_sig_idx = pick_time_signature_index(time_sigs, onsets[0])
    if time_sig_idx is None:
        return None

    return {
        "pitches": pitches,
        "onsets": onsets,
        "durations": durations,
        "velocities": velocities,
        "beat_fraction": beat_fraction,
        "beat_number": beat_number,
        "subdivision": subdivision,
        "time_signature": time_sig_idx,
        "tempo": float(tempo),
    }


def notes_to_key_item(
    notes: List[pretty_midi.Note],
    key_index: int,
) -> Optional[dict]:
    if not notes:
        return None
    pitches = [int(n.pitch) for n in notes]
    durations = [float(n.end - n.start) for n in notes]
    return {
        "pitches": pitches,
        "durations": durations,
        "key_index": int(key_index),
    }


def estimate_key_index_from_notes(notes: List[pretty_midi.Note]) -> Optional[int]:
    if not notes:
        return None
    hist = [0.0] * 12
    for n in notes:
        pc = n.pitch % 12
        hist[pc] += max(1e-6, n.end - n.start)
    total = sum(hist)
    if total <= 0:
        return None
    hist = [v / total for v in hist]

    major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    major_sum = sum(major_profile)
    minor_sum = sum(minor_profile)
    major_profile = [v / major_sum for v in major_profile]
    minor_profile = [v / minor_sum for v in minor_profile]

    key_signatures = [
        ("C", "major"), ("G", "major"), ("D", "major"), ("A", "major"), ("E", "major"),
        ("B", "major"), ("F#", "major"), ("C#", "major"), ("F", "major"), ("Bb", "major"),
        ("Eb", "major"), ("Ab", "major"), ("Db", "major"), ("Gb", "major"), ("Cb", "major"),
        ("A", "minor"), ("E", "minor"), ("B", "minor"), ("F#", "minor"), ("C#", "minor"),
        ("G#", "minor"), ("D#", "minor"), ("A#", "minor"), ("D", "minor"), ("G", "minor"),
        ("C", "minor"), ("F", "minor"), ("Bb", "minor"), ("Eb", "minor"), ("Ab", "minor"),
    ]
    note_to_pc = {
        "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
        "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10,
        "B": 11, "Cb": 11,
    }

    best_idx = None
    best_score = -1e9
    for i, (root, mode) in enumerate(key_signatures):
        pc = note_to_pc[root]
        profile = major_profile if mode == "major" else minor_profile
        rotated = profile[-pc:] + profile[:-pc]
        score = sum(h * p for h, p in zip(hist, rotated))
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def resolve_staff_id(staff, pitch: int) -> int:
    staff_num = None
    if staff is not None:
        try:
            staff_num = int(staff)
        except Exception:
            staff_num = None
    if staff_num == 1:
        return 1
    if staff_num == 2:
        return 0
    return 0 if pitch < 60 else 1


def build_voice_id_map(
    voice_info: Dict[str, dict],
    voice_order: List[str],
    max_voices: int,
    voice_overflow: str,
) -> Tuple[Optional[Dict[str, int]], int, Optional[set]]:
    if len(voice_info) <= max_voices:
        return {key: idx for idx, key in enumerate(voice_order)}, len(voice_info), None
    if voice_overflow == "drop":
        return None, 0, None

    order_index = {key: idx for idx, key in enumerate(voice_order)}
    sorted_keys = sorted(
        voice_info.keys(),
        key=lambda k: (-voice_info[k]["count"], order_index.get(k, 0)),
    )
    anchors = sorted_keys[:max_voices]
    if voice_overflow == "truncate":
        voice_id_map = {key: idx for idx, key in enumerate(anchors)}
        return voice_id_map, len(anchors), set(anchors)

    voice_id_map = {key: idx for idx, key in enumerate(anchors)}
    anchor_stats = {key: voice_info[key] for key in anchors}
    for key in sorted_keys[max_voices:]:
        extra = voice_info[key]
        best_idx = 0
        best_cost = None
        for idx, anchor_key in enumerate(anchors):
            anchor = anchor_stats[anchor_key]
            cost = abs(extra["mean_pitch"] - anchor["mean_pitch"])
            if extra["staff_id"] != anchor["staff_id"]:
                cost += 12.0
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_idx = idx
        voice_id_map[key] = best_idx
    return voice_id_map, len(anchors), None


def parse_musicxml_notes(
    path: Path,
    max_voices: int = 4,
    voice_overflow: str = "merge",
) -> Optional[Tuple[List[dict], int]]:
    if converter is None:
        raise RuntimeError("music21 is required for MusicXML parsing")
    try:
        score = converter.parse(str(path))
    except Exception:
        return None

    tempo = 120.0
    for mm in score.recurse().getElementsByClass("MetronomeMark"):
        if mm.number:
            tempo = float(mm.number)
            break
    sec_per_quarter = 60.0 / tempo

    raw_notes = []
    for n in score.recurse().notes:
        if isinstance(n, m21_chord.Chord):
            chord_notes = n.notes
            duration = max(0.0, float(n.duration.quarterLength) * sec_per_quarter)
            onset = float(n.offset) * sec_per_quarter
            voice_label = getattr(n, "voice", None)
            if voice_label is None:
                voice_ctx = n.getContextByClass("Voice")
                if voice_ctx is not None:
                    voice_label = getattr(voice_ctx, "id", None) or getattr(voice_ctx, "number", None)
            staff = getattr(n, "staff", None)
            if staff is None:
                staff_ctx = n.getContextByClass("Staff")
                if staff_ctx is not None:
                    staff = getattr(staff_ctx, "number", None)
            for cn in chord_notes:
                pitch = int(cn.pitch.midi)
                raw_notes.append({
                    "pitch": pitch,
                    "onset": onset,
                    "duration": duration,
                    "voice": voice_label,
                    "staff": staff,
                    "staff_id": resolve_staff_id(staff, pitch),
                })
        elif isinstance(n, m21_note.Note):
            voice_label = getattr(n, "voice", None)
            if voice_label is None:
                voice_ctx = n.getContextByClass("Voice")
                if voice_ctx is not None:
                    voice_label = getattr(voice_ctx, "id", None) or getattr(voice_ctx, "number", None)
            staff = getattr(n, "staff", None)
            if staff is None:
                staff_ctx = n.getContextByClass("Staff")
                if staff_ctx is not None:
                    staff = getattr(staff_ctx, "number", None)
            pitch = int(n.pitch.midi)
            raw_notes.append({
                "pitch": pitch,
                "onset": float(n.offset) * sec_per_quarter,
                "duration": max(0.0, float(n.duration.quarterLength) * sec_per_quarter),
                "voice": voice_label,
                "staff": staff,
                "staff_id": resolve_staff_id(staff, pitch),
            })

    if not raw_notes:
        return None

    voice_info: Dict[str, dict] = {}
    voice_order: List[str] = []
    for item in raw_notes:
        voice_label = item.get("voice")
        if voice_label is None:
            voice_label = "0"
        voice_key = str(voice_label)
        if voice_key not in voice_info:
            voice_info[voice_key] = {"count": 0, "pitch_sum": 0.0, "staff_counts": {0: 0, 1: 0}}
            voice_order.append(voice_key)
        voice_info[voice_key]["count"] += 1
        voice_info[voice_key]["pitch_sum"] += float(item["pitch"])
        staff_id = item.get("staff_id", 0)
        voice_info[voice_key]["staff_counts"][staff_id] = voice_info[voice_key]["staff_counts"].get(staff_id, 0) + 1

    for voice_key, stats in voice_info.items():
        count = max(1, stats["count"])
        stats["mean_pitch"] = stats["pitch_sum"] / count
        stats["staff_id"] = 1 if stats["staff_counts"].get(1, 0) >= stats["staff_counts"].get(0, 0) else 0

    voice_id_map, num_voices, allowed_voices = build_voice_id_map(
        voice_info,
        voice_order,
        max_voices,
        voice_overflow,
    )
    if voice_id_map is None:
        return None
    if allowed_voices is not None:
        raw_notes = [item for item in raw_notes if str(item.get("voice") if item.get("voice") is not None else "0") in allowed_voices]
        if not raw_notes:
            return None

    for item in raw_notes:
        voice_key = str(item.get("voice") if item.get("voice") is not None else "0")
        item["voice_id"] = voice_id_map[voice_key]

    raw_notes.sort(key=lambda x: (x["onset"], x["pitch"]))
    return raw_notes, num_voices


def musicxml_notes_to_voice_items(
    raw_notes: List[dict],
    max_notes: int,
    min_notes: int,
) -> List[dict]:
    if len(raw_notes) < min_notes:
        return []

    items = []
    for i in range(0, len(raw_notes), max_notes):
        chunk = raw_notes[i:i + max_notes]
        if len(chunk) < min_notes:
            continue
        items.append({
            "pitches": [n["pitch"] for n in chunk],
            "onsets": [n["onset"] for n in chunk],
            "offsets": [n["onset"] + n["duration"] for n in chunk],
            "voice_ids": [n["voice_id"] for n in chunk],
            "staff_ids": [n["staff_id"] for n in chunk],
        })
    return items


def read_acpas_annotations(path: Path) -> List[Tuple[float, str]]:
    labels = []
    with path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            t = safe_float(parts[0])
            label = parts[1]
            labels.append((t, label))
    return labels


def convert_ticks_to_seconds(ticks: float, resolution: int, tempos: List[Tuple[float, float]]) -> float:
    if resolution <= 0:
        resolution = 480
    safe_tempos = [(t, qpm) for t, qpm in tempos if qpm and qpm > 0]
    if not safe_tempos:
        return ticks * 60.0 / (120.0 * resolution)
    tempos = sorted(safe_tempos, key=lambda x: x[0])
    total_seconds = 0.0
    last_tick = 0.0
    for tempo_tick, qpm in tempos:
        if ticks <= tempo_tick:
            break
        dt_ticks = tempo_tick - last_tick
        total_seconds += dt_ticks * 60.0 / (qpm * resolution)
        last_tick = tempo_tick
    remaining_ticks = ticks - last_tick
    qpm = tempos[-1][1]
    for tempo_tick, qpm_value in tempos:
        if tempo_tick <= ticks:
            qpm = qpm_value
        else:
            break
    if qpm <= 0:
        qpm = 120.0
    total_seconds += remaining_ticks * 60.0 / (qpm * resolution)
    return total_seconds


def load_pdmx_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return None


def build_pdmx_rhythm_items(data: dict, max_notes: int, min_notes: int) -> List[dict]:
    if not data:
        return []
    resolution = int(data.get("resolution", 480))
    tempos = [(float(t["time"]), float(t["qpm"])) for t in data.get("tempos", []) if "time" in t and "qpm" in t]
    beat_times = [
        convert_ticks_to_seconds(float(b["time"]), resolution, tempos)
        for b in data.get("beats", [])
        if "time" in b
    ]
    downbeat_times = [
        convert_ticks_to_seconds(float(b["time"]), resolution, tempos)
        for b in data.get("beats", [])
        if b.get("is_downbeat")
    ]
    beat_times.sort()
    downbeat_times.sort()
    time_sigs = [
        (convert_ticks_to_seconds(float(ts["time"]), resolution, tempos), (int(ts["numerator"]), int(ts["denominator"])))
        for ts in data.get("time_signatures", [])
    ]
    if not time_sigs:
        time_sigs = [(0.0, (4, 4))]

    notes = []
    for track in data.get("tracks", []):
        if track.get("is_drum"):
            continue
        program = track.get("program", 0)
        if not is_piano_program(int(program)) and track.get("notes"):
            return []
        for n in track.get("notes", []):
            start_tick = float(n["time"])
            end_tick = float(n["time"]) + float(n["duration"])
            start = convert_ticks_to_seconds(start_tick, resolution, tempos)
            end = convert_ticks_to_seconds(end_tick, resolution, tempos)
            duration = max(0.0, end - start)
            notes.append(pretty_midi.Note(
                velocity=int(n.get("velocity", 64)),
                pitch=int(n.get("pitch", 60)),
                start=start,
                end=start + duration,
            ))
    notes.sort(key=lambda n: (n.start, n.pitch))
    items = []
    for chunk in chunk_notes(notes, max_notes, min_notes):
        item = build_rhythm_item(chunk, beat_times, downbeat_times, time_sigs)
        if item:
            items.append(item)
    return items


def build_pdmx_key_items(data: dict, max_notes: int, min_notes: int) -> List[dict]:
    if not data:
        return []
    resolution = int(data.get("resolution", 480))
    tempos = [(float(t["time"]), float(t["qpm"])) for t in data.get("tempos", []) if "time" in t and "qpm" in t]
    notes: List[pretty_midi.Note] = []
    for track in data.get("tracks", []):
        if track.get("is_drum"):
            continue
        program = track.get("program", 0)
        if not is_piano_program(int(program)) and track.get("notes"):
            return []
        for n in track.get("notes", []):
            start_tick = float(n["time"])
            end_tick = float(n["time"]) + float(n["duration"])
            start = convert_ticks_to_seconds(start_tick, resolution, tempos)
            end = convert_ticks_to_seconds(end_tick, resolution, tempos)
            duration = max(0.0, end - start)
            notes.append(pretty_midi.Note(
                velocity=int(n.get("velocity", 64)),
                pitch=int(n.get("pitch", 60)),
                start=start,
                end=start + duration,
            ))
    notes.sort(key=lambda n: (n.start, n.pitch))
    key_index = estimate_key_index_from_notes(notes)
    if key_index is None:
        return []
    items = []
    for chunk in chunk_notes(notes, max_notes, min_notes):
        item = notes_to_key_item(chunk, key_index)
        if item:
            items.append(item)
    return items


def process_pdmx_rhythm_file(json_path: Path, max_notes: int, min_notes: int) -> List[dict]:
    data = load_pdmx_json(json_path)
    if not data:
        return []
    return build_pdmx_rhythm_items(data, max_notes, min_notes)


def process_pdmx_key_file(json_path: Path, max_notes: int, min_notes: int) -> List[dict]:
    data = load_pdmx_json(json_path)
    if not data:
        return []
    return build_pdmx_key_items(data, max_notes, min_notes)


def build_rhythm_items_from_midi(
    midi_path: Path,
    max_notes: int,
    min_notes: int,
    strict_piano: bool,
) -> List[dict]:
    notes = extract_midi_notes(midi_path, strict_piano=strict_piano)
    if not notes:
        return []
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception:
        return []
    beat_times = list(midi.get_beats())
    downbeat_times = list(midi.get_downbeats())
    time_sigs = build_time_sig_changes_from_midi(midi)

    items = []
    for chunk in chunk_notes(notes, max_notes, min_notes):
        item = build_rhythm_item(chunk, beat_times, downbeat_times, time_sigs)
        if item:
            items.append(item)
    return items


def build_key_items_from_midi(
    midi_path: Path,
    max_notes: int,
    min_notes: int,
    strict_piano: bool,
) -> List[dict]:
    notes = extract_midi_notes(midi_path, strict_piano=strict_piano)
    if not notes:
        return []
    key_index = estimate_key_index_from_notes(notes)
    if key_index is None:
        return []
    items = []
    for chunk in chunk_notes(notes, max_notes, min_notes):
        item = notes_to_key_item(chunk, key_index)
        if item:
            items.append(item)
    return items


def build_asap_paths(asap_root: Path) -> List[dict]:
    metadata_path = asap_root / "metadata.csv"
    rows = []
    with metadata_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def process_asap_rhythm_row(
    row: dict,
    annotations: dict,
    asap_root: Path,
    max_notes: int,
    min_notes: int,
    strict_piano: bool,
) -> List[dict]:
    perf_path = row.get("midi_performance")
    if not perf_path:
        return []
    ann = annotations.get(perf_path)
    if not ann or not ann.get("score_and_performance_aligned", True):
        return []
    perf_full = asap_root / perf_path
    notes = extract_midi_notes(perf_full, strict_piano=strict_piano)
    if not notes:
        return []
    beat_times = sorted(ann.get("performance_beats", []))
    downbeats = sorted(ann.get("performance_downbeats", []))
    time_sigs = build_time_sig_changes_from_asap(ann.get("perf_time_signatures", {}))
    if not time_sigs:
        time_sigs = [(0.0, (4, 4))]
    items = []
    for chunk in chunk_notes(notes, max_notes, min_notes):
        item = build_rhythm_item(chunk, beat_times, downbeats, time_sigs)
        if item:
            items.append(item)
    return items


def process_asap_voice_row(
    row: dict,
    asap_root: Path,
    max_notes: int,
    min_notes: int,
    max_voices: int,
    voice_overflow: str,
) -> List[dict]:
    xml_path = row.get("xml_score")
    if not xml_path:
        return []
    xml_full = asap_root / xml_path
    if not xml_full.exists():
        return []
    parsed = parse_musicxml_notes(xml_full, max_voices=max_voices, voice_overflow=voice_overflow)
    if not parsed:
        return []
    raw_notes, _ = parsed
    return musicxml_notes_to_voice_items(raw_notes, max_notes, min_notes)


def process_asap_key_row(
    row: dict,
    asap_root: Path,
    max_notes: int,
    min_notes: int,
    strict_piano: bool,
) -> List[dict]:
    midi_path = row.get("midi_score") or row.get("midi_performance")
    if not midi_path:
        return []
    midi_full = asap_root / midi_path
    return build_key_items_from_midi(midi_full, max_notes, min_notes, strict_piano)


def build_acpas_rows(acpas_root: Path) -> List[dict]:
    rows = []
    for csv_name in ["metadata_R.csv", "metadata_S.csv"]:
        path = acpas_root / csv_name
        if not path.exists():
            continue
        with path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows


def process_acpas_rhythm_row(
    row: dict,
    acpas_root: Path,
    max_notes: int,
    min_notes: int,
    strict_piano: bool,
) -> List[dict]:
    if row.get("aligned", "").lower() != "true":
        return []
    folder = row.get("folder")
    perf_midi = row.get("performance_MIDI")
    ann_file = row.get("performance_annotation")
    if not folder or not perf_midi or not ann_file:
        return []
    perf_path = acpas_root / folder / perf_midi
    ann_path = acpas_root / folder / ann_file
    if not perf_path.exists() or not ann_path.exists():
        return []
    labels = read_acpas_annotations(ann_path)
    beat_times = [t for t, label in labels if label.startswith("b") or label.startswith("db")]
    downbeats = [t for t, label in labels if label.startswith("db")]
    beat_times.sort()
    downbeats.sort()
    time_sigs = build_time_sig_changes_from_acpas(labels)
    if not time_sigs:
        time_sigs = [(0.0, (4, 4))]
    notes = extract_midi_notes(perf_path, strict_piano=strict_piano)
    if not notes:
        return []
    items = []
    for chunk in chunk_notes(notes, max_notes, min_notes):
        item = build_rhythm_item(chunk, beat_times, downbeats, time_sigs)
        if item:
            items.append(item)
    return items


def process_acpas_key_row(
    row: dict,
    acpas_root: Path,
    max_notes: int,
    min_notes: int,
    strict_piano: bool,
) -> List[dict]:
    folder = row.get("folder")
    midi_score = row.get("MIDI_score")
    if not folder or not midi_score:
        return []
    midi_path = acpas_root / folder / midi_score
    return build_key_items_from_midi(midi_path, max_notes, min_notes, strict_piano)


def build_asap_rhythm(
    asap_root: Path,
    max_notes: int,
    min_notes: int,
    strict_piano: bool,
) -> List[dict]:
    annotations_path = asap_root / "asap_annotations.json"
    with annotations_path.open() as f:
        annotations = json.load(f)

    items = []
    for row in build_asap_paths(asap_root):
        perf_path = row.get("midi_performance")
        if not perf_path:
            continue
        ann = annotations.get(perf_path)
        if not ann or not ann.get("score_and_performance_aligned", True):
            continue
        perf_full = asap_root / perf_path
        notes = extract_midi_notes(perf_full, strict_piano=strict_piano)
        if not notes:
            continue
        beat_times = sorted(ann.get("performance_beats", []))
        downbeats = sorted(ann.get("performance_downbeats", []))
        time_sigs = build_time_sig_changes_from_asap(ann.get("perf_time_signatures", {}))
        if not time_sigs:
            time_sigs = [(0.0, (4, 4))]
        for chunk in chunk_notes(notes, max_notes, min_notes):
            item = build_rhythm_item(chunk, beat_times, downbeats, time_sigs)
            if item:
                items.append(item)
    return items


def build_asap_voice(
    asap_root: Path,
    max_notes: int,
    min_notes: int,
    max_voices: int,
    voice_overflow: str,
) -> List[dict]:
    items = []
    for row in build_asap_paths(asap_root):
        xml_path = row.get("xml_score")
        if not xml_path:
            continue
        xml_full = asap_root / xml_path
        if not xml_full.exists():
            continue
        parsed = parse_musicxml_notes(xml_full, max_voices=max_voices, voice_overflow=voice_overflow)
        if not parsed:
            continue
        raw_notes, _ = parsed
        items.extend(musicxml_notes_to_voice_items(raw_notes, max_notes, min_notes))
    return items


def build_asap_key(
    asap_root: Path,
    max_notes: int,
    min_notes: int,
    strict_piano: bool,
) -> List[dict]:
    items = []
    for row in build_asap_paths(asap_root):
        midi_path = row.get("midi_score") or row.get("midi_performance")
        if not midi_path:
            continue
        midi_full = asap_root / midi_path
        items.extend(build_key_items_from_midi(midi_full, max_notes, min_notes, strict_piano))
    return items


def build_acpas_rhythm(
    acpas_root: Path,
    max_notes: int,
    min_notes: int,
    strict_piano: bool,
) -> List[dict]:
    items = []
    for row in build_acpas_rows(acpas_root):
        if row.get("aligned", "").lower() != "true":
            continue
        folder = row.get("folder")
        perf_midi = row.get("performance_MIDI")
        ann_file = row.get("performance_annotation")
        if not folder or not perf_midi or not ann_file:
            continue
        perf_path = acpas_root / folder / perf_midi
        ann_path = acpas_root / folder / ann_file
        if not perf_path.exists() or not ann_path.exists():
            continue
        labels = read_acpas_annotations(ann_path)
        beat_times = [t for t, label in labels if label.startswith("b") or label.startswith("db")]
        downbeats = [t for t, label in labels if label.startswith("db")]
        beat_times.sort()
        downbeats.sort()
        time_sigs = build_time_sig_changes_from_acpas(labels)
        if not time_sigs:
            time_sigs = [(0.0, (4, 4))]
        notes = extract_midi_notes(perf_path, strict_piano=strict_piano)
        if not notes:
            continue
        for chunk in chunk_notes(notes, max_notes, min_notes):
            item = build_rhythm_item(chunk, beat_times, downbeats, time_sigs)
            if item:
                items.append(item)
    return items


def build_acpas_key(
    acpas_root: Path,
    max_notes: int,
    min_notes: int,
    strict_piano: bool,
) -> List[dict]:
    items = []
    for row in build_acpas_rows(acpas_root):
        folder = row.get("folder")
        midi_score = row.get("MIDI_score")
        if not folder or not midi_score:
            continue
        midi_path = acpas_root / folder / midi_score
        items.extend(build_key_items_from_midi(midi_path, max_notes, min_notes, strict_piano))
    return items


def build_atepp_rows(atepp_root: Path) -> List[dict]:
    meta_path = atepp_root.parent / "ATEPP-metadata-1.2.csv"
    rows = []
    if not meta_path.exists():
        return rows
    with meta_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def process_atepp_rhythm_row(
    row: dict,
    atepp_root: Path,
    max_notes: int,
    min_notes: int,
    strict_piano: bool,
) -> List[dict]:
    midi_path = row.get("midi_path")
    if not midi_path:
        return []
    midi_full = atepp_root / midi_path
    return build_rhythm_items_from_midi(midi_full, max_notes, min_notes, strict_piano)


def process_atepp_voice_row(
    row: dict,
    atepp_root: Path,
    max_notes: int,
    min_notes: int,
    max_voices: int,
    voice_overflow: str,
) -> List[dict]:
    score_path = row.get("score_path")
    if not score_path:
        return []
    xml_full = atepp_root / score_path
    if not xml_full.exists():
        return []
    parsed = parse_musicxml_notes(xml_full, max_voices=max_voices, voice_overflow=voice_overflow)
    if not parsed:
        return []
    raw_notes, _ = parsed
    return musicxml_notes_to_voice_items(raw_notes, max_notes, min_notes)


def process_atepp_key_row(
    row: dict,
    atepp_root: Path,
    max_notes: int,
    min_notes: int,
    strict_piano: bool,
) -> List[dict]:
    midi_path = row.get("midi_path")
    if not midi_path:
        return []
    midi_full = atepp_root / midi_path
    return build_key_items_from_midi(midi_full, max_notes, min_notes, strict_piano)


def build_atepp_rhythm(
    atepp_root: Path,
    max_notes: int,
    min_notes: int,
    strict_piano: bool,
) -> List[dict]:
    items = []
    for row in build_atepp_rows(atepp_root):
        midi_path = row.get("midi_path")
        if not midi_path:
            continue
        midi_full = atepp_root / midi_path
        items.extend(build_rhythm_items_from_midi(midi_full, max_notes, min_notes, strict_piano))
    return items


def build_atepp_voice(
    atepp_root: Path,
    max_notes: int,
    min_notes: int,
    max_voices: int,
    voice_overflow: str,
) -> List[dict]:
    items = []
    for row in build_atepp_rows(atepp_root):
        score_path = row.get("score_path")
        if not score_path:
            continue
        xml_full = atepp_root / score_path
        if not xml_full.exists():
            continue
        parsed = parse_musicxml_notes(xml_full, max_voices=max_voices, voice_overflow=voice_overflow)
        if not parsed:
            continue
        raw_notes, _ = parsed
        items.extend(musicxml_notes_to_voice_items(raw_notes, max_notes, min_notes))
    return items


def build_atepp_key(
    atepp_root: Path,
    max_notes: int,
    min_notes: int,
    strict_piano: bool,
) -> List[dict]:
    items = []
    for row in build_atepp_rows(atepp_root):
        midi_path = row.get("midi_path")
        if not midi_path:
            continue
        midi_full = atepp_root / midi_path
        items.extend(build_key_items_from_midi(midi_full, max_notes, min_notes, strict_piano))
    return items


def build_msmd_midi_files(msmd_root: Path) -> List[Path]:
    midi_files = []
    for piece_dir in msmd_root.iterdir():
        if not piece_dir.is_dir():
            continue
        direct_midi = list(piece_dir.glob("*.midi"))
        if direct_midi:
            midi_files.extend(direct_midi)
            continue
        scores_dir = piece_dir / "scores"
        if scores_dir.exists():
            midi_files.extend(list(scores_dir.rglob("*.midi")))
    return midi_files


def build_msmd_xml_files(msmd_root: Path) -> List[Path]:
    xml_files = []
    for piece_dir in msmd_root.iterdir():
        if not piece_dir.is_dir():
            continue
        scores_dir = piece_dir / "scores"
        if not scores_dir.exists():
            continue
        candidates = sorted(scores_dir.rglob("*.xml"))
        if candidates:
            xml_files.append(candidates[0])
    return xml_files


def process_msmd_midi_file(
    midi_path: Path,
    max_notes: int,
    min_notes: int,
    strict_piano: bool,
    task: str,
) -> List[dict]:
    if task == "rhythm":
        return build_rhythm_items_from_midi(midi_path, max_notes, min_notes, strict_piano)
    if task == "key":
        return build_key_items_from_midi(midi_path, max_notes, min_notes, strict_piano)
    return []


def process_msmd_xml_file(
    xml_path: Path,
    max_notes: int,
    min_notes: int,
    max_voices: int,
    voice_overflow: str,
) -> List[dict]:
    parsed = parse_musicxml_notes(xml_path, max_voices=max_voices, voice_overflow=voice_overflow)
    if not parsed:
        return []
    raw_notes, _ = parsed
    return musicxml_notes_to_voice_items(raw_notes, max_notes, min_notes)


def process_key_midi_file(
    midi_path: Path,
    max_notes: int,
    min_notes: int,
    strict_piano: bool,
) -> List[dict]:
    return build_key_items_from_midi(midi_path, max_notes, min_notes, strict_piano)


def build_msmd_rhythm(
    msmd_root: Path,
    max_notes: int,
    min_notes: int,
    strict_piano: bool,
) -> List[dict]:
    items = []
    for midi_path in build_msmd_midi_files(msmd_root):
        items.extend(build_rhythm_items_from_midi(midi_path, max_notes, min_notes, strict_piano))
    return items


def build_msmd_voice(
    msmd_root: Path,
    max_notes: int,
    min_notes: int,
    max_voices: int,
    voice_overflow: str,
) -> List[dict]:
    items = []
    for piece_dir in msmd_root.iterdir():
        if not piece_dir.is_dir():
            continue
        xml_files = list((piece_dir / "scores").rglob("*.xml")) if (piece_dir / "scores").exists() else []
        if not xml_files:
            continue
        xml_full = xml_files[0]
        parsed = parse_musicxml_notes(xml_full, max_voices=max_voices, voice_overflow=voice_overflow)
        if not parsed:
            continue
        raw_notes, _ = parsed
        items.extend(musicxml_notes_to_voice_items(raw_notes, max_notes, min_notes))
    return items


def build_msmd_key(
    msmd_root: Path,
    max_notes: int,
    min_notes: int,
    strict_piano: bool,
) -> List[dict]:
    items = []
    for midi_path in build_msmd_midi_files(msmd_root):
        items.extend(build_key_items_from_midi(midi_path, max_notes, min_notes, strict_piano))
    return items


def build_maestro_key(
    maestro_root: Path,
    max_notes: int,
    min_notes: int,
    strict_piano: bool,
    max_files: Optional[int],
) -> List[dict]:
    midi_files = list(maestro_root.rglob("*.midi"))
    if max_files:
        midi_files = midi_files[:max_files]
    items = []
    for midi_path in midi_files:
        items.extend(build_key_items_from_midi(midi_path, max_notes, min_notes, strict_piano))
    return items


def build_lakh_key(
    lakh_root: Path,
    max_notes: int,
    min_notes: int,
    strict_piano: bool,
    max_files: Optional[int],
) -> List[dict]:
    midi_files = list(lakh_root.rglob("*.mid"))
    if max_files:
        midi_files = midi_files[:max_files]
    items = []
    for midi_path in midi_files:
        items.extend(build_key_items_from_midi(midi_path, max_notes, min_notes, strict_piano))
    return items


def build_musicnet_key(
    musicnet_root: Path,
    max_notes: int,
    min_notes: int,
    strict_piano: bool,
    max_files: Optional[int],
) -> List[dict]:
    midi_dirs = [musicnet_root / "train_labels_midi", musicnet_root / "test_labels_midi"]
    midi_files: List[Path] = []
    for d in midi_dirs:
        if d.exists():
            midi_files.extend(list(d.rglob("*.mid")))
            midi_files.extend(list(d.rglob("*.midi")))
    if max_files:
        midi_files = midi_files[:max_files]
    items = []
    for midi_path in midi_files:
        items.extend(build_key_items_from_midi(midi_path, max_notes, min_notes, strict_piano))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PM2S manifests")
    parser.add_argument("--data-config", type=str, default="configs/data_config.yaml")
    parser.add_argument("--output-dir", type=str, default="datasets/pm2s_manifests")
    parser.add_argument("--tasks", type=str, default="rhythm,voice,key")
    parser.add_argument("--max-notes", type=int, default=4096)
    parser.add_argument("--min-notes", type=int, default=16)
    parser.add_argument("--max-voices", type=int, default=4)
    parser.add_argument("--voice-overflow", choices=["merge", "truncate", "drop"], default="merge")
    parser.add_argument("--strict-piano", action="store_true")
    parser.add_argument("--no-strict-piano", dest="strict_piano", action="store_false")
    parser.set_defaults(strict_piano=True)
    parser.add_argument("--max-files-pdmx", type=int, default=None)
    parser.add_argument("--max-files-maestro", type=int, default=None)
    parser.add_argument("--max-files-lakh", type=int, default=None)
    parser.add_argument("--max-files-musicnet", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--output-format", choices=["pt", "jsonl", "both"], default="both")
    parser.add_argument("--state-file", type=str, default=None)
    parser.add_argument("--reset-tasks", type=str, default=None)
    args = parser.parse_args()
    args.save_every = max(1, args.save_every)
    args.progress_every = max(1, args.progress_every)

    repo_root = Path(__file__).resolve().parents[1]
    data_cfg = load_yaml(Path(args.data_config))
    dataset_cfg = data_cfg.get("datasets", {})

    def ds_path(name: str, fallback: Path) -> Path:
        entry = dataset_cfg.get(name, {})
        return Path(entry.get("path", fallback))

    asap_root = ds_path("asap", repo_root / "datasets" / "asap")
    acpas_root = ds_path("acpas", repo_root / "datasets" / "acpas")
    pdmx_root = ds_path("pdmx", repo_root / "datasets" / "pdmx" / "PDMX")
    atepp_root = repo_root / "datasets" / "ATEPP" / "ATEPP-1.2"
    msmd_root = ds_path("msmd", repo_root / "datasets" / "msmd" / "msmd_aug_v1-1_no-audio")
    maestro_root = ds_path("maestro", Path("/workspace/datasets/maestro_v3.0.0"))
    lakh_root = ds_path("lakh_midi", repo_root / "datasets" / "lakh_midi" / "lmd_full")
    musicnet_root = ds_path("musicnet", repo_root / "datasets" / "musicnet" / "musicnet_yourmt3_16k")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}
    state_path = Path(args.state_file) if args.state_file else (output_dir / "manifest_state.json")
    error_log_path = output_dir / "manifest_errors.log"

    if args.overwrite:
        for suffix in ["rhythm_piano_all.jsonl", "voice_piano_all.jsonl", "key_piano_all.jsonl",
                       "rhythm_piano_all.pt", "voice_piano_all.pt", "key_piano_all.pt",
                       "manifest_meta.json", "manifest_state.json", "manifest_errors.log"]:
            path = output_dir / suffix
            if path.exists():
                path.unlink()

    resume = args.resume and state_path.exists() and not args.overwrite
    state = load_state(state_path) if resume else {"version": 1, "tasks": {}, "items": {}}
    reset_tasks = {t.strip() for t in (args.reset_tasks or "").split(",") if t.strip()}
    if "all" in reset_tasks:
        reset_tasks = {"rhythm", "voice", "key"}
    for task in sorted(reset_tasks):
        reset_task_state(state, task)
        jsonl_path = output_dir / f"{task}_piano_all.jsonl"
        pt_path = output_dir / f"{task}_piano_all.pt"
        if jsonl_path.exists():
            jsonl_path.unlink()
        if pt_path.exists():
            pt_path.unlink()

    error_log = error_log_path.open("a") if error_log_path.exists() or resume else error_log_path.open("w")
    writers = {}

    if "rhythm" in tasks:
        rhythm_jsonl = output_dir / "rhythm_piano_all.jsonl"
        resume_rhythm = resume and "rhythm" not in reset_tasks
        if resume and not rhythm_jsonl.exists():
            print("Resume requested but rhythm JSONL not found. Restarting rhythm task.")
            reset_task_state(state, "rhythm")
            resume_rhythm = False
        writers["rhythm"] = open_jsonl_writer(rhythm_jsonl, resume_rhythm)

        annotations_path = asap_root / "asap_annotations.json"
        with annotations_path.open() as f:
            asap_annotations = json.load(f)
        asap_rows = sorted(build_asap_paths(asap_root), key=lambda r: r.get("midi_performance", ""))
        process_file_list(
            "rhythm",
            "asap",
            asap_rows,
            lambda row: process_asap_rhythm_row(row, asap_annotations, asap_root, args.max_notes, args.min_notes, args.strict_piano),
            state,
            state_path,
            writers["rhythm"],
            error_log,
            args.save_every,
            args.progress_every,
        )

        acpas_rows = sorted(build_acpas_rows(acpas_root), key=lambda r: r.get("performance_id", ""))
        process_file_list(
            "rhythm",
            "acpas",
            acpas_rows,
            lambda row: process_acpas_rhythm_row(row, acpas_root, args.max_notes, args.min_notes, args.strict_piano),
            state,
            state_path,
            writers["rhythm"],
            error_log,
            args.save_every,
            args.progress_every,
        )

        pdmx_data_root = pdmx_root / "data" if (pdmx_root / "data").exists() else pdmx_root
        pdmx_files = sorted(pdmx_data_root.rglob("*.json"))
        if args.max_files_pdmx:
            pdmx_files = pdmx_files[:args.max_files_pdmx]
        process_file_list(
            "rhythm",
            "pdmx",
            pdmx_files,
            lambda path: process_pdmx_rhythm_file(path, args.max_notes, args.min_notes),
            state,
            state_path,
            writers["rhythm"],
            error_log,
            args.save_every,
            args.progress_every,
        )

        atepp_rows = sorted(build_atepp_rows(atepp_root), key=lambda r: r.get("midi_path", ""))
        process_file_list(
            "rhythm",
            "atepp",
            atepp_rows,
            lambda row: process_atepp_rhythm_row(row, atepp_root, args.max_notes, args.min_notes, args.strict_piano),
            state,
            state_path,
            writers["rhythm"],
            error_log,
            args.save_every,
            args.progress_every,
        )

        msmd_midi_files = sorted(build_msmd_midi_files(msmd_root))
        process_file_list(
            "rhythm",
            "msmd",
            msmd_midi_files,
            lambda path: process_msmd_midi_file(path, args.max_notes, args.min_notes, args.strict_piano, "rhythm"),
            state,
            state_path,
            writers["rhythm"],
            error_log,
            args.save_every,
            args.progress_every,
        )
        writers["rhythm"].close()

    if "voice" in tasks:
        if converter is None:
            raise RuntimeError("music21 is required to build voice manifests")
        voice_jsonl = output_dir / "voice_piano_all.jsonl"
        resume_voice = resume and "voice" not in reset_tasks
        if resume and not voice_jsonl.exists():
            print("Resume requested but voice JSONL not found. Restarting voice task.")
            reset_task_state(state, "voice")
            resume_voice = False
        writers["voice"] = open_jsonl_writer(voice_jsonl, resume_voice)

        asap_rows = sorted(build_asap_paths(asap_root), key=lambda r: r.get("xml_score", ""))
        process_file_list(
            "voice",
            "asap",
            asap_rows,
            lambda row: process_asap_voice_row(
                row,
                asap_root,
                args.max_notes,
                args.min_notes,
                args.max_voices,
                args.voice_overflow,
            ),
            state,
            state_path,
            writers["voice"],
            error_log,
            args.save_every,
            args.progress_every,
        )

        atepp_rows = sorted(build_atepp_rows(atepp_root), key=lambda r: r.get("score_path", ""))
        process_file_list(
            "voice",
            "atepp",
            atepp_rows,
            lambda row: process_atepp_voice_row(
                row,
                atepp_root,
                args.max_notes,
                args.min_notes,
                args.max_voices,
                args.voice_overflow,
            ),
            state,
            state_path,
            writers["voice"],
            error_log,
            args.save_every,
            args.progress_every,
        )

        msmd_xml_files = sorted(build_msmd_xml_files(msmd_root))
        process_file_list(
            "voice",
            "msmd",
            msmd_xml_files,
            lambda path: process_msmd_xml_file(
                path,
                args.max_notes,
                args.min_notes,
                args.max_voices,
                args.voice_overflow,
            ),
            state,
            state_path,
            writers["voice"],
            error_log,
            args.save_every,
            args.progress_every,
        )
        writers["voice"].close()

    if "key" in tasks:
        key_jsonl = output_dir / "key_piano_all.jsonl"
        resume_key = resume and "key" not in reset_tasks
        if resume and not key_jsonl.exists():
            print("Resume requested but key JSONL not found. Restarting key task.")
            reset_task_state(state, "key")
            resume_key = False
        writers["key"] = open_jsonl_writer(key_jsonl, resume_key)

        asap_rows = sorted(build_asap_paths(asap_root), key=lambda r: r.get("midi_score", ""))
        process_file_list(
            "key",
            "asap",
            asap_rows,
            lambda row: process_asap_key_row(row, asap_root, args.max_notes, args.min_notes, args.strict_piano),
            state,
            state_path,
            writers["key"],
            error_log,
            args.save_every,
            args.progress_every,
        )

        acpas_rows = sorted(build_acpas_rows(acpas_root), key=lambda r: r.get("performance_id", ""))
        process_file_list(
            "key",
            "acpas",
            acpas_rows,
            lambda row: process_acpas_key_row(row, acpas_root, args.max_notes, args.min_notes, args.strict_piano),
            state,
            state_path,
            writers["key"],
            error_log,
            args.save_every,
            args.progress_every,
        )

        atepp_rows = sorted(build_atepp_rows(atepp_root), key=lambda r: r.get("midi_path", ""))
        process_file_list(
            "key",
            "atepp",
            atepp_rows,
            lambda row: process_atepp_key_row(row, atepp_root, args.max_notes, args.min_notes, args.strict_piano),
            state,
            state_path,
            writers["key"],
            error_log,
            args.save_every,
            args.progress_every,
        )

        msmd_midi_files = sorted(build_msmd_midi_files(msmd_root))
        process_file_list(
            "key",
            "msmd",
            msmd_midi_files,
            lambda path: process_msmd_midi_file(path, args.max_notes, args.min_notes, args.strict_piano, "key"),
            state,
            state_path,
            writers["key"],
            error_log,
            args.save_every,
            args.progress_every,
        )

        pdmx_data_root = pdmx_root / "data" if (pdmx_root / "data").exists() else pdmx_root
        pdmx_files = sorted(pdmx_data_root.rglob("*.json"))
        if args.max_files_pdmx:
            pdmx_files = pdmx_files[:args.max_files_pdmx]
        process_file_list(
            "key",
            "pdmx",
            pdmx_files,
            lambda path: process_pdmx_key_file(path, args.max_notes, args.min_notes),
            state,
            state_path,
            writers["key"],
            error_log,
            args.save_every,
            args.progress_every,
        )

        maestro_files = sorted(maestro_root.rglob("*.midi"))
        if args.max_files_maestro:
            maestro_files = maestro_files[:args.max_files_maestro]
        process_file_list(
            "key",
            "maestro",
            maestro_files,
            lambda path: process_key_midi_file(path, args.max_notes, args.min_notes, args.strict_piano),
            state,
            state_path,
            writers["key"],
            error_log,
            args.save_every,
            args.progress_every,
        )

        lakh_files = sorted(lakh_root.rglob("*.mid")) + sorted(lakh_root.rglob("*.midi"))
        if args.max_files_lakh:
            lakh_files = lakh_files[:args.max_files_lakh]
        process_file_list(
            "key",
            "lakh",
            lakh_files,
            lambda path: process_key_midi_file(path, args.max_notes, args.min_notes, args.strict_piano),
            state,
            state_path,
            writers["key"],
            error_log,
            args.save_every,
            args.progress_every,
        )

        musicnet_dirs = [musicnet_root / "train_labels_midi", musicnet_root / "test_labels_midi"]
        musicnet_files: List[Path] = []
        for d in musicnet_dirs:
            if d.exists():
                musicnet_files.extend(sorted(d.rglob("*.mid")))
                musicnet_files.extend(sorted(d.rglob("*.midi")))
        if args.max_files_musicnet:
            musicnet_files = musicnet_files[:args.max_files_musicnet]
        process_file_list(
            "key",
            "musicnet",
            musicnet_files,
            lambda path: process_key_midi_file(path, args.max_notes, args.min_notes, args.strict_piano),
            state,
            state_path,
            writers["key"],
            error_log,
            args.save_every,
            args.progress_every,
        )
        writers["key"].close()

    error_log.close()

    if args.output_format in ("pt", "both"):
        if "rhythm" in tasks:
            build_pt_from_jsonl(output_dir / "rhythm_piano_all.jsonl", output_dir / "rhythm_piano_all.pt")
        if "voice" in tasks:
            build_pt_from_jsonl(output_dir / "voice_piano_all.jsonl", output_dir / "voice_piano_all.pt")
        if "key" in tasks:
            build_pt_from_jsonl(output_dir / "key_piano_all.jsonl", output_dir / "key_piano_all.pt")

    manifest_meta = {
        "rhythm_items": state.get("items", {}).get("rhythm", 0),
        "voice_items": state.get("items", {}).get("voice", 0),
        "key_items": state.get("items", {}).get("key", 0),
    }
    meta_path = output_dir / "manifest_meta.json"
    with meta_path.open("w") as f:
        json.dump(manifest_meta, f, indent=2)
    print(f"Wrote manifests to {output_dir}")
    print(json.dumps(manifest_meta, indent=2))


if __name__ == "__main__":
    main()
