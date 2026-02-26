"""
Score Generator Module for PM2S (Stage 2)

Generates MusicXML output from quantized MIDI data.
"""

import torch
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

try:
    from music21 import stream, note, chord, meter, key, clef, metadata, duration, tempo
    from music21 import pitch as m21_pitch
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False


# Standard musical durations in quarterLength, sorted descending.
# Covers whole notes down to 32nd notes, including dotted and triplet variants.
_STANDARD_DURATIONS = sorted([
    8.0,            # breve
    6.0,            # dotted whole
    4.0,            # whole
    3.0,            # dotted half
    2.0,            # half
    4.0 / 3.0,      # half-note triplet
    1.5,            # dotted quarter
    1.0,            # quarter
    2.0 / 3.0,      # quarter-note triplet
    0.75,           # dotted eighth
    0.5,            # eighth
    1.0 / 3.0,      # eighth-note triplet
    0.375,          # dotted sixteenth
    0.25,           # sixteenth
    1.0 / 6.0,      # sixteenth-note triplet
    0.125,          # thirty-second
])


def _snap_duration(ql: float) -> float:
    """Snap a quarterLength value to the nearest standard musical duration."""
    if ql <= 0:
        return 0.125
    best = min(_STANDARD_DURATIONS, key=lambda d: abs(d - ql))
    return best


# Standard onset grid: multiples of 1/12 ql (supports triplets + 16th notes).
_ONSET_GRID = 1.0 / 12.0


def _snap_onset(ql: float) -> float:
    """Snap an onset position to a 1/12 ql grid (triplets + 16th notes)."""
    if ql <= 0:
        return 0.0
    return round(ql / _ONSET_GRID) * _ONSET_GRID


@dataclass
class NoteEvent:
    """Represents a single note event for score generation."""
    pitch: int              # MIDI pitch
    onset_beat: float       # Onset in beats
    duration_beats: float   # Duration in beats
    velocity: int           # MIDI velocity
    voice: int = 0          # Voice number
    staff: int = 1          # Staff number (1=treble, 2=bass)
    spelling: str = None    # Note name with accidental (e.g., "C#4")


@dataclass
class ScoreMetadata:
    """Score metadata."""
    title: str = "Untitled"
    composer: str = "Unknown"
    tempo: float = 120.0
    time_signature: Tuple[int, int] = (4, 4)
    key_signature: Tuple[str, str] = ("C", "major")


class ScoreGenerator:
    """
    Generates MusicXML scores from quantized note data.

    Uses music21 library for score construction and export.
    """

    def __init__(self):
        if not MUSIC21_AVAILABLE:
            raise ImportError(
                "music21 is required for score generation. "
                "Install with: pip install music21"
            )

    def generate_score(
        self,
        notes: List[NoteEvent],
        score_metadata: Optional[ScoreMetadata] = None,
    ) -> "stream.Score":
        """
        Generate a music21 Score from note events.

        Args:
            notes: List of NoteEvent objects
            score_metadata: Optional metadata for the score

        Returns:
            music21 Score object
        """
        meta = score_metadata or ScoreMetadata()

        # Create score
        score = stream.Score()

        # Add metadata
        score.insert(0, metadata.Metadata())
        score.metadata.title = meta.title
        score.metadata.composer = meta.composer

        # Create parts (one for each staff)
        treble_part = stream.Part()
        bass_part = stream.Part()

        treble_part.id = "Piano-Right"
        bass_part.id = "Piano-Left"

        # Add clefs
        treble_part.insert(0, clef.TrebleClef())
        bass_part.insert(0, clef.BassClef())

        # Add key signature
        ks = self._create_key_signature(meta.key_signature)
        treble_part.insert(0, ks)
        bass_part.insert(0, ks)

        # Add time signature
        ts = meter.TimeSignature(f"{meta.time_signature[0]}/{meta.time_signature[1]}")
        treble_part.insert(0, ts)
        bass_part.insert(0, ts)

        # Add tempo (affects MIDI/MusicXML playback timing)
        try:
            bpm = float(meta.tempo)
        except Exception:
            bpm = 120.0
        if not bpm or bpm <= 0:
            bpm = 120.0
        # MusicXML exporters (and some viewers) expect tempo markings to live in a Part.
        # Inserting it once is enough for a grand staff score.
        treble_part.insert(0, tempo.MetronomeMark(number=bpm))

        # Group notes by staff
        treble_notes = [n for n in notes if n.staff == 1]
        bass_notes = [n for n in notes if n.staff == 2]

        # Add notes to parts
        self._add_notes_to_part(treble_part, treble_notes, meta.time_signature)
        self._add_notes_to_part(bass_part, bass_notes, meta.time_signature)

        # Create proper measure structure, barlines, beaming, and ties.
        try:
            treble_part = treble_part.makeNotation()
            treble_part.id = "Piano-Right"
        except Exception:
            pass
        try:
            bass_part = bass_part.makeNotation()
            bass_part.id = "Piano-Left"
        except Exception:
            pass

        # Add parts to score
        score.insert(0, treble_part)
        score.insert(0, bass_part)

        return score

    def _create_key_signature(
        self,
        key_sig: Tuple[str, str],
    ) -> "key.KeySignature":
        """Create music21 key signature."""
        root, mode = key_sig
        return key.Key(root, mode)

    def _add_notes_to_part(
        self,
        part: "stream.Part",
        notes: List[NoteEvent],
        time_sig: Tuple[int, int],
    ):
        """Add notes to a part, handling voice separation."""
        if not notes:
            return

        for n in sorted(notes, key=lambda x: (x.onset_beat, x.pitch, x.voice)):
            if n.duration_beats is None or float(n.duration_beats) <= 0:
                continue
            m21_note = self._create_note(n)
            try:
                m21_note.voice = str(int(n.voice))
            except Exception:
                pass
            offset = _snap_onset(float(max(0.0, n.onset_beat)))
            part.insert(offset, m21_note)

    def _create_note(self, note_event: NoteEvent) -> "note.Note":
        """Create a music21 Note from NoteEvent."""
        if note_event.spelling:
            n = note.Note(note_event.spelling)
        else:
            n = note.Note(note_event.pitch)

        # Snap duration to the nearest standard musical value so that
        # music21 can represent it without complex tuplets.
        beats = _snap_duration(float(note_event.duration_beats))
        n.duration = duration.Duration(beats)

        n.volume.velocity = note_event.velocity
        return n

    def _beats_to_duration(self, beats: float) -> "duration.Duration":
        """Convert beat duration to music21 Duration."""
        # Common duration mappings
        dur_map = {
            4.0: "whole",
            3.0: "half",  # Dotted half
            2.0: "half",
            1.5: "quarter",  # Dotted quarter
            1.0: "quarter",
            0.75: "eighth",  # Dotted eighth
            0.5: "eighth",
            0.375: "16th",  # Dotted 16th
            0.25: "16th",
            0.125: "32nd",
        }

        # Find closest duration
        closest = min(dur_map.keys(), key=lambda x: abs(x - beats))
        d = duration.Duration(dur_map[closest])

        # Add dots if needed
        if beats in [3.0, 1.5, 0.75, 0.375]:
            d.dots = 1

        return d

    def save_musicxml(
        self,
        notes: List[NoteEvent],
        output_path: str,
        score_metadata: Optional[ScoreMetadata] = None,
    ):
        """Save notes as MusicXML file."""
        score = self.generate_score(notes, score_metadata)
        score.write("musicxml", fp=output_path)

    def save_pdf(
        self,
        notes: List[NoteEvent],
        output_path: str,
        score_metadata: Optional[ScoreMetadata] = None,
    ):
        """Save notes as PDF (requires MuseScore or LilyPond)."""
        score = self.generate_score(notes, score_metadata)
        score.write("musicxml.pdf", fp=output_path)


class PM2SModel:
    """
    Complete PM2S (Performance MIDI to Score) Pipeline

    Combines all PM2S components:
    1. Rhythm Quantization
    2. Voice Separation
    3. Key Estimation
    4. Score Generation
    """

    def __init__(
        self,
        rhythm_quantizer: Optional[Any] = None,
        voice_separator: Optional[Any] = None,
        key_estimator: Optional[Any] = None,
    ):
        self.rhythm_quantizer = rhythm_quantizer
        self.voice_separator = voice_separator
        self.key_estimator = key_estimator
        self.score_generator = ScoreGenerator() if MUSIC21_AVAILABLE else None

    def process(
        self,
        midi_data: Dict[str, torch.Tensor],
        title: str = "Transcription",
        *,
        time_scale: float = 1.0,
        tempo_bpm_override: Optional[float] = None,
        time_signature_override: Optional[Tuple[int, int]] = None,
    ) -> "stream.Score":
        """
        Full PM2S pipeline.

        Args:
            midi_data: Dictionary with:
                - pitches: MIDI pitches (seq_len,)
                - onsets: Onset times in seconds (seq_len,)
                - durations: Durations in seconds (seq_len,)
                - velocities: Velocities (seq_len,)
            title: Score title

        Returns:
            music21 Score object
        """
        pitches = midi_data["pitches"]
        onsets = midi_data["onsets"]
        durations = midi_data["durations"]
        velocities = midi_data["velocities"]

        # Ensure batch dimension
        if pitches.dim() == 1:
            pitches = pitches.unsqueeze(0)
            onsets = onsets.unsqueeze(0)
            durations = durations.unsqueeze(0)
            velocities = velocities.unsqueeze(0)

        # 1. Rhythm Quantization
        if self.rhythm_quantizer:
            quant_result = self.rhythm_quantizer.quantize(
                pitches,
                onsets,
                durations,
                velocities,
                time_scale=time_scale,
                tempo_bpm_override=tempo_bpm_override,
                time_signature_override=time_signature_override,
            )
            quantized_beats = quant_result["quantized_beats"][0]
            tempo = quant_result["tempo"][0].item()
            time_sig = quant_result["time_signature"][0]
            beat_unit_ql = float(
                quant_result.get("beat_unit_quarter_length", torch.tensor([1.0], device=quantized_beats.device))[0].item()
            )
        else:
            # Fallback: assume 120 BPM, 4/4
            tempo = 120.0
            quantized_beats = onsets[0] * tempo / 60
            time_sig = (4, 4)
            beat_unit_ql = 1.0

        # 2. Voice Separation
        if self.voice_separator:
            offsets = onsets + durations
            ts = float(time_scale)
            if not ts or ts <= 0:
                ts = 1.0
            voice_ids, staff_ids = self.voice_separator.separate_voices(
                pitches, onsets / ts, offsets / ts
            )
            voice_ids = voice_ids[0]
            staff_ids = staff_ids[0]
        else:
            # Fallback: split at middle C
            voice_ids = torch.zeros_like(pitches[0])
            staff_ids = (pitches[0] >= 60).long()

        # 3. Key Estimation
        if self.key_estimator:
            keys = self.key_estimator.estimate_key(pitches, durations)
            key_sig = (keys[0][0], keys[0][1])
        else:
            key_sig = ("C", "major")

        # 4. Create note events
        beat_durations = durations[0] * (tempo / 60.0) * float(beat_unit_ql)
        note_events = []

        for i in range(len(pitches[0])):
            # VoiceSeparator outputs: 0=bass, 1=treble
            # ScoreGenerator expects: staff=1 (treble), staff=2 (bass)
            staff_assignment = 1 if staff_ids[i].item() == 1 else 2
            note_events.append(NoteEvent(
                pitch=pitches[0, i].item(),
                onset_beat=quantized_beats[i].item(),
                duration_beats=beat_durations[i].item(),
                velocity=int(velocities[0, i].item() * 127),
                voice=voice_ids[i].item(),
                staff=staff_assignment,
            ))

        # 5. Generate score
        # RhythmQuantizer tempo is in "beat" units (per pretty_midi.get_beats). Music21's
        # MetronomeMark expects quarter-note BPM by default, so convert here.
        tempo_quarter = float(tempo) * float(beat_unit_ql)
        metadata = ScoreMetadata(
            title=title,
            tempo=tempo_quarter,
            time_signature=time_sig,
            key_signature=key_sig,
        )

        if self.score_generator:
            return self.score_generator.generate_score(note_events, metadata)
        else:
            return note_events
