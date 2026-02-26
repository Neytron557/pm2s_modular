"""
MIDI Tokenizer for Audio-to-Score

Converts between MIDI events and token sequences.

Vocabulary (519 tokens total):
- NOTE_ON_0 to NOTE_ON_127 (128 tokens)
- NOTE_OFF_0 to NOTE_OFF_127 (128 tokens)
- TIME_SHIFT_10ms to TIME_SHIFT_1000ms (100 tokens)
- VELOCITY_0 to VELOCITY_31 (32 tokens)
- PROGRAM_0 to PROGRAM_127 (128 tokens)
- Special: <PAD>, <SOS>, <EOS> (3 tokens)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

try:
    import pretty_midi
except ImportError:
    pretty_midi = None


@dataclass
class TokenizerConfig:
    """Configuration for MIDI tokenizer."""
    # Token counts
    note_on_tokens: int = 128
    note_off_tokens: int = 128
    time_shift_tokens: int = 100
    velocity_tokens: int = 32
    program_tokens: int = 128
    # Time resolution
    time_shift_ms: int = 10  # Resolution in ms
    max_time_shift_ms: int = 1000
    # Velocity quantization
    velocity_bins: int = 32


class MidiTokenizer:
    """
    Tokenizer for converting MIDI to/from token sequences.
    """

    # Special tokens
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        self._build_vocabulary()

    def _build_vocabulary(self):
        """Build token vocabulary."""
        self.token_to_id = {}
        self.id_to_token = {}

        idx = 0

        # Special tokens
        for token in [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]:
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1

        # Note On tokens
        for pitch in range(self.config.note_on_tokens):
            token = f"NOTE_ON_{pitch}"
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1

        # Note Off tokens
        for pitch in range(self.config.note_off_tokens):
            token = f"NOTE_OFF_{pitch}"
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1

        # Time Shift tokens (10ms to 1000ms)
        for t in range(1, self.config.time_shift_tokens + 1):
            ms = t * self.config.time_shift_ms
            token = f"TIME_SHIFT_{ms}ms"
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1

        # Velocity tokens
        for v in range(self.config.velocity_tokens):
            token = f"VELOCITY_{v}"
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1

        # Program tokens (MIDI instruments)
        for p in range(self.config.program_tokens):
            token = f"PROGRAM_{p}"
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1

        self.vocab_size = idx

        # Store token ranges for easy access
        self._special_start = 0
        self._special_end = 3
        self._note_on_start = 3
        self._note_on_end = 3 + 128
        self._note_off_start = 3 + 128
        self._note_off_end = 3 + 256
        self._time_shift_start = 3 + 256
        self._time_shift_end = 3 + 256 + 100
        self._velocity_start = 3 + 356
        self._velocity_end = 3 + 388
        self._program_start = 3 + 388
        self._program_end = 3 + 516

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.PAD_TOKEN]

    @property
    def sos_token_id(self) -> int:
        return self.token_to_id[self.SOS_TOKEN]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.EOS_TOKEN]

    def _quantize_velocity(self, velocity: int) -> int:
        """Quantize velocity (0-127) to bins."""
        return min(velocity * self.config.velocity_bins // 128, self.config.velocity_bins - 1)

    def _dequantize_velocity(self, bin_idx: int) -> int:
        """Convert velocity bin back to MIDI velocity."""
        return int((bin_idx + 0.5) * 128 / self.config.velocity_tokens)

    def _encode_time_shift(self, time_ms: float) -> List[int]:
        """
        Encode a time shift as a sequence of tokens.

        Args:
            time_ms: Time shift in milliseconds

        Returns:
            List of time shift token IDs
        """
        tokens = []
        remaining_ms = int(time_ms)

        while remaining_ms > 0:
            # Use largest possible time shift
            shift_ms = min(remaining_ms, self.config.max_time_shift_ms)
            # Round to nearest multiple of time_shift_ms
            shift_ms = (shift_ms // self.config.time_shift_ms) * self.config.time_shift_ms
            if shift_ms == 0:
                break
            token = f"TIME_SHIFT_{shift_ms}ms"
            tokens.append(self.token_to_id[token])
            remaining_ms -= shift_ms

        return tokens

    def encode_midi_file(
        self,
        midi_path: str,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """
        Encode a MIDI file to token sequence.

        Args:
            midi_path: Path to MIDI file
            add_special_tokens: Whether to add SOS/EOS tokens

        Returns:
            List of token IDs
        """
        if pretty_midi is None:
            raise ImportError("pretty_midi is required for MIDI encoding")

        midi = pretty_midi.PrettyMIDI(str(midi_path))
        return self.encode_midi(midi, add_special_tokens)

    def encode_midi(
        self,
        midi: "pretty_midi.PrettyMIDI",
        add_special_tokens: bool = True,
    ) -> List[int]:
        """
        Encode PrettyMIDI object to token sequence.

        Args:
            midi: PrettyMIDI object
            add_special_tokens: Whether to add SOS/EOS tokens

        Returns:
            List of token IDs
        """
        # Collect all events
        events = []

        for instrument in midi.instruments:
            program = instrument.program

            for note in instrument.notes:
                # Note on event
                events.append({
                    "time": note.start,
                    "type": "note_on",
                    "pitch": note.pitch,
                    "velocity": note.velocity,
                    "program": program,
                })
                # Note off event
                events.append({
                    "time": note.end,
                    "type": "note_off",
                    "pitch": note.pitch,
                    "program": program,
                })

        # Sort by time, then note_off before note_on at same time
        events.sort(key=lambda x: (x["time"], x["type"] == "note_on"))

        # Convert to tokens
        tokens = []
        if add_special_tokens:
            tokens.append(self.sos_token_id)

        current_time = 0.0
        current_velocity = None
        current_program = None

        for event in events:
            # Time shift
            time_diff_ms = (event["time"] - current_time) * 1000
            if time_diff_ms >= self.config.time_shift_ms:
                tokens.extend(self._encode_time_shift(time_diff_ms))
                current_time = event["time"]

            # Program change (only for note_on)
            program = min(event.get("program", 0), 127)
            if program != current_program:
                tokens.append(self.token_to_id[f"PROGRAM_{program}"])
                current_program = program

            if event["type"] == "note_on":
                # Velocity change
                velocity_bin = self._quantize_velocity(event["velocity"])
                if velocity_bin != current_velocity:
                    tokens.append(self.token_to_id[f"VELOCITY_{velocity_bin}"])
                    current_velocity = velocity_bin

                # Note on
                tokens.append(self.token_to_id[f"NOTE_ON_{event['pitch']}"])
            else:
                # Note off
                tokens.append(self.token_to_id[f"NOTE_OFF_{event['pitch']}"])

        if add_special_tokens:
            tokens.append(self.eos_token_id)

        return tokens

    def decode_to_midi(
        self,
        tokens: List[int],
        tempo: float = 120.0,
    ) -> "pretty_midi.PrettyMIDI":
        """
        Decode token sequence to PrettyMIDI object.

        Args:
            tokens: List of token IDs
            tempo: Tempo in BPM

        Returns:
            PrettyMIDI object
        """
        if pretty_midi is None:
            raise ImportError("pretty_midi is required for MIDI decoding")

        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)

        # Track active notes and instruments
        instruments = {}  # program -> Instrument
        active_notes = {}  # (program, pitch) -> list[(start_time, velocity)]

        current_time = 0.0
        current_velocity = 64
        current_program = 0

        for token_id in tokens:
            token = self.id_to_token.get(token_id, "")

            if token in [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]:
                continue

            elif token.startswith("TIME_SHIFT_"):
                ms = int(token.split("_")[2].replace("ms", ""))
                current_time += ms / 1000.0

            elif token.startswith("VELOCITY_"):
                bin_idx = int(token.split("_")[1])
                current_velocity = self._dequantize_velocity(bin_idx)

            elif token.startswith("PROGRAM_"):
                current_program = int(token.split("_")[1])
                if current_program not in instruments:
                    is_drum = current_program >= 128
                    inst = pretty_midi.Instrument(
                        program=min(current_program, 127),
                        is_drum=is_drum,
                    )
                    instruments[current_program] = inst

            elif token.startswith("NOTE_ON_"):
                pitch = int(token.split("_")[2])
                key = (current_program, pitch)
                active_notes.setdefault(key, []).append((current_time, current_velocity))

            elif token.startswith("NOTE_OFF_"):
                pitch = int(token.split("_")[2])
                key = (current_program, pitch)
                if key in active_notes and active_notes[key]:
                    start_time, velocity = active_notes[key].pop(0)
                    if not active_notes[key]:
                        del active_notes[key]
                    if current_program not in instruments:
                        instruments[current_program] = pretty_midi.Instrument(
                            program=current_program
                        )
                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=start_time,
                        end=current_time,
                    )
                    instruments[current_program].notes.append(note)

        # Close any remaining active notes
        for (program, pitch), notes in active_notes.items():
            if program not in instruments:
                instruments[program] = pretty_midi.Instrument(program=program)
            for start_time, velocity in notes:
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=start_time,
                    end=current_time + 0.1,
                )
                instruments[program].notes.append(note)

        # Add instruments to MIDI
        for inst in instruments.values():
            midi.instruments.append(inst)

        return midi

    def save_midi(
        self,
        tokens: List[int],
        output_path: str,
        tempo: float = 120.0,
    ):
        """Save token sequence as MIDI file."""
        midi = self.decode_to_midi(tokens, tempo)
        midi.write(output_path)

    def tokens_to_string(self, tokens: List[int]) -> str:
        """Convert token IDs to readable string."""
        return " ".join(self.id_to_token.get(t, f"<UNK_{t}>") for t in tokens)

    def __len__(self) -> int:
        return self.vocab_size

    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary dictionary."""
        return self.token_to_id.copy()


class NoteEventTokenizer:
    """
    Alternative tokenizer using note events directly.
    Each note is represented as (pitch, onset_frame, duration_frames, velocity).
    """

    def __init__(
        self,
        frame_rate: int = 100,  # Frames per second
        max_duration_frames: int = 500,  # ~5 seconds at 100fps
    ):
        self.frame_rate = frame_rate
        self.max_duration_frames = max_duration_frames

    def midi_to_note_events(
        self,
        midi: "pretty_midi.PrettyMIDI",
    ) -> List[Tuple[int, int, int, int]]:
        """
        Convert MIDI to list of note events.

        Returns:
            List of (pitch, onset_frame, duration_frames, velocity)
        """
        events = []

        for instrument in midi.instruments:
            for note in instrument.notes:
                onset_frame = int(note.start * self.frame_rate)
                offset_frame = int(note.end * self.frame_rate)
                duration_frames = min(
                    offset_frame - onset_frame,
                    self.max_duration_frames,
                )
                events.append((
                    note.pitch,
                    onset_frame,
                    duration_frames,
                    note.velocity,
                ))

        # Sort by onset time
        events.sort(key=lambda x: x[1])
        return events

    def note_events_to_piano_roll(
        self,
        events: List[Tuple[int, int, int, int]],
        num_frames: Optional[int] = None,
    ) -> np.ndarray:
        """
        Convert note events to piano roll.

        Args:
            events: List of (pitch, onset_frame, duration_frames, velocity)
            num_frames: Total number of frames

        Returns:
            Piano roll array (128, num_frames)
        """
        if not events:
            return np.zeros((128, 1))

        if num_frames is None:
            num_frames = max(e[1] + e[2] for e in events) + 1

        piano_roll = np.zeros((128, num_frames), dtype=np.float32)

        for pitch, onset, duration, velocity in events:
            end_frame = min(onset + duration, num_frames)
            piano_roll[pitch, onset:end_frame] = velocity / 127.0

        return piano_roll
