"""
Full Inference Pipeline for Audio-to-Score

Transcribes audio files to MusicXML scores.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Union, List

import torch
import torchaudio

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.tokenizer import MidiTokenizer
from data.preprocess import AudioPreprocessor, AudioConfig


class AudioToScoreTranscriber:
    """
    End-to-end Audio-to-Score transcription pipeline.

    Stages:
    1. Audio preprocessing
    2. Acoustic model (Audio → MIDI tokens)
    3. Token decoding (tokens → MIDI)
    4. PM2S processing (MIDI → Score)
    5. MusicXML export
    """

    def __init__(
        self,
        acoustic_model_path: Optional[str] = None,
        pm2s_model_path: Optional[str] = None,
        device: str = "auto",
    ):
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Audio preprocessor
        self.preprocessor = AudioPreprocessor(AudioConfig(
            sample_rate=16000,
            n_mels=128,
            hop_length=160,
        ))

        # Tokenizer
        self.tokenizer = MidiTokenizer()

        # Load models
        self.acoustic_model = None
        self.pm2s_model = None

        if acoustic_model_path:
            self._load_acoustic_model(acoustic_model_path)

        if pm2s_model_path:
            self._load_pm2s_model(pm2s_model_path)

    def _load_acoustic_model(self, path: str):
        """Load acoustic model from checkpoint."""
        from models.acoustic_model import AcousticModel

        print(f"Loading acoustic model from {path}...")
        self.acoustic_model = AcousticModel.load(path, map_location=str(self.device))
        self.acoustic_model.to(self.device)
        self.acoustic_model.eval()

    def _load_pm2s_model(self, path: str):
        """Load PM2S model from checkpoint."""
        # TODO: Implement PM2S model loading
        print(f"PM2S model loading not yet implemented: {path}")

    def transcribe(
        self,
        audio_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        output_format: str = "musicxml",
        temperature: float = 0.0,
        beam_size: int = 1,
        max_length: int = 4096,
    ) -> dict:
        """
        Transcribe audio file to score.

        Args:
            audio_path: Path to input audio file
            output_path: Path for output file (optional)
            output_format: Output format ("musicxml", "midi", "both")
            temperature: Sampling temperature (0 = greedy)
            beam_size: Beam size for beam search
            max_length: Maximum output sequence length

        Returns:
            Dictionary with transcription results
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if beam_size != 1:
            print("Warning: beam search is not implemented; using greedy/sampling decoding.")

        print(f"Transcribing: {audio_path}")

        # Step 1: Load and preprocess audio
        print("  Loading audio...")
        waveform, sample_rate = self.preprocessor.load_audio(audio_path)
        waveform = waveform.to(self.device)

        audio_length = torch.tensor([waveform.shape[-1]], device=self.device)

        # Step 2: Run acoustic model
        if self.acoustic_model is None:
            raise RuntimeError("Acoustic model not loaded. Provide acoustic_model_path.")

        print("  Running acoustic model...")
        with torch.no_grad():
            tokens = self.acoustic_model.transcribe(
                waveform.unsqueeze(0),  # Add batch dim
                audio_length,
                max_length=max_length,
                temperature=temperature,
                sos_token_id=self.tokenizer.sos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        tokens = tokens[0].cpu().tolist()  # Remove batch dim
        print(f"  Generated {len(tokens)} tokens")

        # Step 3: Decode tokens to MIDI
        print("  Decoding to MIDI...")
        midi = self.tokenizer.decode_to_midi(tokens)

        # Calculate some statistics
        num_notes = sum(len(inst.notes) for inst in midi.instruments)
        duration = midi.get_end_time()
        print(f"  Detected {num_notes} notes, duration: {duration:.2f}s")

        # Step 4: Save outputs
        results = {
            "tokens": tokens,
            "num_notes": num_notes,
            "duration": duration,
        }

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_format in ["midi", "both"]:
                midi_path = output_path.with_suffix(".mid")
                midi.write(str(midi_path))
                print(f"  Saved MIDI: {midi_path}")
                results["midi_path"] = str(midi_path)

            if output_format in ["musicxml", "both"]:
                # Step 5: Run PM2S and generate MusicXML
                xml_path = output_path.with_suffix(".musicxml")

                if self.pm2s_model:
                    # Use PM2S model for score generation
                    score = self._run_pm2s(midi)
                    score.write("musicxml", fp=str(xml_path))
                else:
                    # Fallback: Direct MIDI to MusicXML conversion
                    self._midi_to_musicxml_simple(midi, xml_path)

                print(f"  Saved MusicXML: {xml_path}")
                results["musicxml_path"] = str(xml_path)

        return results

    def _run_pm2s(self, midi):
        """Run PM2S pipeline on MIDI."""
        from models.pm2s.score_generator import PM2SModel

        # Extract note data from MIDI
        notes = []
        for inst in midi.instruments:
            for note in inst.notes:
                notes.append({
                    "pitch": note.pitch,
                    "onset": note.start,
                    "duration": note.end - note.start,
                    "velocity": note.velocity / 127.0,
                })

        if not notes:
            return None

        # Sort by onset
        notes.sort(key=lambda x: x["onset"])

        # Convert to tensors
        midi_data = {
            "pitches": torch.tensor([n["pitch"] for n in notes]),
            "onsets": torch.tensor([n["onset"] for n in notes]),
            "durations": torch.tensor([n["duration"] for n in notes]),
            "velocities": torch.tensor([n["velocity"] for n in notes]),
        }

        # Run PM2S
        pm2s = PM2SModel()
        return pm2s.process(midi_data)

    def _midi_to_musicxml_simple(self, midi, output_path):
        """Simple MIDI to MusicXML conversion using music21."""
        try:
            from music21 import converter, stream, note as m21_note, meter

            # Create score
            score = stream.Score()
            part = stream.Part()

            # Add time signature
            part.append(meter.TimeSignature("4/4"))

            # Add notes
            for inst in midi.instruments:
                for n in inst.notes:
                    m21_n = m21_note.Note(n.pitch)
                    m21_n.duration.quarterLength = (n.end - n.start) * 2  # Approximate
                    m21_n.offset = n.start * 2
                    part.append(m21_n)

            score.append(part)
            score.write("musicxml", fp=str(output_path))

        except ImportError:
            print("  Warning: music21 not available, skipping MusicXML generation")

    def transcribe_batch(
        self,
        audio_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        **kwargs,
    ) -> List[dict]:
        """Transcribe multiple audio files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for audio_path in audio_paths:
            audio_path = Path(audio_path)
            output_path = output_dir / audio_path.stem

            try:
                result = self.transcribe(audio_path, output_path, **kwargs)
                results.append({"path": str(audio_path), "success": True, **result})
            except Exception as e:
                print(f"Error transcribing {audio_path}: {e}")
                results.append({"path": str(audio_path), "success": False, "error": str(e)})

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio to MusicXML score"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input audio file or directory",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file or directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to acoustic model checkpoint",
    )
    parser.add_argument(
        "--pm2s-model",
        type=str,
        default=None,
        help="Path to PM2S model checkpoint",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["musicxml", "midi", "both"],
        default="musicxml",
        help="Output format",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (cuda, cpu, or auto)",
    )

    args = parser.parse_args()

    # Create transcriber
    transcriber = AudioToScoreTranscriber(
        acoustic_model_path=args.model,
        pm2s_model_path=args.pm2s_model,
        device=args.device,
    )

    input_path = Path(args.input)

    if input_path.is_dir():
        # Batch transcription
        audio_files = list(input_path.glob("*.wav")) + \
                      list(input_path.glob("*.mp3")) + \
                      list(input_path.glob("*.flac"))

        output_dir = args.output or input_path / "transcriptions"

        results = transcriber.transcribe_batch(
            audio_files,
            output_dir,
            output_format=args.format,
            temperature=args.temperature,
        )

        # Summary
        successful = sum(1 for r in results if r["success"])
        print(f"\nTranscribed {successful}/{len(results)} files successfully")

    else:
        # Single file transcription
        output_path = args.output
        if output_path is None:
            output_path = input_path.with_suffix("")

        transcriber.transcribe(
            input_path,
            output_path,
            output_format=args.format,
            temperature=args.temperature,
        )


if __name__ == "__main__":
    main()
