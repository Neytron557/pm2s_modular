"""
Stage 1 inference: Audio to MIDI.
"""

import argparse
from pathlib import Path

from inference.transcribe import AudioToScoreTranscriber


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio to MIDI")
    parser.add_argument("input", type=str, help="Input audio file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output MIDI path")
    parser.add_argument("--model", type=str, required=True, help="Acoustic model checkpoint")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda, cpu, auto)")
    args = parser.parse_args()

    transcriber = AudioToScoreTranscriber(
        acoustic_model_path=args.model,
        pm2s_model_path=None,
        device=args.device,
    )

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".mid")

    transcriber.transcribe(
        input_path,
        output_path,
        output_format="midi",
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
