# Audio-to-Score: Piano Transcription to MusicXML

A two-stage deep learning system for transcribing piano audio recordings to MusicXML scores.

## Overview

This project implements a two-stage Audio-to-Score pipeline:

1. **Stage 1: Audio → MIDI** ✅ COMPLETE
   - Using pre-trained [AR_transcription model](https://github.com/sori-ai/AR_transcription/tree/integrated)
   - Location: `/workspace/projects/AR_transcription`
   - No additional training required

2. **Stage 2: PM2S Model** (MIDI → Score) 🚧 READY TO TRAIN
   - Rhythm quantization (beat grid alignment) - IMPLEMENTED
   - Voice separation (polyphonic voice assignment) - IMPLEMENTED
   - Key estimation (key signature and spelling) - IMPLEMENTED
   - MusicXML generation - IMPLEMENTED
   - Status: Models implemented, piano-only manifest builder available

## Installation

```bash
# Clone the repository
cd /path/to/audio-to-score

# Optional: create conda environment
conda env create -f environment.yml
conda activate a2s_train

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### Transcribe Audio to Score

```bash
# Basic transcription
python -m inference.transcribe input.wav -o output

# With specific model
python -m inference.transcribe input.mp3 --model checkpoints/best.pt -o output

# Batch transcription
python -m inference.transcribe audio_folder/ -o transcriptions/
```

### Training PM2S Models

**Stage 1 (Audio → MIDI) is complete** - using AR_transcription model at `/workspace/projects/AR_transcription`.

**Stage 2 (MIDI → Score) - PM2S Training:**

```bash
# Test with dummy data first
python -m training.train_pm2s --task rhythm_quantizer --use-dummy --epochs 2
python -m training.train_pm2s --task voice_separator --use-dummy --epochs 2
python -m training.train_pm2s --task key_estimator --use-dummy --epochs 2

# Build piano-only manifests
python data/prepare_pm2s_manifests.py \
  --tasks rhythm,voice,key \
  --data-config configs/data_config.yaml \
  --output-dir datasets/pm2s_manifests \
  --strict-piano

# Train with real data (uses config + manifests)
CUDA_VISIBLE_DEVICES=6 \
python -m training.train_pm2s --task rhythm_quantizer \
  --config configs/training_config.yaml --model-config configs/model_config.yaml

CUDA_VISIBLE_DEVICES=7 \
python -m training.train_pm2s --task voice_separator \
  --config configs/training_config.yaml --model-config configs/model_config.yaml

CUDA_VISIBLE_DEVICES=6 \
python -m training.train_pm2s --task key_estimator \
  --config configs/training_config.yaml --model-config configs/model_config.yaml
```

See [training/train_pm2s.py](training/train_pm2s.py) for manifest format details.
WandB logging is enabled via `configs/training_config.yaml` (project `PM2S`, entity `sori-ai`, run name `pm2s_1`).

## Documentation

- PM2S overview for presentations: `PM2S_FRAMEWORK.md`

### Download Datasets

```bash
# List available datasets
python -m data.download_datasets --list

# Download all datasets
python -m data.download_datasets --dataset all --output-dir datasets

# Download specific dataset
python -m data.download_datasets --dataset pdmx
```

## Project Structure

```
audio-to-score/
├── configs/                     # Configuration files
│   ├── model_config.yaml        # Model architecture config
│   ├── training_config.yaml     # Training hyperparameters
│   └── data_config.yaml         # Dataset paths
├── data/                        # Data processing
│   ├── download_datasets.py     # Dataset download scripts
│   ├── preprocess.py            # Audio preprocessing
│   ├── tokenizer.py             # MIDI tokenization
│   ├── dataloader.py            # PyTorch dataloaders
│   └── prepare_pm2s_manifests.py # Piano-only PM2S manifest builder
├── models/                      # Model implementations
│   ├── encoder/
│   │   ├── conformer.py         # Conformer encoder
│   │   └── feature_extractor.py # Mel spectrogram extraction
│   ├── bottleneck/
│   │   └── perceiver.py         # Perceiver IO
│   ├── decoder/
│   │   └── moe_transformer.py   # MoE Transformer decoder
│   ├── acoustic_model.py        # Full Stage 1 model
│   └── pm2s/                    # Stage 2 modules
│       ├── rhythm_quantizer.py
│       ├── voice_separator.py
│       ├── key_estimator.py
│       └── score_generator.py
├── training/
│   ├── train_acoustic.py        # Stage 1 training
│   ├── train_pm2s.py            # Stage 2 training
│   ├── losses.py                # Loss functions
│   └── scheduler.py             # LR scheduler helpers
├── inference/
│   ├── transcribe.py            # Full inference pipeline
│   ├── audio_to_midi.py         # Stage 1 only
│   └── midi_to_score.py         # Stage 2 only
├── evaluation/
│   ├── metrics.py               # Evaluation metrics
│   └── evaluate.py              # Evaluation runner
├── requirements.txt
└── setup.py
```

## Datasets

Dataset paths are configured in `configs/data_config.yaml`; update if your data lives elsewhere.

### Available on System (Piano-Only Focus)
- **ASAP**: `datasets/asap`
- **ACPAS**: `datasets/acpas`
- **PDMX**: `datasets/pdmx/PDMX`
- **ATEPP**: `datasets/ATEPP/ATEPP-1.2`
- **MSMD (no-audio)**: `datasets/msmd/msmd_aug_v1-1_no-audio`
- **Lakh MIDI**: `datasets/lakh_midi`
- **MusicNet**: `datasets/musicnet/musicnet_yourmt3_16k`
- **MAESTRO v3.0.0**: `/workspace/datasets/maestro_v3.0.0`

### To Download
| Dataset | Size | Purpose |
|---------|------|---------|
| PDMX | 1.6 GB | Score JSON for synthetic data |
| Lakh MIDI | ~6 GB | MIDI key pretraining |
| ASAP | ~2 GB | Score-aligned performances |
| ACPAS | ~1 GB | Piano performance annotations |
| MSMD | ~3 GB | Score + MIDI performances (no audio) |

## Model Architecture

### Stage 1: Audio → MIDI (AR_transcription)
- Pre-trained autoregressive transformer model
- Handles multi-instrument audio transcription
- Already trained and production-ready
- Location: `/workspace/projects/AR_transcription`

### Stage 2: PM2S Model (MIDI → Score)
- **Rhythm Quantizer**: 12-layer Bidirectional Transformer (512 dim, 8 heads)
  - Predicts beat positions, subdivisions, time signature, and tempo
- **Voice Separator**: 8-layer Transformer (512 dim, 8 heads)
  - Assigns notes to voices and staves using pairwise attention
- **Key Estimator**: Global attention network (512 dim)
  - Estimates key signature using pitch class distributions
- **Score Generator**: Rule-based MusicXML generator using music21

## Evaluation

```bash
# Evaluate transcription
python -m evaluation.evaluate \
    --predictions output_dir/ \
    --ground_truth maestro_test/ \
    --metrics note_f1,frame_f1,mv2h
```

### Metrics
- **Note F1 (Onset)**: 50ms tolerance
- **Note F1 (Onset+Offset)**: 50ms onset, 20% offset tolerance
- **Frame F1**: 10ms frames
- **MV2H**: Multi-pitch, Voice, Meter, Value, Harmony

## References

Based on research from:
- MIROS and YourMT3+ (2025 AMT Challenge winners)
- Conformer (Speech Recognition)
- Perceiver IO (General Architecture)
- PM2S (Performance MIDI to Score)

## License

Apache 2.0
