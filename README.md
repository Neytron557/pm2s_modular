# PM2S Modular — Piano MIDI to Score

A modular deep learning pipeline for transcribing piano audio recordings into MusicXML sheet music scores. The system is split into two stages:

- **Stage 1 — Audio → MIDI:** Autoregressive acoustic model (Conformer encoder + Perceiver IO bottleneck + MoE Transformer decoder)
- **Stage 2 — MIDI → Score (PM2S):** Four independent modules — rhythm quantization, voice separation, key estimation, and MusicXML generation

---

## Architecture

### Stage 1: Acoustic Model (Audio → MIDI)

| Component | Details |
|---|---|
| Feature extractor | 128-bin log-mel spectrogram, 16 kHz, 10 ms hop |
| Encoder | 12-layer Conformer (512 dim, 8 heads, conv kernel 31) |
| Bottleneck | Perceiver IO (512 latents, 2 cross-attn + 6 self-attn layers) |
| Decoder | 6-layer MoE Transformer (8 experts, top-2 routing, RoPE) |
| Vocabulary | 519 tokens (note-on/off × 128, time-shift × 100, velocity × 32, program × 128, special × 3) |

### Stage 2: PM2S Modules (MIDI → Score)

| Module | Architecture | Task |
|---|---|---|
| **RhythmQuantizer** | 12-layer BiTransformer (512 dim, 8 heads) | Beat grid alignment, tempo, time signature, subdivision |
| **VoiceSeparator** | 8-layer Transformer (512 dim, 8 heads) | Voice ID and treble/bass staff assignment |
| **KeyEstimator** | Global attention network (512 dim, 30 classes) | Key signature (15 major + 15 minor) |
| **ScoreGenerator** | Rule-based, music21 | MusicXML assembly with duration snapping |

---

## Pretrained Checkpoints

Checkpoints are stored via **Git LFS**. After cloning, run `git lfs pull` to download them.

| Component | Path | Size |
|---|---|---|
| RhythmQuantizer | `checkpoints/pm2s/rhythm_quantizer/best.pt` | 468 MB |
| VoiceSeparator | `checkpoints/pm2s/voice_separator/best.pt` | 350 MB |
| KeyEstimator | `checkpoints/pm2s/key_estimator/best.pt` | 12 MB |

---

## Installation

```bash
# Clone (includes LFS checkpoints)
git clone https://github.com/Neytron557/pm2s_modular.git
cd pm2s_modular
git lfs pull

# Create conda environment (recommended)
conda env create -f environment.yml
conda activate a2s_train

# Or install with pip
pip install -r requirements.txt
pip install -e .
```

**Requirements:** Python 3.10+, PyTorch 2.x, CUDA 11.8+

---

## Quick Start

### MIDI → Score (Stage 2 only)

```bash
python -m inference.midi_to_score input.mid -o output.musicxml \
  --rhythm-ckpt checkpoints/pm2s/rhythm_quantizer/best.pt \
  --voice-ckpt  checkpoints/pm2s/voice_separator/best.pt \
  --key-ckpt    checkpoints/pm2s/key_estimator/best.pt
```

### Full Pipeline — Audio → Score

```bash
python -m inference.transcribe input.wav -o output \
  --model checkpoints/acoustic/best.pt \
  --format musicxml
```

### Batch transcription

```bash
python -m inference.transcribe audio_folder/ -o transcriptions/ --format musicxml
```

---

## Training

### 1. Download datasets

```bash
# List available datasets
python -m data.download_datasets --list

# Download a specific dataset
python -m data.download_datasets --dataset asap --output-dir datasets

# Download all PM2S datasets
python -m data.download_datasets --dataset all --output-dir datasets
```

Update dataset paths in `configs/data_config.yaml` if needed (all paths are relative to the repo root).

### 2. Build training manifests

```bash
python data/prepare_pm2s_manifests.py \
  --tasks rhythm,voice,key \
  --data-config configs/data_config.yaml \
  --output-dir datasets/pm2s_manifests \
  --strict-piano
```

### 3. Train PM2S modules

**RhythmQuantizer** (multi-GPU recommended):
```bash
torchrun --nproc_per_node=4 --master_port=29501 -m training.train_pm2s \
  --task rhythm_quantizer \
  --config configs/training_config.yaml \
  --model-config configs/model_config.yaml \
  --batch-size 4 --grad-accum-steps 32 \
  --amp --amp-dtype bf16 --time-scale 100 \
  --epochs 20 --warmup-steps 1000 \
  --tempo-loss log --tempo-loss-weight 5.0 \
  --sample-by-dataset-weights --save-every 5
```

**VoiceSeparator:**
```bash
torchrun --nproc_per_node=2 --master_port=29501 -m training.train_pm2s \
  --task voice_separator \
  --config configs/training_config.yaml \
  --model-config configs/model_config.yaml \
  --batch-size 4 --grad-accum-steps 64 \
  --amp --amp-dtype bf16 --time-scale 100 \
  --epochs 20 --warmup-steps 50 \
  --sample-by-dataset-weights --save-every 5
```

**KeyEstimator:**
```bash
torchrun --nproc_per_node=4 --master_port=29501 -m training.train_pm2s \
  --task key_estimator \
  --config configs/training_config.yaml \
  --model-config configs/model_config.yaml \
  --batch-size 4 --grad-accum-steps 64 \
  --amp --amp-dtype bf16 --time-scale 100 \
  --epochs 20 --warmup-steps 30 \
  --sample-by-dataset-weights --save-every 5
```

> **Note:** `--sample-by-dataset-weights` is important — without it, PDMX (250k+ files) dominates the manifests and crowds out smaller but higher-quality datasets (ASAP, MSMD, ACPAS).

Single-GPU training is also supported; replace `torchrun --nproc_per_node=N` with `python` and remove `--gpu_ids`.

### 4. Test with dummy data (no datasets needed)

```bash
python -m training.train_pm2s --task rhythm_quantizer --use-dummy --epochs 2
python -m training.train_pm2s --task voice_separator  --use-dummy --epochs 2
python -m training.train_pm2s --task key_estimator    --use-dummy --epochs 2
```

---

## Evaluation

```bash
python -m evaluation.evaluate \
  --predictions output_dir/ \
  --ground_truth ground_truth_dir/ \
  --metrics note_f1,frame_f1,mv2h
```

**Metrics:**
- **Note F1 (onset):** 50 ms tolerance
- **Note F1 (onset+offset):** 50 ms onset / 20% offset tolerance
- **Frame F1:** 10 ms frames
- **MV2H:** Multi-pitch, Voice, Meter, Value, Harmony

Evaluate PM2S modules independently:
```bash
python -m evaluation.evaluate_pm2s_components \
  --task rhythm_quantizer \
  --ckpt checkpoints/pm2s/rhythm_quantizer/best.pt \
  --data-config configs/data_config.yaml
```

---

## Project Structure

```
pm2s_modular/
├── checkpoints/pm2s/          # Trained weights (Git LFS)
│   ├── rhythm_quantizer/best.pt
│   ├── voice_separator/best.pt
│   └── key_estimator/best.pt
├── configs/
│   ├── model_config.yaml      # Architecture hyperparameters
│   ├── training_config.yaml   # Training hyperparameters
│   └── data_config.yaml       # Dataset paths
├── data/
│   ├── tokenizer.py           # MIDI tokenizer (519-token vocab)
│   ├── dataloader.py          # PyTorch dataloaders
│   ├── prepare_pm2s_manifests.py  # Piano-only manifest builder
│   ├── preprocess.py          # Audio preprocessing
│   └── download_datasets.py   # Dataset download scripts
├── models/
│   ├── acoustic_model.py      # Stage 1: full acoustic model
│   ├── encoder/
│   │   ├── conformer.py       # Conformer encoder
│   │   └── feature_extractor.py  # Mel spectrogram
│   ├── bottleneck/
│   │   └── perceiver.py       # Perceiver IO
│   ├── decoder/
│   │   └── moe_transformer.py # MoE Transformer with RoPE
│   └── pm2s/                  # Stage 2 modules
│       ├── rhythm_quantizer.py
│       ├── voice_separator.py
│       ├── key_estimator.py
│       └── score_generator.py
├── training/
│   ├── train_pm2s.py          # PM2S training script (DDP, AMP, WandB)
│   ├── losses.py              # Task-specific loss functions
│   └── scheduler.py           # Warmup + cosine LR scheduler
├── inference/
│   ├── transcribe.py          # Full audio → MusicXML pipeline
│   ├── audio_to_midi.py       # Stage 1 only
│   └── midi_to_score.py       # Stage 2 only
├── evaluation/
│   ├── evaluate.py            # General evaluation runner
│   ├── evaluate_pm2s_components.py  # Per-module evaluation
│   └── metrics.py             # Note F1, Frame F1, MV2H
└── utils/
    ├── pm2s_inference.py      # PM2S inference helpers
    └── serialization.py       # Safe checkpoint loading
```

---

## Datasets

All paths are relative to the repo root and configured in `configs/data_config.yaml`.

| Dataset | Purpose | Size |
|---|---|---|
| [ASAP](https://github.com/fosfrancesco/asap-dataset) | Rhythm, voice, key training | ~2 GB |
| [ACPAS](https://github.com/cheriell/ACPAS-dataset) | Rhythm training | ~1 GB |
| [PDMX](https://zenodo.org/record/pdmx) | Synthetic data (250k+ MusicXML) | ~1.6 GB |
| [MSMD](https://github.com/CPJKU/msmd) | All PM2S tasks | ~3 GB |
| [Lakh MIDI](https://colinraffel.com/projects/lmd/) | Key estimator pretraining | ~6 GB |
| [MAESTRO v3](https://magenta.tensorflow.org/datasets/maestro) | Acoustic model training | ~120 GB |
| [MusicNet](https://zenodo.org/record/5120004) | Acoustic model training | ~12 GB |

---

## References

- [PM2S](https://arxiv.org/abs/2209.06318) — Nakamura et al., Performance MIDI-to-Score Conversion
- [Conformer](https://arxiv.org/abs/2005.08100) — Gulati et al.
- [Perceiver IO](https://arxiv.org/abs/2107.14795) — Jaegle et al.
- [MusicFM](https://arxiv.org/abs/2311.03318) — Won et al., pretrained audio encoder

## License

Apache 2.0
