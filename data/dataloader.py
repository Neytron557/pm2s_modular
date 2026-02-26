"""
Data loading utilities for Audio-to-Score.

Provides generic paired audio/MIDI datasets and helpers to build train/val loaders
from configuration files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import random
import yaml
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler

from data.preprocess import AudioPreprocessor, AudioConfig, AudioAugmentor
from data.tokenizer import MidiTokenizer


@dataclass
class AudioMidiPair:
    audio_path: Path
    midi_path: Path


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_data_config(path: str) -> dict:
    return load_yaml(path)


def _find_maestro_csv(root: Path) -> Path:
    candidates = sorted(root.glob("maestro-v*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No MAESTRO metadata CSV found in {root}")
    return candidates[-1]


def load_maestro_pairs(root: Path, split: Optional[str] = None) -> List[AudioMidiPair]:
    csv_path = _find_maestro_csv(root)
    df = pd.read_csv(csv_path)

    split_map = {
        "train": "train",
        "val": "validation",
        "validation": "validation",
        "test": "test",
    }
    desired_split = split_map.get(split, split)

    pairs = []
    for _, row in df.iterrows():
        if desired_split and row.get("split") != desired_split:
            continue
        audio_rel = row.get("audio_filename") or row.get("audio_file")
        midi_rel = row.get("midi_filename") or row.get("midi_file")
        if not audio_rel or not midi_rel:
            continue
        pairs.append(
            AudioMidiPair(
                audio_path=root / audio_rel,
                midi_path=root / midi_rel,
            )
        )
    return pairs


def scan_paired_dataset(
    root: Path,
    audio_exts: Tuple[str, ...] = (".wav", ".flac", ".mp3"),
    midi_exts: Tuple[str, ...] = (".mid", ".midi"),
) -> List[AudioMidiPair]:
    audio_files = {}
    for ext in audio_exts:
        for path in root.rglob(f"*{ext}"):
            audio_files[path.stem] = path

    pairs = []
    for ext in midi_exts:
        for path in root.rglob(f"*{ext}"):
            audio_path = audio_files.get(path.stem)
            if audio_path:
                pairs.append(AudioMidiPair(audio_path=audio_path, midi_path=path))
    return pairs


def split_pairs(
    pairs: List[AudioMidiPair],
    split_ratios: Dict[str, float],
    seed: int = 42,
) -> Dict[str, List[AudioMidiPair]]:
    pairs_copy = pairs[:]
    random.Random(seed).shuffle(pairs_copy)

    train_ratio = split_ratios.get("train", 0.8)
    val_ratio = split_ratios.get("val", 0.1)

    train_end = int(len(pairs_copy) * train_ratio)
    val_end = train_end + int(len(pairs_copy) * val_ratio)

    return {
        "train": pairs_copy[:train_end],
        "val": pairs_copy[train_end:val_end],
        "test": pairs_copy[val_end:],
    }


class AudioMidiDataset(Dataset):
    """Dataset of paired audio and MIDI files."""

    def __init__(
        self,
        pairs: List[AudioMidiPair],
        audio_config: Optional[AudioConfig] = None,
        tokenizer: Optional[MidiTokenizer] = None,
        augmentor: Optional[AudioAugmentor] = None,
        augmentation_config: Optional[dict] = None,
        is_training: bool = True,
    ):
        self.pairs = pairs
        self.preprocessor = AudioPreprocessor(audio_config or AudioConfig())
        self.tokenizer = tokenizer or MidiTokenizer()
        self.augmentor = augmentor
        self.augmentation_config = augmentation_config or {}
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        waveform, _ = self.preprocessor.load_audio(pair.audio_path)

        # Apply augmentation during training
        if self.is_training and self.augmentor is not None and self.augmentation_config.get("enabled", False):
            waveform = self.augmentor.apply_random_augmentation(
                waveform, self.augmentation_config
            )

        waveform = waveform.squeeze(0)

        tokens = self.tokenizer.encode_midi_file(str(pair.midi_path), add_special_tokens=True)
        targets = torch.tensor(tokens, dtype=torch.long)

        return {
            "audio": waveform,
            "audio_length": waveform.shape[-1],
            "targets": targets,
            "target_length": targets.numel(),
        }


def collate_fn(batch: List[dict]) -> dict:
    """Collate function for variable-length sequences."""
    if not batch:
        return {}

    max_audio_len = max(item["audio_length"] for item in batch)
    max_target_len = max(item["target_length"] for item in batch)

    batch_size = len(batch)

    audio_padded = torch.zeros(batch_size, max_audio_len)
    audio_lengths = torch.zeros(batch_size, dtype=torch.long)

    targets_padded = torch.zeros(batch_size, max_target_len, dtype=torch.long)
    target_lengths = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        audio_len = item["audio_length"]
        target_len = item["target_length"]

        audio_padded[i, :audio_len] = item["audio"]
        audio_lengths[i] = audio_len

        targets_padded[i, :target_len] = item["targets"]
        target_lengths[i] = target_len

    return {
        "audio": audio_padded,
        "audio_lengths": audio_lengths,
        "targets": targets_padded,
        "target_lengths": target_lengths,
    }


def build_datasets(
    datasets_cfg: List[dict],
    data_cfg: dict,
    split: str,
    augmentation_cfg: Optional[dict] = None,
) -> Tuple[Optional[Dataset], Optional[WeightedRandomSampler]]:
    datasets = []
    sample_weights = []

    split_cfg = data_cfg.get("splits", {}).get("default", {})
    split_seed = data_cfg.get("splits", {}).get("seed", 42)

    audio_pre_cfg = data_cfg.get("preprocessing", {}).get("audio", {})
    audio_config = AudioConfig(
        sample_rate=audio_pre_cfg.get("target_sample_rate", 16000),
        normalize=audio_pre_cfg.get("normalize", True),
    )

    # Create augmentor for training
    augmentor = None
    is_training = (split == "train")
    if is_training and augmentation_cfg and augmentation_cfg.get("enabled", False):
        if augmentation_cfg.get("time_stretch", {}).get("enabled", False):
            print(
                "Warning: time_stretch augmentation is enabled but not applied. "
                "Stretching audio without retiming MIDI would desync targets."
            )
        noise_paths = {}
        if augmentation_cfg.get("noise", {}).get("enabled", False):
            noise_cfg = augmentation_cfg["noise"]
            if noise_cfg.get("musan_path"):
                noise_paths["musan"] = noise_cfg["musan_path"]
            if noise_cfg.get("mit_ir_path"):
                noise_paths["mit_ir"] = noise_cfg["mit_ir_path"]
        augmentor = AudioAugmentor(
            sample_rate=audio_config.sample_rate,
            noise_paths=noise_paths,
        )

    for dataset_entry in datasets_cfg:
        name = dataset_entry.get("name")
        weight = float(dataset_entry.get("weight", 1.0))

        alias = {"pdmx_synthetic": "pdmx"}
        lookup_name = alias.get(name, name)
        data_def = data_cfg.get("datasets", {}).get(lookup_name, {})
        path = Path(dataset_entry.get("path") or data_def.get("path", ""))

        if not path.exists():
            print(f"Skipping dataset '{name}': path not found: {path}")
            continue
        if not path.is_dir():
            print(f"Skipping dataset '{name}': path is not a directory: {path}")
            continue

        if lookup_name == "maestro":
            pairs = load_maestro_pairs(path, split=split)
        else:
            pairs_all = scan_paired_dataset(path)
            if not pairs_all:
                print(f"Skipping dataset '{name}': no paired audio/MIDI files found")
                continue
            splits = split_pairs(pairs_all, split_cfg, seed=split_seed)
            pairs = splits.get(split, [])

        if not pairs:
            print(f"Skipping dataset '{name}': no pairs for split '{split}'")
            continue

        dataset = AudioMidiDataset(
            pairs,
            audio_config=audio_config,
            augmentor=augmentor,
            augmentation_config=augmentation_cfg,
            is_training=is_training,
        )
        datasets.append(dataset)

        if weight > 0:
            per_sample_weight = weight / len(dataset)
            sample_weights.extend([per_sample_weight] * len(dataset))

    if not datasets:
        return None, None

    concat = ConcatDataset(datasets)
    sampler = None
    if sample_weights:
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    return concat, sampler


def create_dataloaders(
    datasets_cfg: List[dict],
    data_cfg: dict,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    augmentation_cfg: Optional[dict] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    train_dataset, train_sampler = build_datasets(
        datasets_cfg, data_cfg, split="train", augmentation_cfg=augmentation_cfg
    )
    val_dataset, _ = build_datasets(datasets_cfg, data_cfg, split="val", augmentation_cfg=None)

    if train_dataset is None:
        raise RuntimeError("No training datasets available. Check paths in data_config.yaml.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

    return train_loader, val_loader
