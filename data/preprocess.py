"""
Audio Preprocessing Module for Audio-to-Score

Handles:
- Audio loading and resampling
- Feature extraction (Log-Mel Spectrogram)
- Normalization
- Data augmentation
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class AudioConfig:
    """Configuration for audio preprocessing."""
    sample_rate: int = 16000
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 160  # 10ms at 16kHz
    win_length: int = 400  # 25ms at 16kHz
    f_min: float = 20.0
    f_max: float = 8000.0
    normalize: bool = True


class AudioPreprocessor:
    """Preprocesses audio for the acoustic model."""

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self._init_transforms()

    def _init_transforms(self):
        """Initialize audio transforms."""
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            win_length=self.config.win_length,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
            power=2.0,
        )

    def load_audio(
        self,
        path: Union[str, Path],
        target_sr: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Load audio file and resample if needed.

        Args:
            path: Path to audio file
            target_sr: Target sample rate (uses config if None)

        Returns:
            Tuple of (waveform, sample_rate)
        """
        target_sr = target_sr or self.config.sample_rate
        waveform, sr = torchaudio.load(path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)

        return waveform, target_sr

    def extract_mel_spectrogram(
        self,
        waveform: torch.Tensor,
        log_scale: bool = True,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Extract Log-Mel Spectrogram from waveform.

        Args:
            waveform: Audio waveform tensor (1, T)
            log_scale: Whether to apply log scaling
            normalize: Whether to normalize

        Returns:
            Mel spectrogram tensor (n_mels, num_frames)
        """
        mel_spec = self.mel_transform(waveform)

        if log_scale:
            # Log-Mel with small epsilon for stability
            mel_spec = torch.log(mel_spec + 1e-6)

        if normalize:
            # Per-utterance normalization
            mean = mel_spec.mean()
            std = mel_spec.std()
            mel_spec = (mel_spec - mean) / (std + 1e-6)

        # Remove batch dimension if present
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.squeeze(0)

        return mel_spec

    def process_file(
        self,
        path: Union[str, Path],
    ) -> torch.Tensor:
        """
        Full preprocessing pipeline for a single file.

        Args:
            path: Path to audio file

        Returns:
            Log-Mel spectrogram tensor
        """
        waveform, _ = self.load_audio(path)
        mel_spec = self.extract_mel_spectrogram(
            waveform,
            normalize=self.config.normalize,
        )
        return mel_spec

    def get_frame_times(self, num_frames: int) -> np.ndarray:
        """
        Get timestamps for each spectrogram frame.

        Args:
            num_frames: Number of frames

        Returns:
            Array of frame timestamps in seconds
        """
        hop_time = self.config.hop_length / self.config.sample_rate
        return np.arange(num_frames) * hop_time

    def frames_to_time(self, frame_idx: int) -> float:
        """Convert frame index to time in seconds."""
        return frame_idx * self.config.hop_length / self.config.sample_rate

    def time_to_frames(self, time_seconds: float) -> int:
        """Convert time in seconds to frame index."""
        return int(time_seconds * self.config.sample_rate / self.config.hop_length)


class AudioAugmentor:
    """Audio augmentation for training."""

    def __init__(
        self,
        sample_rate: int = 16000,
        noise_paths: Optional[dict] = None,
    ):
        self.sample_rate = sample_rate
        self.noise_paths = noise_paths or {}
        self._noise_cache = {}

    def add_noise(
        self,
        waveform: torch.Tensor,
        noise_type: str = "white",
        snr_db: float = 10.0,
    ) -> torch.Tensor:
        """
        Add noise to waveform.

        Args:
            waveform: Input waveform
            noise_type: Type of noise ("white", "musan", "mit_ir")
            snr_db: Signal-to-noise ratio in dB

        Returns:
            Noisy waveform
        """
        length = waveform.shape[-1]
        if noise_type == "white":
            noise = torch.randn_like(waveform)
        elif noise_type in self.noise_paths:
            noise = self._load_random_noise(noise_type, length)
            if waveform.dim() == 1 and noise.dim() == 2:
                noise = noise.squeeze(0)
            elif waveform.dim() == 2 and noise.dim() == 1:
                noise = noise.unsqueeze(0)
            noise = noise.to(device=waveform.device, dtype=waveform.dtype)
        else:
            noise = torch.randn_like(waveform)

        # Calculate scaling for target SNR
        signal_power = (waveform ** 2).mean()
        noise_power = (noise ** 2).mean()
        snr_linear = 10 ** (snr_db / 10)
        scale = torch.sqrt(signal_power / (snr_linear * noise_power + 1e-8))

        return waveform + scale * noise

    def _load_random_noise(self, noise_type: str, length: int) -> torch.Tensor:
        """Load random noise sample from disk."""
        noise_dir = Path(self.noise_paths.get(noise_type, ""))
        if not noise_dir.exists():
            return torch.randn(1, length)

        noise_files = list(noise_dir.glob("**/*.wav"))
        if not noise_files:
            return torch.randn(1, length)

        # Random file selection
        noise_file = noise_files[np.random.randint(len(noise_files))]
        noise, sr = torchaudio.load(noise_file)

        # Resample if needed
        if sr != self.sample_rate:
            noise = torchaudio.transforms.Resample(sr, self.sample_rate)(noise)

        # Convert to mono
        if noise.shape[0] > 1:
            noise = noise.mean(dim=0, keepdim=True)

        # Adjust length
        if noise.shape[-1] < length:
            # Repeat noise
            repeats = (length // noise.shape[-1]) + 1
            noise = noise.repeat(1, repeats)
        noise = noise[:, :length]

        return noise

    def pitch_shift(
        self,
        waveform: torch.Tensor,
        cents: float,
    ) -> torch.Tensor:
        """
        Pitch shift audio.

        Args:
            waveform: Input waveform
            cents: Pitch shift in cents (100 cents = 1 semitone)

        Returns:
            Pitch-shifted waveform
        """
        # Using torchaudio pitch shift
        n_steps = cents / 100.0
        shifted = torchaudio.functional.pitch_shift(
            waveform,
            self.sample_rate,
            n_steps,
        )
        return shifted

    def time_stretch(
        self,
        waveform: torch.Tensor,
        rate: float,
    ) -> torch.Tensor:
        """
        Time stretch audio without changing pitch.

        Args:
            waveform: Input waveform
            rate: Stretch rate (>1 = slower, <1 = faster)

        Returns:
            Time-stretched waveform
        """
        n_fft = 2048
        hop_length = 512
        n_freq = n_fft // 2 + 1  # Number of frequency bins

        # Compute spectrogram
        spec = torch.stft(
            waveform.squeeze(0),
            n_fft=n_fft,
            hop_length=hop_length,
            return_complex=True,
        )

        # Phase advance: 2 * pi * k * hop_length / n_fft for frequency bin k
        # At Nyquist (k = n_fft // 2), phase_advance = pi * hop_length
        phase_advance = torch.linspace(
            0, np.pi * hop_length, n_freq, device=waveform.device
        ).unsqueeze(1)

        # Phase vocoder for time stretching
        stretched = torchaudio.functional.phase_vocoder(
            spec.unsqueeze(0),
            rate=rate,
            phase_advance=phase_advance,
        )

        # Inverse STFT
        waveform_stretched = torch.istft(
            stretched.squeeze(0),
            n_fft=n_fft,
            hop_length=hop_length,
        )

        return waveform_stretched.unsqueeze(0)

    def gain(
        self,
        waveform: torch.Tensor,
        gain_db: float,
    ) -> torch.Tensor:
        """
        Apply gain to waveform.

        Args:
            waveform: Input waveform
            gain_db: Gain in dB

        Returns:
            Amplified waveform
        """
        gain_linear = 10 ** (gain_db / 20)
        return waveform * gain_linear

    def apply_random_augmentation(
        self,
        waveform: torch.Tensor,
        config: dict,
    ) -> torch.Tensor:
        """
        Apply random augmentation based on config.

        Args:
            waveform: Input waveform
            config: Augmentation configuration dict

        Returns:
            Augmented waveform
        """
        # Noise
        if config.get("noise", {}).get("enabled", False):
            if np.random.random() < config["noise"].get("prob", 0.5):
                snr = np.random.uniform(*config["noise"].get("snr_range", [5, 20]))
                noise_types = config["noise"].get("types", ["white"])
                noise_type = np.random.choice(noise_types)
                waveform = self.add_noise(waveform, noise_type, snr)

        # Pitch shift
        if config.get("pitch_shift", {}).get("enabled", False):
            if np.random.random() < config["pitch_shift"].get("prob", 0.5):
                cents = np.random.uniform(*config["pitch_shift"].get("cents_range", [-10, 10]))
                waveform = self.pitch_shift(waveform, cents)

        # Gain
        if config.get("gain", {}).get("enabled", False):
            if np.random.random() < config["gain"].get("prob", 0.5):
                gain_db = np.random.uniform(*config["gain"].get("db_range", [-6, 6]))
                waveform = self.gain(waveform, gain_db)

        return waveform
