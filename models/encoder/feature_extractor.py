"""
Feature Extractor for Audio-to-Score

Extracts Log-Mel Spectrograms from raw audio waveforms.
"""

import torch
import torch.nn as nn
import torchaudio
from typing import Optional


class AudioFeatureExtractor(nn.Module):
    """
    Extracts audio features (Log-Mel Spectrogram) from raw waveforms.

    This module is designed to be compatible with the Conformer encoder
    and can be integrated into an end-to-end model.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 160,
        win_length: int = 400,
        f_min: float = 20.0,
        f_max: float = 8000.0,
        normalize: bool = True,
        center: bool = True,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.normalize = normalize
        self.n_fft = n_fft
        self.win_length = win_length
        self.center = center

        # Mel spectrogram transform
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
            normalized=False,
            center=center,
        )

        # Learnable normalization parameters (optional)
        self.norm = nn.LayerNorm(n_mels) if normalize else nn.Identity()

    def forward(
        self,
        waveform: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Extract features from waveform.

        Args:
            waveform: Input waveform (batch, time) or (batch, 1, time)
            lengths: Original lengths of waveforms before padding

        Returns:
            features: Log-Mel spectrogram (batch, time, n_mels)
            feature_lengths: Lengths of features
        """
        # Handle channel dimension
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        # Compute Mel spectrogram: (batch, n_mels, time)
        mel = self.mel_spec(waveform)

        # Log scale with small epsilon
        log_mel = torch.log(mel + 1e-6)

        # Transpose to (batch, time, n_mels) for transformer
        log_mel = log_mel.transpose(1, 2)

        # Normalize
        if self.normalize:
            log_mel = self.norm(log_mel)

        # Compute feature lengths
        if lengths is not None:
            feature_lengths = self._compute_feature_lengths(lengths)
        else:
            feature_lengths = torch.full(
                (log_mel.size(0),),
                log_mel.size(1),
                device=log_mel.device,
                dtype=torch.long,
            )

        return log_mel, feature_lengths

    def _compute_feature_lengths(self, audio_lengths: torch.Tensor) -> torch.Tensor:
        """Compute output lengths given input audio lengths."""
        pad = self.n_fft // 2 if self.center else 0
        lengths = (audio_lengths + 2 * pad - self.n_fft) // self.hop_length + 1
        return torch.clamp(lengths, min=0)


class ConvSubsampling(nn.Module):
    """
    Convolutional subsampling layer for reducing sequence length.

    Commonly used before Conformer/Transformer to reduce computational cost.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        subsampling_factor: int = 4,
    ):
        super().__init__()

        self.subsampling_factor = subsampling_factor
        self.kernel_size = 3
        self.stride = 2
        self.padding = 1

        # Two conv layers with stride 2 each -> 4x subsampling
        self.conv1 = nn.Conv2d(
            1,
            output_dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        self.conv2 = nn.Conv2d(
            output_dim,
            output_dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

        self.activation = nn.GELU()

        # Linear projection after flattening
        # After 2x subsampling of input_dim, we get input_dim // 4 * output_dim
        self.linear = nn.Linear(output_dim * (input_dim // 4), output_dim)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Args:
            x: Input features (batch, time, input_dim)
            lengths: Sequence lengths

        Returns:
            output: Subsampled features (batch, time // 4, output_dim)
            new_lengths: Updated lengths
        """
        # Add channel dim: (batch, 1, time, input_dim)
        x = x.unsqueeze(1)

        # Conv layers
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))

        # Reshape: (batch, channels, time, freq) -> (batch, time, channels * freq)
        batch, channels, time, freq = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch, time, channels * freq)

        # Linear projection
        x = self.linear(x)

        # Update lengths
        if lengths is not None:
            new_lengths = self._conv_output_lengths(lengths)
        else:
            new_lengths = None

        return x, new_lengths

    def _conv_output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        """Compute output lengths after two strided conv layers."""
        def conv_out_len(lengths_in: torch.Tensor) -> torch.Tensor:
            return (
                (lengths_in + 2 * self.padding - self.kernel_size) // self.stride + 1
            )

        lengths_out = conv_out_len(lengths)
        lengths_out = conv_out_len(lengths_out)
        return torch.clamp(lengths_out, min=0)
