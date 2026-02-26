"""
Acoustic Model for Audio-to-Score (Stage 1)

Full end-to-end model combining:
- Feature Extractor (audio -> mel spectrogram)
- Conformer Encoder (mel -> encoder states)
- Perceiver IO Bottleneck (encoder states -> latents)
- MoE Transformer Decoder (latents -> MIDI tokens)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

from utils.serialization import safe_torch_load
from .encoder.feature_extractor import AudioFeatureExtractor, ConvSubsampling
from .encoder.conformer import ConformerEncoder
from .bottleneck.perceiver import PerceiverIO
from .decoder.moe_transformer import MoETransformerDecoder


class AcousticModel(nn.Module):
    """
    Stage 1: Acoustic Transcription Model

    Converts audio waveforms to MIDI token sequences.
    """

    def __init__(
        self,
        # Feature extractor config
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 160,
        win_length: int = 400,
        f_min: float = 20.0,
        f_max: float = 8000.0,
        normalize: bool = True,
        # Encoder config
        encoder_dim: int = 512,
        encoder_layers: int = 12,
        encoder_heads: int = 8,
        encoder_ff_dim: int = 2048,
        encoder_conv_kernel: int = 31,
        encoder_dropout: float = 0.1,
        encoder_attention_dropout: float = 0.1,
        encoder_conv_dropout: float = 0.1,
        # Bottleneck config
        num_latents: int = 512,
        bottleneck_layers: int = 6,
        bottleneck_cross_attention_layers: int = 2,
        bottleneck_latent_dim: int = 512,
        bottleneck_num_heads: int = 8,
        bottleneck_dropout: float = 0.1,
        # Decoder config
        vocab_size: int = 519,
        decoder_dim: int = 512,
        decoder_layers: int = 6,
        decoder_heads: int = 8,
        decoder_ff_dim: int = 2048,
        decoder_dropout: float = 0.1,
        max_seq_len: int = 8192,
        # MoE config
        moe_enabled: bool = True,
        num_experts: int = 8,
        top_k: int = 2,
        moe_layer_freq: int = 2,
        moe_load_balancing_weight: float = 0.01,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.normalize = normalize
        self.encoder_layers = encoder_layers
        self.encoder_heads = encoder_heads
        self.encoder_ff_dim = encoder_ff_dim
        self.encoder_conv_kernel = encoder_conv_kernel
        self.encoder_dropout = encoder_dropout
        self.encoder_attention_dropout = encoder_attention_dropout
        self.encoder_conv_dropout = encoder_conv_dropout
        self.num_latents = num_latents
        self.bottleneck_layers = bottleneck_layers
        self.bottleneck_cross_attention_layers = bottleneck_cross_attention_layers
        self.bottleneck_latent_dim = bottleneck_latent_dim
        self.bottleneck_num_heads = bottleneck_num_heads
        self.bottleneck_dropout = bottleneck_dropout
        self.decoder_layers = decoder_layers
        self.decoder_heads = decoder_heads
        self.decoder_ff_dim = decoder_ff_dim
        self.decoder_dropout = decoder_dropout
        self.max_seq_len = max_seq_len
        self.moe_enabled = moe_enabled
        self.num_experts = num_experts
        self.top_k = top_k
        self.moe_layer_freq = moe_layer_freq
        self.moe_load_balancing_weight = moe_load_balancing_weight

        if bottleneck_latent_dim != decoder_dim:
            raise ValueError("bottleneck_latent_dim must match decoder_dim")

        # Feature extractor
        self.feature_extractor = AudioFeatureExtractor(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            f_min=f_min,
            f_max=f_max,
            normalize=normalize,
        )

        # Conformer encoder
        self.encoder = ConformerEncoder(
            input_dim=n_mels,
            d_model=encoder_dim,
            num_layers=encoder_layers,
            num_heads=encoder_heads,
            d_ff=encoder_ff_dim,
            conv_kernel_size=encoder_conv_kernel,
            dropout=encoder_dropout,
            attention_dropout=encoder_attention_dropout,
            conv_dropout=encoder_conv_dropout,
            subsampling=True,
            subsampling_factor=4,
        )

        # Perceiver IO bottleneck
        self.bottleneck = PerceiverIO(
            input_dim=encoder_dim,
            latent_dim=bottleneck_latent_dim,
            num_latents=num_latents,
            num_cross_attention_layers=bottleneck_cross_attention_layers,
            num_self_attention_layers=bottleneck_layers,
            num_heads=bottleneck_num_heads,
            dropout=bottleneck_dropout,
        )

        # MoE Transformer decoder
        moe_config = {
            "enabled": moe_enabled,
            "num_experts": num_experts,
            "top_k": top_k,
            "moe_layer_freq": moe_layer_freq,
            "hidden_dim": decoder_ff_dim,
            "load_balancing_loss_weight": moe_load_balancing_weight,
        }
        self.decoder = MoETransformerDecoder(
            vocab_size=vocab_size,
            dim=decoder_dim,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            ff_dim=decoder_ff_dim,
            dropout=decoder_dropout,
            max_seq_len=max_seq_len,
            moe_config=moe_config,
        )

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            audio: Audio waveforms (batch, time) or (batch, 1, time)
            audio_lengths: Lengths of audio samples
            target_ids: Target token IDs for teacher forcing (batch, seq_len)

        Returns:
            Dictionary containing:
                - logits: Output logits (batch, seq_len, vocab_size)
                - aux_loss: Auxiliary MoE loss
        """
        # Extract features
        features, feature_lengths = self.feature_extractor(audio, audio_lengths)

        # Encode
        encoder_output, encoder_lengths = self.encoder(features, feature_lengths)

        # Create encoder mask
        if encoder_lengths is not None:
            max_len = encoder_output.size(1)
            encoder_mask = torch.arange(max_len, device=encoder_output.device).unsqueeze(0)
            encoder_mask = encoder_mask < encoder_lengths.unsqueeze(1)
        else:
            encoder_mask = None

        # Compress through bottleneck
        latents = self.bottleneck(encoder_output, encoder_mask)

        # Decode
        if target_ids is None:
            raise ValueError("target_ids must be provided during training")

        # Teacher forcing: use target_ids shifted right
        decoder_input = target_ids[:, :-1]  # Remove last token
        logits, _, aux_loss = self.decoder(
            decoder_input,
            latents,
            encoder_mask=None,  # Latents are fixed length
        )

        return {
            "logits": logits,
            "aux_loss": aux_loss,
        }

    def encode(
        self,
        audio: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode audio to latent representation.

        Args:
            audio: Audio waveforms
            audio_lengths: Lengths of audio samples

        Returns:
            Latent representation (batch, num_latents, latent_dim)
        """
        # Extract features
        features, feature_lengths = self.feature_extractor(audio, audio_lengths)

        # Encode
        encoder_output, encoder_lengths = self.encoder(features, feature_lengths)

        # Create encoder mask
        if encoder_lengths is not None:
            max_len = encoder_output.size(1)
            encoder_mask = torch.arange(max_len, device=encoder_output.device).unsqueeze(0)
            encoder_mask = encoder_mask < encoder_lengths.unsqueeze(1)
        else:
            encoder_mask = None

        # Compress through bottleneck
        latents = self.bottleneck(encoder_output, encoder_mask)

        return latents

    @torch.no_grad()
    def transcribe(
        self,
        audio: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        max_length: int = 4096,
        temperature: float = 0.0,
        top_k: int = 0,
        sos_token_id: int = 1,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        """
        Transcribe audio to MIDI tokens.

        Args:
            audio: Audio waveforms
            audio_lengths: Lengths of audio samples
            max_length: Maximum output length
            temperature: Sampling temperature (0 = greedy)
            top_k: Top-k sampling
            sos_token_id: Start of sequence token
            eos_token_id: End of sequence token

        Returns:
            Generated token IDs (batch, seq_len)
        """
        self.eval()

        # Encode audio to latents
        latents = self.encode(audio, audio_lengths)

        # Generate tokens
        tokens = self.decoder.generate(
            latents,
            encoder_mask=None,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            sos_token_id=sos_token_id,
            eos_token_id=eos_token_id,
        )

        return tokens

    def save(self, path: str):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "model_config": self._get_config(),
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
        map_location: Optional[str] = None,
        strict: bool = True,
    ) -> "AcousticModel":
        """Load model checkpoint."""
        import inspect

        checkpoint = safe_torch_load(path, map_location=map_location, weights_only=False)

        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            config = checkpoint.get("model_config") or checkpoint.get("config") or {}
        else:
            state_dict = checkpoint
            config = {}

        if not isinstance(config, dict):
            config = {}

        init_params = inspect.signature(cls.__init__).parameters
        valid_keys = {k for k in init_params if k != "self"}
        model_kwargs = {k: v for k, v in config.items() if k in valid_keys}

        model = cls(**model_kwargs)
        try:
            model.load_state_dict(state_dict, strict=strict)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load checkpoint. Ensure model_config matches the checkpoint."
            ) from exc
        return model

    def _get_config(self) -> dict:
        """Get model configuration."""
        return {
            "vocab_size": self.vocab_size,
            "encoder_dim": self.encoder_dim,
            "decoder_dim": self.decoder_dim,
            "sample_rate": self.sample_rate,
            "n_mels": self.n_mels,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "f_min": self.f_min,
            "f_max": self.f_max,
            "normalize": self.normalize,
            "encoder_layers": self.encoder_layers,
            "encoder_heads": self.encoder_heads,
            "encoder_ff_dim": self.encoder_ff_dim,
            "encoder_conv_kernel": self.encoder_conv_kernel,
            "encoder_dropout": self.encoder_dropout,
            "encoder_attention_dropout": self.encoder_attention_dropout,
            "encoder_conv_dropout": self.encoder_conv_dropout,
            "num_latents": self.num_latents,
            "bottleneck_layers": self.bottleneck_layers,
            "bottleneck_cross_attention_layers": self.bottleneck_cross_attention_layers,
            "bottleneck_latent_dim": self.bottleneck_latent_dim,
            "bottleneck_num_heads": self.bottleneck_num_heads,
            "bottleneck_dropout": self.bottleneck_dropout,
            "decoder_layers": self.decoder_layers,
            "decoder_heads": self.decoder_heads,
            "decoder_ff_dim": self.decoder_ff_dim,
            "decoder_dropout": self.decoder_dropout,
            "max_seq_len": self.max_seq_len,
            "moe_enabled": self.moe_enabled,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "moe_layer_freq": self.moe_layer_freq,
            "moe_load_balancing_weight": self.moe_load_balancing_weight,
        }

    @classmethod
    def from_config(cls, config_path: str) -> "AcousticModel":
        """Create model from YAML config file."""
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        acoustic_config = config.get("acoustic_model", {})

        return cls(
            # Feature extractor
            sample_rate=acoustic_config.get("feature_extractor", {}).get("sample_rate", 16000),
            n_mels=acoustic_config.get("feature_extractor", {}).get("n_mels", 128),
            hop_length=acoustic_config.get("feature_extractor", {}).get("hop_length", 160),
            # Encoder
            encoder_dim=acoustic_config.get("encoder", {}).get("hidden_dim", 512),
            encoder_layers=acoustic_config.get("encoder", {}).get("num_layers", 12),
            encoder_heads=acoustic_config.get("encoder", {}).get("num_heads", 8),
            encoder_ff_dim=acoustic_config.get("encoder", {}).get("ffn_dim", 2048),
            encoder_conv_kernel=acoustic_config.get("encoder", {}).get("conv_kernel_size", 31),
            encoder_dropout=acoustic_config.get("encoder", {}).get("dropout", 0.1),
            # Bottleneck
            num_latents=acoustic_config.get("bottleneck", {}).get("num_latents", 512),
            bottleneck_layers=acoustic_config.get("bottleneck", {}).get("num_self_attention_layers", 6),
            # Decoder
            vocab_size=acoustic_config.get("tokenizer", {}).get("vocab_size", 519),
            decoder_dim=acoustic_config.get("decoder", {}).get("hidden_dim", 512),
            decoder_layers=acoustic_config.get("decoder", {}).get("num_layers", 6),
            decoder_heads=acoustic_config.get("decoder", {}).get("num_heads", 8),
            decoder_ff_dim=acoustic_config.get("decoder", {}).get("ffn_dim", 2048),
            decoder_dropout=acoustic_config.get("decoder", {}).get("dropout", 0.1),
            max_seq_len=acoustic_config.get("decoder", {}).get("max_seq_length", 8192),
            # MoE
            moe_enabled=acoustic_config.get("decoder", {}).get("moe", {}).get("enabled", True),
            num_experts=acoustic_config.get("decoder", {}).get("moe", {}).get("num_experts", 8),
            top_k=acoustic_config.get("decoder", {}).get("moe", {}).get("top_k", 2),
        )
