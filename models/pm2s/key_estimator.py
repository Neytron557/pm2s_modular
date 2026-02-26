"""
Key Estimation Module for PM2S (Stage 2)

Estimates the key signature and provides enharmonic spelling rules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math


# Key signature definitions
KEY_SIGNATURES = [
    # Major keys
    ("C", "major", 0),
    ("G", "major", 1),
    ("D", "major", 2),
    ("A", "major", 3),
    ("E", "major", 4),
    ("B", "major", 5),
    ("F#", "major", 6),
    ("C#", "major", 7),
    ("F", "major", -1),
    ("Bb", "major", -2),
    ("Eb", "major", -3),
    ("Ab", "major", -4),
    ("Db", "major", -5),
    ("Gb", "major", -6),
    ("Cb", "major", -7),
    # Minor keys
    ("A", "minor", 0),
    ("E", "minor", 1),
    ("B", "minor", 2),
    ("F#", "minor", 3),
    ("C#", "minor", 4),
    ("G#", "minor", 5),
    ("D#", "minor", 6),
    ("A#", "minor", 7),
    ("D", "minor", -1),
    ("G", "minor", -2),
    ("C", "minor", -3),
    ("F", "minor", -4),
    ("Bb", "minor", -5),
    ("Eb", "minor", -6),
    ("Ab", "minor", -7),
]

# Pitch class to note name mappings
SHARP_SPELLINGS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_SPELLINGS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


class KeyEstimator(nn.Module):
    """
    Key Estimation Model

    Uses pitch class distributions and global context to estimate key.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_keys: int = 30,  # 15 major + 15 minor
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_keys = num_keys

        # Pitch class histogram encoder
        self.pc_encoder = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Note sequence encoder
        self.note_encoder = nn.Sequential(
            nn.Linear(128, hidden_dim),  # Pitch embedding
            nn.ReLU(),
        )

        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Key classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_keys),
        )

        # Key templates (Krumhansl-Schmuckler profiles)
        self._init_key_templates()

    def _init_key_templates(self):
        """Initialize key profile templates."""
        # Major key profile (Krumhansl-Kessler)
        major_profile = torch.tensor([
            6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
            2.52, 5.19, 2.39, 3.66, 2.29, 2.88
        ])
        major_profile = major_profile / major_profile.sum()

        # Minor key profile
        minor_profile = torch.tensor([
            6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
            2.54, 4.75, 3.98, 2.69, 3.34, 3.17
        ])
        minor_profile = minor_profile / minor_profile.sum()

        if self.num_keys > len(KEY_SIGNATURES):
            raise ValueError(
                f"num_keys ({self.num_keys}) exceeds available key signatures ({len(KEY_SIGNATURES)})"
            )

        note_to_pc = {
            "C": 0,
            "C#": 1,
            "Db": 1,
            "D": 2,
            "D#": 3,
            "Eb": 3,
            "E": 4,
            "F": 5,
            "F#": 6,
            "Gb": 6,
            "G": 7,
            "G#": 8,
            "Ab": 8,
            "A": 9,
            "A#": 10,
            "Bb": 10,
            "B": 11,
            "Cb": 11,
        }

        templates = []
        for root, mode, _ in KEY_SIGNATURES[:self.num_keys]:
            pc = note_to_pc[root]
            profile = major_profile if mode == "major" else minor_profile
            templates.append(torch.roll(profile, pc))

        self.register_buffer("key_templates", torch.stack(templates))

    def compute_pitch_class_histogram(
        self,
        pitches: torch.Tensor,
        durations: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute pitch class distribution.

        Args:
            pitches: MIDI pitches (batch, seq_len)
            durations: Note durations for weighting (batch, seq_len)
            mask: Padding mask (batch, seq_len)

        Returns:
            Pitch class histogram (batch, 12)
        """
        batch_size, seq_len = pitches.shape
        device = pitches.device

        # Convert to pitch classes (0-11)
        pitch_classes = pitches % 12

        # Create one-hot encoding
        pc_one_hot = F.one_hot(pitch_classes, num_classes=12).float()

        # Weight by duration if provided
        if durations is not None:
            pc_one_hot = pc_one_hot * durations.unsqueeze(-1)

        # Apply mask
        if mask is not None:
            pc_one_hot = pc_one_hot * mask.unsqueeze(-1).float()

        # Sum across sequence
        histogram = pc_one_hot.sum(dim=1)

        # Normalize
        histogram = histogram / (histogram.sum(dim=-1, keepdim=True) + 1e-8)

        return histogram

    def forward(
        self,
        pitches: torch.Tensor,
        durations: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            pitches: MIDI pitches (batch, seq_len)
            durations: Note durations (batch, seq_len)
            mask: Padding mask (batch, seq_len)

        Returns:
            Dictionary with key predictions
        """
        batch_size, seq_len = pitches.shape

        # Compute pitch class histogram
        pc_hist = self.compute_pitch_class_histogram(pitches, durations, mask)

        # Template matching scores
        template_scores = torch.einsum("bk,ck->bc", pc_hist, self.key_templates)

        # Encode pitch class distribution
        pc_features = self.pc_encoder(pc_hist)

        # Encode note sequence
        pitch_one_hot = F.one_hot(pitches, num_classes=128).float()
        note_features = self.note_encoder(pitch_one_hot)

        # Apply mask for attention
        if mask is not None:
            note_features = note_features * mask.unsqueeze(-1).float()

        # Attention pooling
        attn_weights = self.attention(note_features)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=1)
        note_pooled = (note_features * attn_weights).sum(dim=1)

        # Combine features
        combined = torch.cat([pc_features, note_pooled], dim=-1)

        # Classify key
        key_logits = self.classifier(combined)

        return {
            "key_logits": key_logits,
            "template_scores": template_scores,
            "pitch_class_histogram": pc_hist,
        }

    def estimate_key(
        self,
        pitches: torch.Tensor,
        durations: Optional[torch.Tensor] = None,
    ) -> List[Tuple[str, str, int]]:
        """
        Estimate key signature.

        Returns:
            List of (root, mode, num_accidentals) tuples
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(pitches, durations)

        # Combine neural and template-based predictions
        combined_scores = outputs["key_logits"] + outputs["template_scores"]
        key_indices = combined_scores.argmax(dim=-1).tolist()

        return [KEY_SIGNATURES[idx] for idx in key_indices]


class EnharmonicSpeller(nn.Module):
    """
    Enharmonic Spelling Module

    Determines the correct spelling (sharp vs flat) for each note
    based on the key context.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()

        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(12 + 30, hidden_dim),  # pitch class hist + key one-hot
            nn.ReLU(),
        )

        # Note spelling classifier
        self.spell_classifier = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # + pitch class
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # 0: flat, 1: sharp
        )

    def forward(
        self,
        pitches: torch.Tensor,
        key_logits: torch.Tensor,
        pc_histogram: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict spelling for each note.

        Args:
            pitches: MIDI pitches (batch, seq_len)
            key_logits: Key prediction logits (batch, num_keys)
            pc_histogram: Pitch class histogram (batch, 12)

        Returns:
            Spelling logits (batch, seq_len, 2)
        """
        batch_size, seq_len = pitches.shape

        # Key probabilities
        key_probs = F.softmax(key_logits, dim=-1)

        # Context features
        context = torch.cat([pc_histogram, key_probs], dim=-1)
        context_features = self.context_encoder(context)

        # Expand context for each note
        context_features = context_features.unsqueeze(1).expand(-1, seq_len, -1)

        # Pitch class for each note
        pitch_classes = (pitches % 12).unsqueeze(-1).float() / 11.0

        # Combine and classify
        combined = torch.cat([context_features, pitch_classes], dim=-1)
        spelling_logits = self.spell_classifier(combined)

        return spelling_logits

    def spell_notes(
        self,
        pitches: torch.Tensor,
        key: Tuple[str, str, int],
    ) -> List[str]:
        """
        Spell notes based on key signature.

        Uses rule-based approach for efficiency.
        """
        root, mode, num_accidentals = key

        # Determine if key uses flats or sharps
        use_flats = num_accidentals < 0

        spellings = []
        for pitch in pitches.flatten().tolist():
            pc = pitch % 12
            octave = pitch // 12 - 1  # MIDI octave

            if use_flats:
                note_name = FLAT_SPELLINGS[pc]
            else:
                note_name = SHARP_SPELLINGS[pc]

            spellings.append(f"{note_name}{octave}")

        return spellings
