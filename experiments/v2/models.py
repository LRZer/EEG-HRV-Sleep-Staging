"""Shared spectral backbone, quality-aware gated residual cardiac fusion."""
import torch
from torch import nn
from deep_learning.models import SpectrogramEncoder, position_encoding, transformer


class FusionModel(nn.Module):
    def __init__(self, variant, protocol):
        super().__init__()
        self.spec = protocol["variants"][variant]
        self.context_length = self.spec["context"]
        self.modality_dropout = self.spec["modality_dropout"]
        d, h, p = protocol["dimension"], protocol["attention_heads"], protocol["dropout"]
        # Common EEG, temporal and classifier parameters are initialized first.
        self.eeg = SpectrogramEncoder(89, d, h, p) if self.spec["modality"] != "hrv" else None
        self.temporal = transformer(d, h, protocol["sequence_layers"], p)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(p), nn.Linear(d, 5))
        self.register_buffer("position", position_encoding(self.context_length, d))
        if self.spec["modality"] != "eeg":
            self.cardiac = nn.Sequential(nn.Linear(55, d), nn.LayerNorm(d), nn.GELU(),
                                         nn.Dropout(p), nn.Linear(d, d), nn.LayerNorm(d))
        if self.spec["fusion"] == "concat":
            self.combine = nn.Sequential(nn.Linear(d * 2, d), nn.LayerNorm(d), nn.GELU())
        if self.spec["fusion"] == "gate":
            self.delta = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
            self.gate = nn.Linear(d * 2 + 3, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, protocol["gate_initial_bias"])

    def forward(self, spectra, cardiac, missing, force_hrv_missing=False, return_gate=False):
        b, length = spectra.shape[:2]
        if force_hrv_missing or (self.training and self.modality_dropout):
            drop = torch.ones(b, 1, 1, dtype=torch.bool, device=spectra.device) if force_hrv_missing else (
                torch.rand(b, 1, 1, device=spectra.device) < self.modality_dropout)
            absent = torch.zeros_like(cardiac)
            absent[..., 26:52] = 1
            cardiac = torch.where(drop, absent, cardiac)
        eeg = self.eeg(spectra.flatten(0, 1)).reshape(b, length, -1) if self.eeg else None
        gates = torch.zeros(b, length, 1, device=spectra.device)
        if self.spec["modality"] == "eeg":
            tokens = eeg
        else:
            heart = self.cardiac(cardiac)
            if self.spec["modality"] == "hrv":
                tokens = heart
            elif self.spec["fusion"] == "concat":
                tokens = self.combine(torch.cat([eeg, heart], dim=-1))
            else:
                quality = cardiac[..., -3:]
                gates = self.gate(torch.cat([eeg, heart, quality], dim=-1)).sigmoid()
                # Valid RR and finite-feature fractions bound the effective gate.
                gates = gates * quality[..., :1] * quality[..., 1:2]
                tokens = eeg + gates * self.delta(heart)
        tokens = tokens + self.position[None]
        encoded = self.temporal(tokens, src_key_padding_mask=missing)
        logits = self.head(encoded[:, -1])
        return (logits, gates[:, -1, 0]) if return_gate else logits
