"""Compact research adaptations; not exact reproductions of published networks.

CNN attention is inspired by AttnSleep (Eldele et al., 2021). The hierarchical
spectrogram transformer is inspired by SleepTransformer (Phan et al., 2022).
All code here is an independent implementation for this project's data/protocol.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def position_encoding(length, dimension):
    positions = torch.arange(length, dtype=torch.float32)[:, None]
    scales = torch.exp(torch.arange(0, dimension, 2, dtype=torch.float32) * (-math.log(10000.0) / dimension))
    result = torch.zeros(length, dimension)
    result[:, 0::2] = torch.sin(positions * scales)
    result[:, 1::2] = torch.cos(positions * scales)
    return result


def transformer(dimension, heads, layers, dropout):
    layer = nn.TransformerEncoderLayer(dimension, heads, dim_feedforward=dimension * 2,
                                       dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(dimension), enable_nested_tensor=False)


class WaveformEncoder(nn.Module):
    def __init__(self, dimension=96, heads=4, dropout=0.2):
        super().__init__()
        width = dimension // 2

        def branch(kernel):
            return nn.Sequential(
                nn.Conv1d(1, 24, kernel, stride=5, padding=kernel // 2), nn.BatchNorm1d(24), nn.GELU(),
                nn.Conv1d(24, 32, 9, stride=5, padding=4), nn.BatchNorm1d(32), nn.GELU(),
                nn.Conv1d(32, width, 5, stride=2, padding=2), nn.BatchNorm1d(width), nn.GELU(),
                nn.Dropout(dropout),
            )

        self.short = branch(9)
        self.long = branch(49)
        self.recalibrate = nn.Sequential(nn.Linear(dimension, dimension // 4), nn.GELU(),
                                         nn.Linear(dimension // 4, dimension), nn.Sigmoid())
        self.attention = transformer(dimension, heads, 1, dropout)
        self.register_buffer("position", position_encoding(60, dimension))

    def forward(self, waveform):
        features = torch.cat([self.short(waveform[:, None]), self.long(waveform[:, None])], dim=1).transpose(1, 2)
        features = features * self.recalibrate(features.mean(dim=1))[:, None]
        features = self.attention(features + self.position[None, :features.shape[1]])
        return features.mean(dim=1)


class SpectrogramEncoder(nn.Module):
    def __init__(self, frequencies, dimension, heads, dropout):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(frequencies, dimension), nn.LayerNorm(dimension))
        self.attention = transformer(dimension, heads, 2, dropout)
        self.register_buffer("position", position_encoding(29, dimension))

    def forward(self, spectrogram):
        tokens = self.project(spectrogram) + self.position[None, :spectrogram.shape[1]]
        return self.attention(tokens).mean(dim=1)


class SleepModel(nn.Module):
    def __init__(self, kind, protocol, hrv_dimensions=52, frequencies=89):
        super().__init__()
        self.kind = kind
        self.context_length = 1 if kind == "eeg_attention" else protocol["context_epochs"]
        dim, heads, dropout = protocol["dimension"], protocol["attention_heads"], protocol["dropout"]
        if kind == "sleep_transformer":
            self.eeg = SpectrogramEncoder(frequencies, dim, heads, dropout)
        else:
            self.eeg = WaveformEncoder(dim, heads, dropout)
        if kind.startswith("fusion"):
            self.hrv = nn.Sequential(nn.Linear(hrv_dimensions, dim), nn.LayerNorm(dim), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(dim, dim), nn.LayerNorm(dim))
        if kind == "fusion_concat":
            self.fuse = nn.Sequential(nn.Linear(2 * dim, dim), nn.LayerNorm(dim), nn.GELU())
        elif kind == "fusion_cross":
            self.cross = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
            self.fuse = nn.Sequential(nn.Linear(2 * dim, dim), nn.LayerNorm(dim), nn.GELU())
        if self.context_length > 1:
            self.temporal = transformer(dim, heads, protocol["sequence_layers"], dropout)
        self.register_buffer("position", position_encoding(protocol["context_epochs"], dim))
        self.classifier = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, 5))

    def forward(self, eeg, hrv, spectra, missing_context):
        if self.kind == "eeg_attention":
            return self.classifier(self.eeg(eeg[:, -1]))
        batch, length = eeg.shape[:2]
        if self.kind == "sleep_transformer":
            tokens = self.eeg(spectra.flatten(0, 1)).reshape(batch, length, -1)
        else:
            tokens = self.eeg(eeg.flatten(0, 1)).reshape(batch, length, -1)
        tokens = tokens + self.position[None, :length]
        if self.kind.startswith("fusion"):
            cardiac = self.hrv(hrv) + self.position[None, :length]
            if self.kind == "fusion_concat":
                tokens = self.fuse(torch.cat([tokens, cardiac], dim=-1))
            else:
                cardiac_context, _ = self.cross(tokens, cardiac, cardiac,
                                                 key_padding_mask=missing_context, need_weights=False)
                tokens = self.fuse(torch.cat([tokens, cardiac_context], dim=-1))
        # All input epochs end at or before the scored epoch. Bidirectional attention
        # inside that available history does not introduce a future sleep epoch.
        encoded = self.temporal(tokens, src_key_padding_mask=missing_context)
        return self.classifier(encoded[:, -1])


def build_model(kind, protocol, hrv_dimensions, frequencies):
    if kind not in protocol["models"]:
        raise ValueError(f"Unregistered model: {kind}")
    return SleepModel(kind, protocol, hrv_dimensions, frequencies)
