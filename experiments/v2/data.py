"""Reuse the frozen EEG/HRV arrays, add quality and 15-epoch context."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from deep_learning.dataset import load_cache, fit_preprocessing, apply_preprocessing


def load_data(cache, quality_cache, fitted=None):
    frame, arrays = load_cache(cache)
    quality_frame = pd.read_csv(Path(quality_cache) / "epochs.csv")
    pd.testing.assert_frame_equal(frame, quality_frame)
    train_rows = np.flatnonzero(frame.split.to_numpy() == "train")
    fitted = fit_preprocessing(arrays, train_rows) if fitted is None else fitted
    arrays = apply_preprocessing(arrays, fitted)
    quality = np.load(Path(quality_cache) / "quality.npy")
    arrays["hrv"] = np.concatenate([arrays["hrv"], quality], axis=1)
    arrays["contexts"] = np.load(Path(quality_cache) / "contexts.npy")
    return frame, arrays, fitted


class Data:
    def __init__(self, arrays, device):
        self.device = device
        self.spectra = torch.as_tensor(arrays["spectra"], device=device)
        self.cardiac = torch.as_tensor(arrays["hrv"], device=device)
        self.contexts = torch.as_tensor(arrays["contexts"], dtype=torch.long, device=device)
        self.labels = torch.as_tensor(arrays["labels"], dtype=torch.long, device=device)

    def batch(self, indices, length, training=False, noise=0.02):
        contexts = self.contexts[indices, -length:]
        missing = contexts < 0
        safe = contexts.clamp_min(0)
        spectra = self.spectra[safe].masked_fill(missing[..., None, None], 0)
        cardiac = self.cardiac[safe].masked_fill(missing[..., None], 0)
        if training:
            spectra = (spectra + torch.randn_like(spectra) * noise).masked_fill(missing[..., None, None], 0)
        return spectra, cardiac, missing, self.labels[indices]
