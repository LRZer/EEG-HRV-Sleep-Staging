"""Load aligned caches; learn all population-dependent preprocessing on train only."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_cache(cache):
    cache = Path(cache)
    frame = pd.read_csv(cache / "epochs.csv")
    arrays = {key: [] for key in ["eeg", "eeg_features", "spectra", "hrv", "labels"]}
    for record, rows in frame.groupby("record", sort=False):
        with np.load(cache / f"{record}.npz") as data:
            if not np.array_equal(data["labels"], rows.stage.to_numpy()):
                raise ValueError("Cache labels are out of order")
            if not np.array_equal(data["epoch"], rows.epoch.to_numpy()):
                raise ValueError("Cache epochs are out of order")
            for key in arrays:
                arrays[key].append(data[key])
    arrays = {key: np.concatenate(values) for key, values in arrays.items()}
    contexts = np.load(cache / "contexts.npy")
    if len(contexts) != len(frame) or not np.array_equal(contexts[:, -1], np.arange(len(frame))):
        raise ValueError("Invalid context targets")
    arrays["contexts"] = contexts
    return frame, arrays


def fit_preprocessing(arrays, training_rows):
    if not len(training_rows):
        raise ValueError("No training samples")
    hrv = arrays["hrv"][training_rows]
    median = np.asarray([np.median(column[np.isfinite(column)]) if np.isfinite(column).any() else 0
                         for column in hrv.T], dtype=np.float32)
    filled = np.where(np.isfinite(hrv), hrv, median)
    hrv_mean, hrv_std = filled.mean(axis=0), np.maximum(filled.std(axis=0), 1e-6)
    spectra = arrays["spectra"][training_rows]
    waveform_scale = max(float(np.median(arrays["eeg"][training_rows].std(axis=1))), 1e-3)
    return {"hrv_median": median, "hrv_mean": hrv_mean, "hrv_std": hrv_std,
            "spectra_mean": spectra.mean(axis=(0, 1)),
            "spectra_std": np.maximum(spectra.std(axis=(0, 1)), 1e-6),
            "waveform_scale": np.asarray(waveform_scale, dtype=np.float32)}


def apply_preprocessing(arrays, fitted):
    present = np.isfinite(arrays["hrv"])
    filled = np.where(present, arrays["hrv"], fitted["hrv_median"])
    cardiac = np.clip((filled - fitted["hrv_mean"]) / fitted["hrv_std"], -8, 8)
    arrays["rf_features"] = np.concatenate([arrays["eeg_features"], cardiac], axis=1).astype(np.float32)
    arrays["hrv"] = np.concatenate([cardiac, (~present).astype(np.float32)], axis=1).astype(np.float32)
    waveform = arrays["eeg"]
    arrays["eeg"] = np.clip((waveform - waveform.mean(axis=1, keepdims=True)) / fitted["waveform_scale"], -12, 12).astype(np.float32)
    arrays["spectra"] = np.clip((arrays["spectra"] - fitted["spectra_mean"]) / fitted["spectra_std"], -8, 8).astype(np.float32)
    return arrays


class DeviceData:
    def __init__(self, arrays, device):
        self.device = device
        self.eeg = torch.as_tensor(arrays["eeg"], device=device)
        self.hrv = torch.as_tensor(arrays["hrv"], device=device)
        self.spectra = torch.as_tensor(arrays["spectra"], device=device)
        self.labels = torch.as_tensor(arrays["labels"], dtype=torch.long, device=device)
        self.contexts = torch.as_tensor(arrays["contexts"], dtype=torch.long, device=device)

    def batch(self, indices, length, augment=False, noise=0.02):
        context = self.contexts[indices, -length:]
        missing = context < 0
        safe = context.clamp_min(0)
        eeg = self.eeg[safe].masked_fill(missing[..., None], 0)
        hrv = self.hrv[safe].masked_fill(missing[..., None], 0)
        spectra = self.spectra[safe].masked_fill(missing[..., None, None], 0)
        if augment:
            amplitude = torch.rand((*eeg.shape[:2], 1), device=self.device) * 0.2 + 0.9
            eeg = (eeg * amplitude + torch.randn_like(eeg) * noise).masked_fill(missing[..., None], 0)
            # Small perturbation in standardized log-power space for the spectral branch.
            spectra = (spectra + torch.randn_like(spectra) * noise).masked_fill(missing[..., None, None], 0)
        return eeg, hrv, spectra, missing, self.labels[indices]
