"""Check subject isolation, contiguous past context and finite gradients before training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .dataset import DeviceData, apply_preprocessing, fit_preprocessing, load_cache
from .models import build_model


def check(cache):
    protocol = json.loads((Path(__file__).parent / "protocol.json").read_text())
    frame, arrays = load_cache(cache)
    train_rows = np.flatnonzero(frame.split.to_numpy() == "train")
    test_rows = np.flatnonzero(frame.split.to_numpy() == "test")
    assert frame.iloc[train_rows].subject.nunique() == 22
    assert frame.iloc[test_rows].subject.nunique() == 4
    assert not set(frame.iloc[train_rows].subject) & set(frame.iloc[test_rows].subject)
    contexts = arrays["contexts"]
    for i, row in enumerate(frame.itertuples()):
        for position, previous in enumerate(contexts[i]):
            if previous < 0:
                continue
            earlier = frame.iloc[previous]
            assert earlier.record == row.record and earlier.subject == row.subject
            assert earlier.split == row.split
            assert earlier.epoch == row.epoch - (contexts.shape[1] - 1 - position)
        assert contexts[i, -1] == i
    fitted = fit_preprocessing(arrays, train_rows)
    # Changing all test HRV features must leave the fitted training statistics unchanged.
    original = arrays["hrv"][test_rows].copy()
    arrays["hrv"][test_rows] = 1e9
    unchanged = fit_preprocessing(arrays, train_rows)
    for key in fitted:
        np.testing.assert_array_equal(fitted[key], unchanged[key])
    arrays["hrv"][test_rows] = original
    arrays = apply_preprocessing(arrays, fitted)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = DeviceData(arrays, device)
    # Include a record's first epoch to exercise left-padding in attention.
    first_rows = [int(group.index[0]) for _, group in frame.loc[frame.split == "train"].groupby("record")]
    idx = torch.as_tensor(first_rows[:8], device=device)
    for kind in protocol["models"]:
        model = build_model(kind, protocol, data.hrv.shape[-1], data.spectra.shape[-1]).to(device)
        eeg, hrv, spectra, missing, y = data.batch(idx, model.context_length)
        output = model(eeg, hrv, spectra, missing)
        assert output.shape == (len(idx), 5)
        assert torch.isfinite(output).all(), kind
        torch.nn.functional.cross_entropy(output, y).backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None), kind
        print(f"PASS {kind}: first-window padding, forward and gradients", flush=True)
        del model
    print(f"PASS: {len(frame)} epochs; subject isolation; past-only contiguous context; train-only preprocessing", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(__file__).resolve().parent.parent / "cache" / "deep_learning")
    check(parser.parse_args().cache)
