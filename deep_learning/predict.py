"""Restore a frozen model and score an aligned cache without fitting preprocessing."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .dataset import DeviceData, apply_preprocessing, load_cache
from .evaluate import STAGES, neural_predictions

PROJECT = Path(__file__).resolve().parent.parent


def predict(cache, results, kind, output, split):
    frame, arrays = load_cache(cache)
    with np.load(results / "preprocessing.npz") as saved:
        fitted = {key: saved[key] for key in saved.files}
    arrays = apply_preprocessing(arrays, fitted)
    rows = np.arange(len(frame)) if split == "all" else np.flatnonzero(frame.split.to_numpy() == split)
    if not len(rows):
        raise ValueError("Selected cache split has no windows")
    checkpoint = torch.load(results / f"{kind}.pt", map_location="cpu", weights_only=False)
    data = DeviceData(arrays, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    probabilities = neural_predictions(checkpoint, data, rows)
    result = frame.iloc[rows].reset_index(drop=True).copy()
    result["prediction"] = np.asarray(STAGES)[probabilities.argmax(axis=1)]
    for i, stage in enumerate(STAGES):
        result[f"p_{stage}"] = probabilities[:, i]
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Predicted {len(result)} windows with {kind}; saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=PROJECT / "cache" / "deep_learning")
    parser.add_argument("--results", type=Path, default=PROJECT / "results" / "deep_learning")
    parser.add_argument("--kind", default="fusion_cross")
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--output", type=Path, default=PROJECT / "results" / "deep_learning" / "restored_predictions.csv")
    args = parser.parse_args()
    predict(args.cache.resolve(), args.results.resolve(), args.kind, args.output.resolve(), args.split)
