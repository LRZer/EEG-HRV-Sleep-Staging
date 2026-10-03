"""Common post-training evaluation and predeclared missing-HRV stress test."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from deep_learning.evaluate import STAGES, score
from .data import Data, load_data
from .models import FusionModel

PROJECT = Path(__file__).resolve().parents[2]


@torch.inference_mode()
def predict(checkpoint, data, rows, condition="normal", batch_size=128):
    model = FusionModel(checkpoint["variant"], checkpoint["protocol"]).to(data.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    probabilities, gates = [], []
    for offset in range(0, len(rows), batch_size):
        idx = torch.as_tensor(rows[offset:offset + batch_size], dtype=torch.long, device=data.device)
        spectra, cardiac, missing, _ = data.batch(idx, model.context_length)
        logits, gate = model(spectra, cardiac, missing, condition == "hrv_missing", True)
        probabilities.append(logits.softmax(1).cpu().numpy())
        gates.append(gate.cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(gates)


def evaluate(cache, quality_cache, results):
    if (results / "metrics.json").exists():
        raise RuntimeError("v2 already evaluated; use stored predictions for figures and analysis")
    protocol_bytes = (results / "protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    digest = hashlib.sha256(protocol_bytes).hexdigest()
    completion = json.loads((results / "training_complete.json").read_text())
    assert completion["protocol_sha256"] == digest and completion["runs"] == 21
    with np.load(results / "preprocessing.npz") as saved:
        fitted = {key: saved[key] for key in saved.files}
    frame, arrays, _ = load_data(cache, quality_cache, fitted)
    rows = np.flatnonzero(frame.split.to_numpy() == "test")
    assert sorted(frame.iloc[rows].subject.unique()) == sorted(protocol["test_subjects"])
    truth = frame.iloc[rows].reset_index(drop=True)
    data = Data(arrays, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    actual = truth.stage.to_numpy()
    quality = arrays["hrv"][rows, -3:]
    table, classes, people, sources, prediction_tables = [], [], [], [], []
    details = {}

    def record_metrics(variant, seed, condition, probability, gate, checkpoint_sha=None):
        predicted = probability.argmax(1)
        metrics = score(actual, predicted)
        key = f"{variant}_seed{seed}_{condition}"
        details[key] = {**metrics, "checkpoint_sha256": checkpoint_sha}
        table.append({"variant": variant, "seed": seed, "condition": condition,
            **{k: metrics[k] for k in ["accuracy", "macro_f1", "balanced_accuracy", "kappa"]}})
        for stage, values in metrics["per_class"].items():
            classes.append({"variant": variant, "seed": seed, "condition": condition, "stage": stage, **values})
        for field, destination in [("subject", people), ("source", sources)]:
            for value in truth[field].unique():
                mask = truth[field].to_numpy() == value
                m = score(actual[mask], predicted[mask])
                destination.append({"variant": variant, "seed": seed, "condition": condition, field: value,
                    "epochs": int(mask.sum()), "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                    **{f"{s}_support": m["per_class"][s]["support"] for s in STAGES}})
        predicted_frame = truth.copy()
        predicted_frame["variant"], predicted_frame["seed"], predicted_frame["condition"] = variant, seed, condition
        predicted_frame["prediction"] = predicted
        predicted_frame["gate"] = gate
        for i, stage in enumerate(STAGES):
            predicted_frame[f"p_{stage}"] = probability[:, i]
        for i, field in enumerate(protocol["hrv_quality_features"]):
            predicted_frame[field] = quality[:, i]
        prediction_tables.append(predicted_frame)
        print(f"SCORE {key}: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}", flush=True)

    for variant in protocol["variants"]:
        for condition in protocol["robustness_conditions"]:
            probabilities, gates = [], []
            for seed in protocol["seeds"]:
                path = results / "checkpoints" / f"{variant}_seed{seed}.pt"
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                assert checkpoint["protocol_sha256"] == digest and checkpoint["epochs"] == 40
                assert not set(checkpoint["train_subjects"]) & set(protocol["test_subjects"])
                probability, gate = predict(checkpoint, data, rows, condition)
                assert np.isfinite(probability).all() and np.allclose(probability.sum(1), 1, atol=1e-5)
                probabilities.append(probability)
                gates.append(gate)
                record_metrics(variant, str(seed), condition, probability, gate, hashlib.sha256(path.read_bytes()).hexdigest())
            record_metrics(variant, "ensemble", condition, np.mean(probabilities, axis=0), np.mean(gates, axis=0))
    scores = pd.DataFrame(table)
    scores.to_csv(results / "run_metrics.csv", index=False)
    pd.DataFrame(classes).to_csv(results / "per_class.csv", index=False)
    pd.DataFrame(people).to_csv(results / "per_subject.csv", index=False)
    pd.DataFrame(sources).to_csv(results / "per_dataset.csv", index=False)
    pd.concat(prediction_tables, ignore_index=True).to_csv(results / "predictions.csv.gz", index=False, compression="gzip")
    individual = scores[scores.seed != "ensemble"]
    aggregate = individual.groupby(["variant", "condition"])[["accuracy", "macro_f1", "balanced_accuracy", "kappa"]].agg(["mean", "std"])
    aggregate.columns = ["_".join(column) for column in aggregate.columns]
    aggregate.reset_index().to_csv(results / "comparison.csv", index=False)
    summary = {"protocol_sha256": digest, "evaluation_status": protocol["evaluation_status"],
        "subjects": 26, "train_subjects": 22, "test_subjects": protocol["test_subjects"],
        "train_epochs": 15761, "test_epochs": len(rows), "runs": 21, "validation_subjects": [],
        "seed_std_note": "sample standard deviation across initializations on the same subjects; not subject-level confidence intervals",
        "per_subject_f1_note": "five fixed classes, absent true classes receive F1=0", "models": details}
    (results / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("EVALUATION COMPLETE: all seeds, ensembles and predeclared missing-HRV condition saved", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=PROJECT / "cache" / "deep_learning")
    parser.add_argument("--quality-cache", type=Path, default=PROJECT / "cache" / "v2")
    parser.add_argument("--results", type=Path, default=PROJECT / "results" / "v2")
    args = parser.parse_args()
    evaluate(args.cache, args.quality_cache, args.results)
