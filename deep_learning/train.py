"""Train predefined models on 22 subjects. Never evaluate the holdout here."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier

from .dataset import DeviceData, apply_preprocessing, fit_preprocessing, load_cache
from .models import build_model

PACKAGE = Path(__file__).resolve().parent
PROJECT = PACKAGE.parent


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)


def environment():
    result = {name: importlib.metadata.version(name) for name in
              ["torch", "numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "joblib"]}
    result["cuda_runtime"] = torch.version.cuda
    result["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return result


def train(cache, results, smoke=False):
    protocol_bytes = (PACKAGE / "protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    audit = json.loads((cache / "data_audit.json").read_text())
    if audit["protocol_sha256"] != protocol_hash:
        raise ValueError("Protocol changed after data preparation; explicitly re-prepare before training")
    if (results / "metrics.json").exists() and not smoke:
        raise RuntimeError("Holdout has already been scored; do not modify models based on it")
    frame, arrays = load_cache(cache)
    rows = np.flatnonzero(frame.split.to_numpy() == "train")
    if frame.iloc[rows].subject.nunique() != 22:
        raise ValueError("Training roster must contain exactly 22 independent subjects")
    if set(frame.iloc[rows].subject) & set(protocol["test_subjects"]):
        raise ValueError("Holdout subject entered training")
    fitted = fit_preprocessing(arrays, rows)
    arrays = apply_preprocessing(arrays, fitted)
    results.mkdir(parents=True, exist_ok=True)
    np.savez(results / "preprocessing.npz", **fitted)
    (results / "protocol.json").write_bytes(protocol_bytes)
    (results / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    frame.to_csv(results / "epoch_manifest.csv", index=False)
    versions = environment()
    (results / "environment.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE {device}; GPU={versions['gpu']}; train epochs={len(rows)}; held-out subjects={protocol['test_subjects']}", flush=True)
    if smoke:
        rng = np.random.default_rng(protocol["training_seed"])
        rows = rng.choice(rows, size=min(256, len(rows)), replace=False)
    else:
        rf_path = results / "random_forest.joblib"
        if not rf_path.exists():
            start = time.perf_counter()
            rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                        class_weight="balanced_subsample", random_state=42, n_jobs=8)
            rf.fit(arrays["rf_features"][rows], arrays["labels"][rows])
            joblib.dump({"model": rf, "protocol_sha256": protocol_hash}, rf_path, compress=3)
            print(f"TRAINED random_forest in {time.perf_counter() - start:.1f}s; no holdout scoring", flush=True)
    data = DeviceData(arrays, device)
    indices = torch.as_tensor(rows, dtype=torch.long, device=device)
    labels = arrays["labels"][rows]
    counts = np.bincount(labels, minlength=5)
    if (counts == 0).any():
        raise ValueError("Training set must contain every sleep stage")
    class_weights = len(rows) / (5 * counts)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device))
    (results / "training_class_counts.json").write_text(json.dumps({"counts": counts.tolist(),
                                                                  "loss_weights": class_weights.tolist()}), encoding="utf-8")
    final_epochs = 2 if smoke else protocol["epochs"]
    batch_size = protocol["batch_size"]
    for kind in protocol["models"]:
        target = results / f"{kind}.pt"
        if target.exists() and not smoke:
            existing = torch.load(target, map_location="cpu", weights_only=False)
            if existing["protocol_sha256"] != protocol_hash or existing["epochs"] != final_epochs:
                raise ValueError(f"Existing checkpoint conflicts with protocol: {kind}")
            print(f"SKIP completed training: {kind}", flush=True)
            continue
        seed_everything(protocol["training_seed"])
        model = build_model(kind, protocol, data.hrv.shape[-1], data.spectra.shape[-1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"], weight_decay=protocol["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=final_epochs,
                                                              eta_min=protocol["learning_rate"] * 0.05)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        history = []
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(f"START {kind}; parameters={parameters:,}; epochs={final_epochs}", flush=True)
        start = time.perf_counter()
        for epoch in range(1, final_epochs + 1):
            model.train()
            order = indices[torch.randperm(len(indices), device=device)]
            loss_sum, correct, seen = 0.0, 0, 0
            epoch_start = time.perf_counter()
            for offset in range(0, len(order), batch_size):
                batch = order[offset:offset + batch_size]
                eeg, hrv, spectra, missing, actual = data.batch(batch, model.context_length, augment=True,
                                                               noise=protocol["augmentation"]["noise_std"])
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    output = model(eeg, hrv, spectra, missing)
                    loss = criterion(output, actual)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss in {kind}, epoch {epoch}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                count = len(batch)
                loss_sum += float(loss.detach()) * count
                correct += int((output.detach().argmax(dim=1) == actual).sum())
                seen += count
            history.append({"epoch": epoch, "train_loss": loss_sum / seen,
                            "train_accuracy": correct / seen, "learning_rate": optimizer.param_groups[0]["lr"],
                            "seconds": time.perf_counter() - epoch_start})
            scheduler.step()
            pd.DataFrame(history).to_csv(results / f"{kind}_history.csv", index=False)
            print(f"{kind} epoch {epoch:02}/{final_epochs}: loss={loss_sum / seen:.4f}, train_acc={correct / seen:.4f}, {history[-1]['seconds']:.1f}s", flush=True)
        torch.save({"kind": kind, "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                    "protocol": protocol, "protocol_sha256": protocol_hash, "epochs": final_epochs,
                    "parameters": parameters, "hrv_dimensions": data.hrv.shape[-1], "frequencies": data.spectra.shape[-1],
                    "train_subjects": audit["train_subjects"], "training_seconds": time.perf_counter() - start}, target)
        print(f"SAVED {kind} final epoch {final_epochs}; no test-based checkpoint selection", flush=True)
        del model, optimizer, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()
    (results / "training_complete.json").write_text(json.dumps({"protocol_sha256": protocol_hash,
                                                               "models": protocol["models"], "epochs": final_epochs,
                                                               "smoke_only": smoke}), encoding="utf-8")
    print("TRAINING COMPLETE; held-out predictions have not been computed", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=PROJECT / "cache" / "deep_learning")
    parser.add_argument("--results", type=Path, default=PROJECT / "results" / "deep_learning")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.results = args.cache / "smoke"
    train(args.cache.resolve(), args.results.resolve(), args.smoke)


if __name__ == "__main__":
    main()
