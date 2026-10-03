"""Train all predeclared variant/seed runs; never read holdout performance."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from deep_learning.train import seed_everything, environment
from .data import Data, load_data
from .models import FusionModel

PROJECT = Path(__file__).resolve().parents[2]


def train(cache, quality_cache, results):
    protocol_bytes = Path(__file__).with_name("protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    digest = hashlib.sha256(protocol_bytes).hexdigest()
    if (results / "metrics.json").exists():
        raise RuntimeError("v2 already scored; preserve frozen experiment")
    audit = json.loads((quality_cache / "quality_audit.json").read_text())
    assert audit["protocol_sha256"] == digest
    frame, arrays, fitted = load_data(cache, quality_cache)
    rows = np.flatnonzero(frame.split.to_numpy() == "train")
    test_rows = np.flatnonzero(frame.split.to_numpy() == "test")
    assert frame.iloc[rows].subject.nunique() == 22
    assert sorted(frame.iloc[test_rows].subject.unique()) == sorted(protocol["test_subjects"])
    assert not set(frame.iloc[rows].subject) & set(frame.iloc[test_rows].subject)
    results.mkdir(parents=True, exist_ok=True)
    if (results / "protocol.json").exists():
        assert (results / "protocol.json").read_bytes() == protocol_bytes
    (results / "protocol.json").write_bytes(protocol_bytes)
    np.savez(results / "preprocessing.npz", **fitted)
    frame.to_csv(results / "epoch_manifest.csv", index=False)
    versions = environment()
    versions["attention_backend"] = "math SDPA; deterministic cuDNN; v1-compatible"
    (results / "environment.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Data(arrays, device)
    indices = torch.as_tensor(rows, dtype=torch.long, device=device)
    counts = np.bincount(arrays["labels"][rows], minlength=5)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.as_tensor(len(rows) / (5 * counts), dtype=torch.float32, device=device))
    (results / "training_class_counts.json").write_text(json.dumps(counts.tolist()), encoding="utf-8")
    completed = []
    for variant in protocol["variants"]:
        for seed in protocol["seeds"]:
            run_name = f"{variant}_seed{seed}"
            path = results / "checkpoints" / f"{run_name}.pt"
            path.parent.mkdir(exist_ok=True)
            if path.exists():
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                assert checkpoint["protocol_sha256"] == digest and checkpoint["epochs"] == protocol["epochs"]
                completed.append(run_name)
                print(f"SKIP {run_name}: complete final checkpoint", flush=True)
                continue
            seed_everything(seed)
            model = FusionModel(variant, protocol).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"], weight_decay=protocol["weight_decay"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, protocol["epochs"], eta_min=protocol["learning_rate"] * 0.05)
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
            history = []
            started = time.perf_counter()
            parameters = sum(p.numel() for p in model.parameters())
            print(f"START {run_name}; parameters={parameters}; context={model.context_length}", flush=True)
            for epoch in range(1, protocol["epochs"] + 1):
                model.train()
                order = indices[torch.randperm(len(indices), device=device)]
                loss_sum = correct = seen = 0
                epoch_started = time.perf_counter()
                for offset in range(0, len(order), protocol["batch_size"]):
                    batch = order[offset:offset + protocol["batch_size"]]
                    spectra, cardiac, missing, actual = data.batch(batch, model.context_length, True, protocol["spectral_noise_std"])
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        logits = model(spectra, cardiac, missing)
                        loss = criterion(logits, actual)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss: {run_name} epoch {epoch}")
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    count = len(batch)
                    loss_sum += float(loss.detach()) * count
                    correct += int((logits.detach().argmax(1) == actual).sum())
                    seen += count
                history.append({"variant": variant, "seed": seed, "epoch": epoch, "train_loss": loss_sum / seen,
                    "train_accuracy": correct / seen, "learning_rate": optimizer.param_groups[0]["lr"],
                    "seconds": time.perf_counter() - epoch_started})
                scheduler.step()
                pd.DataFrame(history).to_csv(results / f"{run_name}_history.csv", index=False)
                if epoch == 1 or epoch % 5 == 0:
                    print(f"{run_name} {epoch}/40: loss={loss_sum / seen:.4f}, train_acc={correct / seen:.4f}, {history[-1]['seconds']:.2f}s", flush=True)
            torch.save({"variant": variant, "seed": seed, "protocol": protocol, "protocol_sha256": digest,
                "epochs": protocol["epochs"], "parameters": parameters,
                "train_subjects": sorted(frame.iloc[rows].subject.unique()),
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "training_seconds": time.perf_counter() - started}, path)
            completed.append(run_name)
            print(f"SAVED {run_name}; final epoch only; {len(completed)}/21 runs", flush=True)
            del model, optimizer, scaler
            if device.type == "cuda":
                torch.cuda.empty_cache()
    (results / "training_complete.json").write_text(json.dumps({"protocol_sha256": digest,
        "completed_runs": completed, "runs": len(completed), "test_scored": False}, indent=2), encoding="utf-8")
    print("COMPLETE: 21 frozen training runs; holdout not scored", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=PROJECT / "cache" / "deep_learning")
    parser.add_argument("--quality-cache", type=Path, default=PROJECT / "cache" / "v2")
    parser.add_argument("--results", type=Path, default=PROJECT / "results" / "v2")
    args = parser.parse_args()
    train(args.cache, args.quality_cache, args.results)
