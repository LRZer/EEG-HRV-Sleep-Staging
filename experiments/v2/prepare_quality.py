"""Add aligned past-only HRV quality and detector diagnostics to the v1 cache."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import sleepecg
import wfdb

from deep_learning.prepare import context_indices, read_edf_channel

PROJECT = Path(__file__).resolve().parents[2]


def match_beats(reference, detected, tolerance=0.15):
    i = j = matches = 0
    while i < len(reference) and j < len(detected):
        difference = detected[j] - reference[i]
        if abs(difference) <= tolerance:
            matches += 1
            i += 1
            j += 1
        elif difference < 0:
            j += 1
        else:
            i += 1
    precision = matches / max(len(detected), 1)
    recall = matches / max(len(reference), 1)
    return matches, precision, recall, 2 * precision * recall / max(precision + recall, 1e-12)


def prepare(cache, mit, isruc, output):
    protocol_path = Path(__file__).with_name("protocol.json")
    protocol = json.loads(protocol_path.read_text())
    audit = json.loads((cache / "data_audit.json").read_text())
    assert audit["protocol_sha256"] == protocol["v1_preparation_protocol_sha256"]
    frame = pd.read_csv(cache / "epochs.csv")
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "contexts.npy", context_indices(frame, 15))
    frame.to_csv(output / "epochs.csv", index=False)
    checksums = dict(line.split()[::-1] for line in (mit / "SHA256SUMS.txt").read_text().splitlines() if line.strip())
    quality, beats_audit, snippets = [], [], []
    rng = np.random.default_rng(42)
    for record, group in frame.groupby("record", sort=False):
        if record.startswith("MIT-"):
            rid = record[4:]
            header = wfdb.rdheader(str(mit / rid))
            channel = header.sig_name.index("ECG")
            ecg = wfdb.rdrecord(str(mit / rid), channels=[channel]).p_signal[:, 0]
            fs = header.fs
        else:
            number = int(record.rsplit("-", 1)[1])
            ecg, fs, _ = read_edf_channel(isruc / str(number) / f"{number}.rec", "X2", "EKG")
        detected_samples = sleepecg.detect_heartbeats(ecg.astype(np.float64), fs)
        beats = detected_samples / fs
        rr, endpoints = np.diff(beats), beats[1:]
        with np.load(cache / f"{record}.npz") as saved:
            finite_fraction = np.isfinite(saved["hrv"]).mean(axis=1)
        local_quality = []
        for row, finite in zip(group.itertuples(), finite_fraction):
            end = (row.epoch + 1) * 30
            lo, hi = np.searchsorted(endpoints, [max(0, end - 300), end], side="left")
            values = rr[lo:hi]
            valid = (values >= 0.3) & (values <= 2.0)
            local_quality.append([float(valid.mean()) if len(values) else 0, float(finite), min(end, 300) / 300])
        quality.append(np.asarray(local_quality, np.float32))
        item = {"record": record, "split": group.iloc[0].split, "detected_beats": len(beats),
                "quality_mean": np.mean(local_quality, axis=0).tolist()}
        if record.startswith("MIT-"):
            path = mit / f"{rid}.ecg"
            if not path.exists():
                response = requests.get(f"https://physionet.org/files/slpdb/1.0.0/{rid}.ecg", timeout=60)
                response.raise_for_status()
                assert hashlib.sha256(response.content).hexdigest() == checksums[path.name]
                path.write_bytes(response.content)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == checksums[path.name]
            annot = wfdb.rdann(str(mit / rid), "ecg")
            beat_symbols = {"N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F", "e", "j", "n", "E", "/", "f", "Q", "?"}
            reference = annot.sample[np.asarray([symbol in beat_symbols for symbol in annot.symbol])] / fs
            tp, precision, recall, f1 = match_beats(reference, beats)
            item.update(reference_beats=len(reference), matched_beats=tp, beat_precision=precision,
                        beat_recall=recall, beat_f1=f1, tolerance_seconds=0.15)
        beats_audit.append(item)
        # Fixed random training-only short segments for waveform/detector visual review.
        if group.iloc[0].split == "train":
            starts = rng.choice(group.epoch.to_numpy(), size=1, replace=False) * 30
            for start in starts:
                start = int(start)
                segment = ecg[int(start * fs):int((start + 10) * fs)].astype(np.float32)
                snippets.append({"record": record, "start_seconds": start, "fs": float(fs),
                                 "ecg": segment, "beats": beats[(beats >= start) & (beats < start + 10)] - start})
        print(f"QUALITY {record}; ECG beats={len(beats)}; labeled windows={len(group)}", flush=True)
    all_quality = np.concatenate(quality)
    assert all_quality.shape == (len(frame), 3) and np.isfinite(all_quality).all()
    np.save(output / "quality.npy", all_quality)
    for index, snippet in enumerate(snippets):
        np.savez_compressed(output / f"ecg_review_{index:02}.npz", **snippet)
    pd.DataFrame(beats_audit).to_csv(output / "ecg_detector_audit.csv", index=False)
    (output / "quality_audit.json").write_text(json.dumps({"protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "epochs": len(frame), "fields": protocol["hrv_quality_features"], "beat_matching": "greedy one-to-one within 150 ms; MIT provided beat annotations",
        "boundary": "RR endpoint within [max(0, epoch_end-300), epoch_end); no future endpoint",
        "note": "quality indicators describe coverage and plausibility, not a clinically validated signal-quality index"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=PROJECT / "cache" / "deep_learning")
    parser.add_argument("--mit-dir", type=Path, required=True)
    parser.add_argument("--isruc-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT / "cache" / "v2")
    args = parser.parse_args()
    prepare(args.cache, args.mit_dir, args.isruc_dir, args.output)
