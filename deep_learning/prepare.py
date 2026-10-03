"""Prepare single-channel EEG, spectrograms, HRV and subject-safe context indices."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import sleepecg
import wfdb
from scipy.integrate import trapezoid
from scipy.signal import butter, resample_poly, sosfiltfilt, stft, welch

PACKAGE = Path(__file__).resolve().parent
PROJECT = PACKAGE.parent
STAGES = ["Wake", "N1", "N2", "N3", "REM"]
RAW_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}
SLEEP_ENUM = [sleepecg.SleepStage.WAKE, sleepecg.SleepStage.N1, sleepecg.SleepStage.N2,
              sleepecg.SleepStage.N3, sleepecg.SleepStage.REM]


def read_edf_channel(path, channel_name, require_transducer=None):
    """Read one EDF-compatible REC channel without allocating all PSG channels."""
    with Path(path).open("rb") as stream:
        head = stream.read(256)
        if head[:8].strip() != b"0":
            raise ValueError("Only standard EDF-compatible REC input is supported")
        header_size, records = int(head[184:192]), int(head[236:244])
        duration, channels = float(head[244:252]), int(head[252:256])
        extra = stream.read(header_size - 256)
    offset, fields = 0, {}
    for name, width in [("label", 16), ("transducer", 80), ("unit", 8), ("pmin", 8),
                        ("pmax", 8), ("dmin", 8), ("dmax", 8), ("prefilter", 80),
                        ("samples", 8), ("reserved", 32)]:
        fields[name] = [extra[offset + i * width:offset + (i + 1) * width].decode("ascii").strip()
                        for i in range(channels)]
        offset += width * channels
    idx = fields["label"].index(channel_name)
    if require_transducer and require_transducer.lower() not in fields["transducer"][idx].lower():
        raise ValueError(f"{channel_name} does not identify {require_transducer}: {fields['transducer'][idx]}")
    ns = np.asarray(fields["samples"], dtype=int)
    expected_bytes = header_size + 2 * records * int(ns.sum())
    if Path(path).stat().st_size != expected_bytes:
        raise ValueError("EDF structural size mismatch")
    mapped = np.memmap(path, dtype="<i2", mode="r", offset=header_size, shape=(records, int(ns.sum())))
    start = int(ns[:idx].sum())
    digital = np.asarray(mapped[:, start:start + ns[idx]], dtype=np.float64).reshape(-1)
    gain = (float(fields["pmax"][idx]) - float(fields["pmin"][idx])) / (float(fields["dmax"][idx]) - float(fields["dmin"][idx]))
    signal = (digital - float(fields["dmin"][idx])) * gain + float(fields["pmin"][idx])
    unit = fields["unit"][idx].replace("\u00b5", "u")
    if unit not in {"uV", "mV", "V"}:
        raise ValueError(f"Unsupported physical unit {unit}")
    signal *= {"uV": 1, "mV": 1000, "V": 1e6}[unit]
    return signal, ns[idx] / duration, fields["transducer"][idx]


def eeg_arrays(signal_uv, fs, count, protocol):
    size = int(round(fs * 30))
    count = min(count, len(signal_uv) // size)
    raw = signal_uv[:count * size].reshape(count, size)
    quality = np.isfinite(raw).all(axis=1)
    quality &= np.nanstd(raw, axis=1) >= protocol["minimum_eeg_std_uv"]
    quality &= np.nanmax(np.abs(raw), axis=1) <= protocol["maximum_eeg_absolute_uv"]
    raw = np.nan_to_num(raw)
    sos = butter(4, protocol["bandpass_hz"], btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, raw, axis=1)
    divisor = math.gcd(int(fs), protocol["sample_rate"])
    eeg = resample_poly(filtered, protocol["sample_rate"] // divisor, int(fs) // divisor, axis=1).astype(np.float32)
    # Each epoch is filtered/resampled independently: no samples after its end enter it.
    freq, psd = welch(eeg, fs=100, nperseg=400, axis=1)
    valid = (freq >= 0.5) & (freq < 30)
    total = np.maximum(trapezoid(psd[:, valid], freq[valid], axis=1), 1e-12)
    features = [np.log10(total)]
    for low, high in [(0.5, 4), (4, 8), (8, 12), (12, 16), (16, 30)]:
        band = (freq >= low) & (freq < high)
        features.append(trapezoid(psd[:, band], freq[band], axis=1) / total)
    norm = psd[:, valid] / np.maximum(psd[:, valid].sum(axis=1, keepdims=True), 1e-12)
    features.append(-(norm * np.log(np.maximum(norm, 1e-12))).sum(axis=1) / np.log(norm.shape[1]))
    sf, _, spectrum = stft(eeg, fs=100, nperseg=200, noverlap=100, nfft=256,
                           boundary=None, padded=False, axis=-1)
    keep = (sf >= 0.3) & (sf <= 35)
    spectrogram = np.log10(np.maximum(np.abs(spectrum[:, keep, :]) ** 2, 1e-8)).transpose(0, 2, 1)
    return eeg, np.column_stack(features).astype(np.float32), spectrogram.astype(np.float32), quality


def hrv_features(record, protocol):
    features, _, names = sleepecg.extract_features(
        [record], lookback=protocol["hrv_lookback_seconds"], lookforward=protocol["hrv_lookforward_seconds"],
        feature_selection=["hrv-time"], min_rri=protocol["rr_interval_seconds"][0],
        max_rri=protocol["rr_interval_seconds"][1],
    )
    values = features[0].astype(np.float32)
    values[~np.isfinite(values)] = np.nan
    return values, names


def context_indices(frame, length):
    lookup = {(row.record, int(row.epoch)): idx for idx, row in enumerate(frame.itertuples())}
    contexts = np.full((len(frame), length), -1, dtype=np.int64)
    for idx, row in enumerate(frame.itertuples()):
        # Only the contiguous segment ending in this epoch is used.
        current = int(row.epoch)
        for position in range(length - 1, -1, -1):
            key = (row.record, current - (length - 1 - position))
            if key not in lookup:
                break
            contexts[idx, position] = lookup[key]
    return contexts


def prepare(mit_dir, isruc_dir, cache):
    protocol_path = PACKAGE / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    cache.mkdir(parents=True, exist_ok=True)
    frames, audits, feature_ids = [], [], None
    enum_to_class = {int(value): idx for idx, value in enumerate(SLEEP_ENUM)}

    def save_record(record_id, subject, source, eeg_uv, fs, stages, sleep_record, channel, extra):
        nonlocal feature_ids
        eeg, engineered, spectra, quality = eeg_arrays(eeg_uv, fs, len(stages), protocol)
        hrv, names = hrv_features(sleep_record, protocol)
        if feature_ids is not None and names != feature_ids:
            raise ValueError("HRV feature order changed between records")
        feature_ids = names
        count = min(len(stages), len(eeg), len(hrv))
        stage = np.asarray(stages[:count], dtype=np.int64)
        keep = (stage >= 0) & quality[:count]
        indices = np.flatnonzero(keep)
        if not len(indices):
            raise ValueError(f"No usable labeled EEG epochs in {record_id}")
        data_path = cache / f"{record_id}.npz"
        np.savez_compressed(data_path, eeg=eeg[:count][keep], eeg_features=engineered[:count][keep],
                            spectra=spectra[:count][keep], hrv=hrv[:count][keep], labels=stage[keep], epoch=indices)
        frame = pd.DataFrame({"record": record_id, "subject": subject, "source": source,
                              "epoch": indices, "stage": stage[keep], "record_row": np.arange(len(indices))})
        frames.append(frame)
        rr = np.diff(sleep_record.heartbeat_times)
        audit = {"record": record_id, "subject": subject, "source": source, "eeg_channel": channel,
                 "source_sample_rate": fs, "retained_epochs": int(keep.sum()), "aligned_epochs": count,
                 "unknown_label_epochs": int((stage < 0).sum()),
                 "bad_eeg_epochs": int(((stage >= 0) & ~quality[:count]).sum()),
                 "hrv_missing_fraction": float(np.isnan(hrv[:count][keep]).mean()),
                 "detected_beats": len(sleep_record.heartbeat_times),
                 "median_heart_rate_bpm": float(60 / np.median(rr)) if len(rr) else None, **extra}
        audits.append(audit)
        print(json.dumps(audit), flush=True)

    for path in sorted(mit_dir.glob("*.hea")):
        rid = path.stem
        header = wfdb.rdheader(str(path.with_suffix("")))
        channel = next(i for i, name in enumerate(header.sig_name) if name.upper().startswith("EEG"))
        signal = wfdb.rdrecord(str(path.with_suffix("")), channels=[channel]).p_signal[:, 0]
        unit = header.units[channel]
        if unit not in {"mV", "uV", "V"}:
            raise ValueError(f"Unexpected MIT EEG unit: {unit}")
        signal *= {"mV": 1000, "uV": 1, "V": 1e6}[unit]
        record = next(iter(sleepecg.read_slpdb(rid, offline=True, data_dir=mit_dir.parent)))
        stages = [enum_to_class.get(int(label), -1) for label in record.sleep_stages]
        subject = rid[:-1] if rid in {"slp01a", "slp01b", "slp02a", "slp02b"} else rid
        save_record(f"MIT-{rid}", f"MIT-{subject}", "MIT-BIH", signal, header.fs, stages,
                    record, header.sig_name[channel], {"ecg_channel": "ECG", "label_standard": "R&K, 3+4 merged"})

    for number in range(1, 11):
        rid = f"ISRUC-III-{number:02}"
        folder = isruc_dir / str(number)
        original = [int(value.strip()) for value in (folder / f"{number}_1.txt").read_text().splitlines() if value.strip()]
        trim = protocol["isruc_drop_last_epochs"]
        stages = [RAW_TO_CLASS.get(label, -1) for label in original[:-trim]]
        eeg, fs, _ = read_edf_channel(folder / f"{number}.rec", "C4-A1", "EEG")
        ecg, ecg_fs, transducer = read_edf_channel(folder / f"{number}.rec", "X2", "EKG")
        # The header itself identifies X2 as EKG_Channel, independently of the paper mapping.
        beats = sleepecg.detect_heartbeats(ecg.astype(np.float64), ecg_fs) / ecg_fs
        record = sleepecg.SleepRecord(id=rid, sleep_stage_duration=30, heartbeat_times=beats,
                                     sleep_stages=np.asarray([int(SLEEP_ENUM[s]) for s in stages]))
        save_record(rid, rid, "ISRUC-III", eeg, fs, stages, record, "C4-A1",
                    {"ecg_channel": "X2", "ecg_transducer": transducer, "label_standard": "AASM, scorer 1",
                     "removed_tail_epochs": trim})

    frame = pd.concat(frames, ignore_index=True)
    frame["split"] = np.where(frame.subject.isin(protocol["test_subjects"]), "test", "train")
    if frame.subject.nunique() != 26:
        raise ValueError("Expected all 26 independent subjects")
    actual_test = sorted(frame.loc[frame.split == "test", "subject"].unique())
    if actual_test != sorted(protocol["test_subjects"]):
        raise ValueError("Test roster mismatch")
    if set(frame.loc[frame.split == "train", "subject"]) & set(actual_test):
        raise ValueError("Subject leakage")
    frame.to_csv(cache / "epochs.csv", index=False)
    np.save(cache / "contexts.npy", context_indices(frame, protocol["context_epochs"]))
    audit = {"protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(), "subjects": 26,
             "train_subjects": sorted(frame.loc[frame.split == "train", "subject"].unique()),
             "test_subjects": actual_test, "epochs": len(frame),
             "train_epochs": int((frame.split == "train").sum()), "test_epochs": int((frame.split == "test").sum()),
             "stages": STAGES, "hrv_feature_ids": feature_ids,
             "class_counts": {STAGES[int(k)]: int(v) for k, v in Counter(frame.stage).items()},
             "records": audits,
             "ecg_mapping_source": "https://research.usq.edu.au/download/fd11f5eee4dc20ffe248a913d3c5ab9f84fbf623e121d58c157ca023d06c911a/1631448/1-s2.0-S0169260723006582-main.pdf",
             "tail_trim_source": "https://sleeptight.isr.uc.pt/?page_id=76"}
    # Read shape from the actual saved file, never rely on a rounded frequency-bin count.
    with np.load(cache / f"{frame.iloc[0].record}.npz") as saved:
        audit["spectrogram_shape"] = list(saved["spectra"].shape[1:])
    (cache / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"PREPARED {len(frame)} epochs; 22 train subjects / 4 test subjects; no validation", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mit-dir", type=Path, required=True)
    parser.add_argument("--isruc-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=PROJECT / "cache" / "deep_learning")
    args = parser.parse_args()
    prepare(args.mit_dir.resolve(), args.isruc_dir.resolve(), args.cache.resolve())


if __name__ == "__main__":
    main()
