"""EEG + ECG-derived HRV sleep staging on the open MIT-BIH PSG database."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import requests
import sleepecg
import wfdb
from scipy.integrate import trapezoid
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline


STAGES = ["Wake", "N1", "N2", "N3", "REM"]
SLEEPECG_TO_CLASS = {
    int(sleepecg.SleepStage.WAKE): 0,
    int(sleepecg.SleepStage.N1): 1,
    int(sleepecg.SleepStage.N2): 2,
    int(sleepecg.SleepStage.N3): 3,
    int(sleepecg.SleepStage.REM): 4,
}
EEG_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "sigma": (12, 16),
    "beta": (16, 30),
}
DATA_URL = "https://physionet-open.s3.amazonaws.com/slpdb/1.0.0"


def download_recordings(data_dir: Path, requested: list[str]) -> list[str]:
    """Download the open PhysioNet files from its public S3 mirror and verify SHA256."""
    db_dir = data_dir / "slpdb"
    db_dir.mkdir(parents=True, exist_ok=True)
    for index_name in ("RECORDS", "SHA256SUMS.txt"):
        index_path = db_dir / index_name
        if not index_path.exists():
            response = requests.get(f"{DATA_URL}/{index_name}", timeout=60)
            response.raise_for_status()
            index_path.write_bytes(response.content)
    available = (db_dir / "RECORDS").read_text(encoding="utf-8").splitlines()
    selected = requested or available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown record IDs: {', '.join(unknown)}")
    checksums = {}
    for line in (db_dir / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        checksum, filename = line.split(maxsplit=1)
        checksums[filename.lstrip("* ")] = checksum

    def download_one(filename: str) -> str:
        target = db_dir / filename
        expected = checksums[filename]
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            return filename
        temp = target.with_name(target.name + ".part")
        digest = hashlib.sha256()
        with requests.get(f"{DATA_URL}/{filename}", stream=True, timeout=120) as response:
            response.raise_for_status()
            with temp.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
                        digest.update(chunk)
        if digest.hexdigest() != expected:
            temp.unlink(missing_ok=True)
            raise ValueError(f"SHA256 mismatch: {filename}")
        temp.replace(target)
        return filename

    filenames = [f"{record_id}{ext}" for record_id in selected for ext in (".hea", ".dat", ".st")]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download_one, filename) for filename in filenames]
        for future in as_completed(futures):
            print(f"Downloaded/verified {future.result()}", flush=True)
    return selected


def subject_id(record_id: str) -> str:
    """Keep two segments of subjects 01 and 02 together during cross-validation."""
    return record_id[:-1] if record_id in {"slp01a", "slp01b", "slp02a", "slp02b"} else record_id


def eeg_features(record_path: Path, n_epochs: int) -> pd.DataFrame:
    header = wfdb.rdheader(str(record_path))
    eeg_channels = [i for i, name in enumerate(header.sig_name) if name.upper().startswith("EEG")]
    if not eeg_channels:
        raise ValueError(f"No EEG channel in {record_path}")
    fs = float(header.fs)
    samples_per_epoch = round(30 * fs)
    signal = wfdb.rdrecord(
        str(record_path),
        sampto=min(header.sig_len, n_epochs * samples_per_epoch),
        channels=[eeg_channels[0]],
    ).p_signal[:, 0]
    available = len(signal) // samples_per_epoch
    epochs = signal[: available * samples_per_epoch].reshape(available, samples_per_epoch)
    freqs, psd = welch(epochs, fs=fs, nperseg=min(1024, samples_per_epoch), axis=1)
    valid = (freqs >= 0.5) & (freqs < 30)
    total = trapezoid(psd[:, valid], freqs[valid], axis=1)
    total = np.maximum(total, 1e-12)
    features = {"eeg_log_total_power": np.log10(total)}
    for name, (low, high) in EEG_BANDS.items():
        band = (freqs >= low) & (freqs < high)
        power = trapezoid(psd[:, band], freqs[band], axis=1)
        features[f"eeg_{name}_relative_power"] = power / total
    normalized_psd = psd[:, valid] / np.maximum(psd[:, valid].sum(axis=1, keepdims=True), 1e-12)
    features["eeg_spectral_entropy"] = -np.sum(
        normalized_psd * np.log(np.maximum(normalized_psd, 1e-12)), axis=1
    ) / np.log(normalized_psd.shape[1])
    return pd.DataFrame(features)


def prepare(data_dir: Path, cache: Path, records: list[str]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    selected = download_recordings(data_dir, records)
    frames = []
    for record_id in selected:
        for record in sleepecg.read_slpdb(record_id, offline=True, data_dir=data_dir):
            record_path = data_dir / "slpdb" / record.id
            eeg = eeg_features(record_path, len(record.sleep_stages))
            hrv_list, _, hrv_names = sleepecg.extract_features(
                [record],
                lookback=270,
                lookforward=30,
                feature_selection=["hrv-time"],
                min_rri=0.3,
                max_rri=2.0,
            )
            hrv = pd.DataFrame(hrv_list[0], columns=[f"hrv_{name}" for name in hrv_names])
            count = min(len(eeg), len(hrv), len(record.sleep_stages))
            frame = pd.concat([eeg.iloc[:count], hrv.iloc[:count]], axis=1)
            frame.insert(0, "stage", [SLEEPECG_TO_CLASS.get(int(x), -1) for x in record.sleep_stages[:count]])
            frame.insert(0, "epoch", np.arange(count))
            frame.insert(0, "subject", subject_id(record.id))
            frame.insert(0, "record", record.id)
            frame = frame.loc[frame.stage >= 0].copy()
            frames.append(frame)
            print(f"{record.id}: {len(frame)} labeled epochs, EEG={eeg.shape[1]}, HRV={hrv.shape[1]}", flush=True)
    if not frames:
        raise ValueError("No labeled records were prepared")
    data = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    data.to_csv(cache, index=False)
    print(f"Saved {len(data)} epochs from {data.subject.nunique()} subjects to {cache}")


def evaluate(cache: Path, results: Path, folds: int) -> None:
    data = pd.read_csv(cache)
    n_subjects = data.subject.nunique()
    if n_subjects < folds:
        raise ValueError(f"Need at least {folds} distinct subjects; cache has {n_subjects}")
    results.mkdir(parents=True, exist_ok=True)
    eeg_cols = [c for c in data if c.startswith("eeg_")]
    hrv_cols = [c for c in data if c.startswith("hrv_")]
    if not eeg_cols or not hrv_cols:
        raise ValueError("Both EEG and HRV features are required")
    feature_sets = {"eeg": eeg_cols, "hrv": hrv_cols, "fusion": eeg_cols + hrv_cols}
    groups = data.subject.to_numpy()
    y = data.stage.to_numpy(dtype=int)
    predictions = data[["record", "subject", "epoch", "stage"]].copy()
    predictions["fold"] = -1
    summary = {
        "dataset": "MIT-BIH Polysomnographic Database 1.0.0",
        "label_mapping": "R&K 3 and 4 merged into N3",
        "epochs": len(data),
        "subjects": n_subjects,
        "class_counts": {STAGES[i]: int((y == i).sum()) for i in range(5)},
        "cross_validation": f"{folds}-fold GroupKFold by subject",
        "software_versions": {
            "python": platform.python_version(),
            **{package: version(package) for package in ("numpy", "scipy", "pandas", "scikit-learn", "sleepecg", "wfdb")},
        },
        "models": {},
    }
    splits = list(GroupKFold(n_splits=folds).split(data, y, groups))
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        assert set(groups[train_idx]).isdisjoint(groups[test_idx])
        predictions.loc[test_idx, "fold"] = fold
    for name, columns in feature_sets.items():
        x = data[columns].to_numpy(dtype=float)
        pred = np.full(len(y), -1, dtype=int)
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestClassifier(
                    n_estimators=120,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Skipping features without any observed values")
                model.fit(x[train_idx], y[train_idx])
            pred[test_idx] = model.predict(x[test_idx])
            print(f"{name}: fold {fold}/{folds} complete", flush=True)
        if np.any(pred < 0):
            raise RuntimeError("Some epochs were not assigned a held-out prediction")
        predictions[f"pred_{name}"] = pred
        matrix = confusion_matrix(y, pred, labels=range(5))
        summary["models"][name] = {
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "macro_f1": round(float(f1_score(y, pred, labels=range(5), average="macro", zero_division=0)), 4),
            "f1_by_stage": {
                STAGES[i]: round(float(value), 4)
                for i, value in enumerate(f1_score(y, pred, labels=range(5), average=None, zero_division=0))
            },
            "confusion_matrix": matrix.tolist(),
        }
        print(f"{name}: Accuracy={summary['models'][name]['accuracy']}, Macro-F1={summary['models'][name]['macro_f1']}")
        if name == "fusion":
            fig, ax = plt.subplots(figsize=(6, 5))
            image = ax.imshow(matrix, cmap="Blues")
            fig.colorbar(image, ax=ax)
            ax.set_xticks(range(5), STAGES)
            ax.set_yticks(range(5), STAGES)
            ax.set_xlabel("Predicted stage")
            ax.set_ylabel("True stage")
            ax.set_title("EEG + HRV: held-out subjects")
            for i in range(5):
                for j in range(5):
                    ax.text(
                        j, i, str(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black",
                    )
            fig.tight_layout()
            fig.savefig(results / "fusion_confusion_matrix.png", dpi=160)
            plt.close(fig)
    predictions.to_csv(results / "predictions.csv", index=False)
    (results / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    final_model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=120,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    )
    final_columns = feature_sets["fusion"]
    final_model.fit(data[final_columns].to_numpy(dtype=float), y)
    joblib.dump({"model": final_model, "features": final_columns, "stages": STAGES}, results / "fusion_model.joblib", compress=3)
    report = [
        "# EEG + HRV 睡眠分期结果",
        "",
        f"数据：MIT-BIH Polysomnographic Database；{n_subjects} 位受试者，{len(data)} 个有效 30 秒窗口。",
        "R&K 第 3、4 期合并为 N3；五折按受试者分组，每个窗口只由未见过该受试者的模型预测。",
        "",
        "| 输入 | Accuracy | Macro-F1 | N1 F1 |",
        "|---|---:|---:|---:|",
    ]
    for name in feature_sets:
        score = summary["models"][name]
        report.append(f"| {name.upper()} | {score['accuracy']:.4f} | {score['macro_f1']:.4f} | {score['f1_by_stage']['N1']:.4f} |")
    report += [
        "",
        "这些数值是特征融合随机森林基线的交叉验证结果，不代表深度学习模型或临床性能。",
        "完整五类 F1 和混淆矩阵见 `metrics.json`；逐窗口预测见 `predictions.csv`。",
        "`fusion_model.joblib` 是交叉验证完成后用全部数据重新训练的模型；对训练数据的预测不能用作测试指标。",
        "",
    ]
    (results / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Saved evaluation to {results}")


def predict(cache: Path, model_path: Path, output: Path) -> None:
    artifact = joblib.load(model_path)
    data = pd.read_csv(cache)
    missing = sorted(set(artifact["features"]) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}")
    model = artifact["model"]
    x = data[artifact["features"]].to_numpy(dtype=float)
    probabilities = model.predict_proba(x)
    result = data[[c for c in ("record", "subject", "epoch") if c in data]].copy()
    result["predicted_stage"] = [artifact["stages"][int(i)] for i in model.predict(x)]
    for class_index, probability in zip(model.classes_, probabilities.T):
        result[f"prob_{artifact['stages'][int(class_index)]}"] = probability
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Saved {len(result)} predictions to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        p = sub.add_parser(name)
        p.add_argument("--data-dir", type=Path, default=Path("data"))
        p.add_argument("--cache", type=Path, default=Path("cache/features.csv"))
        p.add_argument("--records", nargs="*", default=[])
        if name == "run":
            p.add_argument("--results", type=Path, default=Path("results"))
            p.add_argument("--folds", type=int, default=5)
    p = sub.add_parser("evaluate")
    p.add_argument("--cache", type=Path, default=Path("cache/features.csv"))
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--folds", type=int, default=5)
    p = sub.add_parser("predict")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path("results/fusion_model.joblib"))
    p.add_argument("--output", type=Path, default=Path("results/new_predictions.csv"))
    args = parser.parse_args()
    if args.command in ("prepare", "run"):
        prepare(args.data_dir, args.cache, args.records)
    if args.command in ("evaluate", "run"):
        evaluate(args.cache, args.results, args.folds)
    if args.command == "predict":
        predict(args.cache, args.model, args.output)


if __name__ == "__main__":
    main()

