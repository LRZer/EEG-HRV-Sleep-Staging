"""One common held-out evaluation after all predefined training has finished."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

from .dataset import DeviceData, apply_preprocessing, load_cache
from .models import build_model

PACKAGE = Path(__file__).resolve().parent
PROJECT = PACKAGE.parent
STAGES = ["Wake", "N1", "N2", "N3", "REM"]
DISPLAY = {"random_forest": "Random Forest", "eeg_attention": "EEG CNN + Attention",
           "sleep_transformer": "Hierarchical Sleep Transformer", "eeg_sequence": "EEG CNN + Temporal Transformer",
           "fusion_concat": "EEG-HRV Concat Transformer", "fusion_cross": "EEG-HRV Cross-Attention Transformer"}
SHORT = {"random_forest": "RF", "eeg_attention": "EEG-Attn", "sleep_transformer": "Sleep-TF",
         "eeg_sequence": "EEG-Seq", "fusion_concat": "Concat", "fusion_cross": "Cross-Attn"}


def score(actual, predicted):
    precision, recall, f1, support = precision_recall_fscore_support(actual, predicted, labels=np.arange(5), zero_division=0)
    return {"accuracy": float(accuracy_score(actual, predicted)), "macro_f1": float(f1.mean()),
            "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
            "kappa": float(cohen_kappa_score(actual, predicted)),
            "per_class": {stage: {"precision": float(precision[i]), "recall": float(recall[i]),
                                  "f1": float(f1[i]), "support": int(support[i])}
                          for i, stage in enumerate(STAGES)},
            "confusion_matrix": confusion_matrix(actual, predicted, labels=np.arange(5)).tolist()}


@torch.inference_mode()
def neural_predictions(checkpoint, data, rows, batch_size=128):
    device = data.device
    model = build_model(checkpoint["kind"], checkpoint["protocol"], checkpoint["hrv_dimensions"], checkpoint["frequencies"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    probabilities = []
    for start in range(0, len(rows), batch_size):
        indices = torch.as_tensor(rows[start:start + batch_size], device=device)
        eeg, hrv, spectra, missing, _ = data.batch(indices, model.context_length)
        # FP32 scoring avoids hardware-specific half-precision differences in the report.
        output = model(eeg, hrv, spectra, missing)
        probabilities.append(output.softmax(dim=1).cpu().numpy())
    return np.concatenate(probabilities)


def plots(results, summary, predictions, model_order):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    x = np.arange(len(model_order))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for offset, metric, color, label in [(-0.18, "accuracy", "#4477AA", "Accuracy"),
                                       (0.18, "macro_f1", "#EE7733", "Macro-F1")]:
        values = [summary["models"][kind][metric] for kind in model_order]
        bars = ax.bar(x + offset, values, width=0.35, color=color, label=label)
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=8)
    ax.set_xticks(x, [SHORT[kind] for kind in model_order])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Fixed subject holdout: 22 train / 4 test, 3,009 scored epochs")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(results / "model_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, kind in zip(axes.flat, model_order):
        cm = np.asarray(summary["models"][kind]["confusion_matrix"])
        normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if normalized[i, j] > 0.6 else "black")
        ax.set_xticks(range(5), STAGES, rotation=30)
        ax.set_yticks(range(5), STAGES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Expert label")
        ax.set_title(SHORT[kind])
    fig.suptitle("Held-out confusion matrices (row-normalized color, raw counts)")
    fig.tight_layout()
    fig.savefig(results / "confusion_matrices.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for kind in model_order[1:]:
        history = pd.read_csv(results / f"{kind}_history.csv")
        axes[0].plot(history.epoch, history.train_loss, label=SHORT[kind])
        axes[1].plot(history.epoch, history.train_accuracy, label=SHORT[kind])
    axes[0].set_ylabel("Training weighted cross entropy")
    axes[1].set_ylabel("Training accuracy (with augmentation)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Training curves; checkpoints fixed at epoch 40")
    fig.tight_layout()
    fig.savefig(results / "training_curves.png", dpi=180)
    plt.close(fig)

    # Show the research fusion model, independently of which method scored highest.
    records = list(predictions.record.unique())
    fig, axes = plt.subplots(len(records), 1, figsize=(12, 2.2 * len(records)), squeeze=False)
    stage_height = np.asarray([4, 2, 1, 0, 3])
    for ax, record in zip(axes[:, 0], records):
        rows = predictions.loc[predictions.record == record]
        hours = rows.epoch.to_numpy() / 120
        # Gap separators avoid visually linking across excluded EEG/unknown-label epochs.
        truth = stage_height[rows.stage.to_numpy()].astype(float)
        predicted = stage_height[rows.fusion_cross.to_numpy()].astype(float)
        gap = np.r_[False, np.diff(rows.epoch.to_numpy()) != 1]
        truth[gap] = np.nan
        predicted[gap] = np.nan
        ax.step(hours, truth, where="post", color="#222222", lw=1.2, label="Expert")
        ax.step(hours, predicted, where="post", color="#EE7733", lw=0.8, alpha=0.8, label="Cross-attention model")
        ax.set_yticks([0, 1, 2, 3, 4], ["N3", "N2", "N1", "REM", "Wake"])
        ax.set_ylim(-0.4, 4.4)
        ax.set_title(record, loc="left", fontsize=10)
        ax.set_xlabel("Hours from recording start")
    axes[0, 0].legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle("Held-out sleep stage predictions: fusion model vs expert", y=1.0)
    fig.tight_layout()
    fig.savefig(results / "heldout_hypnograms.png", dpi=180)
    plt.close(fig)


def write_report(results, summary, table):
    audit = summary["data"]
    lines = ["# 深度学习阶段实验报告", "", "## 数据与评估设置", "",
             f"两套数据库共 {audit['subjects']} 位受试者、{audit['epochs']:,} 个有效 30 秒窗口。",
             f"固定 22 人训练（{audit['train_epochs']:,} 个窗口）、4 人测试（{audit['test_epochs']:,} 个窗口），不设验证集。",
             "测试受试者：" + "、".join(audit["test_subjects"]) + "。MIT-slp02 的 a/b 两段一起进入测试。",
             "ISRUC 使用第一位专家标注，按提供方说明去掉各记录末尾 30 个噪声窗口；原始文件保留。",
             "EEG 统一到微伏及 100 Hz，逐窗口滤波，不使用当前窗口结束之后的 EEG 样本。HRV 从 ECG 心跳间隔提取 26 项时域特征，窗口最长 5 分钟且截至当前窗口末尾。",
             "连续模型最多读取同一记录内当前及过去共 5 个连续窗口；缺失/未知阶段形成边界，禁止跨记录拼接。",
             "缺失值、HRV 标准化、频谱标准化和 EEG 幅度缩放均只根据训练集拟合。融合神经网络额外接收 26 个 HRV 缺失标记。",
             "", "## 训练规则", "", "五种神经网络都使用预先固定的参数，从头训练 40 轮；保存第 40 轮模型。",
             "AdamW、初始学习率 0.0003、余弦衰减、批量 64、Dropout 0.2、训练类别权重交叉熵及轻微数据增强。",
             "没有验证集、没有早停、没有按测试成绩选检查点。所有候选模型训练结束后才统一进行测试。",
             "CNN 注意力和分层频谱 Transformer 是参考 AttnSleep / SleepTransformer 思路的独立轻量实现，不是原论文的完整复现。",
             "", "## 测试结果", "", "| 模型 | Accuracy | Macro-F1 | N1 F1 | REM F1 | Kappa |", "|---|---:|---:|---:|---:|---:|"]
    for row in table.itertuples():
        lines.append(f"| {DISPLAY[row.model]} | {row.accuracy:.2%} | {row.macro_f1:.4f} | {row.n1_f1:.4f} | {row.rem_f1:.4f} | {row.kappa:.4f} |")
    rf = summary["models"]["random_forest"]
    spectral = summary["models"]["sleep_transformer"]
    sequential = summary["models"]["eeg_sequence"]
    lines += ["", "## 结果解读", "",
              f"分层频谱 Transformer 的整体准确率最高，为 {spectral['accuracy']:.2%}，比同划分随机森林高 {(spectral['accuracy'] - rf['accuracy']) * 100:.2f} 个百分点。",
              f"连续波形 Transformer 的 Macro-F1 最高，为 {sequential['macro_f1']:.4f}。整体准确率最高并不代表每个阶段都最好。",
              "拼接融合与 Cross-Attention 融合没有稳定提升总体指标；不能把多模态融合写成已经证明优于 EEG 单模态。",
              "", "| 模型 | 数据库 | 测试窗口 | Accuracy | Macro-F1 |", "|---|---|---:|---:|---:|"]
    for row in pd.read_csv(results / "per_dataset.csv").itertuples():
        lines.append(f"| {SHORT[row.model]} | {row.source} | {row.epochs} | {row.accuracy:.2%} | {row.macro_f1:.4f} |")
    lines += ["", "融合模型在 ISRUC 测试部分较好、在 MIT 测试部分较差，表现存在数据库差异。导联、人群、标注和 HRV 质量可能相关，目前没有控制实验确认具体原因。",
              "逐受试者 Macro-F1 固定按五类计算；没有真实样本的阶段计为 0，应结合 support 阅读，不能将逐人 Macro-F1 的简单平均等同于全部窗口的 Macro-F1。"]
    lines += ["", "测试结果仅用于报告本轮预设实验，不据此重新调参或重训。",
              "旧的 MIT-BIH 五折准确率 60.42% 与本轮数据/划分不同，不能直接作为本轮提升幅度的参照。",
              "本轮随机森林才是同一训练/测试数据上的对照。", "", "## 各受试者与各数据库", "",
              "详见 `per_subject.csv` 与 `per_dataset.csv`，避免总体平均值掩盖个体差异。",
              "训练数据包含两个数据库，所以本轮是跨受试者测试，不是未见数据库的泛化测试。", "", "## 局限", "",
              "只有 4 位测试受试者、一个固定划分和一个训练随机种子，结果属于探索性成果。",
              "神经网络在无验证集设置下使用固定训练轮数，不能根据验证泛化表现选择最佳轮数。",
              "MIT-BIH 与 ISRUC 的人群、脑电导联及评分标准不同；R&K 的 3/4 期合并并不等价于获得 AASM 原始评分。",
              "心跳检测在完整离线 ECG 上运行；HRV 统计的心跳间隔截止当前窗口末尾。本实验不验证实时采集时的检测延迟。",
              "", "## 参考方法", "",
              "- [AttnSleep 作者代码](https://github.com/emadeldeen24/AttnSleep)",
              "- [SleepTransformer 作者代码](https://github.com/pquochuy/SleepTransformer)",
              "- [ISRUC 提供方的末尾噪声窗口说明](https://sleeptight.isr.uc.pt/?page_id=76)", ""]
    (results / "report.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate(cache, results):
    if (results / "metrics.json").exists():
        raise RuntimeError("This holdout was already evaluated. Preserve its results and do not tune on them.")
    completion = json.loads((results / "training_complete.json").read_text())
    if completion["smoke_only"]:
        raise ValueError("Smoke training is not a formal experiment")
    protocol = json.loads((results / "protocol.json").read_text())
    protocol_hash = hashlib.sha256((results / "protocol.json").read_bytes()).hexdigest()
    if protocol_hash != completion["protocol_sha256"]:
        raise ValueError("Training completion/protocol mismatch")
    audit = json.loads((results / "data_audit.json").read_text())
    frame, arrays = load_cache(cache)
    rows = np.flatnonzero(frame.split.to_numpy() == "test")
    if sorted(frame.iloc[rows].subject.unique()) != sorted(audit["test_subjects"]):
        raise ValueError("Holdout roster mismatch")
    with np.load(results / "preprocessing.npz") as fitted_file:
        fitted = {name: fitted_file[name] for name in fitted_file.files}
    arrays = apply_preprocessing(arrays, fitted)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = DeviceData(arrays, device)
    predictions = frame.iloc[rows].reset_index(drop=True).copy()
    actual = arrays["labels"][rows]
    summary = {"protocol_sha256": protocol_hash, "data": {key: audit[key] for key in
               ["subjects", "epochs", "train_epochs", "test_epochs", "train_subjects", "test_subjects"]},
               "validation_subjects": [], "checkpoint_rule": protocol["checkpoint_rule"],
               "models": {}, "architecture_note": "Independent lightweight research adaptations, not exact paper reproductions"}
    model_order = ["random_forest"] + protocol["models"]
    table, per_subject, per_dataset = [], [], []
    for kind in model_order:
        artifact = results / ("random_forest.joblib" if kind == "random_forest" else f"{kind}.pt")
        if kind == "random_forest":
            saved = joblib.load(artifact)
            if saved["protocol_sha256"] != protocol_hash:
                raise ValueError("Random forest protocol mismatch")
            probabilities = saved["model"].predict_proba(arrays["rf_features"][rows])
        else:
            saved = torch.load(artifact, map_location="cpu", weights_only=False)
            if saved["protocol_sha256"] != protocol_hash or saved["epochs"] != protocol["epochs"]:
                raise ValueError("Incomplete/conflicting neural checkpoint")
            if sorted(saved["train_subjects"]) != sorted(audit["train_subjects"]):
                raise ValueError("Checkpoint training roster mismatch")
            probabilities = neural_predictions(saved, data, rows)
        if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
            raise ValueError("Invalid prediction probabilities")
        predicted = probabilities.argmax(axis=1)
        predictions[kind] = predicted
        for i, stage in enumerate(STAGES):
            predictions[f"{kind}_p_{stage}"] = probabilities[:, i]
        metrics = score(actual, predicted)
        metrics["checkpoint_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if kind != "random_forest":
            metrics["parameters"] = saved["parameters"]
        summary["models"][kind] = metrics
        table.append({"model": kind, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                      "n1_f1": metrics["per_class"]["N1"]["f1"], "rem_f1": metrics["per_class"]["REM"]["f1"],
                      "kappa": metrics["kappa"]})
        for field, destination in [("subject", per_subject), ("source", per_dataset)]:
            for value, group in predictions.groupby(field):
                scored = score(group.stage, group[kind])
                destination.append({"model": kind, field: value, "epochs": len(group),
                                    "accuracy": scored["accuracy"], "macro_f1": scored["macro_f1"],
                                    **{f"{stage}_support": scored["per_class"][stage]["support"] for stage in STAGES}})
        print(f"HOLDOUT {kind}: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}", flush=True)
    table = pd.DataFrame(table)
    table.to_csv(results / "comparison.csv", index=False)
    pd.DataFrame(per_subject).to_csv(results / "per_subject.csv", index=False)
    pd.DataFrame(per_dataset).to_csv(results / "per_dataset.csv", index=False)
    predictions.to_csv(results / "predictions.csv", index=False)
    (results / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plots(results, summary, predictions, model_order)
    write_report(results, summary, table)
    print("EVALUATION COMPLETE: fixed heldout scores, per-subject breakdown, plots and report saved", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=PROJECT / "cache" / "deep_learning")
    parser.add_argument("--results", type=Path, default=PROJECT / "results" / "deep_learning")
    args = parser.parse_args()
    evaluate(args.cache.resolve(), args.results.resolve())
