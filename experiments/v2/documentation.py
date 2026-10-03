"""Generate concise bilingual summaries and complete reports from saved metrics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .figures import LABELS,COLORS,STAGES

PROJECT=Path(__file__).resolve().parents[2]
RELEASE="https://github.com/LRZer/EEG-HRV-Sleep-Staging/releases/tag/v2.0.0"
CN={"A0":"EEG，5 窗口","A1":"EEG，15 窗口","A2":"仅 HRV，5 窗口","A3":"拼接融合，5 窗口",
    "A4":"门控融合，5 窗口","A5":"门控＋模态丢弃，5 窗口","A6":"门控＋模态丢弃，15 窗口"}
CAPTIONS={
    "architecture":("Model architecture and ablation components","模型结构与对照组件"),
    "data_overview":("Subjects, retained epochs and stage distributions","受试者、窗口数及类别分布"),
    "ablation_results":("Seven variants, three individual seeds; mean ± sample SD","七组模型三个独立种子的均值与样本标准差"),
    "training_curves":("840 training-epoch logs; shaded bands are seed SD","840 轮训练日志，阴影为种子标准差"),
    "dataset_results":("MIT and ISRUC results on the same fixed subject split","同一受试者划分下两个数据库的成绩"),
    "per_class_f1":("Five-stage F1 and true class supports","五类 F1 及真实样本数量"),
    "confusion_matrices":("All seven equal-probability ensembles; counts and row percentages","七个概率平均集成的混淆矩阵、计数及行比例"),
    "missing_hrv_robustness":("Normal inputs versus completely unavailable HRV","正常输入与 HRV 完全不可用的预设测试"),
    "subject_results":("Four subject outcomes with fixed-five-class F1","四位受试者的准确率和固定五类 F1"),
    "gate_analysis":("Effective cardiac weights by stage and dataset","按阶段和数据库展示有效心脏权重"),
    "sleep_timelines":("All held-out recordings: expert, A0 ensemble and A6 ensemble","全部测试记录的专家标注、A0 与 A6 集成时间图"),
    "ecg_review":("Fixed random training-only ECG segments and detected beats","固定随机训练 ECG 片段与检测心跳"),
    "ecg_detection":("MIT beat detection against provider annotations; 150 ms tolerance","MIT 心跳检测与官方逐拍标注对比，容差 150 ms")}


def write(path,text):
    path.write_text(text.strip()+"\n",encoding="utf-8")


def generate(results):
    comparison=pd.read_csv(results/"comparison.csv")
    normal=comparison[comparison.condition=="normal"].set_index("variant")
    missing=comparison[comparison.condition=="hrv_missing"].set_index("variant")
    perclass=pd.read_csv(results/"per_class.csv",dtype={"seed":str})
    stage_mean=perclass[(perclass.condition=="normal")&(perclass.seed!="ensemble")].pivot_table(index="variant",columns="stage",values="f1")
    runs=pd.read_csv(results/"run_metrics.csv",dtype={"seed":str})
    ensembles=runs[(runs.condition=="normal")&(runs.seed=="ensemble")].set_index("variant")
    domains=pd.read_csv(results/"per_dataset.csv",dtype={"seed":str})
    domain_mean=domains[(domains.condition=="normal")&(domains.seed!="ensemble")].groupby(["variant","source"])[["accuracy","macro_f1"]].agg(["mean","std"])
    quality=pd.read_csv(PROJECT/"cache"/"v2"/"ecg_detector_audit.csv")
    quality.to_csv(results/"ecg_detector_audit.csv",index=False)
    beat=quality.dropna(subset=["beat_f1"])
    precision=beat.matched_beats.sum()/beat.detected_beats.sum()
    recall=beat.matched_beats.sum()/beat.reference_beats.sum()
    detector_f1=2*precision*recall/(precision+recall)
    best_acc=normal.accuracy_mean.idxmax()
    best_f1=normal.macro_f1_mean.idxmax()
    dacc=(normal.loc["A6","accuracy_mean"]-normal.loc["A0","accuracy_mean"])*100
    df1=normal.loc["A6","macro_f1_mean"]-normal.loc["A0","macro_f1_mean"]
    below=all(normal.loc[v,"accuracy_mean"]<normal.loc["A0","accuracy_mean"] and
              normal.loc[v,"macro_f1_mean"]<normal.loc["A0","macro_f1_mean"] for v in ["A3","A4","A5","A6"])
    conclusion_en=("All multimodal A3–A6 variants scored below EEG control A0 in both mean Accuracy and Macro-F1; this protocol did not show a performance gain from adding HRV." if below else
                   "The complete report lists all predeclared comparisons with the EEG control.")
    conclusion_cn=("本轮所有融合组 A3–A6 的平均 Accuracy 及 Macro-F1 均低于脑电对照 A0，未观察到增加 HRV 的性能收益。" if below else
                   "完整报告列出全部预定组与脑电对照的差值。")
    en_rows=[]
    cn_rows=[]
    for v in COLORS:
        r=normal.loc[v]
        values=f"{r.accuracy_mean*100:.2f} ± {r.accuracy_std*100:.2f}% | {r.macro_f1_mean:.4f} ± {r.macro_f1_std:.4f} | {stage_mean.loc[v,'N1']:.4f} | {stage_mean.loc[v,'REM']:.4f}"
        en_rows.append(f"| {v} | {LABELS[v]} | {values} |")
        cn_rows.append(f"| {v} | {CN[v]} | {values} |")
    en_table="\n".join(en_rows)
    cn_table="\n".join(cn_rows)
    write(PROJECT/"README.md",f"""
# EEG–HRV Sleep Staging

[English](README.md) · [简体中文](README.zh-CN.md) · [Data & release]({RELEASE}) · [Methods](docs/METHODS_V2.md)

Classify each **30-second** sleep segment as **Wake, N1, N2, N3 or REM** using scalp electrical activity (**EEG**) and heartbeat-interval variability (**HRV**) derived from **ECG**. The repository contains data preparation, controlled Transformer fusion experiments, trained weights, predictions and reproducible figures.

## Study

| Data | Subjects | Retained epochs | Training | Test |
|---|---:|---:|---:|---:|
| MIT-BIH PSG + ISRUC-Sleep III | 26 | 18,770 | 22 subjects / 15,761 epochs | 4 subjects / 3,009 epochs |

No validation set. Seven variants × three seeds, **21 runs / 840 training epochs**. Parameters and final-epoch checkpoints were fixed before scoring. The test subjects were already evaluated in version 1: these are **exploratory reused-holdout results**, not new external validation.

![Data and stage distributions](results/v2/figures/data_overview.png)

## Methods

The experiments compare brain signals alone, heartbeat features alone, and their combination. They also compare 2.5-minute versus 7.5-minute context and test what happens when heartbeat features are unavailable.

- Single-channel EEG: independent 0.3–35 Hz filtering, 100 Hz sampling, 29×89 log-power spectra per epoch.
- Cardiac branch: 26 time-domain HRV features, 26 missing flags and 3 coverage/plausibility indicators; statistics end at the current epoch.
- Shared spectral EEG encoder, temporal Transformer, concatenation or quality-constrained residual gating; context contains **5 or 15 epochs in total**, including the current epoch and available past epochs. A5/A6 use 20% training HRV dropout.
- Train-only filling/scaling; weighted cross entropy, AdamW and fixed 40-epoch cosine schedule. Complete methods and boundaries: [methods](docs/METHODS_V2.md).

![Full model and controlled components](results/v2/figures/architecture.png)

## Results

Mean ± **sample SD across three individual seeds** on the same test subjects. N1/REM F1 are seed means. Accuracy is the fraction of correct epochs; Macro-F1 weights all five stages equally.

| ID | Variant | Accuracy | Macro-F1 | N1 F1 | REM F1 |
|---|---|---:|---:|---:|---:|
{en_table}

Highest mean Accuracy: **{best_acc}, {normal.loc[best_acc,'accuracy_mean']:.2%}**. Highest mean Macro-F1: **{best_f1}, {normal.loc[best_f1,'macro_f1_mean']:.4f}**. Full A6 versus EEG control A0: **{dacc:+.2f} percentage points** Accuracy and **{df1:+.4f}** Macro-F1.

{conclusion_en}

![Ablation results with individual seed scores](results/v2/figures/ablation_results.png)

All three seeds are additionally averaged as a probability ensemble for **each** variant; no seed is selected by test score. EEG-only A1 ensemble: Accuracy **{ensembles.loc['A1','accuracy']:.2%}**, Macro-F1 **{ensembles.loc['A1','macro_f1']:.4f}**. Full A6 ensemble: Accuracy **{ensembles.loc['A6','accuracy']:.2%}**, Macro-F1 **{ensembles.loc['A6','macro_f1']:.4f}**. [Complete results](results/v2/report.en.md) · [All figures](docs/FIGURES.md) · [Machine-readable metrics](results/v2/metrics.json).

## Data and running

[Release v2.0.0]({RELEASE}) provides the prepared **complete study inputs** and attributed MIT original files in checksum-verified parts. ISRUC original REC files are downloaded from the [provider](https://sleeptight.isr.uc.pt/?page_id=48); the download code and exact hashes are included. [Data guide](datasets/README.md).

```powershell
conda activate pytorch
python scripts/download_release_data.py --kind prepared
python -m experiments.v2.predict --variant A6 --seed ensemble
```

For installation, raw-signal reconstruction and all 21 runs: [reproduction guide](docs/REPRODUCE_V2.md). Prediction currently accepts the aligned study cache. GPU used: RTX 4060 Laptop, PyTorch 2.6.0+cu118. Histories, 21 checkpoints, preprocessing, per-class/person/dataset metrics and all normal/missing-HRV predictions are under `results/v2/`.

## Scope and attribution

One subject split, four test people, reused holdout. Seed SD is not a population confidence interval. MIT test includes only 7 N3 epochs. Both databases occur in training; this is not unseen-dataset validation. Offline ECG detection and arbitrary unannotated-file deployment are not evaluated. These networks are independent lightweight adaptations, not full reproductions of published models.

[MIT-BIH](https://physionet.org/content/slpdb/1.0.0/) · [ISRUC](https://sleeptight.isr.uc.pt/) · [SleepTransformer](https://github.com/pquochuy/SleepTransformer) · [GMU](https://arxiv.org/abs/1702.01992) · [ModDrop](https://arxiv.org/abs/1501.00102). MIT license covers project software; datasets retain their [original conditions](datasets/DATA_NOTICE.txt). [Version 1](results/deep_learning/report.md) and [historical RF baseline](docs/BASELINE.md) remain archived with different evaluation settings.
""")
    write(PROJECT/"README.zh-CN.md",f"""
# EEG–HRV 五类睡眠分期

[English](README.md) · [简体中文](README.zh-CN.md) · [数据与版本发布]({RELEASE}) · [实验方法](docs/METHODS_V2.md)

根据**脑电 EEG**及由**心电 ECG**计算的**心率变异性 HRV**，识别每个 **30 秒**片段的 **Wake、N1、N2、N3、REM**：分别为清醒、非快速眼动第 1/2/3 期和快速眼动睡眠。仓库包含数据准备、Transformer 融合对照实验、训练权重、逐窗预测及图表生成代码。

EEG 记录头皮电活动；ECG 用于定位心跳；HRV 描述相邻心跳间隔的变化。

## 实验数据与规模

| 数据 | 受试者 | 有效窗口 | 训练集 | 测试集 |
|---|---:|---:|---:|---:|
| MIT-BIH PSG＋ISRUC-Sleep III | 26 人 | 18,770 | 22 人／15,761 窗口 | 4 人／3,009 窗口 |

不设验证集。七组模型各用三个随机种子，共 **21 次训练、840 轮**。参数和第 40 轮检查点规则在评分前固定。这 4 位测试受试者在第一轮已评估，本轮属于**复用历史留出集的探索性结果**。

![数据规模与阶段分布](results/v2/figures/data_overview.png)

## 已完成的方法

实验比较只用脑电、只用心跳特征及同时使用两者，再检查 2.5／7.5 分钟上下文、融合方式和心跳特征缺失对成绩的影响。

- 单通道 EEG 逐窗独立进行 0.3–35 Hz 滤波和 100 Hz 重采样，构造 29×89 对数功率频谱。
- 心脏分支输入 26 项时域 HRV、26 个缺失标记及 3 个覆盖／合理性指示量；HRV 统计截至当前窗口末尾。
- 使用统一频谱 EEG 编码器，比较时序 Transformer、拼接和质量约束的残差门控融合；上下文为当前及过去共 **5 或 15 个窗口**，A5/A6 训练时以 20% 概率丢弃 HRV。
- 填补和标准化仅由训练集拟合；采用加权交叉熵、AdamW 和固定 40 轮余弦衰减。边界及参数见[详细方法](docs/METHODS_V2.md)。

![模型结构与对照组件](results/v2/figures/architecture.png)

## 指标成绩

以下为同一测试集上三个单次模型的**均值±样本标准差**，N1/REM F1 为种子均值。Accuracy 表示预测正确的窗口比例；Macro-F1 对五类阶段等权平均。

| 编号 | 模型 | Accuracy | Macro-F1 | N1 F1 | REM F1 |
|---|---|---:|---:|---:|---:|
{cn_table}

平均 Accuracy 最高为 **{best_acc}，{normal.loc[best_acc,'accuracy_mean']:.2%}**；平均 Macro-F1 最高为 **{best_f1}，{normal.loc[best_f1,'macro_f1_mean']:.4f}**。完整模型 A6 相对脑电对照 A0：Accuracy **{dacc:+.2f} 个百分点**，Macro-F1 **{df1:+.4f}**。

{conclusion_cn}

![七组模型与随机种子波动](results/v2/figures/ablation_results.png)

每组另行报告三个种子的概率平均集成，不按测试成绩挑选种子。仅脑电 A1 集成 Accuracy 为 **{ensembles.loc['A1','accuracy']:.2%}**，Macro-F1 为 **{ensembles.loc['A1','macro_f1']:.4f}**；完整 A6 集成为 **{ensembles.loc['A6','accuracy']:.2%}／{ensembles.loc['A6','macro_f1']:.4f}**。[完整报告](results/v2/report.zh-CN.md) · [图表目录](docs/FIGURES.md) · [机器可读指标](results/v2/metrics.json)。

## 数据与运行

[Release v2.0.0]({RELEASE}) 提供**全部实验输入**及注明来源的 MIT 原始文件，分块、完整包和解压文件均有 SHA-256 校验。ISRUC 原始 REC 从[作者公开来源](https://sleeptight.isr.uc.pt/?page_id=48)下载，仓库保存下载工具及精确文件校验清单。[数据说明](datasets/README.md)。

```powershell
conda activate pytorch
python scripts/download_release_data.py --kind prepared
python -m experiments.v2.predict --variant A6 --seed ensemble
```

环境安装、从原始信号重建及全部训练步骤见[复现说明](docs/REPRODUCE_V2.md)。预测接口目前使用本项目对齐缓存。训练设备为 RTX 4060 Laptop，PyTorch 2.6.0+cu118。`results/v2/` 保存 21 份权重、训练日志、预处理统计、各类／各人／各库指标及正常和缺失 HRV 的全部预测。

## 结果范围与来源

一个划分、4 位测试受试者，复用历史留出集；种子标准差不是人群置信区间。MIT 测试数据 N3 仅有 7 个窗口。训练含两个数据库，本轮不属于未见数据库的外部验证。未评估实时 ECG 检测及任意未标注文件部署。网络为独立轻量实现，不是原论文完整复现。

[MIT-BIH](https://physionet.org/content/slpdb/1.0.0/) · [ISRUC](https://sleeptight.isr.uc.pt/) · [SleepTransformer](https://github.com/pquochuy/SleepTransformer) · [GMU](https://arxiv.org/abs/1702.01992) · [ModDrop](https://arxiv.org/abs/1501.00102)。软件使用 MIT 许可，数据保留[原始权利及使用条件](datasets/DATA_NOTICE.txt)。[第一轮结果](results/deep_learning/report.md)及[早期随机森林基线](docs/BASELINE.md)保留存档，评估设置与本轮不同。
""")
