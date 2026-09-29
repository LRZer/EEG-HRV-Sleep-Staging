# EEG-HRV Sleep Staging

**基于脑电（EEG）与心率变异性（HRV）特征融合的五类睡眠分期项目。** 项目从同步记录的 EEG、ECG 和人工睡眠标注出发，建立可重复运行的数据处理、模型训练及按受试者评估流程。

> 当前实现是随机森林特征融合基线。它没有使用 1D-CNN、Transformer 或 Cross-Attention；这些架构属于后续可比较的改进方向。

## 已验证的结果

在 [MIT-BIH Polysomnographic Database 1.0.0](https://physionet.org/content/slpdb/1.0.0/) 的 18 条记录、16 位受试者上，共保留 10,181 个有标签的 30 秒窗口。五折交叉验证以受试者分组，同一人的记录不会同时出现在训练与测试折。

| 输入 | Accuracy | Macro-F1 | N1 F1 | REM F1 |
|---|---:|---:|---:|---:|
| EEG | 56.85% | 0.4996 | 0.3177 | 0.1618 |
| HRV | 42.70% | 0.3119 | 0.2504 | 0.1583 |
| EEG + HRV | **60.42%** | **0.5299** | **0.4249** | **0.2525** |

融合输入在本数据集上提高了整体指标，但 REM 识别仍较弱。完整每类指标、混淆矩阵和逐窗口预测见 [`results/`](results/)；研究背景、方法和限制见 [`PROJECT.md`](PROJECT.md)。

## 方法概览

1. 从公开数据库下载并校验 EEG、ECG 和睡眠阶段文件，按 30 秒切分并对齐标签。
2. 从一个 EEG 通道计算 7 项特征：δ、θ、α、σ、β 五个频段的相对功率、总功率和谱熵。
3. 使用 [SleepECG](https://github.com/cbrnr/sleepecg)检测 ECG 心跳，并从截至当前窗口末尾、最长 5 分钟的信号计算 26 项时域 HRV 特征。
4. 比较 EEG、HRV、EEG+HRV 三组输入。分类器为带类别平衡权重的随机森林，缺失值只依据训练折填补。
5. 原始 R&K 标签 `W, 1, 2, 3, 4, R` 中的第 3、4 期合并为 N3，输出 Wake、N1、N2、N3、REM 五类。

## 复现

建议使用 **Python 3.11**。Windows PowerShell 示例：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py prepare --data-dir data --cache cache/features.csv
.venv\Scripts\python.exe main.py evaluate --cache cache/features.csv --results results
```

`prepare` 会从 PhysioNet 的公开对象存储下载并校验约 632 MB 原始数据。可先用 `--records slp03 slp04 slp14` 检查处理流程；报告中的正式五折结果使用全部 18 条记录。`evaluate` 保存指标、混淆矩阵、逐窗口留出预测和一个用全部数据重新训练的融合模型。请勿用该最终模型对它的训练数据打分来替代交叉验证指标。

仓库中的训练模型按文件块存放。克隆后可恢复模型文件：

```powershell
.venv\Scripts\python.exe rebuild_model.py
```

对已处理的特征表预测：

```powershell
.venv\Scripts\python.exe main.py predict --cache cache/features.csv --model results/fusion_model.joblib --output results/new_predictions.csv
```

上面使用原训练数据只是演示命令；`results/predictions.csv` 才是逐窗口的留出预测。

## 仓库内容

| 路径 | 内容 |
|---|---|
| `main.py` | 数据下载、特征构建、训练、评估、预测 |
| `requirements.txt` | 已验证的 Python 依赖版本 |
| `PROJECT.md` | 项目详细说明与实验局限 |
| `results/metrics.json` | 三组模型的完整指标 |
| `results/predictions.csv` | 每个窗口的留出预测 |
| `results/fusion_confusion_matrix.png` | 融合模型混淆矩阵 |
| `results/report.md` | 简要实验报告 |
| `artifacts/model_parts/` | 最终训练模型的分块文件和校验清单 |
| `rebuild_model.py` | 校验并恢复模型文件 |

原始数据库文件不提交到仓库；脚本会按需下载。数据集使用 [ODC Attribution 1.0](https://physionet.org/content/slpdb/1.0.0/)；请在使用数据或衍生成果时注明数据库和原论文。SleepECG 使用 BSD-3-Clause 许可，本项目作为依赖调用，没有复制其源码。

