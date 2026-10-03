# EEG–HRV 五类睡眠分期

[English](README.md) · **简体中文** · [数据下载](https://github.com/LRZer/EEG-HRV-Sleep-Staging/releases/tag/v2.0.0) · [图表目录](docs/FIGURES.md)

用脑电和心跳间隔特征，预测每个 **30 秒**片段属于哪个睡眠阶段。本项目已完成两套公开数据的处理、七组深度学习对照实验、三随机种子训练及结果归档，提供代码、21 份模型权重、实验输入、逐窗预测和可重新生成的图表。

| 信号／输出 | 含义 | 在项目中的作用 |
|---|---|---|
| EEG（脑电） | 头皮电极记录的电活动 | 转为时频图，学习每个片段的脑电模式 |
| ECG（心电） | 心脏电活动波形 | 检测心跳位置，计算相邻心跳间隔 RR |
| HRV（心率变异性） | 心跳间隔的变化及其统计特征 | 作为辅助输入；不是另一条原始采集信号 |
| Wake／N1／N2／N3／REM | 清醒、非快速眼动第 1／2／3 期、快速眼动睡眠 | 每个片段输出五类概率及概率最高的类别 |

## 1. 已完成的工作与主要结果

- **数据**：MIT-BIH PSG 与 ISRUC-Sleep III，共 26 人、28 条记录、18,770 个有效窗口；22 人训练、4 人测试，**不设验证集**。
- **方法**：统一频谱 EEG 编码器＋时序 Transformer；比较仅 EEG、仅 HRV、拼接融合、质量约束残差门控、模态丢弃和上下文长度。
- **实验**：7 组×3 个种子，每次固定 40 轮，共 21 次训练、840 轮；同时评估正常输入与 HRV 全缺失。
- **成绩**：单次模型均值最高为 EEG-only A1，Accuracy **{{BEST_ACC}}**、Macro-F1 **{{BEST_F1}}**；A1 三模型概率平均集成为 **{{A1_ENS_ACC}}／{{A1_ENS_F1}}**。
- **融合结果**：本轮 A3–A6 的平均 Accuracy 和 Macro-F1 均低于 EEG 对照 A0；完整 A6 集成为 **{{A6_ENS_ACC}}／{{A6_ENS_F1}}**，未观察到增加 HRV 的整体性能收益。

这 4 位测试受试者在第一轮已经评估，本轮属于**复用历史留出集的探索性实验**。以上成绩描述本项目这一次受试者划分，不作为新外部验证或临床诊断结论。

## 2. 数据、标签与输入如何对应

| 数据库 | 受试者／记录 | 保留窗口 | 训练窗口 | 测试窗口 | 使用通道与标注 |
|---|---:|---:|---:|---:|---|
| MIT-BIH PSG 1.0.0 | 16／18 | 10,181 | 8,850 | 1,331 | 单通道 EEG（导联随记录变化）、ECG、原始睡眠标注 |
| ISRUC-Sleep III | 10／10 | 8,589 | 6,911 | 1,678 | C4-A1、X2 心电、第一位专家标注 |
| 合计 | 26／28 | 18,770 | 15,761 | 3,009 | 按同一人的全部记录划分 |

固定测试名单：`MIT-slp02`、`MIT-slp60`、`ISRUC-III-04`、`ISRUC-III-05`，其中 slp02a/b 属于同一人。未知标签不参与训练，并打断连续上下文。MIT 旧评分的 3／4 期合并为 N3；标签名称映射不消除两个数据库评分标准的差异。ISRUC 每条记录末尾 30 个窗口按提供方的预提取通道噪声说明统一省略，原始文件保留。

**一条预测的输入**：当前及过去共 5 或 15 个 EEG 窗口；每个窗口附带截至该窗口末尾的过去最多 300 秒 HRV 特征。标签对应最后一个、也就是当前窗口。填补和标准化只用训练数据拟合。

![数据窗口、HRV 历史与边界](docs/figures/time_alignment.zh-CN.png)

读图：5／15 窗口对应 EEG 的 2.5／7.5 分钟；叠加各窗口的 HRV 统计历史后，完整历史下的 RR 终点覆盖范围约为 **7／12 分钟**。不跨记录或标签缺口，不足时左侧填充并屏蔽。本轮 ECG 检测在完整离线记录上运行，尚未验证实时检测。[完整数据流程](docs/DATA_PIPELINE.zh-CN.md)

HRV 库的 pNN 分母还依赖整记录填充宽度，可能受较晚窗口影响；局部 RR 终点边界不代表整个特征流程严格因果。具体公式和离线依赖见[特征实现说明](docs/FEATURES.zh-CN.md)。

## 3. 当前模型架构

第二轮主模型采用 **频谱 Transformer＋残差门控融合＋时序 Transformer**。第一轮的 1D-CNN 和 Cross-Attention 保留在历史实验中，当前 A0–A6 不使用 Cross-Attention。

![EEG 对照与完整融合架构](docs/figures/model_architecture.zh-CN.png)

| 模块 | 实际操作与尺寸 | 解决的问题 |
|---|---|---|
| EEG 预处理 | 每 30 秒：0.3–35 Hz 滤波、100 Hz 重采样，3,000 点；STFT→29×89 对数功率频谱 | 保留窗口内的时间和频率变化 |
| 窗口内编码器 | 89→96 投影；位置编码；2 层、4 头 Transformer；29 帧均值池化→96 维 | 将一个 EEG 窗口编码成一个向量 |
| 心脏编码器 | 26 特征＋26 缺失标记＋3 覆盖／合理性量＝55 维；MLP 55→96→96 | 编码 HRV 及其可用性 |
| 残差门控 | `z = EEG + g × delta(HRV)`；`g` 为每窗口一个标量，并受有效 RR 与有限特征比例约束 | 控制心脏分支对 EEG 表示的修正幅度 |
| 窗口间编码器 | 5／15 个 96 维向量＋位置编码；2 层、4 头 Transformer | 学习当前及过去片段之间的关系 |
| 分类头 | 读取当前窗口向量，LayerNorm＋Dropout＋Linear→5；softmax 得到概率 | 预测当前窗口的五类睡眠阶段 |

两个 Transformer 分别处理“**一个片段内部**”和“**多个片段之间**”。输入没有目标窗口之后的 EEG／RR 终点；时序注意力在已提供的历史序列内不使用三角因果掩码。门控值是模型内部权重，不等于某种信号的临床重要性。[层级结构、门控公式与文献对应](docs/ARCHITECTURE.zh-CN.md) · [26 项 HRV 特征字典](docs/FEATURES.zh-CN.md)

## 4. 七组实验与训练设置

| 编号 | 输入与融合 | 上下文窗口 | 训练 HRV 丢弃 | 对照目的 |
|---|---|---:|---:|---|
| A0 | 仅 EEG | 5 | — | 短上下文 EEG 对照 |
| A1 | 仅 EEG | 15 | — | A1−A0：延长 EEG 上下文 |
| A2 | 仅 HRV | 5 | 0 | 检查心脏特征独立预测能力 |
| A3 | EEG＋HRV，拼接 | 5 | 0 | A3−A0：加入 HRV |
| A4 | EEG＋HRV，质量约束残差门控 | 5 | 0 | A4−A3：更换整套融合结构 |
| A5 | 同 A4 | 5 | 20% | A5−A4：模态丢弃 |
| A6 | 同 A5 | 15 | 20% | A6−A5：延长融合上下文 |

A4−A3 同时改变拼接、残差路径和质量约束，不能单独证明质量约束的贡献。A5／A6 按样本以 20% 概率移除整段 HRV 上下文；全缺失时有效门控为 0，保留该融合模型的 EEG 路径，不等于独立训练的 A0／A1。

统一设置：随机种子 **42／123／2026**，固定 **40 轮最后一轮权重**，batch 64，AdamW（学习率 3×10⁻⁴、权重衰减 0.01），Dropout 0.2，余弦衰减至初始学习率的 5%，梯度裁剪 1.0。使用训练集类别频数计算加权交叉熵；标准化频谱加入标准差 0.02 的训练噪声；GPU 训练 AMP，评估 FP32。未按测试成绩选择检查点或种子。[冻结协议](experiments/v2/protocol.json) · [完整方法](docs/METHODS_V2.md)

## 5. 指标与结果解读

下表为三个单次模型的**均值±样本标准差**；N1／REM F1 为种子均值。Accuracy 是正确窗口比例；Macro-F1 对五类 F1 等权平均，是本轮主指标；种子标准差描述初始化波动，不是人群置信区间。

| 编号 | 模型 | Accuracy | Macro-F1 | N1 F1 | REM F1 |
|---|---|---:|---:|---:|---:|
{{RESULT_ROWS}}

![七组指标与种子波动](results/v2/figures/ablation_results.png)

读图：点为三个独立种子，汇总条形和误差线表示均值与样本标准差。**集成**另以三个种子的五类概率平均后取最大值，各组均报告，不能与单次成绩均值混用。完整 A6 相对 A0 的平均 Accuracy 为 **{{DELTA_ACC}} 个百分点**、Macro-F1 为 **{{DELTA_F1}}**。

阶段变化方向并不一致：A4 的 REM F1 为 0.3688（A0 为 0.2521），但 N1 F1 为 0.4263（A0 为 0.5740）。HRV 完全缺失时，A5 相对 A4 的平均 Accuracy 增加 6.85 个百分点、Macro-F1 增加 0.0739；这是本轮压力测试的观察结果。

[结果分析与固定案例](docs/RESULTS_ANALYSIS.zh-CN.md)解释混淆矩阵、分人／分库指标、缺失模态和门控值；[完整报告](results/v2/report.zh-CN.md)提供全部数值；[机器可读指标](results/v2/metrics.json)和[逐窗预测](results/v2/predictions.csv.gz)保留原始输出。

## 6. 下载、推理与复现

[Release v2.0.0](https://github.com/LRZer/EEG-HRV-Sleep-Staging/releases/tag/v2.0.0) 提供 **18,770 窗口的全部对齐实验输入**（368.1 MiB 压缩包）及注明来源的 MIT 原始文件（554.8 MiB）。大文件拆为 48 MiB 分块，下载工具自动拼接并核对分块、完整包和解压文件的 SHA-256。ISRUC 原始 REC 从作者来源下载，仓库提供工具和精确文件校验清单。[数据说明](datasets/README.md)

```powershell
git clone https://github.com/LRZer/EEG-HRV-Sleep-Staging.git
cd EEG-HRV-Sleep-Staging
conda activate pytorch
python -m pip install -r requirements-deep-learning.txt
python scripts/download_release_data.py --kind prepared
# EEG-only A1：本轮平均主指标最高的组
python -m experiments.v2.predict --variant A1 --seed ensemble --output results/A1_restored.csv
# 完整 EEG-HRV 融合模型
python -m experiments.v2.predict --variant A6 --seed ensemble --output results/A6_restored.csv
```

预测接口恢复保存的权重与训练集预处理，输出固定测试集的阶段和概率；目前使用本项目对齐缓存。实际训练环境为 Python 3.11.9、PyTorch 2.6.0+cu118、RTX 4060 Laptop 8 GB。[环境、原始信号重建、21 次训练和结果校验步骤](docs/REPRODUCE_V2.md)

## 7. 仓库导航与版本关系

| 路径 | 内容 |
|---|---|
| `experiments/v2/` | 当前训练、评估、模型、门控检查、预测及绘图 |
| `deep_learning/` | 共享数据处理和频谱编码器，及第一轮深度学习实现 |
| `results/v2/` | 21 份权重、840 轮日志、预处理统计、56 组预测、分阶段／人／库指标和 13 组结果图 |
| `docs/` | 中英文技术说明、复现步骤、图表目录、文档图及生成模板 |
| `datasets/`、`scripts/` | 数据来源、校验清单、下载和检查工具 |
| `main.py` | 早期 MIT-only 随机森林基线 |

研究顺序为[随机森林基线](docs/BASELINE.md)→[第一轮 1D-CNN／Transformer／Cross-Attention](results/deep_learning/report.md)→当前统一骨干的七组对照。数据及评估设置不同，不将历史成绩与本轮直接相减作为改进幅度。[全部文档](PROJECT.md)

## 8. 结果范围、来源与许可

4 位测试受试者、一个划分且复用留出集；两库都参与训练，分库指标不代表未见数据库泛化。MIT 测试 N3 只有 7 个窗口；逐人 Macro-F1 固定按五类计算，缺失类 F1 计 0。未评估任意未标注 EDF 的部署、实时 ECG 检测或临床有效性。HRV 库中的分母、单位和数值稳定性细节见[特征实现说明](docs/FEATURES.zh-CN.md)，解释当前结果时需保留这些条件。

方法为独立轻量实现，借鉴 [SleepTransformer](https://github.com/pquochuy/SleepTransformer)、[GMU](https://arxiv.org/abs/1702.01992) 的门控思想和 [ModDrop](https://arxiv.org/abs/1501.00102) 的模态丢弃思想，未完整复现上述论文，也未据此主张方法创新。数据来自 [MIT-BIH](https://physionet.org/content/slpdb/1.0.0/) 与 [ISRUC-Sleep](https://sleeptight.isr.uc.pt/)。项目软件采用 [MIT 许可](LICENSE)，数据和第三方依赖保留[原始权利及使用条件](datasets/DATA_NOTICE.txt)。
