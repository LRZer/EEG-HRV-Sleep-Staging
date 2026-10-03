# EEG–HRV 五类睡眠分期

[English](README.md) · [简体中文](README.zh-CN.md) · [数据与版本发布](https://github.com/LRZer/EEG-HRV-Sleep-Staging/releases/tag/v2.0.0) · [实验方法](docs/METHODS_V2.md)

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
| A0 | EEG，5 窗口 | 67.05 ± 0.24% | 0.5991 ± 0.0029 | 0.5740 | 0.2521 |
| A1 | EEG，15 窗口 | 67.22 ± 1.82% | 0.6171 ± 0.0221 | 0.5840 | 0.3224 |
| A2 | 仅 HRV，5 窗口 | 36.36 ± 2.40% | 0.3002 ± 0.0311 | 0.1123 | 0.2376 |
| A3 | 拼接融合，5 窗口 | 64.00 ± 2.14% | 0.5789 ± 0.0348 | 0.4185 | 0.2676 |
| A4 | 门控融合，5 窗口 | 63.78 ± 1.47% | 0.5900 ± 0.0188 | 0.4263 | 0.3688 |
| A5 | 门控＋模态丢弃，5 窗口 | 64.44 ± 0.81% | 0.5866 ± 0.0099 | 0.4418 | 0.3198 |
| A6 | 门控＋模态丢弃，15 窗口 | 63.96 ± 0.68% | 0.5834 ± 0.0038 | 0.4136 | 0.3081 |

平均 Accuracy 最高为 **A1，67.22%**；平均 Macro-F1 最高为 **A1，0.6171**。完整模型 A6 相对脑电对照 A0：Accuracy **-3.09 个百分点**，Macro-F1 **-0.0157**。

本轮所有融合组 A3–A6 的平均 Accuracy 及 Macro-F1 均低于脑电对照 A0，未观察到增加 HRV 的性能收益。

![七组模型与随机种子波动](results/v2/figures/ablation_results.png)

每组另行报告三个种子的概率平均集成，不按测试成绩挑选种子。仅脑电 A1 集成 Accuracy 为 **70.22%**，Macro-F1 为 **0.6407**；完整 A6 集成为 **65.10%／0.5862**。[完整报告](results/v2/report.zh-CN.md) · [图表目录](docs/FIGURES.md) · [机器可读指标](results/v2/metrics.json)。

## 数据与运行

[Release v2.0.0](https://github.com/LRZer/EEG-HRV-Sleep-Staging/releases/tag/v2.0.0) 提供**全部实验输入**及注明来源的 MIT 原始文件，分块、完整包和解压文件均有 SHA-256 校验。ISRUC 原始 REC 从[作者公开来源](https://sleeptight.isr.uc.pt/?page_id=48)下载，仓库保存下载工具及精确文件校验清单。[数据说明](datasets/README.md)。

```powershell
conda activate pytorch
python scripts/download_release_data.py --kind prepared
python -m experiments.v2.predict --variant A6 --seed ensemble
```

环境安装、从原始信号重建及全部训练步骤见[复现说明](docs/REPRODUCE_V2.md)。预测接口目前使用本项目对齐缓存。训练设备为 RTX 4060 Laptop，PyTorch 2.6.0+cu118。`results/v2/` 保存 21 份权重、训练日志、预处理统计、各类／各人／各库指标及正常和缺失 HRV 的全部预测。

## 结果范围与来源

一个划分、4 位测试受试者，复用历史留出集；种子标准差不是人群置信区间。MIT 测试数据 N3 仅有 7 个窗口。训练含两个数据库，本轮不属于未见数据库的外部验证。未评估实时 ECG 检测及任意未标注文件部署。网络为独立轻量实现，不是原论文完整复现。

[MIT-BIH](https://physionet.org/content/slpdb/1.0.0/) · [ISRUC](https://sleeptight.isr.uc.pt/) · [SleepTransformer](https://github.com/pquochuy/SleepTransformer) · [GMU](https://arxiv.org/abs/1702.01992) · [ModDrop](https://arxiv.org/abs/1501.00102)。软件使用 MIT 许可，数据保留[原始权利及使用条件](datasets/DATA_NOTICE.txt)。[第一轮结果](results/deep_learning/report.md)及[早期随机森林基线](docs/BASELINE.md)保留存档，评估设置与本轮不同。
