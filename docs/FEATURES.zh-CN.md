# HRV 特征字典与实现细节

[English](FEATURES.en.md) · [README](../README.zh-CN.md) · [数据流程](DATA_PIPELINE.zh-CN.md)

## 1. 定义与计算版本

以下顺序与缓存中 26 列一致，按实际使用的 **SleepECG 0.5.9** 实现说明。[对应版本源码](https://github.com/cbrnr/sleepecg/blob/v0.5.9/src/sleepecg/feature_extraction.py)；本项目调用位置见 [prepare.py](../deep_learning/prepare.py)。库的 `hrv-time` 组同时包含 RR 统计、心率及 Poincaré 几何量，当前未计算 HRV 频域 LF／HF。

令 `NN` 为按时间保留位置的 RR 秒数序列；不合理间隔为 NaN。这里沿用库的 NN 命名，没有通过独立专家逐拍判定每次心跳是否为正常窦性心跳。`D=diff(NN)` 为相邻间隔差，NaN 会保留缺口；mean／median／quantile 忽略 NaN，`sd` 是 `ddof=1` 的样本标准差。RR 范围为 [0.3,2.0] 秒。

| 列 | 名称 | 实际计算／含义 | 单位 |
|---:|---|---|---|
| 1 | meanNN | mean(NN)，平均间隔 | s |
| 2 | maxNN | max(NN) | s |
| 3 | minNN | min(NN) | s |
| 4 | rangeNN | maxNN−minNN | s |
| 5 | SDNN | sd(NN)，间隔离散程度 | s |
| 6 | RMSSD | sqrt(mean(D²))，相邻差的均方根 | s |
| 7 | SDSD | sd(D) | s |
| 8 | NN50 | count(abs(D)>0.05) | 次 |
| 9 | NN20 | count(abs(D)>0.02) | 次 |
| 10 | pNN50 | NN50／(M−1)，M 为库在同一记录中填充后的 RR 行宽，见下文 | 比例 0–1 |
| 11 | pNN20 | NN20／(M−1)，同上 | 比例 0–1 |
| 12 | medianNN | median(NN) | s |
| 13 | madNN | median(abs(NN−medianNN)) | s |
| 14 | iqrNN | Q75(NN)−Q25(NN) | s |
| 15 | cvNN | SDNN／meanNN | 无量纲 |
| 16 | cvSD | SDSD／mean(D)，不是 RMSSD／meanNN | 无量纲，可为负 |
| 17 | meanHR | 60／meanNN，不是 mean(60／NN) | 次／分钟 |
| 18 | maxHR | 60／minNN | 次／分钟 |
| 19 | minHR | 60／maxNN | 次／分钟 |
| 20 | stdHR | sd(60／NN) | 次／分钟 |
| 21 | SD1 | SDSD／sqrt(2) | s |
| 22 | SD2 | sqrt(2×SDNN²−SD1²) | s |
| 23 | S | π×SD1×SD2，椭圆面积量 | s² |
| 24 | SD1_SD2_ratio | SD1／SD2 | 无量纲 |
| 25 | CSI | SD2／SD1 | 无量纲 |
| 26 | CVI | log10(16×SD1×SD2)，代入秒单位的数值 | 单位敏感的对数指标 |

## 2. 当前实验必须保留的实现条件

**pNN 分母**：库将同一记录不同窗口的 RR 序列填充至统一最大宽度 M，空位为 NaN。实现先比较 `abs(D)>threshold` 再取均值；NaN 比较结果为 False，因此分母包含填充及无效差位置，不仅是有效相邻差数。这会让 pNN 数值依赖该记录的最大行宽，不能直接当作常见“有效相邻差中超过阈值的百分比”。缓存中保存的是 0–1 比例，未乘 100。

M 来自完整记录的所有统计窗口，较晚窗口也可能影响这个分母，因此存在整记录的离线依赖。局部 RR 终点不超过当前末尾，并不能保证整个 HRV 特征构建严格因果；加上完整 ECG 检测，本轮不能视作实时流程验证。

例如两个窗口 RR 数分别为 3 和 5，填充后 M=5；若短窗口有 1 个有效差超过 50 ms，库给出 1／4=0.25，而按两对有效相邻差计算会得到 1／2=0.5。这个例子解释实现，不是本轮某个真实窗口的数值。

**cvSD 稳定性**：分母是相邻差的均值，它可能接近 0，所以该特征可能很大、为负或非有限。非有限值进入缺失处理，有限极值经过标准化裁剪。不能把这个名称默认解释为另一种教科书公式。

**CVI 单位**：SD1／SD2 在当前代码中以秒计。改用毫秒会使 `log10(16×SD1×SD2)` 增加 6；不能直接与采用毫秒定义的文献数值比较。CSI／CVI 是库中的指标名称，不据此做自主神经或临床诊断。

上述实现条件在本次文档整理时核对发现。当前冻结结果没有重算特征或重新训练；若以后修正分母或单位约定，需要新协议、新缓存和全部对照重跑，不能将旧成绩解释为修正后的结果。

## 3. 缺失标记与三个覆盖量

心脏输入的列序为 `[26 scaled features, 26 missing flags, 3 quality fields]`。NaN／inf 在填补前标记为缺失；每项按训练集中位数填补，再按训练均值／标准差缩放，标准差下限 1e−6，缩放值裁剪至 [−8,8]。缺失标记不标准化。

| 字段 | 精确定义 | 在模型中的作用 |
|---|---|---|
| valid_rr_fraction | 当前最多 300 秒 RR 终点窗口内，0.3–2.0 秒间隔数／检测间隔总数；无间隔时为 0 | 心脏输入；直接乘到有效门控 |
| feature_finite_fraction | 填补前有限 HRV 特征数／26 | 心脏输入；直接乘到有效门控 |
| history_fraction | min(当前窗口终点秒数,300)／300 | 心脏输入及可学习门控输入，不直接相乘 |

三个量是覆盖／合理性指示，不是经过临床验证的 ECG 质量评分，也不会识别全部运动伪迹或错误心跳。有限的特征不一定无偏；例如 pNN 的分母问题仍可得到有限值。

## 4. 重新计算与审计入口

固定特征名保存在 `cache/deep_learning/data_audit.json`；完整输入随 [Release](https://github.com/LRZer/EEG-HRV-Sleep-Staging/releases/tag/v2.0.0) 提供。RR 合理性与 ECG 检测检查见 [prepare_quality.py](../experiments/v2/prepare_quality.py) 和 [ecg_detector_audit.csv](../results/v2/ecg_detector_audit.csv)。MIT 的独立逐拍标注用于检测审计，不作为睡眠分类模型输入；本轮没有 ISRUC 独立逐拍参考。
