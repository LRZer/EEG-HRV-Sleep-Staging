# Version 2 methods / 第二轮实验方法

[English overview](../README.md) · [中文概览](../README.zh-CN.md)

Detailed companion documents / 详细配套文档：

- [Architecture / 架构](ARCHITECTURE.en.md) · [中文](ARCHITECTURE.zh-CN.md)
- [Data pipeline / 数据流程](DATA_PIPELINE.en.md) · [中文](DATA_PIPELINE.zh-CN.md)
- [HRV dictionary / 特征字典](FEATURES.en.md) · [中文](FEATURES.zh-CN.md)
- [Result interpretation / 结果解读](RESULTS_ANALYSIS.en.md) · [中文](RESULTS_ANALYSIS.zh-CN.md)

## Task and data / 任务与数据

Five-class, 30-second sleep staging from one EEG channel and ECG-derived HRV. EEG is electrical activity measured at the scalp; ECG is the cardiac electrical waveform; HRV describes variation in successive heartbeat intervals. Output classes are Wake, N1, N2, N3 and REM.

以单通道脑电和由心电计算的心率变异性识别每个 30 秒片段的 Wake、N1、N2、N3、REM。HRV 是心跳间隔特征，不是另一条原始采集信号。

| Source / 数据库 | Subjects / 人 | Recordings / 记录 | Retained epochs / 保留窗口 | Train epochs | Test epochs |
|---|---:|---:|---:|---:|---:|
| MIT-BIH PSG 1.0.0 | 16 | 18 | 10,181 | 8,850 | 1,331 |
| ISRUC-Sleep III | 10 | 10 | 8,589 | 6,911 | 1,678 |
| Total | 26 | 28 | 18,770 | 15,761 | 3,009 |

ISRUC uses C4-A1 EEG, X2 ECG (EDF transducer: EKG_Channel) and scorer 1. The last 30 epochs per record are omitted following the provider's extracted-channel noise note; 8,889 original epochs remain unchanged on disk. MIT EEG montages vary by record. R&K stages 3 and 4 are merged as N3; ISRUC labels 0/1/2/3/5 map to the five classes. Unknown labels are excluded and interrupt sequence context.

ISRUC 采用 C4-A1、X2 心电和第一位专家标注，参照提供方关于预提取通道末尾噪声的说明，每条原始记录末尾 30 个窗口按固定协议省略，不代表本项目已逐窗独立证明原始信号存在噪声。原始文件保留。MIT 各记录的脑电导联不同，旧评分的 3/4 期合并为 N3；标签名称对齐不代表评分标准完全一致。

## Split and evaluation status / 划分与评估性质

The fixed split is 22 training subjects and 4 test subjects: MIT-slp02, MIT-slp60, ISRUC-III-04 and ISRUC-III-05. MIT-slp02a/b remain together. There is no validation set. Version 1 already reported these test subjects; version 2 is an exploratory reused-holdout study, not fresh external validation. Both databases occur in training. Seeds measure initialization variability on the same people.

22 人训练、4 人测试，不设验证集；同一人的全部记录归入同一组。这 4 人在第一轮已评估，本轮属于重复使用该留出集的探索性实验。三个随机种子的波动不等同于人群置信区间。

## Signal representations / 信号表示

EEG: each epoch independently filtered at 0.3–35 Hz, converted to microvolts, resampled to 100 Hz. Each waveform has 3,000 samples. STFT uses a 200-sample window, 100-sample overlap and 256-point FFT, retaining 0.3–35 Hz. Log-power shape is 29 frames × 89 frequency bins. Frequency-wise mean/SD, waveform scale, HRV filling medians and HRV mean/SD are fitted on training rows only. EEG epochs with nonfinite values, raw SD below 0.01 µV or absolute amplitude above 2,000 µV are excluded. No labeled epoch was removed by the amplitude rules in this preparation; these basic rules do not constitute complete artifact detection.

脑电逐窗独立滤波和重采样，输入频谱为 29×89。填补与标准化仅由训练集拟合，逐窗处理不混入未来脑电。振幅阈值属于基本检查，不是完整伪迹识别。

HRV: SleepECG heartbeat detection on complete offline ECG; 26 time-domain descriptors, RR plausibility range 0.3–2.0 seconds. Features cover up to 300 seconds ending at the current epoch end. Statistical inputs use no RR endpoint after that end, but the offline detector itself is not a validated streaming detector. Inputs are 26 scaled features + 26 missing flags + 3 quality indicators = 55 values.

HRV 由离线心跳检测计算 26 项时域特征，最多覆盖截至当前窗口末尾的过去 5 分钟。心脏分支另输入缺失标记及三个质量指示量，总计 55 维；本轮不验证实时检测延迟。

| Quality field | Definition / 定义 |
|---|---|
| valid_rr_fraction | Fraction of detected RR intervals in [0.3,2.0] s within the available past 5-minute window / 有效 RR 比例 |
| feature_finite_fraction | Fraction of 26 HRV descriptors finite before filling / 填补前有限特征比例 |
| history_fraction | min(epoch_end,300)/300 / 可用历史时长比例 |

These coverage/plausibility indicators are not a clinically validated ECG quality score. No whole-night normalization or target-label feature is used. Past contexts are contiguous within a record: 5 epochs = 2.5 min, 15 epochs = 7.5 min, including the scored epoch. Left padding is masked; context cannot cross a record or unknown-label gap.

三个指示量用于描述覆盖和合理性，不作为临床验证的心电质量评分。连续输入只包括同一记录当前及过去的窗口，不跨标签缺口；不足时左侧填充并屏蔽。

The EEG spans are 2.5/7.5 minutes; the union of nested HRV endpoint windows spans 7/12 minutes with full history: `300+(L−1)*30` seconds. These are statistical bounds, not a limit on the complete-record ECG detector. / EEG 历史为 2.5／7.5 分钟，嵌套 HRV 终点统计历史联合覆盖约 7／12 分钟，不等于离线检测器全部读取范围。

Actual SleepECG 0.5.9 formulas include padded-row denominators for pNN50/pNN20, `cvSD=SDSD/mean(diff(NN))`, and second-based CVI. These implementation conditions are documented in the feature dictionary; frozen features/results were not changed in the documentation revision. / 特征字典说明 pNN 填充分母、cvSD 公式和 CVI 秒单位等实际条件；文档整理未重算冻结特征或成绩。

pNN padding width is determined over the complete record, potentially depending on later windows. Local RR endpoint selection is historical, but the complete HRV pipeline is not established as strictly causal. / pNN 填充宽度来自完整记录，可能依赖较晚窗口；局部 RR 终点虽按历史选取，整个 HRV 流程不能视作已验证的严格因果实现。

## Model variants / 模型组

| ID | Inputs | Fusion | Context epochs | HRV dropout |
|---|---|---|---:|---:|
| A0 | EEG spectrum | none | 5 | 0 |
| A1 | EEG spectrum | none | 15 | 0 |
| A2 | HRV + quality | none | 5 | 0 |
| A3 | EEG spectrum + HRV + quality | concatenation | 5 | 0 |
| A4 | same multimodal inputs | gated residual | 5 | 0 |
| A5 | same multimodal inputs | gated residual | 5 | 0.20 |
| A6 | same multimodal inputs | gated residual | 15 | 0.20 |

The spectral EEG encoder has a linear 89→96 projection, normalization, positional encoding and two Transformer layers. Each layer uses 4 heads and a 192-dimensional feed-forward block. Frame tokens are mean-pooled per epoch. Cardiac MLP maps 55→96→96. Temporal modeling uses two Transformer layers over the available epoch history; classification reads the current epoch token. Common EEG/temporal/head weights are initialized before variant-specific fusion modules.

窗口内频谱编码和窗口间时序编码均采用两层 Transformer，隐藏维度 96、4 个注意力头。所有包含 EEG 的组共用同一种频谱编码结构，减少脑电分支变化对融合对比的影响。

Gated fusion: `fused = EEG + g * delta(HRV)`, where `g = sigmoid(gate([EEG,HRV,quality])) * valid_rr_fraction * feature_finite_fraction`. The gate starts with zero weights and bias -2. The quality factors force a zero effective gate when cardiac features are unavailable. For A5/A6, 20% of training samples lose the complete HRV context; values become neutral, missing flags 1 and quality 0. EEG remains present. This fallback is checked for exact invariance to removed HRV content; it is not the independently trained A0 model.

门控融合保留 EEG 主路径，以可学习权重加入 HRV 修正；心脏信息全缺失时权重为 0。A5/A6 在训练时随机丢弃整段 HRV 上下文，概率为 20%。回退路径仍使用该融合模型的训练权重，不等同于另行训练的 A0。

Architectures are independent lightweight adaptations. This is not an exact reproduction of SleepTransformer, AttnSleep, GMU or ModDrop, nor a claim of methodological novelty.

A4−A3 changes residual fusion and quality constraints together; no separate gate-without-quality variant is scored. Thus it does not isolate quality-factor benefit. / A4−A3 同时改变残差融合和质量约束，本轮没有仅关闭质量因子的独立对照，不能单独归因于质量约束。

## Fixed training / 固定训练

21 runs = 7 variants × seeds [42,123,2026]. Each starts from scratch and trains 40 epochs; checkpoints are the final epoch. AdamW: learning rate 0.0003, weight decay 0.01; batch 64; Dropout 0.2; cosine LR decay to 5% of initial LR; gradient clipping 1.0. Weighted cross entropy uses `n_train/(5*n_class)`. Spectral training noise SD is 0.02 in standardized log-power space. GPU training uses AMP; scoring uses FP32. No early stopping, seed selection, validation or test-guided checkpoint selection. The frozen protocol and its SHA-256 are saved in every run.

七组各以三个种子从头训练 40 轮，统一使用最后一轮权重。采用加权交叉熵、AdamW、余弦衰减和轻微频谱增强。没有按测试成绩选择种子或检查点，先完成 21 次训练，再统一评分。

## Metrics and stress test / 指标与缺失模态测试

Primary: mean five-class Macro-F1 across individual runs. Secondary: Accuracy, balanced Accuracy, Kappa, per-stage precision/recall/F1/support, per-subject and per-dataset outcomes. Mean ± sample SD across seeds is reported. The equal-probability average of all three seeds is separately reported for each variant; it is not the same metric as mean single-run score.

主指标为单次模型的五类 Macro-F1 在三个种子上的平均值。三模型概率平均集成另行报告，与单次成绩均值明确区分。每类、每人、每库指标完整保存。

The predefined robustness condition removes all HRV inputs while preserving EEG. Individual and ensemble predictions are stored under both conditions. For individual subjects with absent stages, fixed-five-class Macro-F1 counts absent-class F1 as zero; read class supports. Population significance is not inferred from thousands of correlated windows or three seeds.

预设压力测试在保留脑电的同时移除全部 HRV。每人的 Macro-F1 固定按五类计算，无该阶段时该类 F1 为 0，应结合 support 阅读。窗口数及三个种子不能替代独立受试者数量。

## References / 参考

- [SleepTransformer authors' implementation](https://github.com/pquochuy/SleepTransformer)
- [Gated Multimodal Units for Information Fusion](https://arxiv.org/abs/1702.01992)
- [ModDrop: Adaptive Multi-Modal Gesture Recognition](https://arxiv.org/abs/1501.00102)
- [MIT-BIH PSG](https://physionet.org/content/slpdb/1.0.0/)
- [ISRUC source](https://sleeptight.isr.uc.pt/?page_id=48) and [tail-noise note](https://sleeptight.isr.uc.pt/?page_id=76)
