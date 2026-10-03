# 模型架构与设计依据

[English](ARCHITECTURE.en.md) · [返回 README](../README.zh-CN.md) · [数据流程](DATA_PIPELINE.zh-CN.md) · [特征字典](FEATURES.zh-CN.md)

## 1. 两级建模：窗口内与窗口间

模型对当前 30 秒窗口做一次分类。先将每个 EEG 窗口的 29 个频谱帧编码为 96 维向量，再将当前及过去共 L 个窗口的向量送入时序编码器。`B` 是批大小；`L=5` 或 `15`，包括当前窗口。

![A1 与 A6 架构及尺寸](figures/model_architecture.zh-CN.png)

**A1** 用 15 窗口 EEG，是本轮平均 Macro-F1 最高的模型。**A6** 同样用 15 窗口，同时加入 HRV、质量约束门控及训练模态丢弃，是完整融合方案。两者共用 EEG 和时序骨干定义，分别训练；A6 不是在已训练的 A1 权重上添加一个模块。

| 阶段 | 输入 | 输出 | 实现 |
|---|---|---|---|
| 展平批与窗口维 | `[B,L,29,89]` | `[B×L,29,89]` | 各窗口独立编码 |
| 频谱投影 | 每帧 89 个频率值 | `[B×L,29,96]` | Linear 89→96、LayerNorm，加入正弦位置编码 |
| 窗口内 Transformer | 29 帧×96 | 同尺寸 | 2 层、4 头；每头 24 维；FFN 96→192→96 |
| 窗口池化 | `[B×L,29,96]` | `[B,L,96]` | 对 29 帧取均值，不是注意力池化 |
| 心脏编码（A2–A6） | `[B,L,55]` | `[B,L,96]` | Linear 55→96、LN、GELU、Dropout、Linear 96→96、LN |
| 融合（A3–A6） | 两个 96 维表示 | `[B,L,96]` | 拼接或残差门控，见下一节 |
| 时序位置编码 | L 个窗口向量 | `[B,L,96]` | 正弦位置编码，保留顺序 |
| 窗口间 Transformer | `[B,L,96]` | 同尺寸 | 2 层、4 头；FFN 96→192→96；填充键屏蔽 |
| 当前窗口分类 | `encoded[:, -1]` | `[B,5]` | LN、Dropout、Linear 96→5；推理 softmax |

两级 Transformer 均使用 Pre-LN、GELU、Dropout 0.2，编码器末尾再做 LayerNorm。自注意力可写为 `softmax(QKᵀ/√24)V`，4 个头分别学习表示之间的关系。窗口内注意力连接频谱帧；窗口间注意力连接已提供的历史和当前窗口。

上下文中没有未来窗口。时序编码器只使用填充键掩码，没有三角因果掩码，因此较早的历史 token 可以关注已提供的较晚 token；最终只读取当前 token。这是离线的“历史到当前”分类，不是已经验证的逐采样实时模型。

## 2. 融合如何计算

### A3：拼接

将 EEG 表示 `e` 与心脏表示 `h` 拼为 192 维，再通过 Linear 192→96、LayerNorm、GELU 得到窗口向量。A3 不使用独立的有效门控，但 3 个覆盖量仍在心脏的 55 维输入中。

### A4–A6：质量约束残差门控

```text
e = EEGEncoder(spectrum)                  # 96 维
h = CardiacMLP(features, missing, quality) # 96 维
q = [valid_rr, finite_features, history]   # 3 维
u = sigmoid(Linear(concat(e, h, q)))       # 195→1
g = u × valid_rr × finite_features         # 每窗口一个标量
delta = Linear96→96(GELU(Linear96→96(h)))
z = e + g × delta
```

`history_fraction` 作为可学习门控的输入，**不直接乘到 g 上**。由于两个乘法因子都在 0–1 之间，有 `0≤g≤valid_rr×finite_features`。门控权重初始化为 0、偏置为 −2，所以未考虑覆盖约束前的初始 sigmoid 约为 0.1192。残差路径始终保留 EEG 表示；心脏分支提供可缩放的修正。

![门控、丢弃与回退](figures/gated_fusion.zh-CN.png)

图中 `g` 是每窗口一个标量，不是每个特征一个权重，也不是 Cross-Attention。它只描述模型如何合成表示；较大 g 不证明 ECG 对某个睡眠阶段具有生理因果作用。

### A5／A6：训练模态丢弃与缺失回退

按样本以 20% 概率移除 **L 个窗口的全部 HRV 上下文**，并保留 EEG。丢弃后的 26 个已缩放值为 0、26 个缺失标记为 1、3 个覆盖量为 0。正常缺失值先经过训练集统计填补；模态丢弃另在模型内将整段表示改为上述状态。

全缺失使 `g=0`，因此融合向量为 `z=e`。后续时序编码器和分类头仍是融合模型自己的权重，并不恢复独立训练的 A0／A1。保存的检查确认：全部缺失时 A4–A6 的门控恰为 0；EEG-only 模型在两种输入条件下概率不变。

## 3. 参数量与计算量

| 组 | 可训练参数 | 每次训练平均秒数（40 轮） |
|---|---:|---:|
| A0／A1 | 309,029 | 129.0／215.5 |
| A2 | 165,509 | 79.5 |
| A3 | 342,821 | 143.3 |
| A4／A5／A6 | 342,921 | 138.4／142.4／219.6 |

同一组上下文长度增加不改变参数量，但增加参与注意力计算的 token 数。时间来自保存的训练记录，设备为 RTX 4060 Laptop 8 GB，不是跨硬件性能保证。[计算量表](../results/v2/compute.csv)

## 4. 与论文和历史版本的对应

| 来源 | 本项目使用的思想 | 实际差异 |
|---|---|---|
| [SleepTransformer，Phan 等，2022](https://arxiv.org/abs/2105.11043)；[作者代码](https://github.com/pquochuy/SleepTransformer) | 窗口内与窗口间两级 Transformer | 独立 PyTorch 轻量实现、96 维、均值池化、历史到当前的 sequence-to-one 分类；没有完整复现论文的序列输出、注意力解释与不确定性分析 |
| [GMU，Arevalo 等，2017](https://arxiv.org/abs/1702.01992) | 可学习乘法门控调节模态影响 | EEG 主路径＋HRV 修正的非对称标量门控；另加覆盖约束；没有照搬论文单元 |
| [ModDrop，Neverova 等，2015](https://arxiv.org/abs/1501.00102) | 训练中随机移除模态 | 只以 20% 丢弃整段 HRV；不是原论文完整的初始化及渐进融合训练策略 |
| 第一轮本项目 | 1D-CNN、Transformer、拼接、Cross-Attention 的可运行比较 | 第二轮固定频谱 EEG 骨干，减少 EEG 编码差异；21 次训练全部从头开始 |

这些是设计依据，不是完整论文复现或方法创新证明。第二轮没有额外的大规模预训练模型、公开预训练权重迁移或临床验证。

## 5. 代码对应

| 说明 | 源码位置 |
|---|---|
| 共用 Transformer、正弦位置编码、频谱编码器 | [deep_learning/models.py](../deep_learning/models.py) |
| A0–A6、心脏 MLP、门控、模态丢弃、当前 token 分类 | [experiments/v2/models.py](../experiments/v2/models.py) |
| 固定超参数、种子、40 轮规则 | [protocol.json](../experiments/v2/protocol.json) |
| 样本组装与训练集预处理 | [data.py](../experiments/v2/data.py) |
| 门控回退及上下文检查 | [checks.py](../experiments/v2/checks.py) |
| 文档图生成 | [technical_figures.py](../experiments/v2/technical_figures.py) |
