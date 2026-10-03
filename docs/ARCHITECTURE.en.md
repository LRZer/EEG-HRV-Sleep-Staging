# Model architecture and design sources

[简体中文](ARCHITECTURE.zh-CN.md) · [README](../README.md) · [Pipeline](DATA_PIPELINE.en.md) · [Feature dictionary](FEATURES.en.md)

## 1. Within-epoch and across-epoch modeling

Each prediction targets the current 30-second epoch. The EEG encoder compresses its 29 spectral frames into a 96-dimensional vector. The temporal encoder then processes L vectors including the current epoch and its available predecessors. `B` is batch size; `L=5` or `15`.

![A1 and A6 architecture and tensor shapes](figures/model_architecture.en.png)

**A1**, the highest mean Macro-F1 variant, uses 15 EEG epochs. **A6**, the complete fusion design, also uses 15 epochs with HRV, quality-constrained gating and training modality dropout. They share architecture definitions but train separately; A6 does not add modules to a pretrained A1 checkpoint.

| Stage | Input | Output | Operations |
|---|---|---|---|
| Flatten batch/epoch axes | `[B,L,29,89]` | `[B×L,29,89]` | Encode each epoch independently |
| Spectrum projection | 89 frequencies per frame | `[B×L,29,96]` | Linear 89→96, LayerNorm, sinusoidal position encoding |
| Within-epoch Transformer | 29 frames×96 | Same shape | 2 layers, 4 heads of width 24; FFN 96→192→96 |
| Epoch pooling | `[B×L,29,96]` | `[B,L,96]` | Mean over 29 frames; not attention pooling |
| Cardiac encoder, A2–A6 | `[B,L,55]` | `[B,L,96]` | Linear 55→96, LN, GELU, Dropout, Linear 96→96, LN |
| Fusion, A3–A6 | Two width-96 vectors | `[B,L,96]` | Concatenation or residual gating |
| Temporal position encoding | L epoch vectors | `[B,L,96]` | Sinusoidal encoding of epoch order |
| Across-epoch Transformer | `[B,L,96]` | Same shape | 2 layers, 4 heads; FFN 96→192→96; padding-key mask |
| Current-epoch classifier | `encoded[:, -1]` | `[B,5]` | LN, Dropout, Linear 96→5; inference softmax |

Both Transformers use Pre-LN, GELU, Dropout 0.2 and final encoder LayerNorm. Attention is `softmax(QKᵀ/√24)V` per head. Within-epoch attention links spectral frames; across-epoch attention links supplied historical/current epochs.

No future epoch is supplied. The temporal encoder uses a padding-key mask but no triangular causal mask: earlier historical tokens may attend to later tokens already supplied. Only the current token is classified. This is offline history-to-current classification, not validated sample-by-sample streaming inference.

## 2. Fusion equations

### A3: concatenation

EEG `e` and cardiac `h` vectors concatenate into 192 values, followed by Linear 192→96, LayerNorm and GELU. There is no separate effective gate; the three coverage fields still enter the 55-value cardiac input.

### A4–A6: quality-constrained residual gate

```text
e = EEGEncoder(spectrum)                  # 96 values
h = CardiacMLP(features, missing, quality) # 96 values
q = [valid_rr, finite_features, history]   # 3 values
u = sigmoid(Linear(concat(e, h, q)))       # 195→1
g = u × valid_rr × finite_features         # one scalar per epoch
delta = Linear96→96(GELU(Linear96→96(h)))
z = e + g × delta
```

`history_fraction` enters the learned gate but **does not multiply g directly**. Thus `0≤g≤valid_rr×finite_features`. Gate weights start at zero and bias at −2, giving initial sigmoid ≈0.1192 before coverage constraints. EEG is retained as a residual path; HRV supplies a scaled correction.

![Gating, dropout and fallback](figures/gated_fusion.en.png)

The gate is one scalar per epoch, not one weight per feature and not Cross-Attention. Its value describes representation composition; larger g does not establish a physiological causal role for ECG.

### A5/A6: modality dropout and fallback

For 20% of training samples, the **complete L-epoch HRV context** is removed while EEG remains. The 26 scaled values become zero, 26 missing flags become one, and three coverage fields become zero. Ordinary missing features are first filled with training statistics; modality dropout independently replaces whole contexts inside the model.

Completely missing HRV forces `g=0`, hence `z=e`. Downstream temporal/classifier weights still belong to the fusion model; this does not restore independently trained A0/A1. Preserved checks confirm exactly zero missing-HRV gates for A4–A6 and unchanged EEG-only probabilities under both conditions.

## 3. Parameters and compute

| Variants | Trainable parameters | Mean seconds per 40-epoch run |
|---|---:|---:|
| A0 / A1 | 309,029 | 129.0 / 215.5 |
| A2 | 165,509 | 79.5 |
| A3 | 342,821 | 143.3 |
| A4 / A5 / A6 | 342,921 | 138.4 / 142.4 / 219.6 |

Longer context changes token count and attention cost, not parameter count. Times come from recorded RTX 4060 Laptop 8 GB runs, not a hardware-independent guarantee. [Compute CSV](../results/v2/compute.csv)

## 4. Literature and historical versions

| Source | Adopted idea | Differences |
|---|---|---|
| [SleepTransformer, Phan et al., 2022](https://arxiv.org/abs/2105.11043); [author code](https://github.com/pquochuy/SleepTransformer) | Within/across-epoch Transformers | Independent compact PyTorch implementation, width 96, mean pooling, history-to-current sequence-to-one output; not full sequence output, attention interpretability or uncertainty reproduction |
| [GMU, Arevalo et al., 2017](https://arxiv.org/abs/1702.01992) | Learned multiplicative gating | Asymmetric EEG residual plus scalar HRV correction, additional coverage constraints; not the paper's original unit |
| [ModDrop, Neverova et al., 2015](https://arxiv.org/abs/1501.00102) | Randomly dropping modalities during training | Only complete HRV context is dropped at 20%; not the original initialization/progressive-fusion strategy |
| Project version 1 | Runnable CNN, Transformer, concat and Cross-Attention comparisons | Version 2 fixes the spectral EEG backbone; all 21 runs train from scratch |

These are design sources, not exact paper reproductions or evidence of methodological novelty. Version 2 uses no additional large pretrained network, pretrained checkpoint transfer or clinical validation.

## 5. Code map

| Component | Source |
|---|---|
| Shared Transformer, position encoding, spectral encoder | [deep_learning/models.py](../deep_learning/models.py) |
| A0–A6, cardiac MLP, gate, modality dropout, current-token head | [experiments/v2/models.py](../experiments/v2/models.py) |
| Frozen hyperparameters, seeds and 40-epoch rule | [protocol.json](../experiments/v2/protocol.json) |
| Sample assembly and training-only preprocessing | [data.py](../experiments/v2/data.py) |
| Gate fallback and context checks | [checks.py](../experiments/v2/checks.py) |
| Technical diagrams | [technical_figures.py](../experiments/v2/technical_figures.py) |
