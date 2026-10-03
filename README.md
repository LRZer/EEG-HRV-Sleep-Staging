# EEG–HRV Sleep Staging

[English](README.md) · [简体中文](README.zh-CN.md) · [Data & release](https://github.com/LRZer/EEG-HRV-Sleep-Staging/releases/tag/v2.0.0) · [Methods](docs/METHODS_V2.md)

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
| A0 | EEG / 5 epochs | 67.05 ± 0.24% | 0.5991 ± 0.0029 | 0.5740 | 0.2521 |
| A1 | EEG / 15 epochs | 67.22 ± 1.82% | 0.6171 ± 0.0221 | 0.5840 | 0.3224 |
| A2 | HRV / 5 epochs | 36.36 ± 2.40% | 0.3002 ± 0.0311 | 0.1123 | 0.2376 |
| A3 | Concat / 5 epochs | 64.00 ± 2.14% | 0.5789 ± 0.0348 | 0.4185 | 0.2676 |
| A4 | Gated / 5 epochs | 63.78 ± 1.47% | 0.5900 ± 0.0188 | 0.4263 | 0.3688 |
| A5 | Gated + drop / 5 epochs | 64.44 ± 0.81% | 0.5866 ± 0.0099 | 0.4418 | 0.3198 |
| A6 | Gated + drop / 15 epochs | 63.96 ± 0.68% | 0.5834 ± 0.0038 | 0.4136 | 0.3081 |

Highest mean Accuracy: **A1, 67.22%**. Highest mean Macro-F1: **A1, 0.6171**. Full A6 versus EEG control A0: **-3.09 percentage points** Accuracy and **-0.0157** Macro-F1.

All multimodal A3–A6 variants scored below EEG control A0 in both mean Accuracy and Macro-F1; this protocol did not show a performance gain from adding HRV.

![Ablation results with individual seed scores](results/v2/figures/ablation_results.png)

All three seeds are additionally averaged as a probability ensemble for **each** variant; no seed is selected by test score. EEG-only A1 ensemble: Accuracy **70.22%**, Macro-F1 **0.6407**. Full A6 ensemble: Accuracy **65.10%**, Macro-F1 **0.5862**. [Complete results](results/v2/report.en.md) · [All figures](docs/FIGURES.md) · [Machine-readable metrics](results/v2/metrics.json).

## Data and running

[Release v2.0.0](https://github.com/LRZer/EEG-HRV-Sleep-Staging/releases/tag/v2.0.0) provides the prepared **complete study inputs** and attributed MIT original files in checksum-verified parts. ISRUC original REC files are downloaded from the [provider](https://sleeptight.isr.uc.pt/?page_id=48); the download code and exact hashes are included. [Data guide](datasets/README.md).

```powershell
conda activate pytorch
python scripts/download_release_data.py --kind prepared
python -m experiments.v2.predict --variant A6 --seed ensemble
```

For installation, raw-signal reconstruction and all 21 runs: [reproduction guide](docs/REPRODUCE_V2.md). Prediction currently accepts the aligned study cache. GPU used: RTX 4060 Laptop, PyTorch 2.6.0+cu118. Histories, 21 checkpoints, preprocessing, per-class/person/dataset metrics and all normal/missing-HRV predictions are under `results/v2/`.

## Scope and attribution

One subject split, four test people, reused holdout. Seed SD is not a population confidence interval. MIT test includes only 7 N3 epochs. Both databases occur in training; this is not unseen-dataset validation. Offline ECG detection and arbitrary unannotated-file deployment are not evaluated. These networks are independent lightweight adaptations, not full reproductions of published models.

[MIT-BIH](https://physionet.org/content/slpdb/1.0.0/) · [ISRUC](https://sleeptight.isr.uc.pt/) · [SleepTransformer](https://github.com/pquochuy/SleepTransformer) · [GMU](https://arxiv.org/abs/1702.01992) · [ModDrop](https://arxiv.org/abs/1501.00102). MIT license covers project software; datasets retain their [original conditions](datasets/DATA_NOTICE.txt). [Version 1](results/deep_learning/report.md) and [historical RF baseline](docs/BASELINE.md) remain archived with different evaluation settings.
