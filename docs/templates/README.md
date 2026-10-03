# EEG–HRV Sleep Staging

**English** · [简体中文](README.zh-CN.md) · [Data release](https://github.com/LRZer/EEG-HRV-Sleep-Staging/releases/tag/v2.0.0) · [Figure catalogue](docs/FIGURES.md)

Predict the sleep stage of each **30-second** segment from brain activity and heartbeat-interval features. Completed work includes preparation of two public datasets, seven controlled deep learning variants, three-seed training and result preservation. Code, 21 checkpoints, study inputs, epoch-level predictions and reproducible figures are included.

| Signal / output | Meaning | Role in this project |
|---|---|---|
| EEG | Electrical activity recorded by scalp electrodes | Converted to a time–frequency representation for learning brain-signal patterns |
| ECG | Electrical waveform of the heart | Used to detect beats and derive successive RR intervals |
| HRV | Variability and statistics of heartbeat intervals | Auxiliary features derived from ECG; not a separately acquired raw signal |
| Wake / N1 / N2 / N3 / REM | Wakefulness, non-REM stages 1/2/3, and REM sleep | Five probabilities and their highest-probability class per segment |

## 1. Completed work and main outcomes

- **Data:** MIT-BIH PSG and ISRUC-Sleep III; 26 subjects, 28 recordings, 18,770 retained epochs; 22 training and 4 test subjects, **no validation set**.
- **Methods:** shared spectral EEG encoder and temporal Transformer; EEG-only, HRV-only, concatenation, quality-constrained residual gating, modality dropout and context-length comparisons.
- **Experiments:** 7 variants × 3 seeds × 40 fixed epochs; 21 runs and 840 training epochs; normal-input and fully missing-HRV evaluation.
- **Scores:** highest mean individual-run scores belong to EEG-only A1: Accuracy **{{BEST_ACC}}**, Macro-F1 **{{BEST_F1}}**. Its three-model probability ensemble scores **{{A1_ENS_ACC}} / {{A1_ENS_F1}}**.
- **Fusion outcome:** mean Accuracy and Macro-F1 of A3–A6 are below EEG control A0. Full A6 ensemble scores **{{A6_ENS_ACC}} / {{A6_ENS_F1}}**; adding HRV did not improve overall performance under this protocol.

The four test subjects were already evaluated in version 1. These are **exploratory reused-holdout results** for one subject split, rather than fresh external validation or clinical diagnostic evidence.

## 2. Data, labels and aligned inputs

| Dataset | Subjects / records | Retained epochs | Train epochs | Test epochs | Channels and labels |
|---|---:|---:|---:|---:|---|
| MIT-BIH PSG 1.0.0 | 16 / 18 | 10,181 | 8,850 | 1,331 | One EEG channel, varying montages; ECG; original sleep annotations |
| ISRUC-Sleep III | 10 / 10 | 8,589 | 6,911 | 1,678 | C4-A1 EEG, X2 ECG, scorer 1 |
| Total | 26 / 28 | 18,770 | 15,761 | 3,009 | All recordings from one person stay in one split |

Fixed test IDs: `MIT-slp02`, `MIT-slp60`, `ISRUC-III-04`, `ISRUC-III-05`; slp02a/b belong to the same person. Unknown labels are excluded and interrupt context. MIT R&K stages 3/4 merge into N3; matching label names does not harmonize scoring standards. The final 30 epochs of every ISRUC record are omitted following the provider's note about noisy pre-extracted channels; original files are retained.

**One prediction:** 5 or 15 EEG epochs including the current epoch and its available predecessors. Each carries HRV statistics from up to 300 seconds ending at that epoch's end. The target is the last, current epoch. Filling and scaling statistics are fitted on training data only.

![Context, HRV history and boundaries](docs/figures/time_alignment.en.png)

Read the figure: 5/15 EEG epochs cover 2.5/7.5 minutes; the union of their RR-endpoint statistical histories spans approximately **7/12 minutes** when full history is available. Context cannot cross records or unknown-label gaps; unavailable history is left-padded and masked. ECG beat detection runs on complete offline recordings; streaming detection has not been validated. [Detailed data pipeline](docs/DATA_PIPELINE.en.md)

Library pNN denominators also depend on the complete record's padded width, potentially influenced by later windows. Local RR endpoint bounds do not establish strict causality of the full feature pipeline. [Formulas and offline dependencies](docs/FEATURES.en.md)

## 3. Current model architecture

Version 2 uses a **spectral Transformer, residual gated fusion and temporal Transformer**. Version-1 1D-CNN and Cross-Attention experiments remain archived; A0–A6 do not use Cross-Attention.

![EEG control and full fusion architecture](docs/figures/model_architecture.en.png)

| Module | Operations and dimensions | Purpose |
|---|---|---|
| EEG preprocessing | 30 s: 0.3–35 Hz filtering, 100 Hz resampling, 3,000 samples; STFT→29×89 log-power spectrum | Preserve within-epoch time and frequency patterns |
| Within-epoch encoder | 89→96 projection; position encoding; 2 Transformer layers, 4 heads; mean-pool 29 frames→96 values | Encode one EEG epoch as one vector |
| Cardiac encoder | 26 features + 26 missing flags + 3 coverage/plausibility fields = 55 values; MLP 55→96→96 | Encode HRV and its availability |
| Residual gate | `z = EEG + g × delta(HRV)`; one scalar `g` per epoch, bounded by valid-RR and finite-feature fractions | Control the cardiac correction to the EEG representation |
| Across-epoch encoder | 5/15 vectors of width 96 plus position encoding; 2 Transformer layers, 4 heads | Learn relationships among past and current epochs |
| Classifier | Current token→LayerNorm, Dropout, Linear→5; softmax probabilities | Predict the current sleep stage |

The two Transformers operate **within one epoch** and **across epochs**, respectively. No EEG epoch or statistical RR endpoint after the target end is supplied. Temporal attention has no triangular causal mask within the supplied history. Gates are internal model weights, not clinical importance scores. [Layers, gating equations and literature mapping](docs/ARCHITECTURE.en.md) · [26-feature HRV dictionary](docs/FEATURES.en.md)

## 4. Seven variants and fixed training

| ID | Inputs and fusion | Context epochs | Training HRV dropout | Comparison purpose |
|---|---|---:|---:|---|
| A0 | EEG only | 5 | — | Short-context EEG control |
| A1 | EEG only | 15 | — | A1−A0: longer EEG context |
| A2 | HRV only | 5 | 0 | Independent predictive ability of cardiac features |
| A3 | EEG + HRV, concatenation | 5 | 0 | A3−A0: add HRV |
| A4 | EEG + HRV, quality-constrained residual gate | 5 | 0 | A4−A3: replace the fusion design |
| A5 | Same as A4 | 5 | 20% | A5−A4: modality dropout |
| A6 | Same as A5 | 15 | 20% | A6−A5: longer fusion context |

A4−A3 changes concatenation, the residual path and quality constraints together; it does not isolate the quality factor. A5/A6 remove the complete HRV context for 20% of training samples. Fully missing HRV forces the effective gate to zero, retaining the fusion model's EEG path; this is not the independently trained A0/A1 network.

Shared settings: seeds **42/123/2026**, **40 epochs with final-epoch checkpoints**, batch 64, AdamW (learning rate 3×10⁻⁴, weight decay 0.01), Dropout 0.2, cosine decay to 5% of initial LR, gradient clipping 1.0. Class-weighted cross entropy uses training frequencies. Spectral training noise has SD 0.02 after scaling. GPU training uses AMP; scoring uses FP32. No checkpoint or seed is selected by test score. [Frozen protocol](experiments/v2/protocol.json) · [Complete methods](docs/METHODS_V2.md)

## 5. Scores and interpretation

Three individual models: **mean ± sample SD**; N1/REM F1 are seed means. Accuracy is the fraction of correct epochs. Macro-F1 equally averages five class F1 scores and is the primary metric. Seed SD measures initialization variability, not a population confidence interval.

| ID | Variant | Accuracy | Macro-F1 | N1 F1 | REM F1 |
|---|---|---:|---:|---:|---:|
{{RESULT_ROWS}}

![Seven variants and seed variability](results/v2/figures/ablation_results.png)

Read the figure: points show three individual seeds; bars and error bars show means and sample SD. **Ensembles** separately average the three models' class probabilities before argmax, for every variant. Ensemble scores and mean individual-run scores are distinct. Full A6 versus A0 changes mean Accuracy by **{{DELTA_ACC}} percentage points** and Macro-F1 by **{{DELTA_F1}}**.

Stage changes differ in direction: A4 REM F1 is 0.3688 (A0: 0.2521), while N1 F1 is 0.4263 (A0: 0.5740). With HRV fully unavailable, A5 versus A4 increases mean Accuracy by 6.85 percentage points and Macro-F1 by 0.0739. These are observed outcomes of this predefined stress test.

[Result analysis and fixed cases](docs/RESULTS_ANALYSIS.en.md) explain confusion matrices, subject/source metrics, missing modalities and gates. [Full report](results/v2/report.en.md), [machine-readable metrics](results/v2/metrics.json) and [epoch predictions](results/v2/predictions.csv.gz) preserve all scores and outputs.

## 6. Download, inference and reproduction

[Release v2.0.0](https://github.com/LRZer/EEG-HRV-Sleep-Staging/releases/tag/v2.0.0) provides **all aligned inputs for 18,770 epochs** (368.1 MiB compressed) and attributed MIT original files (554.8 MiB). Archives use 48 MiB parts; the downloader verifies part, archive and extracted-file SHA-256 hashes. ISRUC original REC files are downloaded from the provider, with tools and exact hashes included. [Data guide](datasets/README.md)

```powershell
git clone https://github.com/LRZer/EEG-HRV-Sleep-Staging.git
cd EEG-HRV-Sleep-Staging
conda activate pytorch
python -m pip install -r requirements-deep-learning.txt
python scripts/download_release_data.py --kind prepared
# EEG-only A1: highest mean primary score in this study
python -m experiments.v2.predict --variant A1 --seed ensemble --output results/A1_restored.csv
# Complete EEG-HRV fusion model
python -m experiments.v2.predict --variant A6 --seed ensemble --output results/A6_restored.csv
```

Inference restores weights and training-fitted preprocessing and returns stages/probabilities for the fixed study test cache. Recorded training environment: Python 3.11.9, PyTorch 2.6.0+cu118, RTX 4060 Laptop 8 GB. [Environment, raw reconstruction, all 21 runs and verification](docs/REPRODUCE_V2.md)

## 7. Repository map and version history

| Path | Contents |
|---|---|
| `experiments/v2/` | Current training, evaluation, models, gate checks, inference and plotting |
| `deep_learning/` | Shared preparation and spectral encoder; version-1 deep models |
| `results/v2/` | 21 weights, 840 epoch logs, preprocessing, 56 prediction sets, stage/person/source scores and 13 result figure sets |
| `docs/` | Bilingual technical documentation, reproduction, gallery, technical figures and generation templates |
| `datasets/`, `scripts/` | Provenance, checksums, download and inspection tools |
| `main.py` | Historical MIT-only random forest baseline |

Progression: [random forest](docs/BASELINE.md)→[version-1 CNN/Transformer/Cross-Attention](results/deep_learning/report.md)→current seven-variant shared-backbone study. Different data and evaluation settings prevent treating historical score differences as a single improvement estimate. [Documentation index](PROJECT.md)

## 8. Scope, sources and license

Four test subjects, one reused split. Both sources occur in training; source-specific metrics do not establish unseen-dataset generalization. MIT test has only seven N3 epochs. Per-person Macro-F1 fixes all five classes and assigns zero F1 to absent classes. Arbitrary unannotated EDF deployment, streaming ECG detection and clinical effectiveness are untested. Retain the library denominator, unit and numerical-stability conditions in the [feature implementation notes](docs/FEATURES.en.md) when interpreting these results.

Independent lightweight implementation informed by [SleepTransformer](https://github.com/pquochuy/SleepTransformer), gating ideas in [GMU](https://arxiv.org/abs/1702.01992), and modality dropping in [ModDrop](https://arxiv.org/abs/1501.00102). It is not a full reproduction of those papers or a claim of methodological novelty. Data: [MIT-BIH](https://physionet.org/content/slpdb/1.0.0/) and [ISRUC-Sleep](https://sleeptight.isr.uc.pt/). [MIT license](LICENSE) covers project software; datasets and dependencies retain their [original rights and conditions](datasets/DATA_NOTICE.txt).
