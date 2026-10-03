# Version 2 controlled experiment report

22 training subjects, 4 test subjects, 18,770 epochs. Seven variants × three seeds × 40 epochs: 21 runs and 840 training epochs.

No validation set; final epoch checkpoints only. The same subjects were evaluated in version 1, so these are exploratory reused-holdout outcomes.

Protocol SHA-256: `75eed49a8e3f3206f30813c4c6d89573f35f135baa2b2d1889a28f7dad03698c`.

## Three individual runs per variant

± is sample SD across three initialization seeds, not a subject-population confidence interval.

| ID | Variant | Accuracy | Macro-F1 | Balanced Acc | Kappa |
|---|---|---:|---:|---:|---:|
| A0 | EEG / 5 epochs | 67.05% ± 0.24 pp | 0.5991 ± 0.0029 | 0.5850 | 0.5638 |
| A1 | EEG / 15 epochs | 67.22% ± 1.82 pp | 0.6171 ± 0.0221 | 0.6033 | 0.5688 |
| A2 | HRV / 5 epochs | 36.36% ± 2.40 pp | 0.3002 ± 0.0311 | 0.3146 | 0.1540 |
| A3 | Concat / 5 epochs | 64.00% ± 2.14 pp | 0.5789 ± 0.0348 | 0.5654 | 0.5222 |
| A4 | Gated / 5 epochs | 63.78% ± 1.47 pp | 0.5900 ± 0.0188 | 0.5771 | 0.5205 |
| A5 | Gated + drop / 5 epochs | 64.44% ± 0.81 pp | 0.5866 ± 0.0099 | 0.5709 | 0.5281 |
| A6 | Gated + drop / 15 epochs | 63.96% ± 0.68 pp | 0.5834 ± 0.0038 | 0.5692 | 0.5232 |

## Predeclared pairwise contrasts

| Contrast | Changed factor | Δ Accuracy (pp) | Δ Macro-F1 |
|---|---|---:|---:|
| A1 − A0 | EEG context 5→15 | +0.17 | +0.0181 |
| A3 − A0 | Add HRV via concat | -3.06 | -0.0202 |
| A4 − A3 | Concat→gated residual | -0.22 | +0.0112 |
| A5 − A4 | 20% HRV dropout | +0.66 | -0.0034 |
| A6 − A5 | Gated context 5→15 | -0.48 | -0.0033 |
| A6 − A0 | Full model vs EEG control | -3.09 | -0.0157 |

## Observed outcomes

The table above gives the overall fusion contrasts; this run did not show a multimodal variant exceeding the EEG control in both Accuracy and Macro-F1.
A4 versus A0: REM F1 0.3688 versus 0.2521, but N1 F1 0.4263 versus 0.5740; stage-specific changes differ in direction.
With HRV unavailable, A5 versus A4 changes mean Accuracy by +6.85 pp and Macro-F1 by +0.0739. These are observed mean differences, without a population-significance claim.

A4−A3 changes residual fusion and quality constraints together, so quality-factor benefit is not isolated. Companion technical documents describe feature implementation conditions and error cases.

## Per-stage metrics

| ID | Stage | Precision mean | Recall mean | F1 mean | F1 SD | Support |
|---|---|---:|---:|---:|---:|---:|
| A0 | Wake | 0.7855 | 0.8560 | 0.8190 | 0.0112 | 882 |
| A0 | N1 | 0.5107 | 0.6552 | 0.5740 | 0.0052 | 583 |
| A0 | N2 | 0.7053 | 0.7421 | 0.7228 | 0.0119 | 875 |
| A0 | N3 | 0.9868 | 0.4608 | 0.6275 | 0.0310 | 361 |
| A0 | REM | 0.3154 | 0.2110 | 0.2521 | 0.0270 | 308 |
| A1 | Wake | 0.7967 | 0.8432 | 0.8187 | 0.0199 | 882 |
| A1 | N1 | 0.5201 | 0.6667 | 0.5840 | 0.0394 | 583 |
| A1 | N2 | 0.7108 | 0.7051 | 0.7078 | 0.0216 | 875 |
| A1 | N3 | 0.9477 | 0.4995 | 0.6528 | 0.0341 | 361 |
| A1 | REM | 0.3461 | 0.3019 | 0.3224 | 0.0561 | 308 |
| A2 | Wake | 0.4410 | 0.6266 | 0.5173 | 0.0160 | 882 |
| A2 | N1 | 0.2265 | 0.0749 | 0.1123 | 0.0063 | 583 |
| A2 | N2 | 0.3596 | 0.3802 | 0.3690 | 0.0228 | 875 |
| A2 | N3 | 0.2780 | 0.2576 | 0.2647 | 0.0775 | 361 |
| A2 | REM | 0.2426 | 0.2338 | 0.2376 | 0.0472 | 308 |
| A3 | Wake | 0.7578 | 0.8783 | 0.8135 | 0.0322 | 882 |
| A3 | N1 | 0.4057 | 0.4345 | 0.4185 | 0.0800 | 583 |
| A3 | N2 | 0.6586 | 0.7051 | 0.6794 | 0.0200 | 875 |
| A3 | N3 | 0.9130 | 0.5928 | 0.7154 | 0.0472 | 361 |
| A3 | REM | 0.3732 | 0.2165 | 0.2676 | 0.0581 | 308 |
| A4 | Wake | 0.6934 | 0.8987 | 0.7825 | 0.0257 | 882 |
| A4 | N1 | 0.4418 | 0.4151 | 0.4263 | 0.0448 | 583 |
| A4 | N2 | 0.6930 | 0.6514 | 0.6714 | 0.0082 | 875 |
| A4 | N3 | 0.8919 | 0.5826 | 0.7012 | 0.0136 | 361 |
| A4 | REM | 0.4106 | 0.3377 | 0.3688 | 0.0402 | 308 |
| A5 | Wake | 0.7437 | 0.8874 | 0.8089 | 0.0166 | 882 |
| A5 | N1 | 0.4425 | 0.4420 | 0.4418 | 0.0149 | 583 |
| A5 | N2 | 0.6643 | 0.7059 | 0.6842 | 0.0158 | 875 |
| A5 | N3 | 0.9133 | 0.5411 | 0.6784 | 0.0250 | 361 |
| A5 | REM | 0.3839 | 0.2781 | 0.3198 | 0.0177 | 308 |
| A6 | Wake | 0.7887 | 0.8677 | 0.8262 | 0.0219 | 882 |
| A6 | N1 | 0.4184 | 0.4105 | 0.4136 | 0.0462 | 583 |
| A6 | N2 | 0.6393 | 0.7181 | 0.6754 | 0.0279 | 875 |
| A6 | N3 | 0.8963 | 0.5660 | 0.6935 | 0.0300 | 361 |
| A6 | REM | 0.3391 | 0.2835 | 0.3081 | 0.0534 | 308 |

## Dataset-specific outcomes

| ID | Dataset | Epochs | Accuracy mean ± SD | Macro-F1 mean ± SD |
|---|---|---:|---:|---:|
| A0 | MIT-BIH | 1331 | 70.35% ± 0.72 pp | 0.5063 ± 0.0055 |
| A0 | ISRUC-III | 1678 | 64.44% ± 0.97 pp | 0.5755 ± 0.0072 |
| A1 | MIT-BIH | 1331 | 69.17% ± 2.89 pp | 0.5019 ± 0.0322 |
| A1 | ISRUC-III | 1678 | 65.67% ± 1.08 pp | 0.6055 ± 0.0126 |
| A2 | MIT-BIH | 1331 | 33.66% ± 1.80 pp | 0.2459 ± 0.0286 |
| A2 | ISRUC-III | 1678 | 38.50% ± 3.98 pp | 0.3187 ± 0.0487 |
| A3 | MIT-BIH | 1331 | 59.85% ± 3.15 pp | 0.4499 ± 0.0975 |
| A3 | ISRUC-III | 1678 | 67.28% ± 1.82 pp | 0.6047 ± 0.0149 |
| A4 | MIT-BIH | 1331 | 59.08% ± 2.59 pp | 0.4249 ± 0.0508 |
| A4 | ISRUC-III | 1678 | 67.50% ± 1.07 pp | 0.6294 ± 0.0065 |
| A5 | MIT-BIH | 1331 | 61.81% ± 2.31 pp | 0.4889 ± 0.0656 |
| A5 | ISRUC-III | 1678 | 66.53% ± 0.58 pp | 0.6066 ± 0.0061 |
| A6 | MIT-BIH | 1331 | 60.88% ± 0.61 pp | 0.4796 ± 0.0645 |
| A6 | ISRUC-III | 1678 | 66.41% ± 1.55 pp | 0.6069 ± 0.0069 |

## Subject-specific outcomes

| ID | Subject | Epochs | Accuracy mean | Macro-F1 mean |
|---|---|---:|---:|---:|
| A0 | MIT-slp02 | 621 | 68.92% | 0.3796 |
| A0 | MIT-slp60 | 710 | 71.60% | 0.4623 |
| A0 | ISRUC-III-04 | 764 | 54.14% | 0.5130 |
| A0 | ISRUC-III-05 | 914 | 73.05% | 0.6121 |
| A1 | MIT-slp02 | 621 | 65.43% | 0.3524 |
| A1 | MIT-slp60 | 710 | 72.44% | 0.5125 |
| A1 | ISRUC-III-04 | 764 | 54.54% | 0.5339 |
| A1 | ISRUC-III-05 | 914 | 74.98% | 0.6552 |
| A2 | MIT-slp02 | 621 | 57.97% | 0.3224 |
| A2 | MIT-slp60 | 710 | 12.39% | 0.1075 |
| A2 | ISRUC-III-04 | 764 | 34.82% | 0.2904 |
| A2 | ISRUC-III-05 | 914 | 41.58% | 0.2875 |
| A3 | MIT-slp02 | 621 | 66.72% | 0.4099 |
| A3 | MIT-slp60 | 710 | 53.85% | 0.3739 |
| A3 | ISRUC-III-04 | 764 | 57.55% | 0.5462 |
| A3 | ISRUC-III-05 | 914 | 75.42% | 0.6493 |
| A4 | MIT-slp02 | 621 | 62.27% | 0.3657 |
| A4 | MIT-slp60 | 710 | 56.29% | 0.3729 |
| A4 | ISRUC-III-04 | 764 | 55.45% | 0.5396 |
| A4 | ISRUC-III-05 | 914 | 77.57% | 0.7189 |
| A5 | MIT-slp02 | 621 | 66.61% | 0.4255 |
| A5 | MIT-slp60 | 710 | 57.61% | 0.4036 |
| A5 | ISRUC-III-04 | 764 | 54.93% | 0.5280 |
| A5 | ISRUC-III-05 | 914 | 76.22% | 0.6699 |
| A6 | MIT-slp02 | 621 | 69.08% | 0.4264 |
| A6 | MIT-slp60 | 710 | 53.71% | 0.4084 |
| A6 | ISRUC-III-04 | 764 | 53.40% | 0.5184 |
| A6 | ISRUC-III-05 | 914 | 77.28% | 0.6699 |

## Completely unavailable HRV

| ID | Normal Accuracy | Missing Accuracy | Normal Macro-F1 | Missing Macro-F1 |
|---|---:|---:|---:|---:|
| A0 | 67.05% | 67.05% | 0.5991 | 0.5991 |
| A1 | 67.22% | 67.22% | 0.6171 | 0.6171 |
| A2 | 36.36% | 23.39% | 0.3002 | 0.0744 |
| A3 | 64.00% | 60.88% | 0.5789 | 0.5347 |
| A4 | 63.78% | 58.77% | 0.5900 | 0.5213 |
| A5 | 64.44% | 65.61% | 0.5866 | 0.5952 |
| A6 | 63.96% | 65.44% | 0.5834 | 0.5964 |

## Equal-probability three-model ensembles

Ensembles and mean individual-run scores are separate; no best seed is selected.

| ID | Ensemble Accuracy | Ensemble Macro-F1 |
|---|---:|---:|
| A0 | 69.26% | 0.6120 |
| A1 | 70.22% | 0.6407 |
| A2 | 36.66% | 0.3030 |
| A3 | 65.80% | 0.5936 |
| A4 | 64.31% | 0.5939 |
| A5 | 65.87% | 0.5980 |
| A6 | 65.10% | 0.5862 |

## Compute and data checks

| ID | Parameters | Context epochs | Mean training seconds |
|---|---:|---:|---:|
| A0 | 309,029 | 5 | 129.0 |
| A1 | 309,029 | 15 | 215.5 |
| A2 | 165,509 | 5 | 79.5 |
| A3 | 342,821 | 5 | 143.3 |
| A4 | 342,921 | 5 | 138.4 |
| A5 | 342,921 | 5 | 142.4 |
| A6 | 342,921 | 15 | 219.6 |

MIT beat detector: precision=0.9912, recall=0.9989, F1=0.9950; greedy 150 ms one-to-one tolerance.
This is a beat-detection check, distinct from sleep-stage scores. No independent ISRUC beat reference is used in this study.

## Interpretation boundaries

- One subject split, four test people and a reused holdout; no fresh external validation.
- Both databases occur in training; per-dataset scores are not unseen-dataset generalization.
- MIT test has seven N3 epochs. Subject-level fixed-class F1 includes zero for absent stages; inspect supports.
- Gate weights are descriptive, not physiological causality. Quality fields are not clinically validated indices.
- Historical RF, version 1 and version 2 use distinct study settings; do not combine their scores as one improvement estimate.

## Files and figures

[方法 / Methods](../../docs/METHODS_V2.md) · [复现 / Reproduction](../../docs/REPRODUCE_V2.md) · [图表 / Figures](../../docs/FIGURES.md)

Full per-run metrics: `run_metrics.csv`; all 56 prediction sets: `predictions.csv.gz`; complete confusion matrices and checkpoint hashes: `metrics.json`.
