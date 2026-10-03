# Results, errors and evidence boundaries

[简体中文](RESULTS_ANALYSIS.zh-CN.md) · [README](../README.md) · [Full numeric report](../results/v2/report.en.md) · [Breakdowns and cases](tables/result_breakdown.md)

## 1. Three distinct score types

1. **Individual seed:** one model trained from scratch on the fixed split.
2. **Mean ± SD:** mean and sample SD of three individual-run scores, the primary controlled-comparison summary.
3. **Probability ensemble:** average the three models' five probabilities per epoch, then classify and score. This is not average Accuracy/F1.

A1 has the highest individual-run means: Accuracy 67.22%, Macro-F1 0.6171; its ensemble scores 70.22%/0.6407. A6 means are 63.96%/0.5834, ensemble 65.10%/0.5862. Comparing one variant's mean to another variant's ensemble does not estimate an improvement.

## 2. What the comparisons support

| Contrast | Observation | Supported scope |
|---|---|---|
| A1−A0 | Accuracy +0.17 pp, Macro-F1 +0.0181 | Longer EEG context has higher means on this split |
| A3–A6 versus A0 | All fusion variants have lower mean Accuracy/Macro-F1 | This input/fusion/training combination does not improve overall scores |
| A4 versus A0 | REM F1 0.3688 versus 0.2521; N1 0.4263 versus 0.5740 | Stage-specific improvement is not overall improvement |
| A5−A4, missing HRV | Accuracy +6.85 pp, Macro-F1 +0.0739 | The dropout variant has higher means in this missing-modality test |
| A6−A5, normal | Accuracy −0.48 pp, Macro-F1 −0.0033 | Longer fusion context does not improve means here |

A4−A3 changes several fusion components and does not isolate quality multiplication. A5/A6 missing-HRV scores also exceed their normal-input scores. These observations do not identify noise, overfitting, feature formulas, source differences or gate design as a proven cause.

## 3. Reading stage, source and subject figures

![Stage F1 and supports](../results/v2/figures/per_class_f1.png)

**Stage figure:** read F1 alongside true supports: Wake 882, N1 583, N2 875, N3 361, REM 308. Class-weighted training does not ensure equal class performance.

**Confusion matrices:** rows are true stages, columns predictions. The diagonal is correct; off-diagonal cells identify error destinations. Read counts and row percentages together. The largest A1/A6 ensemble error destination per class is in the [breakdown table](tables/result_breakdown.md); all seven matrices are in the [gallery](FIGURES.md).

**Source figures:** both sources appear in training; these scores describe unseen people within each source, not external cross-dataset validation. MIT test contains only seven N3 epochs, so a few errors can substantially change that stage's result.

**Subject figures:** recording lengths and stage distributions differ among four people. Pooled epoch scores are not the unweighted mean of four subject scores. Subject Macro-F1 fixes five classes and assigns zero F1 to absent classes. [Generated tables](tables/result_breakdown.md) preserve A0/A1/A6 subject/source means and supports; the full report covers all variants.

## 4. Fixed correct/error cases

![Fixed A1/A6 cases](figures/prediction_cases.en.png)

Categories: both correct, only A1 correct, only A6 correct, both wrong. The deterministic rule uses normal-input three-model ensembles, sorts record/epoch, and takes the first case with `epoch≥14` in each category. **Cases are not selected by confidence and are not population-representative evidence.**

Green shading indicates the expert stage; bars show five probabilities. Titles identify original record, zero-based epoch, expert label and current A6 gate. [Case CSV](tables/prediction_cases.csv) includes original probabilities, all three coverage fields and labels. Higher gate/probability does not establish a physiological mechanism for an error or success. Softmax outputs have not been calibrated as reliability estimates.

Full sleep timelines cover all five test recordings, rather than only successful excerpts. Expert labels are the evaluation reference, not independently re-adjudicated clinical truth.

## 5. Why this does not establish that HRV is ineffective

The conclusion concerns **this feature/data/model/training combination**. Conditions include four test people, reused holdout, different EEG montages/scoring, overlapping HRV histories, padded pNN denominators and cvSD stability, and only 26 RR/heart-rate/geometry descriptors. Each requires a separate test before being treated as a cause.

| Unperformed experiment | Question addressed |
|---|---|
| Correct denominator/unit conventions under a new protocol; rebuild and rerun all controls | Do feature implementation conditions affect the conclusion? |
| Keep residual fusion fixed and toggle quality multiplication | Independent quality-constraint contribution |
| Fix features/training and vary HRV history length | Appropriate HRV timescale |
| Independent final evaluation on new subjects/source | Generalization beyond study/design participants |
| Streaming ECG, unannotated-file inference and probability calibration | Deployment boundary, detection latency and reliability |

These are unperformed work, not completed achievements. This documentation revision does not retrain or change frozen scores.

## 6. Preservation and verification

Preserved artifacts: 21 final checkpoints, 840 training-log rows, 56×3,009=168,504 predictions, class/person/source metrics, confusion matrices and checkpoint hashes. The existing [verification record](../results/v2/verification.json) checks metrics, subject separation, weights and A6 restoration; recorded restoration has identical labels and maximum probability difference zero. This establishes artifact restoration, not additional generalization evidence.
