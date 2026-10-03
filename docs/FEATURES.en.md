# HRV feature dictionary and implementation notes

[简体中文](FEATURES.zh-CN.md) · [README](../README.md) · [Pipeline](DATA_PIPELINE.en.md)

## 1. Definitions and version

The order below matches the 26 cached columns and the actual **SleepECG 0.5.9** implementation. [Versioned source](https://github.com/cbrnr/sleepecg/blob/v0.5.9/src/sleepecg/feature_extraction.py); project call: [prepare.py](../deep_learning/prepare.py). Its `hrv-time` group includes RR statistics, heart-rate features and Poincaré geometry. HRV frequency-domain LF/HF features are not computed here.

`NN` denotes ordered RR intervals in seconds, with implausible values retained as NaN. NN is the library's naming convention: beats are not independently adjudicated by experts as normal sinus beats. `D=diff(NN)` retains missing gaps. Means, medians and quantiles ignore NaN; `sd` denotes sample SD with `ddof=1`. Accepted RR range: [0.3,2.0] seconds.

| Column | Name | Actual calculation / meaning | Unit |
|---:|---|---|---|
| 1 | meanNN | mean(NN) | s |
| 2 | maxNN | max(NN) | s |
| 3 | minNN | min(NN) | s |
| 4 | rangeNN | maxNN−minNN | s |
| 5 | SDNN | sd(NN) | s |
| 6 | RMSSD | sqrt(mean(D²)) | s |
| 7 | SDSD | sd(D) | s |
| 8 | NN50 | count(abs(D)>0.05) | count |
| 9 | NN20 | count(abs(D)>0.02) | count |
| 10 | pNN50 | NN50/(M−1), where M is the library's padded RR row width within a record | fraction 0–1 |
| 11 | pNN20 | NN20/(M−1), same condition | fraction 0–1 |
| 12 | medianNN | median(NN) | s |
| 13 | madNN | median(abs(NN−medianNN)) | s |
| 14 | iqrNN | Q75(NN)−Q25(NN) | s |
| 15 | cvNN | SDNN/meanNN | dimensionless |
| 16 | cvSD | SDSD/mean(D), not RMSSD/meanNN | dimensionless; may be negative |
| 17 | meanHR | 60/meanNN, not mean(60/NN) | beats/min |
| 18 | maxHR | 60/minNN | beats/min |
| 19 | minHR | 60/maxNN | beats/min |
| 20 | stdHR | sd(60/NN) | beats/min |
| 21 | SD1 | SDSD/sqrt(2) | s |
| 22 | SD2 | sqrt(2×SDNN²−SD1²) | s |
| 23 | S | π×SD1×SD2 | s² |
| 24 | SD1_SD2_ratio | SD1/SD2 | dimensionless |
| 25 | CSI | SD2/SD1 | dimensionless |
| 26 | CVI | log10(16×SD1×SD2), with numeric inputs in seconds | unit-sensitive log index |

## 2. Conditions required to interpret these frozen results

**pNN denominator:** the library pads window RR rows to the record's largest width M using NaN. It compares `abs(D)>threshold` before taking a mean. NaN comparisons become False, so the denominator includes padded/invalid positions, not just valid adjacent differences. Thus values depend on maximum row width in the recording and are not directly the conventional percentage of valid adjacent differences above threshold. Stored values are fractions, without multiplication by 100.

M is derived from all statistical windows in the complete record, so later windows can affect the denominator. Local RR endpoint bounds do not make the entire HRV feature pipeline strictly causal. This record-level dependence and full-record ECG detection preclude treating these experiments as streaming validation.

Illustration: windows with 3 and 5 RR intervals share padded width M=5. If the shorter window has one valid difference above 50 ms, this implementation returns 1/4=0.25; a valid-pair denominator of two would give 1/2=0.5. This is an implementation example, not a measured study epoch.

**cvSD stability:** mean successive difference may approach zero. The feature can therefore be large, negative or nonfinite. Nonfinite values are filled; finite extremes are clipped after scaling. The name should not be assumed to imply a different textbook formula.

**CVI units:** SD1/SD2 are in seconds here. Changing to milliseconds shifts `log10(16×SD1×SD2)` upward by 6. Direct comparisons with millisecond-based definitions are inappropriate. CSI/CVI are library feature names, not autonomic or clinical diagnoses.

These conditions were identified while auditing documentation. Frozen features and training results have not been recomputed. A denominator/unit change requires a new protocol, cache and full controlled rerun; old scores must not be presented as corrected-feature results.

## 3. Missing flags and coverage fields

Cardiac columns are `[26 scaled features, 26 missing flags, 3 quality fields]`. NaN/inf are flagged before filling. Fill each feature using its training median; scale using training mean/SD with SD floor 1e−6; clip to [−8,8]. Missing flags are not standardized.

| Field | Definition | Model use |
|---|---|---|
| valid_rr_fraction | Plausible [0.3,2.0] s RR count / all detected RR endpoints in the available past-300-second window; zero when none | Cardiac input; multiplies the effective gate |
| feature_finite_fraction | Finite descriptors before filling / 26 | Cardiac input; multiplies the gate |
| history_fraction | min(current epoch end in seconds,300)/300 | Cardiac and learned-gate input; no direct multiplication |

These describe coverage/plausibility, not clinically validated ECG quality or comprehensive artifact/beat-error detection. Finite does not imply unbiased: padded pNN features can remain finite.

## 4. Recalculation and audit

Feature names are recorded in `cache/deep_learning/data_audit.json`; complete inputs are available in the [release](https://github.com/LRZer/EEG-HRV-Sleep-Staging/releases/tag/v2.0.0). RR plausibility and detector audits: [prepare_quality.py](../experiments/v2/prepare_quality.py), [ecg_detector_audit.csv](../results/v2/ecg_detector_audit.csv). Independent MIT beat annotations are used for audit, not sleep-classification inputs. No independent ISRUC beat reference is used.
