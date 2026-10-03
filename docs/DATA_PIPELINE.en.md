# From raw signals to one prediction

[简体中文](DATA_PIPELINE.zh-CN.md) · [README](../README.md) · [Feature dictionary](FEATURES.en.md)

## 1. Signals and labels

Pipeline: read physical EEG/ECG and expert annotations→30-second epochs→five-class mapping→EEG filtering/spectra→ECG beats/HRV→subject split→training-fitted preprocessing→historical contexts→five probabilities.

![Original signals to predictions](figures/data_pipeline.en.png)

MIT-BIH contributes 16 subjects, 18 recordings and 10,181 retained epochs. ISRUC III contributes 10 subjects and 10 recordings: 8,889 original labels minus a fixed final 30 epochs per recording gives 8,589 retained epochs. The subject split has 15,761 training and 3,009 test epochs, with no validation set. Source manifests, `epoch_manifest.csv` and the frozen protocol preserve original files, epoch indices and exclusions.

ISRUC uses C4-A1 EEG, X2 ECG (EDF transducer EKG_Channel), and scorer 1. MIT uses the first EEG channel in each record with varying montages. MIT R&K 3/4 merge into N3; ISRUC 0/1/2/3/5 map to Wake/N1/N2/N3/REM. Unknown labels are excluded and break context, not treated as another output class.

## 2. EEG waveform to spectrum

| Step | Exact settings |
|---|---|
| Units | Read physical values and convert to µV |
| Basic check | Finite values; SD≥0.01 µV; absolute amplitude≤2,000 µV |
| Filtering | Independent 30-second epochs, fourth-order Butterworth 0.3–35 Hz, `sosfiltfilt` |
| Resampling | `resample_poly` to 100 Hz, 3,000 samples per epoch |
| STFT | 200-sample/2-second window, 100-sample/1-second overlap, FFT 256; `boundary=None`, `padded=False` |
| Frequencies | 0.3–35 Hz, 89 bins; FFT spacing 100/256=0.390625 Hz |
| Transform | `log10(max(abs(Z)², 1e-8))`; 29 frames×89 bins |
| Training scaling | Frequency-wise mean/SD over training epochs and frames; clip scaled values to [−8,8] |

This is STFT log squared magnitude, not a defined µV²/Hz spectral-density measurement. Basic amplitude rules removed no labeled epoch here and are not complete artifact detection. Epoch-local processing reads no later EEG epoch; zero-phase filtering does use later samples within that epoch.

![Actual EEG, spectrum, ECG and RR](figures/signal_walkthrough.en.png)

The EEG/spectrum pair comes from one fixed training epoch. ECG/RR come from a separate saved training segment, with their own record/time labels; the figure does not imply synchronization between the two examples. Ten-second ECG illustrates RR only; model HRV statistics use up to 300 seconds. Display scaling is not training preprocessing.

## 3. ECG→RR→HRV timing

SleepECG 0.5.9 detects beats on complete offline ECG. For beat times `b₀,b₁,…`, RR=`bᵢ−bᵢ₋₁` with timestamp at **endpoint bᵢ**. RR outside [0.3,2.0] s become NaN without deleting sequence positions.

For epoch start `t` and end `T=t+30`, extraction uses lookback 270 s and lookforward 30 s, equivalent to RR endpoints in **[T−300,T)**. No endpoint after T enters that feature vector. The first included RR may start before the left boundary because grouping uses endpoints. The complete-record detector itself has not been validated as strictly causal streaming processing.

Nonfinite values among 26 features→training median fill→training mean/SD scaling (SD floor 1e−6)→clip [−8,8]. Generate 26 missing flags before filling, then append three coverage fields for 55 values. [Feature dictionary](FEATURES.en.md)

Endpoint bounds describe selected local RR values. Library pNN denominators additionally depend on the complete record's maximum padded width, potentially affected by later windows; local bounds therefore do not establish strict streaming causality of all features.

## 4. Historical context, padding and effective history

![Alignment and statistical history](figures/time_alignment.en.png)

- `L=5`: EEG covers 2.5 minutes including the current epoch. The union of 5-minute HRV endpoint windows covers `300+(5−1)×30=420` s, or 7 minutes.
- `L=15`: EEG covers 7.5 minutes; the union covers `300+(15−1)×30=720` s, or 12 minutes.
- These are full-history statistical spans. Early epochs have less history, and these bounds do not describe all data read by the offline ECG detector.
- Context stays within one record/split and consecutive known-label epochs. Unknown-label gaps restart context accumulation. Short histories are left-padded with index −1 and masked as temporal-attention keys.
- `history_fraction=min(T,300)/300` measures time available from recording start, not valid-beat coverage. Unknown sleep-label gaps do not reset it.

## 5. Cached arrays, fitting statistics and outputs

| File | Fields / role |
|---|---|
| `cache/deep_learning/<record>.npz` | EEG `(n,3000)`, spectrum `(n,29,89)`, HRV `(n,26)`, labels, original epoch; also seven engineered EEG features unused in version 2 |
| `epochs.csv` / `results/v2/epoch_manifest.csv` | record, subject, source, epoch, stage, record_row, split; exact row alignment |
| `cache/v2/quality.npy` | 18,770×3, same row order as the manifest |
| `cache/v2/contexts.npy` | 18,770×15 history indices; five-epoch variants use the last five columns |
| `results/v2/preprocessing.npz` | Training-only filling/scaling statistics |
| `results/v2/predictions.csv.gz` | Labels, prediction, probabilities, current gate and coverage fields; 56 groups×3,009 rows |

Augmentation changes inputs without increasing independent subject count. Adjacent epochs and overlapping HRV histories are correlated. Outputs predict expert stage labels, not insomnia, apnea or a direct sleep-quality measurement.

Code: [signal preparation](../deep_learning/prepare.py) · [quality/context preparation](../experiments/v2/prepare_quality.py) · [loading/preprocessing](../experiments/v2/data.py) · [inference](../experiments/v2/predict.py).
