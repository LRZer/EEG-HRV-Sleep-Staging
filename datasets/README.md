# Research data / 研究数据

[Release v2.0.0](https://github.com/LRZer/EEG-HRV-Sleep-Staging/releases/tag/v2.0.0)

| Archive | Size | Contents / 内容 |
|---|---:|---|
| mit-bih-psg-1.0.0.zip | 554.8 MiB | Original MIT files with attribution / MIT 原始实验文件 |
| prepared-eeg-hrv-v2.zip | 368.1 MiB | All 18,770 prepared EEG-HRV epochs and contexts / 全部对齐输入和上下文 |

```powershell
python scripts/download_release_data.py --kind prepared
# Optional MIT original files / 可选 MIT 原始文件
python scripts/download_release_data.py --kind mit
```

Archives use 48 MiB upload parts; all part, whole-archive and extracted-file hashes are verified. / 压缩包以 48 MiB 分块提供，逐层校验文件。

ISRUC original REC: [provider download](https://sleeptight.isr.uc.pt/?page_id=48). Exact hashes: [isruc-original-files.json](isruc-original-files.json). Download and inspect using `scripts/download_isruc_s3.py` and `scripts/inspect_isruc_s3.py`. / ISRUC 原始文件从作者来源获取，下载与检查工具已包含。

[Attribution and data conditions / 归属与使用条件](DATA_NOTICE.txt) · [Full reconstruction / 完整重建](../docs/REPRODUCE_V2.md)

### Prepared schema / 对齐输入结构

Per-record `.npz`: `eeg` float32 (n,3000), `spectra` float32 (n,29,89), `hrv` float32 (n,26), `eeg_features` float32 (n,7), `labels` integer, `epoch` integer. These arrays are before train-fitted scaling; models use saved preprocessing.

`epochs.csv`: record, subject, source, epoch, stage, record_row, split. `cache/v2/quality.npy`: (18770,3), matching this exact row order. `cache/v2/contexts.npy`: (18770,15), -1 for unavailable history. Stages 0..4 map to Wake/N1/N2/N3/REM.

Rows are aligned across modalities and labels. No raw data is silently added to training; augmented examples remain the same subjects. / 模态与标签按窗口对齐，增强不增加独立受试者。
