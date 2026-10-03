# Reproduce version 2 / 复现第二轮实验

## 1. Download ready-to-use inputs / 下载实验输入

Python 3.11 is recommended. For the recorded training environment use PyTorch 2.6.0+cu118, NumPy 1.26.4, SciPy 1.13.1, pandas 2.2.2, scikit-learn 1.5.0 and Matplotlib 3.9.0. `requirements-deep-learning.txt` lists the corresponding base packages; the exact GPU build is recorded in `results/v2/environment.json`. Keep an already working CUDA environment when available.

建议 Python 3.11。实际训练环境版本保存在 `results/v2/environment.json`；已有可用 CUDA 版 PyTorch 时可以复用，不需要重复安装整个环境。

Install any missing dependencies in the activated environment / 在已激活环境中补齐依赖：

```powershell
python -m pip install -r requirements-deep-learning.txt
```

```powershell
conda activate pytorch
python scripts/download_release_data.py --kind prepared
python -m experiments.v2.predict --variant A6 --seed ensemble --output results/restored_v2_predictions.csv
python -m experiments.v2.verify
```

The release contains aligned inputs for all 18,770 epochs: per-record EEG waveforms, 29×89 log-power spectra, 26 HRV descriptors, labels, subject split, 15-window indices and 3 quality fields. Assets are split into 48 MiB parts. The downloader verifies part, archive and extracted-file SHA-256, reuses verified parts, and writes caches beneath the repository root. Full prepared archive size is recorded in `datasets/release_assets.json`. Reserve approximately 2 GB for parts, reassembly and extraction.

Release 包含全部 18,770 个窗口的对齐输入、标签、受试者名单、序列索引及质量字段。下载脚本自动拼接 48 MiB 分块，并逐层核对 SHA-256。准备约 2 GB 空间用于分块、完整压缩包和解压文件。输入缓存位于 `cache/deep_learning` 与 `cache/v2`。

The predictor restores preserved weights and training-fitted preprocessing. Default A6 ensemble averages the three predeclared seeds. It operates on the labeled, aligned study cache; it is not an arbitrary unannotated EDF reader or streaming deployment service. Prediction labels and probabilities are restored for the fixed 3,009 test epochs.

The verifier recalculates all 56 score groups from saved predictions, checks all checkpoint hashes and subject separation, and compares restored A6 ensemble labels and probabilities with the published predictions.

预测命令恢复已保存模型和训练集预处理统计。默认 A6 集成为三个预定种子的概率平均；目前接口使用本项目对齐缓存，不是通用未标注 EDF 或实时服务。

验证命令从保存的预测重新核算全部 56 组成绩，核对权重及受试者隔离，并比较恢复模型与已发表预测的标签和概率。

## 2. Repeat the frozen training / 重复训练

Do not overwrite the scored `results/v2` experiment. Use an independent output directory and keep the original outputs.

```powershell
python -m experiments.v2.checks
python -m experiments.v2.train --results results/reproduction-v2
python -m experiments.v2.evaluate --results results/reproduction-v2
python -m experiments.v2.figures --results results/reproduction-v2
python -m experiments.v2.predict --results results/reproduction-v2 --variant A6 --seed ensemble --output results/reproduction-v2/restored.csv
```

All seven variants and all three seeds train before the common evaluation. Training resumes only completed final checkpoints with a matching protocol hash; intermediate interrupted runs restart from scratch. Scoring refuses an already evaluated results directory. The same historical holdout remains exploratory even after a fresh repeat of training. Hardware and library differences can change floating-point outputs.

先完成七组、三个种子的全部训练，再统一评分。中断时仅复用协议一致的完整第 40 轮检查点；未完成的单次训练从头开始。重复训练不改变本轮复用历史留出集的探索性质。

## 3. Rebuild from original signals / 从原始信号重建

MIT original files used in the study are also packaged with ODC-By attribution in the release. ISRUC original files are obtained directly from the provider; the exact 50 source-file hashes are in `datasets/isruc-original-files.json`. The provider offers research access and requests citation but does not state a separate permissive redistribution license on its download page. The release provides transformed study inputs, while the original REC files remain available from the provider.

MIT 原始实验文件随 Release 提供并保留 ODC-By 归属说明。ISRUC 原始 REC 文件从作者公开分享获取，50 个文件的校验清单已保存；本项目提供变换后的实验输入及原始下载工具，不另行镜像原始 REC。数据权利及使用条件仍属于原作者。

Data-preparation environment uses the versions in `requirements.txt` plus download dependencies in `requirements-download.txt`. Run from the repository root:

```powershell
python -m venv .venv-data
.venv-data\Scripts\python.exe -m pip install -r requirements.txt -r requirements-download.txt
.venv-data\Scripts\python.exe scripts/download_release_data.py --kind mit
.venv-data\Scripts\python.exe scripts/download_isruc_s3.py --destination data/isrucIII
.venv-data\Scripts\python.exe scripts/inspect_isruc_s3.py --destination data/isrucIII
.venv-data\Scripts\python.exe -m deep_learning.prepare --mit-dir data/slpdb --isruc-dir data/isrucIII
.venv-data\Scripts\python.exe -m experiments.v2.prepare_quality --mit-dir data/slpdb --isruc-dir data/isrucIII
```

MIT can alternatively be downloaded with `main.py prepare --data-dir data --cache cache/features.csv`. Quality preparation checks or downloads only the small `.ecg` beat annotation files using the official checksums. ECG detection is deterministic for these recorded inputs and library versions. The two environments share the prepared files, not fitted statistics from the test set.

MIT 也可通过原有 `main.py prepare` 下载。质量准备程序仅补齐小型逐拍标注文件，并按官方清单校验，不增加新的受试者或大型数据库。数据准备与训练环境通过缓存衔接。

## 4. Output files / 输出文件

| File | Contents / 内容 |
|---|---|
| `protocol.json` | Frozen variants, seeds, training settings and scoring rules / 固定协议 |
| `checkpoints/A*_seed*.pt` | 21 final checkpoints with protocol hashes / 21 份最终权重 |
| `A*_seed*_history.csv` | 840 epoch logs / 840 轮训练日志 |
| `preprocessing.npz` | Training-only filling and scaling statistics / 训练集预处理统计 |
| `run_metrics.csv`, `comparison.csv` | Individual, ensemble and three-seed aggregate scores / 单次、集成、均值指标 |
| `per_class.csv`, `per_subject.csv`, `per_dataset.csv` | Stage, subject and source breakdowns / 分阶段、受试者、数据库指标 |
| `predictions.csv.gz` | 56 prediction sets: 7 variants × 4 seed/ensemble entries × 2 conditions / 56 组逐窗预测 |
| `metrics.json` | Complete confusion matrices and checkpoint hashes / 完整指标和权重校验 |
| `figures/` | 13 figure sets, each PNG + SVG + PDF / 13 组图表 |
| `verification.json` | Preserved-artifact and restoration checks / 产物及恢复检查 |

See [methods](METHODS_V2.md) for interpretation and [datasets notice](../datasets/DATA_NOTICE.txt) for attribution. Software licensing does not relicense datasets or third-party dependencies.
