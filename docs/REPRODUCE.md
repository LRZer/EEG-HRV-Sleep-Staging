# 深度学习运行与复现

## 本轮实际环境

数据准备：Python 3.11.7、NumPy 1.26.4、SciPy 1.17.1、pandas 1.5.3、scikit-learn 1.2.2、WFDB 4.1.2、SleepECG 0.5.9，见根目录 `requirements.txt`。

训练及测试：现有 Conda `pytorch` 环境，Python 3.11.9、PyTorch 2.6.0+cu118、NumPy 1.26.4、SciPy 1.13.1、pandas 2.2.2、scikit-learn 1.5.0。实际记录在 `results/deep_learning/environment.json`。硬件为 RTX 4060 Laptop，8 GB 显存。

原始数据准备与训练分两个环境运行，因此不需要向现有 Conda 环境安装 WFDB / SleepECG。已经存在数据和缓存时，可以直接恢复仓库权重进行预测。

## 首次运行

以下命令均从仓库根目录运行。数据准备环境：

```powershell
python -m venv .venv-data
.venv-data\Scripts\python.exe -m pip install -r requirements.txt -r requirements-download.txt
.venv-data\Scripts\python.exe main.py prepare --data-dir data --cache cache/features.csv
.venv-data\Scripts\python.exe scripts/download_isruc_s3.py --destination data/isrucIII
.venv-data\Scripts\python.exe scripts/inspect_isruc_s3.py --destination data/isrucIII
.venv-data\Scripts\python.exe -m deep_learning.prepare --mit-dir data/slpdb --isruc-dir data/isrucIII
```

`main.py prepare` 同时下载 MIT 数据并生成历史基线特征；深度学习准备程序读取它下载的原始文件，独立构建波形与频谱缓存。ISRUC 文件来自作者公开 MEGA 分享，不需要账号。输入结构：

```text
data/slpdb/slp01a.dat, slp01a.hea, slp01a.st ...
data/isrucIII/1/1.rec, 1_1.txt, 1_2.txt, 1_1.xlsx, 1_2.xlsx
data/isrucIII/2/... 到 10/...
```

训练环境：可以复用已有 `pytorch`，或另建 Python 3.11 环境。若已有可用 CUDA 版 PyTorch，可以保留它，只补齐必要库。精确的 CUDA 轮子版本依赖本机驱动；本轮使用 cu118。

```powershell
conda activate pytorch
python -m deep_learning.checks
python -m deep_learning.train --results results/reproduction
python -m deep_learning.evaluate --results results/reproduction
```

`requirements-deep-learning.txt` 给出了对应基础库版本，含首次运行所需的 WFDB / SleepECG 和下载依赖。严格匹配原实验时，仍应使用上述单独的数据环境。GPU/依赖版本差异可能使浮点结果略有不同。

仓库中的 `results/deep_learning` 已完成评估，训练程序会拒绝在该目录重训；新复现实验使用 `results/reproduction`。评估程序也拒绝重复打分已存在 `metrics.json` 的结果目录。请保存原实验，勿依据这 4 位受试者的测试成绩反复调参。

## 使用已保存的网络权重

先用相同参数准备缓存，然后：

```powershell
python -m deep_learning.predict --kind fusion_cross --output results/restored_predictions.csv
```

默认为固定测试集，输出阶段名称及五类概率。可选 `--kind eeg_attention`、`sleep_transformer`、`eeg_sequence`、`fusion_concat`。`--split all` 包含训练样本，只适合查看预测，不代表独立测试成绩。

本次已验证 Cross-Attention 权重恢复：3,009 个测试窗口的标签与概率和原评估完全一致，见 `results/deep_learning/verification.json`。

预测模块接收本项目的对齐缓存；当前准备程序要求这两套数据库及已有标签。通用的未标注 EDF 输入、实时 ECG 检测、部署服务尚未实现。

## 文件对应关系

- `deep_learning/protocol.json`：准备和训练前固定的全部关键参数、测试名单。
- `cache/deep_learning/epochs.csv`：受试者、记录、窗口、标签、训练/测试归属。
- `cache/deep_learning/contexts.npy`：同一记录的当前及过去 5 个连续窗口索引，左侧空位为 -1。
- `cache/deep_learning/*.npz`：每条记录的 EEG、频谱、HRV、标签。缓存不上传。
- `results/deep_learning/preprocessing.npz`：仅训练集拟合的预处理统计。
- `results/deep_learning/*.pt`：五种网络各自固定第 40 轮的权重。
- `results/deep_learning/random_forest.joblib`：本轮在 22 位训练受试者上训练的对照模型，和历史全数据模型不同。
- `results/deep_learning/predictions.csv`：六种模型对 3,009 个独立测试窗口的类别和概率。

## 未来实验约束

这一轮测试已经完成。新结构或新超参数需要新的实验协议与独立评估安排，不能继续把原来的 4 位测试受试者当作选方案的数据。若继续保持不设验证集，应在训练前固定方案，保留新的最终评估数据，并清楚报告实验变化。
