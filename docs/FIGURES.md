# Figures / 图表

All figures are generated from preserved data and scores. Each is available as PNG, SVG and PDF. / 所有图表由保存的数据和指标生成，提供 PNG、SVG、PDF。

Sources: `python -m experiments.v2.figures`. Score bars use three individual seeds; confusion matrices and timelines explicitly use equal-probability ensembles.

## Model architecture and ablation components / 模型结构与对照组件

[PNG](../results/v2/figures/architecture.png) · [SVG](../results/v2/figures/architecture.svg) · [PDF](../results/v2/figures/architecture.pdf)

![Model architecture and ablation components](../results/v2/figures/architecture.png)

## Subjects, retained epochs and stage distributions / 受试者、窗口数及类别分布

[PNG](../results/v2/figures/data_overview.png) · [SVG](../results/v2/figures/data_overview.svg) · [PDF](../results/v2/figures/data_overview.pdf)

![Subjects, retained epochs and stage distributions](../results/v2/figures/data_overview.png)

## Seven variants, three individual seeds; mean ± sample SD / 七组模型三个独立种子的均值与样本标准差

[PNG](../results/v2/figures/ablation_results.png) · [SVG](../results/v2/figures/ablation_results.svg) · [PDF](../results/v2/figures/ablation_results.pdf)

![Seven variants, three individual seeds; mean ± sample SD](../results/v2/figures/ablation_results.png)

## 840 training-epoch logs; shaded bands are seed SD / 840 轮训练日志，阴影为种子标准差

[PNG](../results/v2/figures/training_curves.png) · [SVG](../results/v2/figures/training_curves.svg) · [PDF](../results/v2/figures/training_curves.pdf)

![840 training-epoch logs; shaded bands are seed SD](../results/v2/figures/training_curves.png)

## MIT and ISRUC results on the same fixed subject split / 同一受试者划分下两个数据库的成绩

[PNG](../results/v2/figures/dataset_results.png) · [SVG](../results/v2/figures/dataset_results.svg) · [PDF](../results/v2/figures/dataset_results.pdf)

![MIT and ISRUC results on the same fixed subject split](../results/v2/figures/dataset_results.png)

## Five-stage F1 and true class supports / 五类 F1 及真实样本数量

[PNG](../results/v2/figures/per_class_f1.png) · [SVG](../results/v2/figures/per_class_f1.svg) · [PDF](../results/v2/figures/per_class_f1.pdf)

![Five-stage F1 and true class supports](../results/v2/figures/per_class_f1.png)

## All seven equal-probability ensembles; counts and row percentages / 七个概率平均集成的混淆矩阵、计数及行比例

[PNG](../results/v2/figures/confusion_matrices.png) · [SVG](../results/v2/figures/confusion_matrices.svg) · [PDF](../results/v2/figures/confusion_matrices.pdf)

![All seven equal-probability ensembles; counts and row percentages](../results/v2/figures/confusion_matrices.png)

## Normal inputs versus completely unavailable HRV / 正常输入与 HRV 完全不可用的预设测试

[PNG](../results/v2/figures/missing_hrv_robustness.png) · [SVG](../results/v2/figures/missing_hrv_robustness.svg) · [PDF](../results/v2/figures/missing_hrv_robustness.pdf)

![Normal inputs versus completely unavailable HRV](../results/v2/figures/missing_hrv_robustness.png)

## Four subject outcomes with fixed-five-class F1 / 四位受试者的准确率和固定五类 F1

[PNG](../results/v2/figures/subject_results.png) · [SVG](../results/v2/figures/subject_results.svg) · [PDF](../results/v2/figures/subject_results.pdf)

![Four subject outcomes with fixed-five-class F1](../results/v2/figures/subject_results.png)

## Effective cardiac weights by stage and dataset / 按阶段和数据库展示有效心脏权重

[PNG](../results/v2/figures/gate_analysis.png) · [SVG](../results/v2/figures/gate_analysis.svg) · [PDF](../results/v2/figures/gate_analysis.pdf)

![Effective cardiac weights by stage and dataset](../results/v2/figures/gate_analysis.png)

## All held-out recordings: expert, A0 ensemble and A6 ensemble / 全部测试记录的专家标注、A0 与 A6 集成时间图

[PNG](../results/v2/figures/sleep_timelines.png) · [SVG](../results/v2/figures/sleep_timelines.svg) · [PDF](../results/v2/figures/sleep_timelines.pdf)

![All held-out recordings: expert, A0 ensemble and A6 ensemble](../results/v2/figures/sleep_timelines.png)

## Fixed random training-only ECG segments and detected beats / 固定随机训练 ECG 片段与检测心跳

[PNG](../results/v2/figures/ecg_review.png) · [SVG](../results/v2/figures/ecg_review.svg) · [PDF](../results/v2/figures/ecg_review.pdf)

![Fixed random training-only ECG segments and detected beats](../results/v2/figures/ecg_review.png)

## MIT beat detection against provider annotations; 150 ms tolerance / MIT 心跳检测与官方逐拍标注对比，容差 150 ms

[PNG](../results/v2/figures/ecg_detection.png) · [SVG](../results/v2/figures/ecg_detection.svg) · [PDF](../results/v2/figures/ecg_detection.pdf)

![MIT beat detection against provider annotations; 150 ms tolerance](../results/v2/figures/ecg_detection.png)
