"""Write bilingual reports, data guide and figure catalogue from frozen results."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
import torch
from .documentation import generate,write,CAPTIONS,CN,RELEASE
from .figures import COLORS,LABELS,STAGES

PROJECT=Path(__file__).resolve().parents[2]

READING={
    "architecture":("Overview of A6; detailed tensor shapes and A1 comparison are in the technical diagrams below.","A6 的模块概览；详细尺寸与 A1 对照见下方技术图。"),
    "data_overview":("Read subject counts separately from epoch counts; windows from one person are correlated.","区分人数与窗口数；同一人的相邻窗口不是独立受试者。"),
    "ablation_results":("Points are individual seeds; bars/error bars summarize means/sample SD, not ensembles or confidence intervals.","点为单个种子；条形和误差线为均值／样本标准差，不是集成或人群置信区间。"),
    "training_curves":("Training loss and training accuracy only; no validation curve or test-guided checkpoint selection.","只有训练损失和训练准确率；无验证曲线或按测试成绩选择权重。"),
    "dataset_results":("Both sources occur in training; these are source-specific held-out subjects, not external source validation.","两个库都参与训练；这是各库留出受试者成绩，不是外部跨库验证。"),
    "per_class_f1":("Read F1 alongside support; a change in REM alone does not establish overall improvement.","结合支持量阅读 F1；仅 REM 改善不等于整体提高。"),
    "confusion_matrices":("Rows are expert stages, columns predictions; off-diagonal cells show errors. These use probability ensembles.","行为专家阶段、列为预测；离对角线是错误去向。此图使用概率平均集成。"),
    "missing_hrv_robustness":("The missing condition removes all HRV; compare the same variant across conditions, then compare dropout variants.","缺失条件移除全部 HRV；先比较同一模型两种条件，再比较有／无模态丢弃。"),
    "subject_results":("Pooled scores are not an unweighted average of people. Fixed-five-class F1 assigns zero to absent stages.","窗口汇总不是受试者简单平均；逐人固定五类 F1 对不存在阶段计 0。"),
    "gate_analysis":("Effective scalar weights describe model behavior, not physiological causality or validated signal quality.","有效标量权重描述模型行为，不证明生理因果或经过验证的信号质量。"),
    "sleep_timelines":("All five test records are shown. Expert labels are references; trace disagreements identify temporal errors.","展示全部五条测试记录；专家标签是参考，曲线偏离标出错误发生的时间。"),
    "ecg_review":("Display-normalized fixed training segments; beat markers are detector outputs, not independent annotations.","固定训练片段仅作显示标准化；心跳标记是检测输出，不是独立专家标注。"),
    "ecg_detection":("MIT beat matching at 150 ms; truncated axes are labeled. Beat F1 is separate from sleep-stage F1.","MIT 逐拍匹配容差 150 ms，截断坐标已注明；心跳 F1 与睡眠阶段 F1 不同。")}

TECHNICAL={
    "data_pipeline":("Signals to predictions","信号到预测","Follow EEG, ECG and expert labels to aligned inputs; labels remain targets and scaling is training-only.","按 EEG、ECG 和专家标签追踪至对齐输入；标签是目标，预处理统计仅由训练集拟合。"),
    "model_architecture":("A1 and A6 layer structure","A1 与 A6 分层结构","Follow the dimensions from one spectrum to one epoch vector, then to the current-stage output.","从频谱尺寸读到窗口向量，再读到当前阶段输出。"),
    "gated_fusion":("Residual gate and fallback","残差门控与回退","The EEG path is retained; q1/q2 bound one cardiac weight per epoch.","保留 EEG 主路径，q1／q2 约束每窗口一个心脏权重。"),
    "time_alignment":("EEG context and nested HRV histories","EEG 上下文与嵌套 HRV 历史","5/15 EEG epochs cover 2.5/7.5 min; their RR-endpoint histories jointly cover 7/12 min.","5／15 个 EEG 窗口覆盖 2.5／7.5 分钟，RR 终点历史联合覆盖 7／12 分钟。"),
    "signal_walkthrough":("Actual signal representations","真实信号表示","EEG/spectrum are one training epoch; ECG/RR are a separate training example with explicit provenance.","EEG／频谱为同一训练窗口；ECG／RR 为另一个注明来源的训练片段。"),
    "prediction_cases":("Fixed probability examples","固定概率案例","First sorted example per outcome category, not confidence-selected or representative of population frequency.","每类取排序后的首例，不按置信度选取，也不代表总体频率。")}


def reports(results):
    generate(results)
    summary=pd.read_csv(results/"comparison.csv")
    normal=summary[summary.condition=="normal"].set_index("variant")
    missing=summary[summary.condition=="hrv_missing"].set_index("variant")
    run=pd.read_csv(results/"run_metrics.csv",dtype={"seed":str})
    ensemble=run[(run.condition=="normal")&(run.seed=="ensemble")].set_index("variant")
    per_class=pd.read_csv(results/"per_class.csv",dtype={"seed":str})
    domain=pd.read_csv(results/"per_dataset.csv",dtype={"seed":str})
    people=pd.read_csv(results/"per_subject.csv",dtype={"seed":str})
    normal_domain=domain[(domain.condition=="normal")&(domain.seed!="ensemble")]
    normal_people=people[(people.condition=="normal")&(people.seed!="ensemble")]
    compute=[]
    for v in COLORS:
        runs=[]
        for s in [42,123,2026]:
            checkpoint=torch.load(results/"checkpoints"/f"{v}_seed{s}.pt",map_location="cpu",weights_only=False)
            runs.append(checkpoint["training_seconds"])
        compute.append({"variant":v,"parameters":checkpoint["parameters"],"context_epochs":checkpoint["protocol"]["variants"][v]["context"],
                        "mean_training_seconds":sum(runs)/3,"total_training_seconds":sum(runs)})
    pd.DataFrame(compute).to_csv(results/"compute.csv",index=False)
    qc=pd.read_csv(results/"ecg_detector_audit.csv").dropna(subset=["beat_f1"])
    precision=qc.matched_beats.sum()/qc.detected_beats.sum()
    recall=qc.matched_beats.sum()/qc.reference_beats.sum()
    f1=2*precision*recall/(precision+recall)
    protocol_hash=hashlib.sha256((results/"protocol.json").read_bytes()).hexdigest()
    for lang in ["en","zh-CN"]:
        cn=lang=="zh-CN"
        lines=["# 第二轮受控实验报告" if cn else "# Version 2 controlled experiment report","",
            "22 人训练、4 人测试、18,770 个窗口。七组各三个种子，每次 40 轮，共 21 次训练、840 轮。" if cn else
            "22 training subjects, 4 test subjects, 18,770 epochs. Seven variants × three seeds × 40 epochs: 21 runs and 840 training epochs.","",
            "不设验证集，统一保存最后一轮。测试受试者已在第一轮评估，以下属于复用留出集的探索性结果。" if cn else
            "No validation set; final epoch checkpoints only. The same subjects were evaluated in version 1, so these are exploratory reused-holdout outcomes.","",
            f"Protocol SHA-256: `{protocol_hash}`.","",
            "## 三个单次模型的成绩" if cn else "## Three individual runs per variant","",
            "表中 ± 为三个种子的样本标准差，不是人群置信区间。" if cn else
            "± is sample SD across three initialization seeds, not a subject-population confidence interval.","",
            "| ID | Variant | Accuracy | Macro-F1 | Balanced Acc | Kappa |","|---|---|---:|---:|---:|---:|"]
        for v in COLORS:
            r=normal.loc[v]
            lines.append(f"| {v} | {CN[v] if cn else LABELS[v]} | {r.accuracy_mean:.2%} ± {r.accuracy_std*100:.2f} pp | {r.macro_f1_mean:.4f} ± {r.macro_f1_std:.4f} | {r.balanced_accuracy_mean:.4f} | {r.kappa_mean:.4f} |")
        lines += ["","## 预定对照差值" if cn else "## Predeclared pairwise contrasts","",
            "| Contrast | Changed factor | Δ Accuracy (pp) | Δ Macro-F1 |","|---|---|---:|---:|"]
        for a,b,factor in [("A1","A0","EEG context 5→15"),("A3","A0","Add HRV via concat"),("A4","A3","Concat→gated residual"),
                           ("A5","A4","20% HRV dropout"),("A6","A5","Gated context 5→15"),("A6","A0","Full model vs EEG control")]:
            lines.append(f"| {a} − {b} | {factor} | {(normal.loc[a,'accuracy_mean']-normal.loc[b,'accuracy_mean'])*100:+.2f} | {normal.loc[a,'macro_f1_mean']-normal.loc[b,'macro_f1_mean']:+.4f} |")
        stage_means=per_class[(per_class.condition=="normal")&(per_class.seed!="ensemble")].groupby(["variant","stage"]).f1.mean()
        lines += ["","## 观察到的结果" if cn else "## Observed outcomes","",
            ("整体融合差值见上表，本轮未观察到融合组同时超过脑电对照的 Accuracy 与 Macro-F1。" if cn else
             "The table above gives the overall fusion contrasts; this run did not show a multimodal variant exceeding the EEG control in both Accuracy and Macro-F1."),
            (f"A4 相对 A0，REM F1 为 {stage_means['A4','REM']:.4f}／{stage_means['A0','REM']:.4f}，N1 F1 为 {stage_means['A4','N1']:.4f}／{stage_means['A0','N1']:.4f}；不同阶段的变化方向不同。" if cn else
             f"A4 versus A0: REM F1 {stage_means['A4','REM']:.4f} versus {stage_means['A0','REM']:.4f}, but N1 F1 {stage_means['A4','N1']:.4f} versus {stage_means['A0','N1']:.4f}; stage-specific changes differ in direction."),
            (f"完全缺失 HRV 时，A5 相对 A4 的平均 Accuracy 差值为 {(missing.loc['A5','accuracy_mean']-missing.loc['A4','accuracy_mean'])*100:+.2f} 个百分点，Macro-F1 差值为 {missing.loc['A5','macro_f1_mean']-missing.loc['A4','macro_f1_mean']:+.4f}。这是本轮均值对比，未作人群显著性结论。" if cn else
             f"With HRV unavailable, A5 versus A4 changes mean Accuracy by {(missing.loc['A5','accuracy_mean']-missing.loc['A4','accuracy_mean'])*100:+.2f} pp and Macro-F1 by {missing.loc['A5','macro_f1_mean']-missing.loc['A4','macro_f1_mean']:+.4f}. These are observed mean differences, without a population-significance claim.")]
        lines += ["", "A4−A3 同时更换残差融合与质量约束，不能单独证明质量因子的贡献。特征实现条件及错误案例见配套技术文档。" if cn else
                  "A4−A3 changes residual fusion and quality constraints together, so quality-factor benefit is not isolated. Companion technical documents describe feature implementation conditions and error cases."]
        lines += ["","## 分阶段指标" if cn else "## Per-stage metrics","",
            "| ID | Stage | Precision mean | Recall mean | F1 mean | F1 SD | Support |","|---|---|---:|---:|---:|---:|---:|"]
        selected=per_class[(per_class.condition=="normal")&(per_class.seed!="ensemble")]
        for v in COLORS:
            for stage in STAGES:
                rows=selected[(selected.variant==v)&(selected.stage==stage)]
                lines.append(f"| {v} | {stage} | {rows.precision.mean():.4f} | {rows.recall.mean():.4f} | {rows.f1.mean():.4f} | {rows.f1.std():.4f} | {int(rows.support.iloc[0])} |")
        lines += ["","## 分数据库结果" if cn else "## Dataset-specific outcomes","",
            "| ID | Dataset | Epochs | Accuracy mean ± SD | Macro-F1 mean ± SD |","|---|---|---:|---:|---:|"]
        for (v,source),rows in normal_domain.groupby(["variant","source"],sort=False):
            lines.append(f"| {v} | {source} | {int(rows.epochs.iloc[0])} | {rows.accuracy.mean():.2%} ± {rows.accuracy.std()*100:.2f} pp | {rows.macro_f1.mean():.4f} ± {rows.macro_f1.std():.4f} |")
        lines += ["","## 分受试者结果" if cn else "## Subject-specific outcomes","",
            "| ID | Subject | Epochs | Accuracy mean | Macro-F1 mean |","|---|---|---:|---:|---:|"]
        for (v,subject),rows in normal_people.groupby(["variant","subject"],sort=False):
            lines.append(f"| {v} | {subject} | {int(rows.epochs.iloc[0])} | {rows.accuracy.mean():.2%} | {rows.macro_f1.mean():.4f} |")
        lines += ["","## 完全缺失 HRV" if cn else "## Completely unavailable HRV","",
            "| ID | Normal Accuracy | Missing Accuracy | Normal Macro-F1 | Missing Macro-F1 |","|---|---:|---:|---:|---:|"]
        for v in COLORS:
            lines.append(f"| {v} | {normal.loc[v,'accuracy_mean']:.2%} | {missing.loc[v,'accuracy_mean']:.2%} | {normal.loc[v,'macro_f1_mean']:.4f} | {missing.loc[v,'macro_f1_mean']:.4f} |")
        lines += ["","## 三模型概率平均集成" if cn else "## Equal-probability three-model ensembles","",
            "集成与单次模型均值分别报告，不按测试成绩选种子。" if cn else "Ensembles and mean individual-run scores are separate; no best seed is selected.","",
            "| ID | Ensemble Accuracy | Ensemble Macro-F1 |","|---|---:|---:|"]
        for v in COLORS:
            lines.append(f"| {v} | {ensemble.loc[v,'accuracy']:.2%} | {ensemble.loc[v,'macro_f1']:.4f} |")
        lines += ["","## 计算量与数据检查" if cn else "## Compute and data checks","",
            "| ID | Parameters | Context epochs | Mean training seconds |","|---|---:|---:|---:|"]
        for row in compute:
            lines.append(f"| {row['variant']} | {row['parameters']:,} | {row['context_epochs']} | {row['mean_training_seconds']:.1f} |")
        lines += ["",f"MIT beat detector: precision={precision:.4f}, recall={recall:.4f}, F1={f1:.4f}; greedy 150 ms one-to-one tolerance.",
            "这是心跳检测检查，与五类睡眠分期指标不同。ISRUC 没有本轮使用的独立逐拍参考标注。" if cn else
            "This is a beat-detection check, distinct from sleep-stage scores. No independent ISRUC beat reference is used in this study.","",
            "## 结果范围" if cn else "## Interpretation boundaries","",
            "- 一个受试者划分、4 位测试受试者、复用历史留出集；没有新外部验证。" if cn else "- One subject split, four test people and a reused holdout; no fresh external validation.",
            "- 两库均参与训练；分库成绩不等同于未见数据库泛化。" if cn else "- Both databases occur in training; per-dataset scores are not unseen-dataset generalization.",
            "- MIT 测试中 N3 仅 7 个窗口；逐人 Macro-F1 对缺失阶段计 0，应结合 support。" if cn else "- MIT test has seven N3 epochs. Subject-level fixed-class F1 includes zero for absent stages; inspect supports.",
            "- 门控权重是模型内部描述，不证明生理因果关系；质量字段不是临床验证评分。" if cn else "- Gate weights are descriptive, not physiological causality. Quality fields are not clinically validated indices.",
            "- 第一轮、早期随机森林及本轮的不同评估设置不能混作同一改进幅度。" if cn else "- Historical RF, version 1 and version 2 use distinct study settings; do not combine their scores as one improvement estimate.","",
            "## 文件与图表" if cn else "## Files and figures","",
            "[方法 / Methods](../../docs/METHODS_V2.md) · [复现 / Reproduction](../../docs/REPRODUCE_V2.md) · [图表 / Figures](../../docs/FIGURES.md)","",
            "Full per-run metrics: `run_metrics.csv`; all 56 prediction sets: `predictions.csv.gz`; complete confusion matrices and checkpoint hashes: `metrics.json`."]
        write(results/f"report.{lang}.md","\n".join(lines))
    gallery=["# Figures / 图表","","All figures are generated from preserved data and scores. Each is available as PNG, SVG and PDF. / 所有图表由保存的数据和指标生成，提供 PNG、SVG、PDF。","",
        "Sources: `python -m experiments.v2.figures`. Score bars use three individual seeds; confusion matrices and timelines explicitly use equal-probability ensembles.",""]
    for name,(en,cn) in CAPTIONS.items():
        gallery += [f"## {en} / {cn}","",f"[PNG](../results/v2/figures/{name}.png) · [SVG](../results/v2/figures/{name}.svg) · [PDF](../results/v2/figures/{name}.pdf)","",
                    f"![{en}](../results/v2/figures/{name}.png)","",f"{READING[name][0]} / {READING[name][1]}",""]
    gallery += ["# Technical diagrams / 技术说明图","",
                "Generated with `python -m experiments.v2.technical_figures`; each topic has English and Chinese PNG/SVG/PDF. Preserved inputs and outputs are used without retraining. / 六个主题均提供中英文三种格式，使用既有输入与预测，不进行新训练。",""]
    for name,(en,cn,reading_en,reading_cn) in TECHNICAL.items():
        gallery += [f"## {en} / {cn}","",f"{reading_en} / {reading_cn}",""]
        for lang in ["en","zh-CN"]:
            gallery += [f"**{lang}**: [PNG](figures/{name}.{lang}.png) · [SVG](figures/{name}.{lang}.svg) · [PDF](figures/{name}.{lang}.pdf)","",
                        f"![{en} / {cn}](figures/{name}.{lang}.png)",""]
    gallery += ["[Signal provenance / 信号示例来源](figures/provenance.json) · [Fixed case data / 固定案例数据](tables/prediction_cases.csv) · [Numeric breakdown / 数值分解](tables/result_breakdown.md)",""]
    write(PROJECT/"docs"/"FIGURES.md","\n".join(gallery))
    manifest=json.loads((PROJECT/"datasets"/"release_assets.json").read_text())
    guide=["# Research data / 研究数据","",f"[Release v2.0.0]({RELEASE})","",
        "| Archive | Size | Contents / 内容 |","|---|---:|---|"]
    for a in manifest["archives"]:
        guide.append(f"| {a['archive']} | {a['bytes']/1024**2:.1f} MiB | {'Original MIT files with attribution / MIT 原始实验文件' if a['archive'].startswith('mit') else 'All 18,770 prepared EEG-HRV epochs and contexts / 全部对齐输入和上下文'} |")
    guide += ["","```powershell","python scripts/download_release_data.py --kind prepared","# Optional MIT original files / 可选 MIT 原始文件","python scripts/download_release_data.py --kind mit","```","",
        "Archives use 48 MiB upload parts; all part, whole-archive and extracted-file hashes are verified. / 压缩包以 48 MiB 分块提供，逐层校验文件。","",
        "ISRUC original REC: [provider download](https://sleeptight.isr.uc.pt/?page_id=48). Exact hashes: [isruc-original-files.json](isruc-original-files.json). Download and inspect using `scripts/download_isruc_s3.py` and `scripts/inspect_isruc_s3.py`. / ISRUC 原始文件从作者来源获取，下载与检查工具已包含。","",
        "[Attribution and data conditions / 归属与使用条件](DATA_NOTICE.txt) · [Full reconstruction / 完整重建](../docs/REPRODUCE_V2.md)","",
        "### Prepared schema / 对齐输入结构","",
        "Per-record `.npz`: `eeg` float32 (n,3000), `spectra` float32 (n,29,89), `hrv` float32 (n,26), `eeg_features` float32 (n,7), `labels` integer, `epoch` integer. These arrays are before train-fitted scaling; models use saved preprocessing.","",
        "`epochs.csv`: record, subject, source, epoch, stage, record_row, split. `cache/v2/quality.npy`: (18770,3), matching this exact row order. `cache/v2/contexts.npy`: (18770,15), -1 for unavailable history. Stages 0..4 map to Wake/N1/N2/N3/REM.","",
        "Rows are aligned across modalities and labels. No raw data is silently added to training; augmented examples remain the same subjects. / 模态与标签按窗口对齐，增强不增加独立受试者。"]
    write(PROJECT/"datasets"/"README.md","\n".join(guide))
    write(PROJECT/"PROJECT.md","""# Project documentation / 项目文档

## Current experiment / 当前实验

- [English overview and results](README.md)
- [中文概览与成绩](README.zh-CN.md)
- [Complete methods / 完整方法](docs/METHODS_V2.md)
- [Model architecture](docs/ARCHITECTURE.en.md) · [模型架构](docs/ARCHITECTURE.zh-CN.md)
- [Data pipeline and timing](docs/DATA_PIPELINE.en.md) · [数据流程与时间边界](docs/DATA_PIPELINE.zh-CN.md)
- [HRV definitions and implementation](docs/FEATURES.en.md) · [HRV 特征与实现条件](docs/FEATURES.zh-CN.md)
- [Result interpretation and error cases](docs/RESULTS_ANALYSIS.en.md) · [结果解读与错误案例](docs/RESULTS_ANALYSIS.zh-CN.md)
- [Full English report](results/v2/report.en.md)
- [完整中文实验报告](results/v2/report.zh-CN.md)
- [Figure catalogue / 图表目录](docs/FIGURES.md)
- [Reproduce all runs / 复现全部实验](docs/REPRODUCE_V2.md)
- [Data, files and conditions / 数据、文件和来源](datasets/README.md)

## Archived experiments / 历史实验

- [Version 1 detailed project description / 第一轮详细项目说明](PROJECT.v1.md)
- [Version 1: five deep models and RF on the mixed subject holdout](results/deep_learning/report.md)
- [Original MIT-only grouped five-fold random forest baseline](docs/BASELINE.md)

Historical outputs are preserved. Current results report exploratory reuse of the version-1 test subjects; differences in datasets and evaluation settings are stated explicitly.
历史结果完整保留。本轮明确报告复用第一轮测试受试者的探索性质，不将不同数据与划分的成绩合并计算提升。
""")
    print("Generated bilingual README, full reports, figure catalogue, data guide and project index")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",type=Path,default=PROJECT/"results"/"v2")
    reports(parser.parse_args().results)
