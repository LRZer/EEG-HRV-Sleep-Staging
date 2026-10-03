"""Generate concise bilingual summaries and complete reports from saved metrics."""
from pathlib import Path
import pandas as pd
from .figures import LABELS,COLORS,STAGES

PROJECT=Path(__file__).resolve().parents[2]
RELEASE="https://github.com/LRZer/EEG-HRV-Sleep-Staging/releases/tag/v2.0.0"
CN={"A0":"EEG，5 窗口","A1":"EEG，15 窗口","A2":"仅 HRV，5 窗口","A3":"拼接融合，5 窗口",
    "A4":"门控融合，5 窗口","A5":"门控＋模态丢弃，5 窗口","A6":"门控＋模态丢弃，15 窗口"}
CAPTIONS={
    "architecture":("Model architecture and ablation components","模型结构与对照组件"),
    "data_overview":("Subjects, retained epochs and stage distributions","受试者、窗口数及类别分布"),
    "ablation_results":("Seven variants, three individual seeds; mean ± sample SD","七组模型三个独立种子的均值与样本标准差"),
    "training_curves":("840 training-epoch logs; shaded bands are seed SD","840 轮训练日志，阴影为种子标准差"),
    "dataset_results":("MIT and ISRUC results on the same fixed subject split","同一受试者划分下两个数据库的成绩"),
    "per_class_f1":("Five-stage F1 and true class supports","五类 F1 及真实样本数量"),
    "confusion_matrices":("All seven equal-probability ensembles; counts and row percentages","七个概率平均集成的混淆矩阵、计数及行比例"),
    "missing_hrv_robustness":("Normal inputs versus completely unavailable HRV","正常输入与 HRV 完全不可用的预设测试"),
    "subject_results":("Four subject outcomes with fixed-five-class F1","四位受试者的准确率和固定五类 F1"),
    "gate_analysis":("Effective cardiac weights by stage and dataset","按阶段和数据库展示有效心脏权重"),
    "sleep_timelines":("All held-out recordings: expert, A0 ensemble and A6 ensemble","全部测试记录的专家标注、A0 与 A6 集成时间图"),
    "ecg_review":("Fixed random training-only ECG segments and detected beats","固定随机训练 ECG 片段与检测心跳"),
    "ecg_detection":("MIT beat detection against provider annotations; 150 ms tolerance","MIT 心跳检测与官方逐拍标注对比，容差 150 ms")}


def write(path,text):
    path.write_text(text.strip()+"\n",encoding="utf-8")


def generate(results):
    """Render maintained overview templates from preserved score tables."""
    if results.resolve() != (PROJECT / "results/v2").resolve():
        # Canonical overview prose describes the preserved study, not a new run.
        return
    comparison = pd.read_csv(results / "comparison.csv")
    normal = comparison[comparison.condition == "normal"].set_index("variant")
    perclass = pd.read_csv(results / "per_class.csv", dtype={"seed": str})
    stage_mean = perclass[(perclass.condition == "normal") & (perclass.seed != "ensemble")].pivot_table(
        index="variant", columns="stage", values="f1")
    runs = pd.read_csv(results / "run_metrics.csv", dtype={"seed": str})
    ensembles = runs[(runs.condition == "normal") & (runs.seed == "ensemble")].set_index("variant")
    best = normal.macro_f1_mean.idxmax()
    values = {
        "BEST_ACC": f"{normal.accuracy_mean.max():.2%}",
        "BEST_F1": f"{normal.loc[best, 'macro_f1_mean']:.4f}",
        "DELTA_ACC": f"{100 * (normal.loc['A6', 'accuracy_mean'] - normal.loc['A0', 'accuracy_mean']):+.2f}",
        "DELTA_F1": f"{normal.loc['A6', 'macro_f1_mean'] - normal.loc['A0', 'macro_f1_mean']:+.4f}",
    }
    for variant in ["A1", "A6"]:
        values[f"{variant}_ENS_ACC"] = f"{ensembles.loc[variant, 'accuracy']:.2%}"
        values[f"{variant}_ENS_F1"] = f"{ensembles.loc[variant, 'macro_f1']:.4f}"
    for filename, labels in [("README.md", LABELS), ("README.zh-CN.md", CN)]:
        rows = []
        for variant in COLORS:
            row = normal.loc[variant]
            rows.append(f"| {variant} | {labels[variant]} | {row.accuracy_mean:.2%} ± {row.accuracy_std * 100:.2f} pp | "
                        f"{row.macro_f1_mean:.4f} ± {row.macro_f1_std:.4f} | "
                        f"{stage_mean.loc[variant, 'N1']:.4f} | {stage_mean.loc[variant, 'REM']:.4f} |")
        replacements = dict(values, RESULT_ROWS="\n".join(rows))
        text = (PROJECT / "docs" / "templates" / filename).read_text(encoding="utf-8")
        for key, value in replacements.items():
            text = text.replace("{{" + key + "}}", value)
        if "{{" in text:
            raise ValueError("Unresolved README template variable")
        write(PROJECT / filename, text)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=PROJECT / "results/v2")
    generate(parser.parse_args().results)
