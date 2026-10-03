"""Bilingual technical diagrams and fixed examples from preserved artifacts.

Does not train, score new models or modify results/v2. Signal examples require
the released prepared cache; architecture and result cases use tracked files.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from .figures import save, STAGES

PROJECT = Path(__file__).resolve().parents[2]
BLUE, GREEN, PURPLE, GOLD = "#DCEAF6", "#E0F0E8", "#E9E4F5", "#FFF0CE"
INK = "#20324A"


def setup(cn):
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    families = ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei"]
    chosen = next((name for name in families if name in fonts), None)
    if cn and chosen is None:
        raise RuntimeError("Chinese figures require Microsoft YaHei, Noto Sans CJK SC or SimHei")
    plt.rcParams.update({"font.family": chosen if cn else "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.unicode_minus": False,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
        "savefig.facecolor": "white", "svg.fonttype": "none", "pdf.fonttype": 42})


def box(ax, x, y, w, h, title, detail, color=BLUE):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.06,rounding_size=0.09",
                              facecolor=color,edgecolor="#B9C8D8",lw=1))
    ax.text(x+w/2,y+h*.73,title,ha="center",va="center",weight="bold",color=INK,fontsize=10)
    ax.text(x+w/2,y+h*.32,detail,ha="center",va="center",color=INK,fontsize=8.8,linespacing=1.5)


def arrow(ax, a, b, color="#667A91"):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=12,lw=1.5,color=color))


def architecture(root, cn):
    fig,ax=plt.subplots(figsize=(14,7.2))
    ax.set(xlim=(0,14),ylim=(0,7.2)); ax.axis("off")
    ax.text(.1,6.85,"两级 Transformer：EEG 对照 A1 与完整融合 A6" if cn else
            "Two-level Transformer: EEG control A1 and full fusion A6",fontsize=18,weight="bold",color=INK)
    ax.text(.1,6.42,"B：批大小；L=15（A0/A3/A4/A5 为 5）；尺寸包含当前及过去窗口" if cn else
            "B: batch size; L=15 (5 for A0/A3/A4/A5); contexts include current and past epochs",color="#52667F")
    box(ax,.15,4.55,2.05,1.28,"EEG 频谱" if cn else "EEG spectra","[B,L,29,89]\n30 s / epoch")
    box(ax,2.65,4.55,3.2,1.28,"窗口内编码" if cn else "Within-epoch encoder",
        "89→96 + positional encoding\n2-layer / 4-head Transformer\nMean pool 29 frames → [B,L,96]")
    box(ax,6.4,4.55,3.2,1.28,"窗口间编码" if cn else "Across-epoch encoder",
        "[B,L,96] + positional encoding\n2-layer / 4-head Transformer\nPadding-key mask",PURPLE)
    box(ax,10.15,4.55,2.8,1.28,"当前窗口分类" if cn else "Current-epoch head",
        "Last token [B,96]\nLayerNorm / Dropout / Linear\n5 logits → softmax",GOLD)
    for a,b in [(2.27,2.57),(5.92,6.32),(9.67,10.07)]:arrow(ax,(a,5.2),(b,5.2))
    ax.text(.15,4.12,"A1：仅 EEG；309,029 参数；15 窗口" if cn else
            "A1: EEG only; 309,029 parameters; 15 epochs",fontsize=11,weight="bold",color="#0072B2")
    box(ax,.15,1.5,2.05,1.65,"A6 双模态输入" if cn else "A6 inputs",
        "EEG [B,L,29,89]\nHRV [B,L,55]\n26 values + 26 flags + 3 q")
    box(ax,2.65,2.6,3.2,.88,"同一 EEG 编码定义" if cn else "Same EEG encoder",
        "29×89 → 96 / epoch")
    box(ax,2.65,.9,3.2,1.08,"心脏 MLP" if cn else "Cardiac MLP",
        "55→96→96\n20% whole-context HRV dropout",GREEN)
    box(ax,6.4,1.5,3.2,1.65,"质量约束残差门控" if cn else "Quality-constrained gate",
        "g = sigmoid(195→1) × q1 × q2\nz = EEG + g × delta(HRV)\n[B,L,96]",GOLD)
    box(ax,10.15,1.5,2.8,1.65,"时序编码与分类" if cn else "Temporal encoder + head",
        "Same layer definitions as A1\n2 layers / 4 heads / 96 dims\nCurrent token → [B,5]",PURPLE)
    arrow(ax,(2.27,2.8),(2.57,3.04));arrow(ax,(2.27,1.9),(2.57,1.45))
    arrow(ax,(5.92,3.04),(6.32,2.7));arrow(ax,(5.92,1.45),(6.32,1.96));arrow(ax,(9.67,2.3),(10.07,2.3))
    ax.text(.15,.42,"A6：342,921 参数；全缺失 HRV 时 g=0；使用 A6 自己的 EEG／时序权重" if cn else
            "A6: 342,921 parameters; missing HRV forces g=0; fallback retains A6's own trained weights",
            fontsize=10,color="#52667F")
    save(fig,root,f"model_architecture.{'zh-CN' if cn else 'en'}")


def gate(root,cn):
    fig,ax=plt.subplots(figsize=(13,6.4));ax.set(xlim=(0,13),ylim=(0,6.4));ax.axis("off")
    ax.text(.1,6.03,"残差门控：保留 EEG，按覆盖程度加入 HRV 修正" if cn else
            "Residual gating: retain EEG and scale the HRV correction",fontsize=17,weight="bold",color=INK)
    box(ax,.2,3.6,2.15,1.15,"EEG 向量 e" if cn else "EEG vector e","96 values",BLUE)
    box(ax,.2,1.6,2.15,1.15,"心脏向量 h" if cn else "Cardiac vector h","55→96→96",GREEN)
    box(ax,3.05,2.9,3.05,1.35,"可学习门控 u" if cn else "Learned gate u",
        "[e,h,q] = 195 values\nLinear 195→1 / sigmoid\nInitial bias −2; u ≈ 0.1192",GOLD)
    box(ax,3.05,.5,3.05,1.15,"心脏修正 delta" if cn else "Cardiac correction delta",
        "Linear 96→96 / GELU\nLinear 96→96",GREEN)
    box(ax,6.8,2.9,2.7,1.35,"有效权重 g" if cn else "Effective weight g",
        "g = u × q1 × q2\n0 ≤ g ≤ q1×q2\nOne scalar / epoch",GOLD)
    box(ax,10.2,2.9,2.35,1.35,"融合向量 z" if cn else "Fused vector z",
        "z = e + g × delta\n96 values\n→ temporal Transformer",PURPLE)
    arrow(ax,(2.42,4.15),(2.97,3.9));arrow(ax,(2.42,2.2),(2.97,3.3));arrow(ax,(2.42,2.0),(2.97,1.1))
    arrow(ax,(6.17,3.55),(6.72,3.55));arrow(ax,(9.57,3.55),(10.12,3.55))
    ax.plot([6.17,12.75,12.75],[1.1,1.1,2.55],color="#667A91",lw=1.5)
    arrow(ax,(12.75,2.55),(11.35,2.82))
    ax.plot([1.27,1.27,11.35],[4.82,5.1,5.1],color="#0072B2",lw=1.5);arrow(ax,(11.35,5.1),(11.35,4.32))
    ax.text(6.3,5.3,"EEG 主路径（残差）" if cn else "EEG residual path",ha="center",color="#0072B2")
    ax.text(6.8,2.12,"q1: valid RR fraction\nq2: finite feature fraction\nq3: history fraction (learned input only)",fontsize=9,color=INK,linespacing=1.6)
    ax.text(6.8,.55,"HRV 全缺失：值 0／标记 1／q=0 → g=0" if cn else
            "Missing HRV: values 0 / flags 1 / q=0 → g=0",fontsize=10,color="#52667F")
    save(fig,root,f"gated_fusion.{'zh-CN' if cn else 'en'}")


def timing(root,cn):
    fig,axes=plt.subplots(3,1,figsize=(13,7.3),gridspec_kw={"height_ratios":[1.1,1.1,.9]})
    for ax,L in zip(axes[:2],[5,15]):
        eeg_start=-L*30; hrv_start=-300-(L-1)*30
        ax.broken_barh([(eeg_start,L*30)],(2.4,.65),facecolors="#0072B2")
        for i in range(L):ax.plot([-i*30,-i*30],[2.4,3.05],color="white",lw=1)
        ax.broken_barh([(hrv_start,300)],(1.25,.45),facecolors="#B5D9C3")
        ax.broken_barh([(-300,300)],(.52,.45),facecolors="#009E73")
        ax.axvline(0,color="#D55E00",lw=1.4,ls="--")
        ax.set(yticks=[2.72,1.47,.74],yticklabels=["EEG", "最早 HRV" if cn else "Earliest HRV","当前 HRV" if cn else "Current HRV"],
               xlim=(-750,40),ylim=(.15,3.45),xlabel="距当前窗口终点的秒数 T=0" if cn else "Seconds relative to current epoch end T=0")
        ax.set_title(f"L={L}: EEG {L*30/60:g} min; " + (f"RR 终点历史联合覆盖 {(300+(L-1)*30)/60:g} min" if cn else
                     f"RR-endpoint history union {(300+(L-1)*30)/60:g} min"),loc="left")
        ax.text(7,2.68,"当前\n终点" if cn else "Target\nend",fontsize=9,color="#D55E00",va="center")
        ax.grid(axis="x",alpha=.15);ax.spines["left"].set_visible(False);ax.tick_params(axis="y",length=0)
    ax=axes[2];ax.axis("off");ax.set(xlim=(0,13),ylim=(0,1.8))
    for i,label in enumerate(["−1","−1","0","1","2"]):
        box(ax,1.7+i*1.75,.64,1.4,.8,label,"masked" if i<2 else ("当前" if cn and i==4 else "current" if i==4 else "valid"),"#E7EAEE" if i<2 else BLUE)
        if i<4:arrow(ax,(3.16+i*1.75,1.02),(3.37+i*1.75,1.02))
    ax.text(.15,1.55,"记录开头示例 L=5；未知标签缺口后同样重新积累历史" if cn else
            "Record-start example, L=5; unknown-label gaps also restart epoch context",fontsize=11,color=INK)
    ax.text(.15,.17,"左侧填充屏蔽；不跨记录；HRV 统计边界不代表离线 ECG 检测器的全部读取范围" if cn else
            "Left padding masked; no cross-record context; HRV bounds do not bound the offline ECG detector",fontsize=9,color="#52667F")
    fig.suptitle("时间窗、HRV 嵌套历史与标签对齐" if cn else "Epoch alignment and nested HRV histories",fontsize=17,weight="bold")
    fig.tight_layout(rect=(0,0,1,.95),h_pad=2)
    save(fig,root,f"time_alignment.{'zh-CN' if cn else 'en'}")


def pipeline(root,cn):
    fig,ax=plt.subplots(figsize=(15,6.4));ax.set(xlim=(0,15),ylim=(0,6.4));ax.axis("off")
    ax.text(.1,6.02,"数据处理全流程：原始信号到五类预测" if cn else
            "Data pipeline: original signals to five-stage predictions",fontsize=17,weight="bold",color=INK)
    box(ax,.15,4.15,2,1.15,"脑电 EEG" if cn else "Raw EEG","1 channel / µV",BLUE)
    box(ax,2.65,4.15,3.1,1.15,"逐窗处理" if cn else "Epoch-local processing",
        "0.3–35 Hz / 100 Hz\n30 s → 3,000 samples\nSTFT → 29×89 log power",BLUE)
    box(ax,.15,2.15,2,1.15,"心电 ECG" if cn else "Raw ECG","1 channel / offline",GREEN)
    box(ax,2.65,2.15,3.1,1.15,"心跳与历史特征" if cn else "Beats and history features",
        "SleepECG → RR endpoints\nUp to 300 s ending at epoch end\n26 features + 26 flags + 3 q",GREEN)
    box(ax,.15,.25,2,1.15,"专家标签" if cn else "Expert labels","30 s / Wake–REM",GOLD)
    box(ax,2.65,.25,3.1,1.15,"映射与保留规则" if cn else "Mapping and exclusions",
        "MIT 3+4 → N3\nISRUC scorer 1 / trim final 30\nUnknown labels break context",GOLD)
    box(ax,6.45,2.2,2.8,2.35,"对齐、按人划分" if cn else "Align; split by subject",
        "18,770 epochs / 26 people\nTrain: 22 / 15,761\nTest: 4 / 3,009\nNo validation set",PURPLE)
    box(ax,9.95,2.2,2.8,2.35,"预处理与上下文" if cn else "Scaling and contexts",
        "Fit fill/scale on training only\nL=5 or 15 / past + current\nMask unavailable history\nNo cross-record context",BLUE)
    box(ax,13.35,2.55,1.4,1.65,"模型" if cn else "Model",
        "A0–A6\n5 probabilities\nCurrent stage",GOLD)
    for y in [4.72,2.72,.82]:arrow(ax,(2.22,y),(2.57,y))
    arrow(ax,(5.82,4.72),(6.37,4.02));arrow(ax,(5.82,2.72),(6.37,3.2));arrow(ax,(5.82,.82),(6.37,2.54))
    arrow(ax,(9.32,3.38),(9.87,3.38));arrow(ax,(12.82,3.38),(13.27,3.38))
    ax.text(6.45,.84,"标签作为训练目标和评估参考，不作为模型特征输入" if cn else
            "Labels are targets/evaluation references, not model input features",fontsize=10,color="#52667F")
    save(fig,root,f"data_pipeline.{'zh-CN' if cn else 'en'}")


def signals(root,cache,quality,cn):
    manifest=pd.read_csv(PROJECT/"results/v2/epoch_manifest.csv")
    row=manifest[(manifest.split=="train")&(manifest.stage==2)].sort_values(["record","epoch"]).iloc[0]
    with np.load(cache/f"{row.record}.npz") as data:
        eeg=data["eeg"][row.record_row]; spectrum=data["spectra"][row.record_row]
    ecg_path=sorted(quality.glob("ecg_review_*.npz"))[0]
    with np.load(ecg_path) as data:
        ecg=data["ecg"];fs=float(data["fs"]);beats=data["beats"]
        record=str(data["record"]);start=int(data["start_seconds"])
    assert row.split=="train" and record in set(manifest[manifest.split=="train"].record)
    fig,axes=plt.subplots(2,2,figsize=(13,7.8))
    axes[0,0].plot(np.arange(3000)/100,eeg,color="#0072B2",lw=.65)
    axes[0,0].set(title=f"EEG · {row.record} · epoch {row.epoch} · N2",xlabel="秒" if cn else "Seconds",ylabel="µV",xlim=(0,30))
    freq=np.fft.rfftfreq(256,1/100);freq=freq[(freq>=.3)&(freq<=35)]
    mesh=axes[0,1].pcolormesh(np.arange(29)+1,freq,spectrum.T,cmap="magma",shading="nearest",rasterized=True)
    axes[0,1].set(title="同一 EEG 窗口的 29×89 频谱" if cn else "Same EEG epoch: 29×89 spectrum",xlabel="帧中心秒数" if cn else "Frame-center seconds",ylabel="Hz")
    fig.colorbar(mesh,ax=axes[0,1],label="log10 |STFT|²",fraction=.046,pad=.04)
    normalized=(ecg-np.median(ecg))/max(float(ecg.std()),1e-8)
    axes[1,0].plot(np.arange(len(ecg))/fs,normalized,color=INK,lw=.8)
    idx=np.minimum((beats*fs).astype(int),len(ecg)-1)
    axes[1,0].scatter(beats,normalized[idx],marker="x",s=25,color="#D55E00",label="检测心跳" if cn else "Detected beat")
    axes[1,0].set(title=f"ECG · {record} · start {start} s",xlabel="片段内秒数" if cn else "Seconds within segment",ylabel="显示标准化幅值" if cn else "Display-normalized amplitude",xlim=(0,10));axes[1,0].legend(frameon=False,fontsize=9)
    axes[1,1].plot(beats[1:],np.diff(beats),"o-",color="#009E73",lw=1.2,ms=4)
    axes[1,1].set(title="相邻检测心跳间隔：RR = diff(beats)" if cn else "RR intervals = diff(detected beat times)",
                  xlabel="RR 末端秒数" if cn else "RR endpoint seconds",ylabel="RR (s)",xlim=(0,10))
    for ax in axes.flat:ax.grid(alpha=.13)
    fig.suptitle("真实训练信号示例：波形、频谱、心跳与间隔" if cn else "Actual training examples: waveform, spectrum, beats and intervals",fontsize=16,weight="bold")
    fig.text(.5,.015,"EEG 与 ECG 是分别选取的训练片段；不是同步示例。10 秒 RR 仅作说明，模型 HRV 最多取 300 秒。" if cn else
             "EEG and ECG are separate training examples, not a synchronized pair. Ten-second RR illustrates intervals; model HRV uses up to 300 s.",ha="center",fontsize=9,color="#52667F")
    fig.tight_layout(rect=(0,.05,1,.95),pad=2)
    save(fig,root,f"signal_walkthrough.{'zh-CN' if cn else 'en'}")
    return {"eeg_record":str(row.record),"eeg_epoch":int(row.epoch),"ecg_record":record,"ecg_start_seconds":start,
            "ecg_source":ecg_path.name,"split":"train"}


def select_cases(predictions):
    normal=predictions[(predictions.seed=="ensemble")&(predictions.condition=="normal")]
    a=normal[normal.variant=="A1"].set_index(["record","epoch"])
    b=normal[normal.variant=="A6"].set_index(["record","epoch"])
    cases=[]
    for code,correct_a,correct_b in [("both_correct",True,True),("A1_only_correct",True,False),
                                     ("A6_only_correct",False,True),("both_wrong",False,False)]:
        keep=(a.prediction.eq(a.stage)==correct_a)&(b.prediction.eq(b.stage)==correct_b)&(a.index.get_level_values("epoch")>=14)
        record,epoch=a[keep].sort_index().index[0]
        for variant,frame in [("A1",a),("A6",b)]:
            row=frame.loc[(record,epoch)].to_dict()
            row.update(case=code,record=record,epoch=int(epoch),variant=variant)
            cases.append(row)
    return pd.DataFrame(cases)


def case_figures(root,cases,cn):
    names={"both_correct":("两者正确","Both correct"),"A1_only_correct":("仅 A1 正确","Only A1 correct"),
           "A6_only_correct":("仅 A6 正确","Only A6 correct"),"both_wrong":("两者错误","Both wrong")}
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for ax,(case,group) in zip(axes.flat,cases.groupby("case",sort=False)):
        a=group[group.variant=="A1"].iloc[0]; b=group[group.variant=="A6"].iloc[0]
        x=np.arange(5)
        ax.bar(x-.18,[a[f"p_{s}"] for s in STAGES],width=.34,color="#0072B2",label=f"A1 → {STAGES[int(a.prediction)]}")
        ax.bar(x+.18,[b[f"p_{s}"] for s in STAGES],width=.34,color="#332288",label=f"A6 → {STAGES[int(b.prediction)]}")
        ax.set(xticks=x,xticklabels=STAGES,ylim=(0,1.05),ylabel="预测概率" if cn else "Predicted probability")
        ax.set_title(f"{names[case][0 if cn else 1]} · {a.record} / epoch {a.epoch}\n"+
                     ("专家" if cn else "Expert")+f": {STAGES[int(a.stage)]} · A6 gate={b.gate:.3f}",fontsize=11)
        ax.axvspan(int(a.stage)-.47,int(a.stage)+.47,color="#009E73",alpha=.09,zorder=0)
        ax.legend(frameon=False,fontsize=9);ax.grid(axis="y",alpha=.15)
    fig.suptitle("固定预测案例：EEG A1 与融合 A6 的概率输出" if cn else "Fixed prediction cases: EEG A1 versus fusion A6",fontsize=16,weight="bold")
    fig.text(.5,.015,"每类按 record／epoch 排序，取 epoch≥14 的首例；展示三模型概率平均，不表示总体频率或已验证机制。" if cn else
             "First record/epoch-sorted case in each category with epoch≥14; three-model ensembles; examples do not establish frequency or mechanism.",ha="center",fontsize=8.5,color="#52667F")
    fig.tight_layout(rect=(0,.05,1,.95),pad=2.3)
    save(fig,root,f"prediction_cases.{'zh-CN' if cn else 'en'}")


def breakdown(predictions,cases):
    lines=["# Preserved result breakdown / 保存结果分解","",
           "Generated from frozen CSVs; mean individual-run metrics unless explicitly marked ensemble. / 从冻结 CSV 生成；除案例与混淆外均为单次种子均值。","",
           "## Dataset means / 分库均值","","| Model | Dataset | Epochs | Accuracy | Macro-F1 |","|---|---|---:|---:|---:|"]
    for name,key in [("per_dataset.csv","source"),("per_subject.csv","subject")]:
        if key=="subject":lines += ["","## Subject means / 分人均值","","| Model | Subject | Epochs | Accuracy | Macro-F1 |","|---|---|---:|---:|---:|"]
        frame=pd.read_csv(PROJECT/"results/v2"/name,dtype={"seed":str})
        frame=frame[(frame.condition=="normal")&(frame.seed!="ensemble")&(frame.variant.isin(["A0","A1","A6"]))]
        for (v,group),rows in frame.groupby(["variant",key]):
            lines.append(f"| {v} | {group} | {int(rows.epochs.iloc[0])} | {rows.accuracy.mean():.2%} | {rows.macro_f1.mean():.4f} |")
    lines += ["","## Largest off-diagonal errors per true stage / 每个真实阶段最大的错误去向","",
              "Ensembles; percentage denominator is all epochs of the true class. / 概率平均集成；分母为该真实阶段的全部窗口。","",
              "| Model | True | Most frequent wrong prediction | Count / support | Fraction |","|---|---|---|---:|---:|"]
    normal=predictions[(predictions.seed=="ensemble")&(predictions.condition=="normal")]
    for v in ["A1","A6"]:
        for stage in range(5):
            rows=normal[(normal.variant==v)&(normal.stage==stage)]
            counts=rows[rows.prediction!=stage].prediction.value_counts().reindex(range(5),fill_value=0)
            wrong=int(counts.idxmax());n=int(counts.max())
            lines.append(f"| {v} | {STAGES[stage]} | {STAGES[wrong]} | {n} / {len(rows)} | {n/len(rows):.2%} |")
    lines += ["","## Fixed cases / 固定案例","","| Case | Record | Epoch (0-based) | Expert | A1 | A6 | A6 gate |","|---|---|---:|---|---|---|---:|"]
    for case,group in cases.groupby("case",sort=False):
        a=group[group.variant=="A1"].iloc[0];b=group[group.variant=="A6"].iloc[0]
        lines.append(f"| {case} | {a.record} | {a.epoch} | {STAGES[int(a.stage)]} | {STAGES[int(a.prediction)]} | {STAGES[int(b.prediction)]} | {b.gate:.4f} |")
    folder=PROJECT/"docs/tables";folder.mkdir(exist_ok=True)
    (folder/"result_breakdown.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    cases.to_csv(folder/"prediction_cases.csv",index=False)


def main(cache,quality):
    root=PROJECT/"docs/figures";root.mkdir(exist_ok=True)
    predictions=pd.read_csv(PROJECT/"results/v2/predictions.csv.gz",dtype={"seed":str})
    cases=select_cases(predictions)
    for cn in [False,True]:
        setup(cn);architecture(root,cn);gate(root,cn);timing(root,cn);pipeline(root,cn)
        provenance=signals(root,cache,quality,cn);case_figures(root,cases,cn)
    breakdown(predictions,cases)
    provenance["case_rule"]="First lexicographic record/epoch per outcome category with epoch>=14; normal A1/A6 ensembles"
    (root/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
    print("Generated six bilingual technical figure sets (36 files), fixed cases and result tables")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,default=PROJECT/"cache/deep_learning")
    parser.add_argument("--quality-cache",type=Path,default=PROJECT/"cache/v2")
    args=parser.parse_args();main(args.cache,args.quality_cache)
