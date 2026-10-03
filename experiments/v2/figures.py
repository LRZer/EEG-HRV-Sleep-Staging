"""Publication-style figures generated only from preserved experiment outputs."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
STAGES = ["Wake", "N1", "N2", "N3", "REM"]
COLORS = {"A0": "#0072B2", "A1": "#56B4E9", "A2": "#CC79A7", "A3": "#D55E00", "A4": "#009E73", "A5": "#E69F00", "A6": "#332288"}
LABELS = {"A0": "EEG / 5 epochs", "A1": "EEG / 15 epochs", "A2": "HRV / 5 epochs",
          "A3": "Concat / 5 epochs", "A4": "Gated / 5 epochs", "A5": "Gated + drop / 5 epochs", "A6": "Gated + drop / 15 epochs"}


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "svg.fonttype": "none", "pdf.fonttype": 42, "axes.titleweight": "semibold"})


def save(fig, root, name):
    root.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "svg", "pdf"]:
        target=root / f"{name}.{extension}"
        fig.savefig(target, dpi=190, bbox_inches="tight", pad_inches=0.15)
        if extension=="svg":
            # Matplotlib emits trailing spaces in path coordinates; preserve a clean text artifact.
            target.write_bytes(("\n".join(line.rstrip() for line in target.read_text(encoding="utf-8").splitlines())+"\n").encode("utf-8"))
    plt.close(fig)


def architecture(root):
    fig, ax = plt.subplots(figsize=(12, 5.3))
    ax.set(xlim=(0, 12), ylim=(0, 5.3))
    ax.axis("off")
    def box(x, y, w, h, title, subtitle, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.1", facecolor=color, edgecolor="#CED5DF", lw=1))
        ax.text(x+w/2, y+h*.67, title, ha="center", va="center", weight="bold", fontsize=10)
        ax.text(x+w/2, y+h*.28, subtitle, ha="center", va="center", fontsize=9, color="#394658")
    def arrow(x1, y1, x2, y2, text=None):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2), arrowstyle="-|>", mutation_scale=13, lw=1.4, color="#5F6C80"))
        if text:
            ax.text((x1+x2)/2,(y1+y2)/2+.14,text,ha="center",fontsize=8)
    box(.25,3.15,2.0,1.15,"EEG spectrogram","30 s · 100 Hz\n29 frames × 89 bins","#E7F1FA")
    box(2.85,3.15,2.15,1.15,"Epoch Transformer","2 layers · 4 heads\n96-dimensional token","#E7F1FA")
    box(.25,.95,2.0,1.15,"ECG-derived HRV","26 values + 26 flags\n3 coverage indicators","#E4F2EA")
    box(2.85,.95,2.15,1.15,"Cardiac MLP","55 → 96\n20% modality dropout","#E4F2EA")
    box(5.6,2.05,2.15,1.25,"Residual gated fusion","EEG + gate × correction\nGate constrained by quality","#FFF1D9")
    box(8.35,2.05,2.15,1.25,"Temporal Transformer","15 past/current epochs\n2 layers · 4 heads","#EAE8F6")
    box(10.95,2.05,.7,1.25,"5","stages","#F0F2F5")
    arrow(2.4,3.72,2.72,3.72)
    arrow(2.4,1.52,2.72,1.52)
    arrow(5.14,3.72,5.5,2.95)
    arrow(5.14,1.52,5.5,2.35)
    arrow(7.9,2.68,8.22,2.68)
    arrow(10.64,2.68,10.81,2.68)
    ax.text(.13,5.05,"Controlled EEG–HRV fusion study",fontsize=17,weight="bold",color="#182638")
    ax.text(.13,4.68,"A6 shown; variants compare modality, fusion and context under a fixed protocol",fontsize=10,color="#536276")
    ax.text(.2,.25,"Training: 22 subjects · 3 seeds · final epoch 40       Evaluation: 4 subjects · 3,009 epochs · exploratory reused holdout",fontsize=9,color="#536276")
    save(fig,root,"architecture")


def data_overview(root, frame):
    fig, axes = plt.subplots(2,2,figsize=(12,7.4),gridspec_kw={"height_ratios":[1,1.25]})
    groups = [("train","MIT-BIH"),("train","ISRUC-III"),("test","MIT-BIH"),("test","ISRUC-III")]
    x = np.arange(4)
    counts = [len(frame[(frame.split==s)&(frame.source==d)]) for s,d in groups]
    people = [frame[(frame.split==s)&(frame.source==d)].subject.nunique() for s,d in groups]
    bars=axes[0,0].bar(x,counts,color=["#0072B2","#009E73","#56B4E9","#8CC9AB"])
    axes[0,0].bar_label(bars,labels=[f"{n:,}" for n in counts],padding=3)
    axes[0,0].set_xticks(x,["MIT train","ISRUC train","MIT test","ISRUC test"])
    axes[0,0].set_ylim(0,max(counts)*1.17)
    axes[0,0].set(title="Labeled 30-second epochs",ylabel="Epochs")
    bars=axes[0,1].bar(x,people,color=["#0072B2","#009E73","#56B4E9","#8CC9AB"])
    axes[0,1].bar_label(bars,padding=3)
    axes[0,1].set_xticks(x,["MIT train","ISRUC train","MIT test","ISRUC test"])
    axes[0,1].set(title="Independent subjects",ylabel="Subjects",ylim=(0,17))
    for ax, split in zip(axes[1], ["train","test"]):
        for offset, source, color in [(-.19,"MIT-BIH","#0072B2"),(.19,"ISRUC-III","#009E73")]:
            subset=frame[(frame.split==split)&(frame.source==source)]
            values=subset.stage.value_counts().reindex(range(5),fill_value=0).to_numpy()
            bars=ax.bar(np.arange(5)+offset,values,width=.36,color=color,label=source)
            ax.bar_label(bars,padding=3,fontsize=8)
        ax.set_xticks(range(5),STAGES)
        ax.set(title=f"{split.title()} stage distribution",ylabel="Epochs")
        ax.set_ylim(0,ax.get_ylim()[1]*1.14)
        ax.legend(frameon=False,fontsize=9)
    fig.suptitle("26 subjects · 28 recordings · 18,770 retained epochs",fontsize=16,weight="bold")
    fig.tight_layout(pad=2)
    save(fig,root,"data_overview")


def comparison(root, runs):
    normal=runs[(runs.condition=="normal")&(runs.seed!="ensemble")]
    fig, axes=plt.subplots(1,2,figsize=(12,5.1),sharey=True)
    positions=np.arange(7)
    for ax,metric,title in zip(axes,["accuracy","macro_f1"],["Accuracy","Five-class Macro-F1"]):
        for pos,variant in enumerate(COLORS):
            values=normal.loc[normal.variant==variant,metric].to_numpy()
            mean,std=values.mean(),values.std(ddof=1)
            ax.barh(pos,mean,xerr=std,color=COLORS[variant],height=.62,capsize=3,error_kw={"elinewidth":1,"ecolor":"#202D3C"})
            ax.scatter(values,np.full(3,pos),s=18,color="#182638",zorder=3,alpha=.75)
            ax.text(mean+std+.017,pos,f"{mean:.3f} ± {std:.3f}",va="center",fontsize=8)
        ax.set_yticks(positions,[f"{v}  {LABELS[v]}" for v in COLORS])
        ax.set(xlim=(0,1.02),xlabel="Score",title=title)
        ax.grid(axis="x",alpha=.16)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle("Controlled ablations: mean ± sample SD across 3 seeds",fontsize=15,weight="bold")
    fig.text(.5,.005,"Dots = seeds 42, 123, 2026 · same 4 subjects · SD measures initialization variability, not population uncertainty",ha="center",fontsize=8,color="#536276")
    fig.tight_layout(rect=(0,.045,1,.97))
    save(fig,root,"ablation_results")


def training_curves(root, results):
    fig,axes=plt.subplots(1,2,figsize=(12,4.9))
    for variant in COLORS:
        frames=[pd.read_csv(results/f"{variant}_seed{s}_history.csv") for s in [42,123,2026]]
        for ax,field in zip(axes,["train_loss","train_accuracy"]):
            values=np.stack([f[field] for f in frames])
            mean,std=values.mean(0),values.std(0,ddof=1)
            ax.plot(np.arange(1,41),mean,color=COLORS[variant],label=variant,lw=1.6)
            ax.fill_between(np.arange(1,41),mean-std,mean+std,color=COLORS[variant],alpha=.1)
    axes[0].set(ylabel="Weighted cross entropy",title="Training loss")
    axes[1].set(ylabel="Training accuracy (augmented)",title="Training accuracy",ylim=(0,1))
    for ax in axes:
        ax.set(xlabel="Training epoch",xlim=(1,40))
        ax.grid(alpha=.15)
        ax.legend(ncol=4,frameon=False,fontsize=8,loc="upper right" if ax==axes[0] else "lower right")
    fig.suptitle("840 training epochs · fixed final checkpoints · no validation curves",fontsize=15,weight="bold")
    fig.tight_layout(pad=2)
    save(fig,root,"training_curves")


def domain_results(root, domains):
    normal=domains[(domains.condition=="normal")&(domains.seed.astype(str)!="ensemble")]
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,sharey=True)
    for column,source in enumerate(["MIT-BIH","ISRUC-III"]):
        for row,metric in enumerate(["accuracy","macro_f1"]):
            ax=axes[row,column]
            for pos,variant in enumerate(COLORS):
                values=normal[(normal.variant==variant)&(normal.source==source)][metric].to_numpy()
                ax.bar(pos,values.mean(),yerr=values.std(ddof=1),capsize=3,color=COLORS[variant],width=.6)
                ax.scatter(np.full(3,pos),values,color="#182638",s=15,zorder=3)
            ax.set_xticks(range(7),list(COLORS))
            ax.set(ylim=(0,1),ylabel="Accuracy" if metric=="accuracy" else "Macro-F1")
            n=int(normal[normal.source==source].epochs.iloc[0])
            ax.set_title(f"{source} · 2 subjects · {n:,} epochs")
            ax.grid(axis="y",alpha=.15)
            ax.set_axisbelow(True)
    fig.suptitle("Within-study dataset differences: three-seed results",fontsize=15,weight="bold")
    fig.text(.5,.01,"Training includes both datasets; these are unseen subjects, not unseen-dataset validation. MIT test contains only 7 N3 epochs.",ha="center",fontsize=8,color="#536276")
    fig.tight_layout(rect=(0,.05,1,.98),pad=2)
    save(fig,root,"dataset_results")


def class_f1(root, classes):
    normal=classes[(classes.condition=="normal")&(classes.seed.astype(str)!="ensemble")]
    mean=normal.pivot_table(index="variant",columns="stage",values="f1",aggfunc="mean").reindex(index=list(COLORS),columns=STAGES)
    deviation=normal.pivot_table(index="variant",columns="stage",values="f1",aggfunc="std").reindex(index=list(COLORS),columns=STAGES)
    support=normal.groupby("stage").support.first().reindex(STAGES)
    fig,ax=plt.subplots(figsize=(10,5.5))
    im=ax.imshow(mean.to_numpy(),vmin=0,vmax=1,cmap="YlGnBu",aspect="auto")
    for i in range(7):
        for j in range(5):
            value=mean.iloc[i,j]
            ax.text(j,i,f"{value:.3f}\n± {deviation.iloc[i,j]:.3f}",ha="center",va="center",fontsize=9,color="white" if value>.7 else "#182638")
    ax.set_xticks(range(5),[f"{s}\nn={int(support[s]):,}" for s in STAGES])
    ax.set_yticks(range(7),[f"{v}  {LABELS[v]}" for v in COLORS])
    ax.set_title("Per-stage F1: mean ± SD across three seeds",pad=14)
    fig.colorbar(im,ax=ax,label="F1",fraction=.035,pad=.025)
    fig.tight_layout()
    save(fig,root,"per_class_f1")


def confusions(root, summary):
    fig,axes=plt.subplots(2,4,figsize=(16,8))
    for ax,variant in zip(axes.flat,COLORS):
        metrics=summary["models"][f"{variant}_seedensemble_normal"]
        cm=np.asarray(metrics["confusion_matrix"])
        fractions=cm/np.maximum(cm.sum(1,keepdims=True),1)
        ax.imshow(fractions,vmin=0,vmax=1,cmap="Blues")
        for i in range(5):
            for j in range(5):
                ax.text(j,i,f"{cm[i,j]}\n{fractions[i,j]:.0%}",ha="center",va="center",fontsize=7.5,color="white" if fractions[i,j]>.6 else "#182638")
        ax.set_xticks(range(5),STAGES,rotation=35)
        ax.set_yticks(range(5),STAGES)
        ax.set(xlabel="Predicted stage",ylabel="Expert stage")
        ax.set_title(f"{variant} ensemble\nAcc {metrics['accuracy']:.3f} · F1 {metrics['macro_f1']:.3f}",fontsize=10)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(.1,.8,"READING THE MATRIX\n\nRows: expert labels\nColumns: predictions\n\nCell: epoch count\nand row percentage\n\nEnsemble = probability mean\nof seeds 42, 123 and 2026\n\n3,009 epochs from 4 subjects",va="top",fontsize=10,color="#536276",linespacing=1.7)
    fig.suptitle("All seven predeclared three-seed ensembles",fontsize=16,weight="bold")
    fig.tight_layout(pad=2)
    save(fig,root,"confusion_matrices")


def robustness(root,runs):
    individual=runs[runs.seed.astype(str)!="ensemble"]
    fig,axes=plt.subplots(1,2,figsize=(11.5,5))
    for ax,metric,title in zip(axes,["accuracy","macro_f1"],["Accuracy","Macro-F1"]):
        for variant in COLORS:
            means,stds=[],[]
            for condition in ["normal","hrv_missing"]:
                values=individual[(individual.variant==variant)&(individual.condition==condition)][metric].to_numpy()
                means.append(values.mean())
                stds.append(values.std(ddof=1))
            ax.errorbar([0,1],means,yerr=stds,color=COLORS[variant],marker="o",lw=1.4,capsize=3,label=variant)
        ax.set_xticks([0,1],["Normal inputs","All HRV unavailable"])
        ax.set(xlim=(-.18,1.18),ylim=(0,1),title=title,ylabel="Score")
        ax.grid(axis="y",alpha=.15)
        ax.legend(ncol=4,frameon=False,fontsize=8)
    fig.suptitle("Predeclared missing-modality stress test",fontsize=15,weight="bold")
    fig.text(.5,.015,"HRV values set to neutral; missing flags set to 1; all quality indicators set to 0. EEG inputs are unchanged.",ha="center",fontsize=8,color="#536276")
    fig.tight_layout(rect=(0,.055,1,.98),pad=2)
    save(fig,root,"missing_hrv_robustness")


def subjects(root, subjects):
    normal=subjects[(subjects.condition=="normal")&(subjects.seed.astype(str)!="ensemble")]
    people=list(normal.subject.unique())
    fig,axes=plt.subplots(1,2,figsize=(12,5.3),sharey=True)
    for ax,metric,title in zip(axes,["accuracy","macro_f1"],["Accuracy","Five-class Macro-F1"]):
        for index,person in enumerate(people):
            values=normal[normal.subject==person].groupby("variant")[metric].agg(["mean","std"]).reindex(COLORS)
            ax.errorbar(range(7),values["mean"],yerr=values["std"],label=person,marker=["o","s","^","D"][index],capsize=2,lw=1)
        ax.set_xticks(range(7),list(COLORS))
        ax.set(ylim=(0,1),title=title,xlabel="Variant",ylabel="Score")
        ax.grid(alpha=.15)
        ax.legend(frameon=False,fontsize=8)
    fig.suptitle("Individual subject outcomes: mean ± SD across seeds",fontsize=15,weight="bold")
    fig.text(.5,.01,"Per-subject Macro-F1 uses all five classes; absent true stages have F1=0. See per_subject.csv for class support.",ha="center",fontsize=8,color="#536276")
    fig.tight_layout(rect=(0,.05,1,.98))
    save(fig,root,"subject_results")


def gate_analysis(root,predictions):
    subset=predictions[(predictions.condition=="normal")&(predictions.seed.astype(str)=="ensemble")]
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for column,variant in enumerate(["A4","A5","A6"]):
        selected=subset[subset.variant==variant]
        groups=[selected[selected.stage==i].gate.to_numpy() for i in range(5)]
        axes[0,column].boxplot(groups,tick_labels=STAGES,showfliers=False,patch_artist=True,
            boxprops={"facecolor":COLORS[variant],"alpha":.35},medianprops={"color":"#182638"})
        axes[0,column].set(title=f"{variant}: gate by expert stage",ylabel="Effective gate",ylim=(0,1))
        for index,source in enumerate(["MIT-BIH","ISRUC-III"]):
            values=selected[selected.source==source].gate.to_numpy()
            axes[1,column].hist(values,bins=np.linspace(0,1,21),density=True,alpha=.55,label=source,
                color=["#0072B2","#009E73"][index])
        axes[1,column].set(title=f"{variant}: gate by dataset",xlabel="Effective gate",ylabel="Density")
        axes[1,column].legend(frameon=False,fontsize=8)
    # Histograms have a density axis, not the shared 0..1 gate axis.
    for ax in axes[1]:
        ax.set_ylim(0, max(ax.get_ylim()[1], 1))
    fig.suptitle("Learned effective HRV weights: three-seed averages",fontsize=15,weight="bold")
    fig.text(.5,.01,"Effective gate = sigmoid gate × valid-RR fraction × finite-feature fraction. Descriptive weights do not establish physiological causality.",ha="center",fontsize=8,color="#536276")
    fig.tight_layout(rect=(0,.05,1,.98),pad=2)
    save(fig,root,"gate_analysis")


def hypnograms(root,predictions):
    subset=predictions[(predictions.condition=="normal")&(predictions.seed.astype(str)=="ensemble")]
    records=list(subset.record.unique())
    fig,axes=plt.subplots(len(records),1,figsize=(13,2.0*len(records)),squeeze=False)
    heights=np.asarray([4,2,1,0,3])
    for ax,record in zip(axes[:,0],records):
        truth=subset[(subset.record==record)&(subset.variant=="A0")].sort_values("epoch")
        times=truth.epoch.to_numpy()/120
        gaps=np.r_[False,np.diff(truth.epoch.to_numpy())!=1]
        stages=heights[truth.stage.to_numpy()].astype(float)
        stages[gaps]=np.nan
        ax.step(times,stages,where="post",color="#182638",lw=1.45,label="Expert")
        for variant,color in [("A0",COLORS["A0"]),("A6",COLORS["A6"])]:
            selected=subset[(subset.record==record)&(subset.variant==variant)].sort_values("epoch")
            values=heights[selected.prediction.to_numpy()].astype(float)
            values[gaps]=np.nan
            ax.step(times,values,where="post",color=color,lw=.8,alpha=.8,label=f"{variant} ensemble")
        ax.set_yticks(range(5),["N3","N2","N1","REM","Wake"])
        ax.set(title=record,xlabel="Hours from recording start",ylim=(-.35,4.35))
        ax.grid(axis="x",alpha=.14)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc="upper center",bbox_to_anchor=(.5,.958),frameon=False,fontsize=9)
    fig.suptitle("Held-out sleep timelines: expert, EEG control and full fusion model",fontsize=15,weight="bold",y=.994)
    fig.tight_layout(rect=(0,0,1,.93),pad=1.8)
    save(fig,root,"sleep_timelines")


def ecg_review(root,quality_cache):
    files=sorted(quality_cache.glob("ecg_review_*.npz"))
    mit=[]
    isruc=[]
    for path in files:
        with np.load(path) as data:
            record=str(data["record"])
        (mit if record.startswith("MIT") else isruc).append(path)
    selected=mit[:3]+isruc[:3]
    fig,axes=plt.subplots(3,2,figsize=(12,7.5))
    for ax,path in zip(axes.T.flat,selected):
        with np.load(path) as data:
            ecg=data["ecg"]
            fs=float(data["fs"])
            beats=data["beats"]
            record=str(data["record"])
            start=int(data["start_seconds"])
        normalized=(ecg-np.median(ecg))/max(float(ecg.std()),1e-8)
        time=np.arange(len(ecg))/fs
        ax.plot(time,normalized,color="#182638",lw=.8)
        beat_indices=np.minimum((beats*fs).astype(int),len(ecg)-1)
        ax.scatter(beats,normalized[beat_indices],color="#D55E00",s=22,marker="x",label="Detected heartbeat",zorder=3)
        ax.set(xlim=(0,10),title=f"{record} · start {start} s",xlabel="Seconds",ylabel="Normalized amplitude")
        ax.grid(alpha=.12)
    fig.suptitle("Training-only ECG review: fixed random 10-second segments",fontsize=15,weight="bold")
    fig.text(.5,.01,"Per-segment normalization is for display only. Markers are SleepECG detections, not independent expert annotations.",ha="center",fontsize=8,color="#536276")
    fig.tight_layout(rect=(0,.05,1,.98),pad=2)
    save(fig,root,"ecg_review")


def ecg_detection(root,audit):
    mit=audit.dropna(subset=["beat_f1"])
    tp=mit.matched_beats.sum()
    precision=tp/mit.detected_beats.sum()
    recall=tp/mit.reference_beats.sum()
    f1=2*precision*recall/(precision+recall)
    fig,axes=plt.subplots(1,2,figsize=(12,5.2),gridspec_kw={"width_ratios":[1,2]})
    bars=axes[0].bar(["Precision","Recall","F1"],[precision,recall,f1],color=["#0072B2","#009E73","#332288"])
    axes[0].bar_label(bars,labels=[f"{v:.4f}" for v in [precision,recall,f1]],padding=3)
    axes[0].set(ylim=(.94,1.01),title="Pooled MIT beat matching",ylabel="Score (axis starts at 0.94)")
    axes[1].barh(mit.record,mit.beat_f1,color="#0072B2")
    axes[1].set(xlim=(.97,1.005),title="Per-record detector F1",xlabel="Beat F1 (axis starts at 0.97)")
    axes[1].tick_params(axis="y",labelsize=8)
    axes[1].grid(axis="x",alpha=.15)
    fig.suptitle("ECG detector check against provided MIT annotations",fontsize=15,weight="bold")
    fig.text(.5,.015,"Greedy one-to-one matching within 150 ms; recognized WFDB beat symbols only. This evaluates beat detection, not sleep-stage classification.",ha="center",fontsize=8,color="#536276")
    fig.tight_layout(rect=(0,.06,1,.98),pad=2)
    save(fig,root,"ecg_detection")


def figures(results,quality_cache):
    style()
    root=results/"figures"
    frame=pd.read_csv(results/"epoch_manifest.csv")
    architecture(root)
    data_overview(root,frame)
    ecg_review(root,quality_cache)
    ecg_detection(root,pd.read_csv(quality_cache/"ecg_detector_audit.csv"))
    if not (results/"metrics.json").exists():
        print("Data, architecture and ECG figures generated; classification figures await frozen scores")
        return
    runs=pd.read_csv(results/"run_metrics.csv",dtype={"seed":str})
    predictions=pd.read_csv(results/"predictions.csv.gz",dtype={"seed":str})
    summary=json.loads((results/"metrics.json").read_text())
    comparison(root,runs)
    training_curves(root,results)
    domain_results(root,pd.read_csv(results/"per_dataset.csv",dtype={"seed":str}))
    class_f1(root,pd.read_csv(results/"per_class.csv",dtype={"seed":str}))
    confusions(root,summary)
    robustness(root,runs)
    subjects(root,pd.read_csv(results/"per_subject.csv",dtype={"seed":str}))
    gate_analysis(root,predictions)
    hypnograms(root,predictions)
    print("Generated 13 figure sets in PNG, SVG and PDF from preserved outputs")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",type=Path,default=PROJECT/"results"/"v2")
    parser.add_argument("--quality-cache",type=Path,default=PROJECT/"cache"/"v2")
    args=parser.parse_args()
    figures(args.results,args.quality_cache)
