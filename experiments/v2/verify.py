"""Verify preserved scores, subject isolation, checkpoint hashes and restoration."""
import argparse
import ast
import hashlib
import json
import re
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from deep_learning.evaluate import STAGES,score
from .predict import restore

PROJECT=Path(__file__).resolve().parents[2]


def verify(cache,quality_cache,results):
    RESULTS=results
    protocol_bytes=(RESULTS/"protocol.json").read_bytes()
    assert protocol_bytes==(PROJECT/"experiments"/"v2"/"protocol.json").read_bytes()
    protocol=json.loads(protocol_bytes)
    digest=hashlib.sha256(protocol_bytes).hexdigest()
    completion=json.loads((RESULTS/"training_complete.json").read_text(encoding="utf-8"))
    assert completion["runs"]==21 and completion["protocol_sha256"]==digest
    manifest=pd.read_csv(RESULTS/"epoch_manifest.csv")
    assert manifest[manifest.split=="train"].subject.nunique()==22
    assert manifest[manifest.split=="test"].subject.nunique()==4
    assert not set(manifest[manifest.split=="train"].subject)&set(manifest[manifest.split=="test"].subject)
    metadata={}
    for variant in protocol["variants"]:
        for seed in protocol["seeds"]:
            path=RESULTS/"checkpoints"/f"{variant}_seed{seed}.pt"
            checkpoint=torch.load(path,map_location="cpu",weights_only=False)
            assert checkpoint["protocol_sha256"]==digest and checkpoint["epochs"]==40
            assert sorted(checkpoint["train_subjects"])==sorted(manifest[manifest.split=="train"].subject.unique())
            history=pd.read_csv(RESULTS/f"{variant}_seed{seed}_history.csv")
            assert len(history)==40 and history.epoch.tolist()==list(range(1,41))
            metadata[path.name]={"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    summary=json.loads((RESULTS/"metrics.json").read_text(encoding="utf-8"))
    predictions=pd.read_csv(RESULTS/"predictions.csv.gz",dtype={"seed":str})
    assert len(predictions)==7*4*2*3009
    for (variant,seed,condition),group in predictions.groupby(["variant","seed","condition"]):
        assert len(group)==3009
        recalculated=score(group.stage,group.prediction)
        reported=summary["models"][f"{variant}_seed{seed}_{condition}"]
        for key in ["accuracy","macro_f1","balanced_accuracy","kappa"]:
            assert abs(reported[key]-recalculated[key])<1e-12
        assert recalculated["confusion_matrix"]==reported["confusion_matrix"]
        if seed!="ensemble":
            assert reported["checkpoint_sha256"]==metadata[f"{variant}_seed{seed}.pt"]["sha256"]
    for variant in ["A0","A1"]:
        normal=predictions[(predictions.variant==variant)&(predictions.condition=="normal")]
        missing=predictions[(predictions.variant==variant)&(predictions.condition=="hrv_missing")]
        np.testing.assert_array_equal(normal[[f"p_{s}" for s in STAGES]].to_numpy(),missing[[f"p_{s}" for s in STAGES]].to_numpy())
    for variant in ["A4","A5","A6"]:
        assert (predictions[(predictions.variant==variant)&(predictions.condition=="hrv_missing")].gate==0).all()
    with tempfile.TemporaryDirectory(prefix="sleep-staging-verify-") as folder:
        restored_path=Path(folder)/"restored.csv"
        restore(cache,quality_cache,RESULTS,"A6","ensemble",restored_path)
        restored=pd.read_csv(restored_path)
    reference=predictions[(predictions.variant=="A6")&(predictions.seed=="ensemble")&(predictions.condition=="normal")]
    delta=float(np.max(np.abs(restored[[f"p_{s}" for s in STAGES]].to_numpy()-reference[[f"p_{s}" for s in STAGES]].to_numpy())))
    assert delta<1e-6
    assert np.array_equal(restored.predicted_stage.to_numpy(),np.asarray(STAGES)[reference.prediction.to_numpy()])
    for folder in [PROJECT/"experiments"/"v2",PROJECT/"scripts"]:
        for path in folder.glob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
    docs=[PROJECT/"README.md",PROJECT/"README.zh-CN.md",PROJECT/"PROJECT.md",*list((PROJECT/"docs").glob("*.md")),
          RESULTS/"report.en.md",RESULTS/"report.zh-CN.md",PROJECT/"datasets"/"README.md"]
    for path in docs:
        for link in re.findall(r"\]\(([^)]+)\)",path.read_text(encoding="utf-8")):
            if not link.startswith(("https://","http://","#")):
                assert (path.parent/link).exists(),(path,link)
    for asset in json.loads((PROJECT/"datasets"/"release_assets.json").read_text(encoding="utf-8"))["archives"]:
        assert sum(p["bytes"] for p in asset["parts"])==asset["bytes"]
        assert (PROJECT/"datasets"/(asset["archive"]+".manifest.json")).exists()
    result={"protocol_sha256":digest,"runs":21,"training_epochs":840,"prediction_rows":len(predictions),
        "recalculated_prediction_groups":56,"subject_isolation":True,"EEG_only_unchanged_by_missing_HRV":True,
        "missing_HRV_gates_exactly_zero":True,"restored_A6_ensemble_labels_identical":True,
        "restored_max_probability_difference":delta,"syntax_and_document_links":True,
        "cache_used_for_restoration":str(cache.resolve()),"checkpoints":metadata}
    (RESULTS/"verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("VERIFIED: 21 checkpoints, 840 logs, 56 score groups, all document links and exact A6 ensemble restoration")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,default=PROJECT/"cache/deep_learning")
    parser.add_argument("--quality-cache",type=Path,default=PROJECT/"cache/v2")
    parser.add_argument("--results",type=Path,default=PROJECT/"results/v2")
    args=parser.parse_args()
    verify(args.cache,args.quality_cache,args.results)
