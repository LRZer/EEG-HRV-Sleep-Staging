"""Restore one frozen v2 run or the predeclared three-seed ensemble."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .data import Data,load_data
from .evaluate import predict as predict_run,STAGES

PROJECT=Path(__file__).resolve().parents[2]


def restore(cache,quality_cache,results,variant,seed,output):
    with np.load(results/"preprocessing.npz") as saved:
        fitted={k:saved[k] for k in saved.files}
    frame,arrays,_=load_data(cache,quality_cache,fitted)
    rows=np.flatnonzero(frame.split.to_numpy()=="test")
    data=Data(arrays,torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    protocol=json.loads((results/"protocol.json").read_text())
    seeds=protocol["seeds"] if seed=="ensemble" else [int(seed)]
    probabilities=[]
    for value in seeds:
        checkpoint=torch.load(results/"checkpoints"/f"{variant}_seed{value}.pt",map_location="cpu",weights_only=False)
        p,_=predict_run(checkpoint,data,rows)
        probabilities.append(p)
    probability=np.mean(probabilities,axis=0)
    result=frame.iloc[rows].reset_index(drop=True).copy()
    result["predicted_stage"]=np.asarray(STAGES)[probability.argmax(1)]
    for i,stage in enumerate(STAGES):
        result[f"p_{stage}"]=probability[:,i]
    output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(output,index=False)
    print(f"Restored {variant} {seed}; predicted {len(rows)} aligned test epochs; {output}")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,default=PROJECT/"cache"/"deep_learning")
    parser.add_argument("--quality-cache",type=Path,default=PROJECT/"cache"/"v2")
    parser.add_argument("--results",type=Path,default=PROJECT/"results"/"v2")
    parser.add_argument("--variant",choices=[f"A{i}" for i in range(7)],default="A6")
    parser.add_argument("--seed",choices=["42","123","2026","ensemble"],default="ensemble")
    parser.add_argument("--output",type=Path,default=PROJECT/"results"/"restored_v2_predictions.csv")
    args=parser.parse_args()
    restore(args.cache,args.quality_cache,args.results,args.variant,args.seed,args.output)
