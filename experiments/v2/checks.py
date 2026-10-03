"""Check the 15-window boundaries and quality-gated fallback before formal training."""
import json
from pathlib import Path
import numpy as np
import torch
from .data import Data, load_data
from .models import FusionModel

PROJECT = Path(__file__).resolve().parents[2]


def check():
    frame, arrays, _ = load_data(PROJECT / "cache" / "deep_learning", PROJECT / "cache" / "v2")
    contexts = arrays["contexts"]
    record, epoch, split = frame.record.to_numpy(), frame.epoch.to_numpy(), frame.split.to_numpy()
    for position in range(15):
        keep = contexts[:, position] >= 0
        current = np.flatnonzero(keep)
        past = contexts[keep, position]
        assert (record[current] == record[past]).all()
        assert (split[current] == split[past]).all()
        assert (epoch[current] - epoch[past] == 14 - position).all()
    assert np.isfinite(arrays["hrv"]).all() and arrays["hrv"].shape[1] == 55
    protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())
    data = Data(arrays, torch.device("cuda"))
    first = [int(g.index[0]) for _, g in frame.loc[frame.split == "train"].groupby("record")]
    idx = torch.as_tensor(first[:8], device=data.device)
    for variant in protocol["variants"]:
        model = FusionModel(variant, protocol).to(data.device)
        spectra, cardiac, missing, labels = data.batch(idx, model.context_length)
        logits = model(spectra, cardiac, missing)
        assert torch.isfinite(logits).all() and logits.shape == (8, 5)
        torch.nn.functional.cross_entropy(logits, labels).backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if model.spec["fusion"] == "gate":
            model.eval()
            with torch.inference_mode():
                x = model(spectra, cardiac, missing, force_hrv_missing=True)
                y = model(spectra, cardiac + 100, missing, force_hrv_missing=True)
                torch.testing.assert_close(x, y, atol=0, rtol=0)
        print(f"PASS {variant}: padded context, forward/backward and missing-HRV fallback", flush=True)
    print("PASS: 15-epoch same-record past-only contexts; 55-dimensional finite cardiac input", flush=True)


if __name__ == "__main__":
    check()
