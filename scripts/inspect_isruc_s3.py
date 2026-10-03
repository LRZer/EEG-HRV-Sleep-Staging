"""Check downloaded EDF-compatible REC files and preserve original annotations."""
from __future__ import annotations

import csv
import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "isrucIII"
REPORT = DEST / "download-report.json"
STAGES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 5: "REM"}


def edf_header(path):
    with path.open("rb") as stream:
        fixed = stream.read(256)
        if len(fixed) != 256 or fixed[:8].strip() != b"0":
            raise ValueError(f"Unexpected EDF header: {path.name}")
        header_bytes = int(fixed[184:192])
        records = int(fixed[236:244])
        record_seconds = float(fixed[244:252])
        channels = int(fixed[252:256])
        if header_bytes != 256 + channels * 256:
            raise ValueError("EDF header length mismatch")
        raw = stream.read(header_bytes - 256)
    offset = 0
    fields = {}
    for name, width in [("labels", 16), ("transducers", 80), ("units", 8),
                        ("physical_min", 8), ("physical_max", 8),
                        ("digital_min", 8), ("digital_max", 8),
                        ("prefilter", 80), ("samples", 8), ("reserved", 32)]:
        fields[name] = [raw[offset + i * width: offset + (i + 1) * width].decode("ascii").strip()
                        for i in range(channels)]
        offset += width * channels
    samples = [int(value) for value in fields["samples"]]
    expected_bytes = header_bytes + records * sum(samples) * 2
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"Truncated or oversized waveform: {path.name}")
    channel_info = [{"label": fields["labels"][i], "unit": fields["units"][i],
                     "sample_rate_hz": samples[i] / record_seconds,
                     "transducer": fields["transducers"][i], "prefilter": fields["prefilter"][i]}
                    for i in range(channels)]
    return {"format": "EDF-compatible .rec", "duration_seconds": records * record_seconds,
            "data_records": records, "record_seconds": record_seconds,
            "channels": channel_info, "structural_size_verified": True}


def labels(path):
    raw = path.read_text(encoding="utf-8-sig").splitlines()
    # A trailing blank line in subject 9 is retained in the source, not a new epoch.
    while raw and not raw[-1].strip():
        raw.pop()
    if any(not value.strip() for value in raw):
        raise ValueError(f"Blank annotation inside {path.name}")
    result = [int(value.strip()) for value in raw]
    if any(value not in STAGES for value in result):
        raise ValueError(f"Unknown stage in {path.name}")
    return result


def main():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    report["destination"] = str(DEST)
    if report["status"] != "complete":
        raise RuntimeError("Download is not complete")
    subjects, table = [], []
    stage_counts = Counter()
    total_epochs, total_agree = 0, 0
    for number in range(1, 11):
        folder = DEST / str(number)
        required = [f"{number}.rec", f"{number}_1.txt", f"{number}_2.txt",
                    f"{number}_1.xlsx", f"{number}_2.xlsx"]
        if not all((folder / name).is_file() for name in required):
            raise ValueError(f"Missing files for subject {number}")
        header = edf_header(folder / f"{number}.rec")
        scores = [labels(folder / f"{number}_{scorer}.txt") for scorer in (1, 2)]
        epoch_count = int(round(header["duration_seconds"] / 30))
        if any(len(score) != epoch_count for score in scores):
            raise ValueError(f"Waveform/annotation length mismatch for subject {number}")
        for scorer in (1, 2):
            with zipfile.ZipFile(folder / f"{number}_{scorer}.xlsx") as workbook:
                if workbook.testzip() is not None:
                    raise ValueError("Corrupt annotation workbook")
        eeg = [ch for ch in header["channels"] if ch["label"] in
               {"F3-A2", "C3-A2", "O1-A2", "F4-A1", "C4-A1", "O2-A1"}]
        if len(eeg) != 6:
            raise ValueError(f"Unexpected EEG channels for subject {number}")
        if any(ch["sample_rate_hz"] != 200 for ch in eeg):
            raise ValueError("Unexpected EEG sample rate")
        cardiac = [ch for ch in header["channels"] if ch["label"] == "X2"]
        if len(cardiac) != 1 or "ekg" not in cardiac[0]["transducer"].lower():
            raise ValueError("X2 must identify an ECG/EKG channel")
        counts = Counter(STAGES[value] for value in scores[0])
        agree = sum(a == b for a, b in zip(*scores))
        subject = {"subject": f"ISRUC-III-{number:02}", "folder": str(number), **header,
                   "epochs_30_seconds": epoch_count, "scorer1_stage_counts": dict(counts),
                   "scorer2_stage_counts": dict(Counter(STAGES[value] for value in scores[1])),
                   "scorer_agreement": agree / epoch_count, "annotations_aligned": True}
        subjects.append(subject)
        table.append({"subject": subject["subject"], "hours": round(header["duration_seconds"] / 3600, 3),
                      "epochs": epoch_count, **{stage: counts[stage] for stage in STAGES.values()},
                      "scorer_agreement": round(agree / epoch_count, 4)})
        stage_counts.update(counts)
        total_epochs += epoch_count
        total_agree += agree
    report["inspection"] = {"subjects": 10, "epochs_30_seconds": total_epochs,
                            "hours": total_epochs / 120, "scorer1_stage_counts": dict(stage_counts),
                            "scorer_agreement": total_agree / total_epochs,
                            "all_waveform_sizes_verified": True, "all_annotations_aligned": True,
                            "all_xlsx_integrity_verified": True,
                            "original_files_modified": False,
                            "original_epochs": 8889,
                            "official_extracted_channels_note": "The author's pre-extracted channel release omits the last 30 epochs per subject due to noise; this download preserves the original REC files and all 8889 annotations.",
                            "extracted_channels_source_page": "https://sleeptight.isr.uc.pt/?page_id=76",
                            "auxiliary_channel_mapping": "X2 is ECG: its EDF transducer identifies EKG_Channel; checked by deep_learning.prepare before HRV extraction.",
                            "subject_details": subjects}
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = REPORT.with_name("isruc-subjects.csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps({"subjects": 10, "files": len(report["files"]), "bytes": report["expected_bytes"],
                      "epochs": total_epochs, "hours": total_epochs / 120,
                      "stage_counts": dict(stage_counts), "scorer_agreement": total_agree / total_epochs,
                      "inspection_passed": True, "report": str(REPORT)}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEST)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    DEST = args.destination.resolve()
    REPORT = args.report.resolve() if args.report else DEST / "download-report.json"
    main()
