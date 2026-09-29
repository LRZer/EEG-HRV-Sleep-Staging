"""Rebuild the trained model from checksummed repository parts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PARTS_DIR = ROOT / "artifacts" / "model_parts"
MANIFEST = PARTS_DIR / "manifest.json"
OUTPUT = ROOT / "results" / "fusion_model.joblib"


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if OUTPUT.exists() and digest_file(OUTPUT) == manifest["sha256"]:
        print(f"Model already verified: {OUTPUT}")
        return
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(OUTPUT.name + ".part")
    with temporary.open("wb") as target:
        for part in manifest["parts"]:
            path = PARTS_DIR / part["name"]
            if digest_file(path) != part["sha256"]:
                raise ValueError(f"SHA256 mismatch: {path}")
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(chunk)
    if digest_file(temporary) != manifest["sha256"]:
        raise ValueError("Rebuilt model SHA256 mismatch")
    temporary.replace(OUTPUT)
    print(f"Restored model: {OUTPUT}")


if __name__ == "__main__":
    main()

