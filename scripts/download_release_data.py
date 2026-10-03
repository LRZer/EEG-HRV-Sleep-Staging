"""Download, verify, reassemble and extract the versioned research data assets."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
import requests

PROJECT=Path(__file__).resolve().parent.parent


def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def download(kind, destination, download_dir):
    manifest=json.loads((PROJECT/"datasets"/"release_assets.json").read_text())
    archives=[a for a in manifest["archives"] if kind=="all" or (kind=="prepared" and a["archive"].startswith("prepared"))
              or (kind=="mit" and a["archive"].startswith("mit"))]
    destination=destination.resolve()
    download_dir=download_dir.resolve()
    destination.mkdir(parents=True,exist_ok=True)
    download_dir.mkdir(parents=True,exist_ok=True)
    for archive in archives:
        if shutil.disk_usage(download_dir).free<archive["bytes"]*2:
            raise RuntimeError("Insufficient free space for parts and reassembled archive")
        for part in archive["parts"]:
            path=download_dir/part["asset"]
            if path.exists() and path.stat().st_size==part["bytes"] and sha(path)==part["sha256"]:
                print(f"Verified existing {path.name}",flush=True)
                continue
            url=f"https://github.com/{manifest['repository']}/releases/download/{manifest['release']}/{part['asset']}"
            response=requests.get(url,stream=True,timeout=(20,120))
            response.raise_for_status()
            partial=path.with_suffix(path.suffix+".partial")
            with partial.open("wb") as stream:
                for block in response.iter_content(1024*1024):
                    stream.write(block)
            assert partial.stat().st_size==part["bytes"] and sha(partial)==part["sha256"], "Part checksum mismatch"
            partial.replace(path)
            print(f"Downloaded and verified {path.name}",flush=True)
        path=download_dir/archive["archive"]
        with path.open("wb") as stream:
            for part in archive["parts"]:
                with (download_dir/part["asset"]).open("rb") as source:
                    shutil.copyfileobj(source,stream,4*1024*1024)
        assert path.stat().st_size==archive["bytes"] and sha(path)==archive["sha256"], "Archive checksum mismatch"
        with zipfile.ZipFile(path) as bundle:
            for entry in bundle.infolist():
                target=(destination/entry.filename).resolve()
                if not target.is_relative_to(destination):
                    raise ValueError("Archive path escapes destination")
            assert bundle.testzip() is None
            checksums=bundle.read("FILES.sha256").decode().splitlines()
            bundle.extractall(destination)
        for entry in checksums:
            expected,name=entry.split("  ",1)
            assert sha(destination/name)==expected, name
        print(f"Extracted {archive['archive']}; all per-file hashes verified",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind",choices=["prepared","mit","all"],default="prepared")
    parser.add_argument("--destination",type=Path,default=PROJECT)
    parser.add_argument("--download-dir",type=Path,default=PROJECT/"downloads")
    args=parser.parse_args()
    download(args.kind,args.destination,args.download_dir)
