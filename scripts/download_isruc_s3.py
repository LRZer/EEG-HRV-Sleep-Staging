"""Download the original ISRUC-Sleep III files from the author's public MEGA folder.

Only the public share key linked by the dataset authors is used. Each decrypted
file is checked against MEGA's file authentication tag, then locally SHA-256 hashed.
Partial downloads can be resumed. No account or credentials are needed.
"""
from __future__ import annotations

import base64
import argparse
import hashlib
import json
import shutil
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import requests
from Crypto.Cipher import AES

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "isrucIII"
REPORT = DEST / "download-report.json"
FOLDER_HANDLE = "hMIDEBBJ"
SHARE_KEY = "jzU5WOF7V5fQCgykEkOptA"
SOURCE_PAGE = "https://sleeptight.isr.uc.pt/?page_id=48"
SOURCE_SHARE = f"https://mega.nz/folder/{FOLDER_HANDLE}#{SHARE_KEY}"
API = "https://g.api.mega.co.nz/cs"
PRINT_LOCK = threading.Lock()


def log(message):
    with PRINT_LOCK:
        print(message, flush=True)


def unb64(value):
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def pack(words):
    return struct.pack(">" + "I" * len(words), *words)


def words(value):
    return struct.unpack(">" + "I" * (len(value) // 4), value)


def api(command):
    for attempt in range(5):
        try:
            response = requests.post(
                API, params={"id": time.time_ns(), "n": FOLDER_HANDLE},
                json=[command], timeout=(15, 45),
            )
            response.raise_for_status()
            result = response.json()
            result = result[0] if isinstance(result, list) else result
            if isinstance(result, int):
                raise RuntimeError(f"MEGA API code {result}")
            return result
        except (requests.RequestException, RuntimeError, ValueError):
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 8))


def inventory():
    folder_key = unb64(SHARE_KEY)
    decoded = {}
    for item in api({"a": "f", "c": 1, "r": 1})["f"]:
        for wrapping in item.get("k", "").split("/"):
            if ":" not in wrapping:
                continue
            try:
                raw_key = AES.new(folder_key, AES.MODE_ECB).decrypt(
                    unb64(wrapping.split(":", 1)[1])
                )
                key_words = words(raw_key)
                aes_key = pack([key_words[i] ^ key_words[i + 4] for i in range(4)]) if len(key_words) == 8 else raw_key
                attributes = AES.new(aes_key, AES.MODE_CBC, bytes(16)).decrypt(unb64(item["a"]))
                if not attributes.startswith(b"MEGA"):
                    continue
                attributes = json.loads(attributes[4:].rstrip(b"\0"))
                decoded[item["h"]] = {**item, "name": attributes["n"], "key_words": key_words}
                break
            except (ValueError, KeyError, UnicodeError):
                continue
        else:
            raise RuntimeError(f"Could not decode public node {item['h']}")

    def relative_path(node):
        pieces = [node["name"]]
        while node.get("p") in decoded:
            node = decoded[node["p"]]
            if node.get("p") in decoded:
                pieces.append(node["name"])
        result = PurePosixPath(*reversed(pieces))
        if result.is_absolute() or any(p in {"..", "."} for p in result.parts):
            raise ValueError("Invalid remote path")
        if len(result.parts) != 2 or result.parts[0] not in {str(i) for i in range(1, 11)}:
            raise ValueError(f"Unexpected subject path: {result}")
        return result

    files = []
    for item in decoded.values():
        if item["t"] == 0:
            item["relative_path"] = str(relative_path(item))
            files.append(item)
    files.sort(key=lambda n: (int(PurePosixPath(n["relative_path"]).parts[0]), n["name"]))
    return files


def chunk_sizes(size):
    position, chunk = 0, 128 * 1024
    while position < size:
        length = min(chunk, size - position)
        yield length
        position += length
        chunk = min(chunk + 128 * 1024, 1024 * 1024)


def authenticate(path, file_key):
    aes_key = pack([file_key[i] ^ file_key[i + 4] for i in range(4)])
    iv = pack([file_key[4], file_key[5], file_key[4], file_key[5]])
    aggregate = bytes(16)
    sha256 = hashlib.sha256()
    size = path.stat().st_size
    with path.open("rb") as source:
        for length in chunk_sizes(size):
            plaintext = source.read(length)
            if len(plaintext) != length:
                raise IOError("Incomplete file during authentication")
            sha256.update(plaintext)
            if len(plaintext) % 16:
                plaintext += bytes(16 - len(plaintext) % 16)
            chunk_mac = AES.new(aes_key, AES.MODE_CBC, iv).encrypt(plaintext)[-16:]
            aggregate = AES.new(aes_key, AES.MODE_CBC, aggregate).encrypt(chunk_mac)
    mac = words(aggregate)
    if (mac[0] ^ mac[1], mac[2] ^ mac[3]) != (file_key[6], file_key[7]):
        raise ValueError(f"MEGA authentication failed for {path.name}")
    return sha256.hexdigest()


def fetch_range(url, start, end, aes_key, initial_counter):
    """Fetch an exact encrypted byte range and decrypt at its original offset."""
    for attempt in range(4):
        try:
            with requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                              stream=True, timeout=(20, 60)) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise IOError("Server did not honor exact byte range")
                if not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
                    raise IOError("Incorrect response byte range")
                cipher = AES.new(aes_key, AES.MODE_CTR, nonce=b"",
                                 initial_value=initial_counter + start // 16)
                if start % 16:
                    cipher.decrypt(bytes(start % 16))
                plaintext = bytearray()
                for encrypted in response.iter_content(1024 * 1024):
                    plaintext.extend(cipher.decrypt(encrypted))
                    if len(plaintext) > end - start + 1:
                        raise IOError("Range response exceeded requested size")
                if len(plaintext) != end - start + 1:
                    raise IOError("Incomplete byte range")
                return plaintext
        except (requests.RequestException, IOError):
            if attempt == 3:
                raise
            time.sleep(min(2 ** attempt, 4))


def download(item):
    target = DEST / item["relative_path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_size = item["s"]
    key = item["key_words"]
    if target.exists() and target.stat().st_size == expected_size:
        checksum = authenticate(target, key)
        log(f"VERIFIED {item['relative_path']} ({expected_size:,} bytes)")
    else:
        partial = target.with_name(target.name + ".part")
        aes_key = pack([key[i] ^ key[i + 4] for i in range(4)])
        initial_counter = int.from_bytes(pack([key[4], key[5], 0, 0]), "big")
        for attempt in range(6):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                if offset > expected_size:
                    raise ValueError(f"Oversized partial file {partial.name}")
                if offset == expected_size:
                    break
                info = api({"a": "g", "g": 1, "n": item["h"]})
                if "g" not in info or info.get("s", expected_size) != expected_size:
                    raise RuntimeError("Invalid MEGA download metadata")
                if expected_size >= 10 * 1024 * 1024:
                    # Several exact ranges prevent one slow transfer from blocking a record.
                    with partial.open("ab" if offset else "wb") as output, ThreadPoolExecutor(max_workers=4) as ranges:
                        written, last_log = offset, time.monotonic()
                        while written < expected_size:
                            starts = list(range(written, min(written + 16 * 1024 * 1024, expected_size), 4 * 1024 * 1024))
                            pieces = [ranges.submit(fetch_range, info["g"], start,
                                      min(start + 4 * 1024 * 1024, expected_size) - 1,
                                      aes_key, initial_counter) for start in starts]
                            for piece in pieces:
                                plaintext = piece.result()
                                output.write(plaintext)
                                output.flush()
                                written += len(plaintext)
                            if time.monotonic() - last_log >= 20:
                                log(f"PROGRESS {item['relative_path']}: {written / expected_size:.0%} ({written / 1e6:.1f} MB)")
                                last_log = time.monotonic()
                    if partial.stat().st_size != expected_size:
                        raise IOError("Download length mismatch")
                    break
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with requests.get(info["g"], headers=headers, stream=True, timeout=(20, 60)) as response:
                    response.raise_for_status()
                    if offset and response.status_code != 206:
                        raise IOError("Server did not honor resume range")
                    cipher = AES.new(aes_key, AES.MODE_CTR, nonce=b"", initial_value=initial_counter + offset // 16)
                    if offset % 16:
                        cipher.decrypt(bytes(offset % 16))
                    mode = "ab" if offset else "wb"
                    written, last_log = offset, time.monotonic()
                    with partial.open(mode) as output:
                        for encrypted in response.iter_content(1024 * 1024):
                            if not encrypted:
                                continue
                            if written + len(encrypted) > expected_size:
                                raise IOError("Download exceeded declared file size")
                            output.write(cipher.decrypt(encrypted))
                            written += len(encrypted)
                            if time.monotonic() - last_log >= 20:
                                log(f"PROGRESS {item['relative_path']}: {written / expected_size:.0%} ({written / 1e6:.1f} MB)")
                                last_log = time.monotonic()
                if partial.stat().st_size != expected_size:
                    raise IOError("Download length mismatch")
                break
            except (requests.RequestException, RuntimeError, IOError) as error:
                if attempt == 5:
                    raise
                log(f"RETRY {item['relative_path']}: {type(error).__name__}")
                time.sleep(min(2 ** attempt, 10))
        checksum = authenticate(partial, key)
        partial.replace(target)
        log(f"DOWNLOADED {item['relative_path']} ({expected_size:,} bytes; authenticated)")
    return {"path": item["relative_path"], "bytes": expected_size, "sha256": checksum,
            "source_node": item["h"], "mega_mac_verified": True}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    files = inventory()
    total = sum(item["s"] for item in files)
    existing = sum(min((DEST / item["relative_path"]).stat().st_size, item["s"])
                   for item in files if (DEST / item["relative_path"]).exists())
    if shutil.disk_usage(DEST).free < total - existing + 500 * 1024 * 1024:
        raise RuntimeError("Insufficient free disk space")
    log(f"INVENTORY: {len(files)} files, {total:,} bytes ({total / 1024**3:.3f} GiB), 10 subjects")
    report = {"dataset": "ISRUC-Sleep Subgroup III", "source_page": SOURCE_PAGE,
              "source_folder": SOURCE_SHARE, "destination": str(DEST),
              "expected_files": len(files), "expected_bytes": total,
              "status": "downloading", "files": []}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Download annotations first, then four waveform files concurrently.
    files.sort(key=lambda n: n["s"])
    errors = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(download, item): item for item in files}
        for job in as_completed(jobs):
            try:
                report["files"].append(job.result())
            except Exception as error:
                item = jobs[job]
                errors.append({"path": item["relative_path"], "error": str(error)})
                log(f"FAILED {item['relative_path']}: {type(error).__name__}: {error}")
            report["files"].sort(key=lambda n: n["path"])
            report["errors"] = errors
            REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["status"] = "complete" if not errors and len(report["files"]) == len(files) else "incomplete"
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"RESULT: {report['status']}; {len(report['files'])}/{len(files)} files; report={REPORT}")
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEST)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    DEST = args.destination.resolve()
    REPORT = args.report.resolve() if args.report else DEST / "download-report.json"
    main()
