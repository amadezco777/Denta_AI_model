"""Recover the original research archive with checksums and safe extraction.

The destination is local application data, outside the OneDrive workspace.
No clinical approval, label completion, or patient identity is inferred.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import stat
import time
import urllib.request
from zipfile import ZipFile

NAME = "panoramic_radiography_yolo_dataset_14_classes.zip"
URL = "https://zenodo.org/records/15487430/files/" + NAME + "?download=1"
SIZE = 1397393531
MD5 = "18d5acaea021f4f3b05b76e9a27fbd42"


def file_hash(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def recover(root, workers=10):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / NAME
    prefix = root / (NAME + ".part")
    if target.exists():
        if target.stat().st_size != SIZE or file_hash(target, "md5") != MD5:
            raise ValueError("Existing archive differs from the published checksum")
        return target
    start = prefix.stat().st_size if prefix.exists() else 0
    if start > SIZE:
        raise ValueError("Partial archive exceeds the expected size")
    parts = root / "download_parts"
    parts.mkdir(exist_ok=True)
    ranges = [(a, min(a + 16 * 1024 * 1024 - 1, SIZE - 1)) for a in range(start, SIZE, 16 * 1024 * 1024)]

    def download(pair):
        a, b = pair
        destination = parts / f"{a}-{b}.part"
        size = b - a + 1
        if destination.exists() and destination.stat().st_size == size:
            return destination
        for attempt in range(5):
            try:
                req = urllib.request.Request(URL, headers={"Range": f"bytes={a}-{b}"})
                with urllib.request.urlopen(req, timeout=90) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {a}-{b}/{SIZE}":
                        raise ValueError("Server did not honor the exact byte range")
                    with destination.open("wb") as out:
                        shutil.copyfileobj(response, out, 1 << 20)
                if destination.stat().st_size != size:
                    raise ValueError("Incomplete range")
                return destination
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 10))

    completed = start
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(download, pair): pair for pair in ranges}
        for future in as_completed(pending):
            path = future.result()
            completed += path.stat().st_size
            print(json.dumps({"downloaded_bytes": completed, "total_bytes": SIZE,
                              "percent": round(100 * completed / SIZE, 1)}), flush=True)
    assembled = root / (NAME + ".assembling")
    with assembled.open("wb") as out:
        if start:
            with prefix.open("rb") as inp:
                shutil.copyfileobj(inp, out, 1 << 20)
        for a, b in ranges:
            with (parts / f"{a}-{b}.part").open("rb") as inp:
                shutil.copyfileobj(inp, out, 1 << 20)
    if assembled.stat().st_size != SIZE or file_hash(assembled, "md5") != MD5:
        raise ValueError("Archive does not match the published MD5")
    assembled.rename(target)
    (root / "archive_verification.json").write_text(json.dumps({
        "url": URL, "bytes": SIZE, "published_md5": MD5, "md5_verified": True,
        "sha256": file_hash(target), "annotation_status": "source_annotations_not_locally_adjudicated",
        "intended_use": "noncommercial_academic_research",
    }, indent=2), encoding="utf-8")
    return target


def extract(archive, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Extraction destination already exists")
    with ZipFile(archive) as source:
        for entry in source.infolist():
            name = PurePosixPath(entry.filename.replace("\\", "/"))
            if name.is_absolute() or ".." in name.parts or PureWindowsPath(entry.filename).drive:
                raise ValueError("Unsafe archive member")
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError("Symbolic link in source archive")
            if not (destination / Path(*name.parts)).resolve().is_relative_to(destination):
                raise ValueError("Archive member outside destination")
        destination.mkdir(parents=True)
        source.extractall(destination)
        bad = source.testzip()
        if bad:
            raise ValueError(f"CRC failure: {bad}")
    return destination


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=10)
    args = p.parse_args()
    archive = recover(args.root, args.workers)
    destination = args.root / "raw_verified"
    if not destination.exists():
        extract(archive, destination)
    print(json.dumps({"archive": str(archive), "extracted": str(destination)}), flush=True)
