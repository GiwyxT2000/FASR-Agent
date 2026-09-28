from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

from .common import RAW_DIR, ensure_dirs, write_json


DATASETS = {
    "queries": {
        "repo": "rnapberkeley/asrs",
        "suffixes": [".csv"],
        "known_files": [
            "train.csv",
            "validation.csv",
            "test.csv",
            "data/train.csv",
            "data/validation.csv",
            "data/test.csv",
        ],
    },
    "reports": {
        "repo": "elihoole/asrs-aviation-reports",
        "suffixes": [".jsonl"],
        "known_files": [
            "asrs-aviation-reports-train.jsonl",
            "asrs-aviation-reports-validation.jsonl",
            "asrs-aviation-reports-test.jsonl",
        ],
    },
}


def hf_api_url(repo: str) -> str:
    return f"https://huggingface.co/api/datasets/{repo}"


def hf_resolve_url(repo: str, filename: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}?download=true"


def fetch_json(url: str, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "fasr-agent-data-check/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def discover_files(repo: str, suffixes: Iterable[str], known_files: Iterable[str]) -> list[str]:
    discovered: list[str] = []
    try:
        meta = fetch_json(hf_api_url(repo))
        for sibling in meta.get("siblings", []):
            name = sibling.get("rfilename") or ""
            if any(name.endswith(suffix) for suffix in suffixes):
                discovered.append(name)
    except Exception as exc:
        print(f"[warn] Failed to query HuggingFace API for {repo}: {exc}")

    for name in known_files:
        if name not in discovered:
            discovered.append(name)
    return discovered


def download_file(url: str, target: Path, timeout: int = 600) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        print(f"[skip] {target} already exists ({target.stat().st_size} bytes)")
        return True

    print(f"[download] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "fasr-agent-data-check/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            tmp = target.with_suffix(target.suffix + ".tmp")
            with tmp.open("wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.replace(target)
        print(f"[ok] {target} ({target.stat().st_size} bytes)")
        return True
    except urllib.error.HTTPError as exc:
        print(f"[miss] {target.name}: HTTP {exc.code}")
        return False
    except Exception as exc:
        print(f"[error] {target.name}: {exc}")
        return False


def main() -> None:
    ensure_dirs()
    manifest: dict[str, dict] = {}
    for group, spec in DATASETS.items():
        repo = spec["repo"]
        target_dir = RAW_DIR / group
        files = discover_files(repo, spec["suffixes"], spec["known_files"])
        downloaded: list[str] = []
        attempted: list[str] = []
        print(f"\n== {group}: {repo} ==")
        for filename in files:
            attempted.append(filename)
            target = target_dir / filename.replace("/", "__")
            ok = download_file(hf_resolve_url(repo, filename), target)
            if ok:
                downloaded.append(str(target.relative_to(RAW_DIR)))
            time.sleep(0.2)
        manifest[group] = {
            "repo": repo,
            "attempted": attempted,
            "downloaded": downloaded,
        }
    write_json(RAW_DIR / "download_manifest.json", manifest)
    print(f"\nWrote manifest: {RAW_DIR / 'download_manifest.json'}")


if __name__ == "__main__":
    main()

