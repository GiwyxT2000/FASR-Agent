from __future__ import annotations

import gzip
import json
import lzma
import re
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
NOTES_DIR = PROJECT_ROOT / "notes"


def ensure_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, OUTPUTS_DIR, NOTES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def normalize_id(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    text = text.replace(",", "")
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    digits = re.findall(r"\d+", text)
    if len(digits) == 1 and digits[0] == text:
        return digits[0]
    if text.isdigit():
        return text
    return text


def compact_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    # The report corpus is far past GitHub's per-file size limit, so a rebuilt
    # corpus is normally kept compressed. Callers always pass the logical name,
    # so transparently fall back to a compressed sibling when the plain file is
    # absent. Behaviour is unchanged when an uncompressed file exists.
    if not path.exists():
        for suffix in (".xz", ".gz"):
            candidate = path.with_name(path.name + suffix)
            if candidate.exists():
                path = candidate
                break
    if path.suffix == ".xz":
        opener = lzma.open
    elif path.suffix == ".gz":
        opener = gzip.open
    else:
        opener = open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

