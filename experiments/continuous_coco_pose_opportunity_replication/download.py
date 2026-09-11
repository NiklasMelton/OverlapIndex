"""Download the hash-planned COCO image cohort from the official image host."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
from urllib.request import Request, urlopen

from PIL import Image

from experiments.continuous_coco_pose_opportunity_replication import prepare


def _url(protocol: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    key = "train_image_url_template" if row["source_split"] == "train" else "validation_image_url_template"
    return str(protocol["dataset"][key]).format(file_name=row["file_name"])


def _valid(path: Path, row: Mapping[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            return image.size == (int(row["width"]), int(row["height"])) and image.format == "JPEG"
    except OSError:
        return False


def _download_one(
    input_root: Path, protocol: Mapping[str, Any], row: Mapping[str, Any]
) -> tuple[str, bool]:
    path = input_root / row["image_path"]
    if _valid(path, row):
        return str(row["sample_id"]), True
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    url = _url(protocol, row)
    last_error: BaseException | None = None
    for attempt in range(6):
        try:
            request = Request(url, headers={"User-Agent": "OverlapIndex-research/1.0"})
            with urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    handle.write(block)
            if not _valid(temporary, row):
                raise ValueError("downloaded JPEG identity is malformed")
            temporary.replace(path)
            return str(row["sample_id"]), False
        except BaseException as error:
            last_error = error
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            if attempt < 5:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {row['sample_id']} from {url}") from last_error


def run(
    input_root: Path = prepare.DEFAULT_INPUT_ROOT,
    prepared_root: Path = prepare.DEFAULT_OUTPUT_ROOT,
    *,
    workers: int = 8,
) -> dict[str, Any]:
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("workers must be an integer from 1 through 16")
    protocol = prepare.load_protocol()
    plan = prepare.load_plan(prepared_root.resolve())
    rows = plan["rows"]
    reused = 0
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_one, input_root.resolve(), protocol, row): row
            for row in rows
        }
        for future in as_completed(futures):
            _sample_id, was_reused = future.result()
            reused += int(was_reused)
            completed += 1
            if completed % 100 == 0 or completed == len(rows):
                print(f"downloaded/validated {completed}/{len(rows)}", flush=True)
    return {
        "status": "completed_coco_pose_image_download",
        "image_count": len(rows),
        "reused": reused,
        "downloaded": len(rows) - reused,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=prepare.DEFAULT_INPUT_ROOT)
    parser.add_argument("--prepared-root", type=Path, default=prepare.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    print(prepare.canonical_json(run(args.input_root, args.prepared_root, workers=args.workers)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
