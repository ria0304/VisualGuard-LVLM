#!/usr/bin/env python
"""Convert the lmms-lab/POPE parquet release into the layout this repo expects.

The repo's loader (``src/data/pope.py``) wants, per split::

    <data-root>/coco_pope_<setting>.jsonl      one row per line, fields
                                               image / text / label / question_id
    <image-root>/<image filename>              real JPEG on disk

lmms-lab/POPE ships those same POPE rows as parquet with the JPEG bytes
embedded in an ``image`` column. This script unpacks them, so the benchmark can
run without separately downloading COCO val2017.

Labels are re-emitted as ``yes`` / ``no`` because ``_label_to_bool`` in the repo
expects those strings; the parquet stores 0/1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet-dir", required=True,
                    help="Directory holding <setting>-00000-of-00001.parquet")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Cap rows per split. Images are still written for "
                         "every capped row so no sample dangles.")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from PIL import Image
    import io

    data_root = Path(args.data_root)
    image_root = Path(args.image_root)
    data_root.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)

    for setting in ("random", "popular", "adversarial"):
        pq_path = Path(args.parquet_dir) / f"{setting}-00000-of-00001.parquet"
        if not pq_path.is_file():
            print(f"missing {pq_path}", file=sys.stderr)
            return 1

        table = pq.read_table(pq_path)
        cols = set(table.column_names)
        if "image" not in cols:
            print(f"{pq_path} has no 'image' column; got {sorted(cols)}",
                  file=sys.stderr)
            return 1

        rows = table.to_pylist()
        if args.max_rows is not None:
            rows = rows[: args.max_rows]

        out_path = data_root / f"coco_pope_{setting}.jsonl"
        written = 0
        with out_path.open("w", encoding="utf-8") as fh:
            for i, row in enumerate(rows):
                cell = row["image"]
                if isinstance(cell, dict):
                    raw = cell.get("bytes")
                    filename = cell.get("path")
                else:
                    raw, filename = cell, None
                if raw is None:
                    continue
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                name = Path(str(filename or row.get("id") or i)).name
                if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                    name = f"{Path(name).stem}.jpg"
                img.save(image_root / name, format="JPEG", quality=95)

                answer = row.get("answer")
                label = str(answer).strip().lower()
                if label in ("yes", "true", "1"):
                    label = "yes"
                elif label in ("no", "false", "0"):
                    label = "no"
                else:
                    print(f"unexpected label {answer!r} in row {i}", file=sys.stderr)
                    return 1

                fh.write(json.dumps({
                    "question_id": str(row.get("question_id", i)),
                    "image": name,
                    "text": row.get("question", ""),
                    "label": label,
                }) + "\n")
                written += 1
        print(f"{setting}: wrote {written} rows -> {out_path}")

    print(f"images -> {image_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())