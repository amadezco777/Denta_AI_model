"""Datos y revisiones ficticias EXCLUSIVAMENTE para pruebas de software."""

import csv
import json

import cv2
import numpy as np
import yaml

from src.data.contract import create_contract, write_review_template
from src.data.provenance import DATASET_EXCLUDE, fingerprint


def synthetic_dataset(root, n=2, splits=("train", "val"), *, external=False):
    for split_no, split in enumerate(splits):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(n):
            img = np.random.default_rng(100 * split_no + i).integers(30, 50, (64, 64, 3), dtype=np.uint8)
            cv2.rectangle(img, (20, 20), (40, 40), (220, 220, 220), -1)
            cv2.imwrite(str(root / "images" / split / f"{i}.png"), img)
            (root / "labels" / split / f"{i}.txt").write_text(f"{i % 2} 0.47 0.47 0.33 0.33\n")
    meta = {"taxonomy_version": "0.2.0", "role": "external_evaluation" if external else "development",
            "provenance": {"output_fingerprint": fingerprint(root, DATASET_EXCLUDE)}}
    data = {"path": str(root.resolve()), "names": {0: "caries", 1: "implante"}, "meta": meta}
    if external:
        data.update(val=f"images/{splits[0]}", test=f"images/{splits[0]}")
    else:
        data.update({split: f"images/{split}" for split in splits})
    data_yaml = root / "dataset.yaml"
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    review = root.parent / f"{root.name}.review.csv"
    lock = root.parent / f"{root.name}.lock.json"
    write_review_template(data_yaml, review)
    approve_synthetic_review(review)
    create_contract(data_yaml, lock, review)
    return data_yaml, lock, review


def approve_synthetic_review(path):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        row.update(label_status="approved", reviewer="synthetic-test-only", group_id=row["archivo"],
                   group_basis="image_only", group_verified="true")
        row["coverage"] = json.dumps({name: "complete" for name in json.loads(row["coverage"])})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
