"""Source-annotated research cohorts, separate from clinician-approved contracts.

These manifests attest byte integrity and partition separation only. They do
not assert local clinical adjudication, complete annotations, or patient IDs.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tempfile
import os

import cv2
import numpy as np
import yaml

from src.data.annotations import read_yolo_boxes
from src.data.contract import sha256_file
from src.data.preprocessing import DEFAULT_MASK, apply_mask, boxes_under_mask
from src.data.similarity import thumbnail, similar_pairs, connected_groups
from src.data.splitting import stratified_group_split
from src.data.provenance import git_state, environment

NAMES = ["caries", "lesion_periapical", "diente_impactado", "restauracion",
         "endodoncia", "protesis_fija", "implante"]
MAPPING = {4: 0, 7: 1, 6: 2, 2: 3, 3: 4, 1: 5, 0: 6}
SPLITS = {"train": "train", "valid": "valid", "test": "test",
          "external": "test_alte_cabinete/Ext-validation"}
REFERENCE = "source_annotations_not_locally_adjudicated"
PREPROCESSING = {"version": "research-gray-mask-max1280-v1", "mask": DEFAULT_MASK, "max_side": 1280}


def prepare_image(image, max_side=1280):
    if image is None or image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError("An 8-bit decoded image is required")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = apply_mask(image.copy())
    h, w = image.shape
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        image = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return image


def inventory(root):
    return {p.relative_to(root).as_posix(): sha256_file(p)
            for p in sorted(Path(root).rglob("*")) if p.is_file()
            and not (p.parent.name == "labels" and p.suffix == ".cache")}


def verify_research_contract(contract_path, for_training=False):
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    if contract.get("reference_status") != REFERENCE or contract.get("schema") != "source-research-v1":
        raise ValueError("Not a source-annotated research contract")
    root = Path(contract["root"]).resolve()
    if not root.is_dir() or any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("Missing research view or symbolic links")
    if inventory(root) != contract["files"]:
        raise ValueError("Research dataset bytes changed after freezing")
    cfg = yaml.safe_load((root / "dataset.yaml").read_text(encoding="utf-8"))
    if cfg.get("names") != NAMES or cfg["meta"].get("reference_status") != REFERENCE:
        raise ValueError("Research taxonomy or reference status changed")
    rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    groups = defaultdict(set)
    for row in rows:
        groups[row["group_id"]].add(row["split"])
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("Similarity group crosses a research partition")
    if for_training and not all(any(r["split"] == s for r in rows) for s in ["train", "val"]):
        raise ValueError("Training requires nonempty development partitions")
    return contract, rows


def build_research_view(source, destination, contract_path, seed=42):
    source, destination, contract_path = Path(source).resolve(), Path(destination).resolve(), Path(contract_path).resolve()
    if destination.exists() or contract_path.exists():
        raise FileExistsError("Research views and contracts are immutable; choose a new version")
    rows, rejected = [], []
    for split, rel in SPLITS.items():
        folder = source / rel
        if not (folder / "images").is_dir():
            raise ValueError(f"Missing source partition {rel}")
        for image in sorted((folder / "images").iterdir()):
            if image.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            label = folder / "labels" / (image.stem + ".txt")
            row = {"source_split": split, "source_image": image.relative_to(source).as_posix(),
                   "source_image_sha256": sha256_file(image), "source_label_sha256": sha256_file(label),
                   "image": image, "label": label, "reason": None}
            try:
                boxes, _ = read_yolo_boxes(label, range(14))
                row["boxes"] = [(MAPPING[c], x, y, w, h) for c, x, y, w, h in boxes if c in MAPPING]
            except ValueError as exc:
                row["reason"] = "invalid_annotation"
                row["detail"] = str(exc)
                row["boxes"] = []
            if split in {"train", "valid"} and image.stem.isdigit():
                number = int(image.stem)
                if 101 <= number <= 199:
                    row["reason"] = "unconfirmed_class_remap"
                elif number <= 30 or number >= 1392:
                    row["reason"] = "historically_ambiguous_annotation_block"
            if not row["reason"] and not row["boxes"]:
                row["reason"] = "no_target_objects_and_unverified_negative_coverage"
            if not row["reason"] and boxes_under_mask(row["boxes"]):
                row["reason"] = "target_box_intersects_border_mask"
            rows.append(row)
    print(f"Indexed {len(rows)} source images", flush=True)
    # Group ALL source images before exclusions so rejected copies cannot hide
    # a relation between a development image and a source holdout.
    vectors = [thumbnail(row["image"]) for row in rows]
    near_pairs = similar_pairs(vectors, .90)
    groups = connected_groups(len(rows), near_pairs)
    exact = defaultdict(list)
    for i, row in enumerate(rows):
        exact[row["source_image_sha256"]].append(i)
    for members in exact.values():
        labels = {tuple(sorted(rows[i]["boxes"])) for i in members}
        if len(members) > 1 and len(labels) > 1:
            for i in members:
                rows[i]["reason"] = "annotation_disagreement_between_exact_copies"
        else:
            # Prefer the source holdout, then an unflagged member.
            members.sort(key=lambda i: (rows[i]["source_split"] not in {"test", "external"}, rows[i]["reason"] is not None, rows[i]["source_image"]))
            for i in members[1:]:
                rows[i]["reason"] = "redundant_exact_copy"
    for i, j in near_pairs:
        # A near-reexport requires adjudication. Do not select or average its
        # annotations automatically. Exact byte copies are handled above.
        if rows[i]["source_image_sha256"] != rows[j]["source_image_sha256"] and float(vectors[i] @ vectors[j]) > .98:
            rows[i]["reason"] = rows[j]["reason"] = "unconfirmed_visual_reexport"
    pool_groups, fixed = [], []
    for group in groups:
        gid = "similarity_" + hashlib.sha256("|".join(sorted(rows[i]["source_image_sha256"] for i in group)).encode()).hexdigest()[:16]
        holdouts = {rows[i]["source_split"] for i in group} & {"test", "external"}
        accepted = []
        for i in group:
            row = rows[i]
            row["group_id"] = gid
            if len(holdouts) > 1:
                row["reason"] = "relation_between_source_holdouts"
            elif holdouts and row["source_split"] not in holdouts:
                row["reason"] = "development_related_to_source_holdout"
            if row["reason"]:
                rejected.append(row)
            else:
                accepted.append(row)
        if accepted:
            if holdouts:
                split = next(iter(holdouts))
                for row in accepted:
                    row["split"] = split
                    fixed.append(row)
            else:
                pool_groups.append(accepted)
    summaries = [(len(g), Counter(b[0] for row in g for b in row["boxes"])) for g in pool_groups]
    val = stratified_group_split(summaries, len(NAMES), .15, seed)
    selected = fixed[:]
    for i, group in enumerate(pool_groups):
        for row in group:
            row["split"] = "val" if i in val else "train"
            selected.append(row)
    counts = {s: Counter() for s in ["train", "val", "test", "external"]}
    for row in selected:
        counts[row["split"]].update(b[0] for b in row["boxes"])
    if any(counts[s][c] == 0 for s in ["train", "val"] for c in range(len(NAMES))):
        raise ValueError("A development class has no reference boxes")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".research-", dir=destination.parent) as temp:
        staged = Path(temp) / "view"
        for split in counts:
            for kind in ["images", "labels"]:
                (staged / kind / split).mkdir(parents=True)
        manifest = []
        for row in selected:
            image = cv2.imdecode(np.frombuffer(row["image"].read_bytes(), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("Undecodable accepted image")
            sid = row["source_split"] + "_" + row["image"].stem
            rel_image = f"images/{row['split']}/{sid}.png"
            derivative = prepare_image(image)
            if not cv2.imwrite(str(staged / rel_image), derivative, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                raise OSError("Could not write image derivative")
            label_out = staged / "labels" / row["split"] / (sid + ".txt")
            label_out.write_text("".join(" ".join(map(repr, box)) + "\n" for box in row["boxes"]), encoding="utf-8")
            manifest.append({k: v for k, v in row.items() if k not in {"image", "label", "boxes", "reason"}} | {
                "sample_id": sid, "image": rel_image, "label": label_out.relative_to(staged).as_posix(),
                "original_shape": list(image.shape), "derived_shape": list(derivative.shape),
                "boxes": len(row["boxes"]), "reference_status": REFERENCE,
                "patient_identity": "unknown", "group_basis": "unverified_conservative_similarity",
            })
        meta = {"taxonomy_version": "0.2.0", "reference_status": REFERENCE,
                "preprocessing": PREPROCESSING, "role": "development", "seed": seed,
                "source_doi": "10.5281/zenodo.15487430", "patient_independence": "not_established"}
        cfg = {"path": str(destination), "train": "images/train", "val": "images/val", "test": "images/test", "names": NAMES, "meta": meta}
        (staged / "dataset.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        external = {"path": str(destination), "val": "images/external", "test": "images/external", "names": NAMES,
                    "meta": meta | {"role": "external_evaluation"}}
        (staged / "external.yaml").write_text(yaml.safe_dump(external, allow_unicode=True, sort_keys=False), encoding="utf-8")
        (staged / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest), encoding="utf-8")
        exclusion_rows = [{k: v for k, v in row.items() if k not in {"image", "label", "boxes"}} for row in rejected]
        (staged / "exclusions.json").write_text(json.dumps(exclusion_rows, indent=2), encoding="utf-8")
        report = {"raw_images": len(rows), "accepted_images": len(selected), "excluded_images": len(rejected),
                  "exclusion_reasons": dict(Counter(r["reason"] for r in rejected)),
                  "partitions": {s: {"images": sum(r["split"] == s for r in selected),
                                      "boxes": {NAMES[c]: counts[s][c] for c in range(len(NAMES))}} for s in counts},
                  "reference_status": REFERENCE, "preprocessing": PREPROCESSING}
        (staged / "cohort_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        os.rename(staged, destination)
    contract = {"schema": "source-research-v1", "root": str(destination), "reference_status": REFERENCE,
                "files": inventory(destination), "code": git_state(Path(__file__).resolve().parents[2]), "environment": environment()}
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    verify_research_contract(contract_path, for_training=True)
    print(json.dumps(report, indent=2), flush=True)
    return report
