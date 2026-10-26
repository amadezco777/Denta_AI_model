import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.research import build_research_view, prepare_image, verify_research_contract, REFERENCE


def test_research_contract_never_claims_review_and_rejects_mutation(tmp_path):
    source = tmp_path / "source"
    rng = np.random.default_rng(42)
    for folder, count, start in [("train", 12, 300), ("valid", 3, 400), ("test", 2, 500), ("test_alte_cabinete/Ext-validation", 2, 600)]:
        for kind in ["images", "labels"]:
            (source / folder / kind).mkdir(parents=True)
        for n in range(start, start + count):
            image = rng.integers(0, 255, (48, 96), dtype=np.uint8)
            cv2.imwrite(str(source / folder / "images" / f"{n}.png"), image)
            (source / folder / "labels" / f"{n}.txt").write_text("".join(f"{c} 0.5 0.5 0.1 0.1\n" for c in [0, 1, 2, 3, 4, 6, 7]))
    contract = tmp_path / "contract.json"
    view = tmp_path / "view"
    report = build_research_view(source, view, contract)
    assert report["accepted_images"] == 19
    record, rows = verify_research_contract(contract, for_training=True)
    assert record["reference_status"] == REFERENCE
    assert "reviewer" not in record
    assert all(r["patient_identity"] == "unknown" for r in rows)
    (view / rows[0]["label"]).write_text("0 0.5 0.5 0.2 0.2\n")
    with pytest.raises(ValueError, match="bytes changed"):
        verify_research_contract(contract)


def test_preprocessing_masks_boundaries_preserves_dimensions_and_dtype():
    image = np.full((1600, 3200), 200, dtype=np.uint8)
    out = prepare_image(image)
    assert out.shape == (640, 1280)
    assert out.dtype == np.uint8
    assert np.all(out[:, :20] == 0)
    assert np.all(out[-20:, :] == 0)
    assert out[300, 600] == 200
