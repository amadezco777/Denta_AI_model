import io

import cv2
import numpy as np
from PIL import Image
import pytest

from src.data.research import prepare_image
from src.inference.engine import MAX_BYTES, decode_upload


def test_upload_and_training_decoder_have_identical_preprocessing():
    rng = np.random.default_rng(123)
    image = rng.integers(0, 255, (160, 320, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", image)
    original = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(prepare_image(decode_upload(encoded.tobytes())), prepare_image(original))


@pytest.mark.parametrize("payload", [b"", b"not an image"], ids=["empty", "invalid"])
def test_invalid_uploads_rejected(payload):
    with pytest.raises(ValueError):
        decode_upload(payload)


def test_oversize_upload_rejected_before_decode():
    with pytest.raises(ValueError, match="12 MB"):
        decode_upload(b"x" * (MAX_BYTES + 1))


def test_size_and_format_validation():
    for fmt, shape in [("GIF", (64, 64)), ("PNG", (16, 64)), ("PNG", (5000, 5000))]:
        output = io.BytesIO()
        Image.new("L", shape).save(output, format=fmt)
        with pytest.raises(ValueError):
            decode_upload(output.getvalue())


def test_16bit_image_is_not_silently_reduced():
    output = io.BytesIO()
    Image.new("I;16", (64, 64)).save(output, "PNG")
    with pytest.raises(ValueError, match="8 bits"):
        decode_upload(output.getvalue())


@pytest.mark.parametrize("field,value",[("threshold",float("nan")),("imgsz",63),("nms_multi_label",False),("clinical_approval",True)])
def test_invalid_model_metadata_rejected_before_loading_framework(tmp_path,field,value):
    import json
    from src.data.research import NAMES,PREPROCESSING,REFERENCE
    from src.inference.engine import ResearchDetector
    meta={"names":NAMES,"preprocessing":PREPROCESSING,"reference_status":REFERENCE,
          "nms_multi_label":True,"clinical_approval":False,"imgsz":640,"threshold":.25}
    meta[field]=value
    path=tmp_path/"metadata.json"
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        ResearchDetector(path)
