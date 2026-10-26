import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as application


class FakeDetector:
    def __init__(self, path, device):
        self.metadata = {"version": "unit-test", "threshold": .25, "names": ["caries"], "reference_status": "unit-test"}

    def predict(self, image):
        return {"detections": [], "image_shape": list(image.shape), "elapsed_ms": 1.}


def valid_image():
    output = io.BytesIO()
    Image.new("L", (128, 64), 100).save(output, "PNG")
    return output.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    metadata = tmp_path / "model.json"
    metadata.write_text("{}")
    monkeypatch.setenv("DENTAL_MODEL_METADATA", str(metadata))
    monkeypatch.setenv("DENTAL_DATA_CONTRACT", str(tmp_path / "missing.json"))
    monkeypatch.setattr(application, "ResearchDetector", FakeDetector)
    with TestClient(application.app) as c:
        yield c


def test_health_upload_and_cache_control(client):
    assert client.get("/health").json()["status"] == "ready"
    response = client.post("/predict", files={"file": ("image.png", valid_image(), "image/png")})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["image_shape"] == [64, 128]
    assert client.get("/").status_code == 200
    control = client.get("/demo/control.png")
    assert control.status_code == 200 and control.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_bad_input_and_path_like_example_are_rejected(client):
    response = client.post("/predict", files={"file": ("x.png", b"not PNG", "image/png")})
    assert response.status_code == 422
    assert client.post("/predict").status_code == 422
    assert client.post("/predict-example", json={"example_id": "../../README.md"}).status_code == 404


def test_health_unavailable_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("DENTAL_MODEL_METADATA", str(tmp_path / "missing.json"))
    with TestClient(application.app) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/predict", files={"file": ("a.png", valid_image(), "image/png")}).status_code == 503


def test_user_gets_safe_error_if_inference_fails(client, monkeypatch):
    def fail(image):
        raise RuntimeError("internal path C:/private/model.pt")
    monkeypatch.setattr(application.detector, "predict", fail)
    response = client.post("/predict", files={"file": ("image.png", valid_image(), "image/png")})
    assert response.status_code == 500
    assert "private" not in response.text


def test_csv_attachment_has_real_rows_and_no_cache(client, monkeypatch):
    rows = [{"class": "caries", "score": .6, "box_xyxy": [1, 2, 30, 40]},
            {"class": "endodoncia", "score": .7, "box_xyxy": [5, 6, 70, 80]}]
    monkeypatch.setattr(application.detector, "predict", lambda image: {"detections": rows})
    result = client.post("/predict", files={"file": ("image.png", valid_image(), "image/png")}).json()
    response = client.get(result["csv_download_url"])
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert "attachment" in response.headers["content-disposition"]
    import csv
    parsed = list(csv.reader(io.StringIO(response.text)))
    assert len(parsed) == 3 and parsed[0] == ["clase", "puntuacion", "x1", "y1", "x2", "y2"]
    assert parsed[1] == ["caries", "0.6", "1", "2", "30", "40"]


def test_csv_expires_and_result_memory_is_bounded(client, monkeypatch):
    for index in range(application.CSV_MAX_RESULTS+2):
        application.attach_csv({"detections": []})
    assert len(application.csv_results) == application.CSV_MAX_RESULTS
    result = application.attach_csv({"detections": []})
    now = application.time.monotonic()
    monkeypatch.setattr(application.time, "monotonic", lambda: now+121)
    assert client.get(result["csv_download_url"]).status_code == 404
    assert client.get("/exports/unknown.csv").status_code == 404
