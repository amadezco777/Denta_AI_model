"""Verify the running local app and CSV attachment without a browser download API."""
import base64
import csv
import hashlib
import io
import json
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]

def main():
    metadata = json.loads((ROOT/"models/research_v2/model.json").read_text(encoding="utf-8"))
    rows = []
    with httpx.Client(base_url="http://127.0.0.1:8089", timeout=30) as client:
        assert client.get("/health").status_code == 200
        for identifier in ["1", "2"]:
            start = time.perf_counter()
            response = client.post("/predict-example", json={"example_id": identifier})
            response.raise_for_status()
            result = response.json()
            assert result["weights_sha256"] == metadata["weights_sha256"]
            assert base64.b64decode(result["image_png_base64"]).startswith(b"\x89PNG\r\n\x1a\n")
            download = client.get(result["csv_download_url"])
            assert download.status_code == 200 and "attachment" in download.headers["content-disposition"]
            assert download.headers["cache-control"] == "no-store"
            parsed = list(csv.reader(io.StringIO(download.text)))
            assert parsed[0] == ["clase", "puntuacion", "x1", "y1", "x2", "y2"]
            assert len(parsed)-1 == len(result["detections"])
            for record, detection in zip(parsed[1:], result["detections"]):
                assert record[0] == detection["class"]
                assert list(map(float, record[1:])) == [detection["score"], *detection["box_xyxy"]]
            rows.append({"example_id": identifier, "detections": len(parsed)-1,
                         "prediction_and_csv_ms": (time.perf_counter()-start)*1000,
                         "csv_sha256": hashlib.sha256(download.content).hexdigest()})
        control = (ROOT/"app/demo/control_sintetico.png").read_bytes()
        result = client.post("/predict", files={"file": ("control.png", control, "image/png")})
        assert result.status_code == 200
        assert client.post("/predict", files={"file": ("invalid.png", b"no image", "image/png")}).status_code == 422
        assert client.post("/predict", files={"file": ("oversize.png", b"0"*(12*1024*1024+1), "image/png")}).status_code == 422
        assert client.post("/predict-example", json={"example_id": "../../README.md"}).status_code == 404
    report = {"status": "passed", "weights_sha256": metadata["weights_sha256"],
              "examples_and_csv": rows, "synthetic_control_verified": True,
              "invalid_and_oversize_422": True, "path_like_example_404": True,
              "browser_native_download_not_claimed": True,
              "app_source_sha256": hashlib.sha256((ROOT/"app/main.py").read_bytes()).hexdigest()}
    (ROOT/"reports/completion/application_final_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
