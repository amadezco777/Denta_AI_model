"""Export and check numerical parity and actual local API inference."""
import base64
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[key] = "4"
os.environ["YOLO_AUTOINSTALL"] = "false"
import cv2
import numpy as np
from src.data.contract import sha256_file
from src.data.research import prepare_image, verify_research_contract
from scripts.training.complete_research import write_json


def main():
    import torch
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    import onnxruntime as ort
    torch.set_num_threads(4)
    reports = ROOT / "reports/completion"
    metadata = json.loads((ROOT / "models/research_v2/model.json").read_text())
    weights, size = Path(metadata["weights"]), metadata["imgsz"]
    if sha256_file(weights) != metadata["weights_sha256"]:
        raise ValueError("Model hash changed")
    contract, rows = verify_research_contract(ROOT / "configs/data/contracts/research_v2.json")
    dataset = Path(contract["root"])
    val = [r for r in rows if r["split"] == "val"][:12]
    model = YOLO(str(weights))
    parameter_count = sum(p.numel() for p in model.model.parameters())
    onnx_path = weights.with_suffix(".onnx")
    if not onnx_path.exists():
        exported = model.export(format="onnx", imgsz=size, batch=1, dynamic=False, simplify=False,
                                opset=17, device="cpu", half=False, nms=False)
        onnx_path = Path(exported)
    else:
        model.to("cpu")
    # Predictor performs the same fusion used by export before comparing raw
    # tensors. NMS is deliberately external in this portable FP32 artifact.
    model.model.fuse(verbose=False)
    model.model.eval()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    differences, torch_ms, onnx_ms = [], [], []
    for row in val:
        image = cv2.imread(str(dataset / row["image"]))
        resized = LetterBox(new_shape=(size, size), auto=False)(image=image)
        tensor = np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.
        t = torch.from_numpy(tensor)
        with torch.inference_mode():
            native = model.model(t)
            native = native[0] if isinstance(native, (list, tuple)) else native
        expected = native.detach().numpy()
        actual = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        if expected.ndim != 3 or expected.shape[1] != 4+len(metadata["names"]) or actual.shape != expected.shape:
            raise ValueError("Unexpected raw detector output geometry")
        delta = np.abs(actual - expected)
        differences.append({"sample_id": row["sample_id"], "max_absolute_error": float(delta.max()),
                            "mean_absolute_error": float(delta.mean()),
                            "max_absolute_coordinate_error": float(delta[:, :4, :].max()),
                            "max_absolute_class_score_error": float(delta[:, 4:, :].max()),
                            "allclose_atol0.02_rtol0.001": bool(np.allclose(actual, expected, atol=.02, rtol=.001)),
                            "class_scores_allclose_atol0.0001_rtol0.0001": bool(np.allclose(actual[:, 4:, :], expected[:, 4:, :], atol=.0001, rtol=.0001))})
        for _ in range(3):
            tick = time.perf_counter()
            with torch.inference_mode():
                model.model(t)
            torch_ms.append((time.perf_counter() - tick) * 1000)
            tick = time.perf_counter()
            session.run(None, {session.get_inputs()[0].name: tensor})
            onnx_ms.append((time.perf_counter() - tick) * 1000)
    result = {"format": "ONNX_FP32_opset17_batch1_static_no_embedded_NMS", "path": str(onnx_path),
              "checkpoint_bytes": weights.stat().st_size, "model_parameters": parameter_count,
              "sha256": sha256_file(onnx_path), "bytes": onnx_path.stat().st_size, "validation_images": len(val),
              "raw_output_parity": differences,
              "all_parity_checks_passed": all(r["allclose_atol0.02_rtol0.001"] and r["class_scores_allclose_atol0.0001_rtol0.0001"] for r in differences),
              "forward_only_cpu_4threads": {"repeats": len(torch_ms),
                  "torch_median_ms": float(np.median(torch_ms)), "torch_p95_ms": float(np.percentile(torch_ms, 95)),
                  "onnx_median_ms": float(np.median(onnx_ms)), "onnx_p95_ms": float(np.percentile(onnx_ms, 95))},
              "nms_and_end_to_end_equivalence": "Application continues using verified PyTorch; raw ONNX tensor parity does not certify every downstream implementation."}
    write_json(reports / "export.json", result)
    if not result["all_parity_checks_passed"]:
        raise ValueError("ONNX raw tensor parity failed")
    # Real TestClient calls load the actual final weights, execute preprocessing,
    # model prediction, postprocessing, PNG encoding and input validation.
    from fastapi.testclient import TestClient
    from app.main import app
    api_results, durations = [], []
    raw_root = dataset.parent / "raw_verified"
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/").status_code == 200
        for index, row in enumerate(val):
            payload = (raw_root / row["source_image"]).read_bytes()
            # A derivative must be exactly reproducible from the original.
            decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            assert np.array_equal(prepare_image(decoded), cv2.imread(str(dataset / row["image"]), cv2.IMREAD_GRAYSCALE))
            tick = time.perf_counter()
            response = client.post("/predict", files={"file": ("anonymous.jpg", payload, "image/jpeg")})
            elapsed = (time.perf_counter() - tick) * 1000
            assert response.status_code == 200, response.text[:200]
            r = response.json()
            assert r["weights_sha256"] == metadata["weights_sha256"]
            png = cv2.imdecode(np.frombuffer(base64.b64decode(r["image_png_base64"]), dtype=np.uint8), cv2.IMREAD_COLOR)
            assert png is not None
            api_results.append({"sample_id": row["sample_id"], "detections": len(r["detections"]), "request_ms": elapsed})
            if index:
                durations.append(elapsed)
        error = client.post("/predict", files={"file": ("invalid.png", b"invalid", "image/png")})
        assert error.status_code == 422
        unknown = client.post("/predict-example", json={"example_id": "../../README.md"})
        assert unknown.status_code == 404
        for example in client.get("/about").json()["examples"]:
            assert client.post("/predict-example", json={"example_id": example["id"]}).status_code == 200
    write_json(reports / "application_verification.json", {
        "actual_final_weights": True, "model_sha256": metadata["weights_sha256"], "device": "cpu",
        "tested_images": api_results, "preprocessing_byte_identical": True,
        "invalid_image_status": error.status_code, "path_traversal_example_status": unknown.status_code,
        "warm_end_to_end_request_median_ms": float(np.median(durations)),
        "warm_end_to_end_request_p95_ms": float(np.percentile(durations, 95)), "warm_images": len(durations),
        "includes": "decode, grayscale, mask, resize, inference, NMS, annotation, PNG/base64 and HTTP serialization",
        "human_usability_study": False, "clinical_validation": False})
    print("REAL MODEL EXPORT AND API CHECKS PASSED")


if __name__ == "__main__":
    main()
