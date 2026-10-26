"""Serve only on localhost: python -m uvicorn app.main:app --host 127.0.0.1 --port 8089."""
import csv
import hashlib
import io
import json
import logging
import os
import secrets
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from src.data.research import verify_research_contract
from src.inference.engine import MAX_BYTES, ResearchDetector, decode_upload

ROOT = Path(__file__).resolve().parents[1]
detector = None
examples = {}
logger = logging.getLogger(__name__)
csv_results = OrderedDict()
csv_lock = threading.Lock()
CSV_TTL_SECONDS = 120
CSV_MAX_RESULTS = 32


def attach_csv(result):
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["clase", "puntuacion", "x1", "y1", "x2", "y2"])
    for row in result["detections"]:
        writer.writerow([row["class"], row["score"], *row["box_xyxy"]])
    now = time.monotonic()
    token = secrets.token_urlsafe(24)
    with csv_lock:
        for key in list(csv_results):
            if csv_results[key][0] <= now:
                del csv_results[key]
        while len(csv_results) >= CSV_MAX_RESULTS:
            csv_results.popitem(last=False)
        csv_results[token] = (now + CSV_TTL_SECONDS, output.getvalue().encode("utf-8"))
    result["csv_download_url"] = "/exports/"+token+".csv"
    return result


@asynccontextmanager
async def lifespan(app):
    global detector, examples
    path = Path(os.environ.get("DENTAL_MODEL_METADATA", ROOT / "models/research_v2/model.json"))
    detector = ResearchDetector(path, os.environ.get("DENTAL_DEVICE", "cpu")) if path.is_file() else None
    examples = {}
    contract_path = Path(os.environ.get("DENTAL_DATA_CONTRACT", ROOT / "configs/data/contracts/research_v2.json"))
    if detector is not None and contract_path.is_file():
        contract, rows = verify_research_contract(contract_path)
        raw_root = Path(contract["root"]).parent / "raw_verified"
        # Source-authorized examples are local only and use development images.
        eligible = [r for r in rows if r["split"] == "val"]
        for index, row in enumerate(sorted(eligible, key=lambda r: r["sample_id"])[:2]):
            candidate = (raw_root / row["source_image"]).resolve()
            if candidate.is_relative_to(raw_root.resolve()) and candidate.is_file():
                examples[str(index + 1)] = {"path": candidate, "sha256": row["source_image_sha256"],
                                            "description": f"Panorámica de investigación {index + 1}"}
    yield
    detector = None
    examples = {}
    with csv_lock:
        csv_results.clear()


app = FastAPI(title="Dental AI · Investigación", version="2.0.0", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(ROOT / "app/index.html", headers={"Cache-Control": "no-store"})


@app.get("/demo/control.png")
def technical_control():
    return FileResponse(ROOT / "app/demo/control_sintetico.png", media_type="image/png",
                        headers={"Cache-Control": "no-store"})


@app.get("/health")
def health():
    if detector is None:
        raise HTTPException(503, "Modelo verificado no disponible.")
    return {"status": "ready", "model_version": detector.metadata["version"],
            "threshold": detector.metadata["threshold"], "names": detector.metadata["names"],
            "reference_status": detector.metadata["reference_status"]}


@app.get("/about")
def about():
    path = ROOT / "reports/completion/evaluation.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    if report and (detector is None or any(report["decision"][k] != detector.metadata.get(k) for k in ["weights_sha256","threshold","imgsz"])):
        report = None
    return {"description": "Detector YOLO de siete clases con aprendizaje por transferencia y anotaciones publicadas como referencia.",
            "metrics": {k: {m: v[m] for m in ["images", "boxes", "map50", "map50_95", "false_positives_per_image"]}
                        for k, v in report["partitions"].items()} if report else {},
            "limitations": "Estudio retrospectivo. Sin validación clínica prospectiva ni equidad demográfica acreditada.",
            "examples": [{"id": key, "description": value["description"]} for key, value in examples.items()]}


class ExampleRequest(BaseModel):
    example_id: str


@app.get("/exports/{token}.csv")
def export_csv(token: str):
    with csv_lock:
        entry = csv_results.get(token)
        if entry is None or entry[0] <= time.monotonic():
            csv_results.pop(token, None)
            raise HTTPException(404, "La descarga caduc?. Vuelve a analizar la imagen.")
        payload = entry[1]
    return Response(payload, media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="dental_ai_detecciones.csv"',
                             "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})


@app.post("/predict-example")
async def predict_example(request: ExampleRequest):
    if detector is None:
        raise HTTPException(503, "Modelo verificado no disponible.")
    entry = examples.get(request.example_id)
    if entry is None:
        raise HTTPException(404, "Ejemplo local no disponible.")
    payload = entry["path"].read_bytes()
    if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
        raise HTTPException(409, "El ejemplo cambió respecto del inventario congelado.")
    image = await run_in_threadpool(decode_upload, payload)
    from fastapi.responses import JSONResponse
    result=await run_in_threadpool(detector.predict, image)
    return JSONResponse(attach_csv(result),headers={"Cache-Control":"no-store"})


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File()]):
    started = time.perf_counter()
    if detector is None:
        raise HTTPException(503, "Modelo verificado no disponible.")
    try:
        payload = await file.read(MAX_BYTES + 1)
        image = await run_in_threadpool(decode_upload, payload)
        result = await run_in_threadpool(detector.predict, image)
        result["request_elapsed_ms"] = (time.perf_counter() - started) * 1000
        from fastapi.responses import JSONResponse
        return JSONResponse(attach_csv(result), headers={"Cache-Control": "no-store"})
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except (RuntimeError, OSError) as exc:
        logger.exception("Research inference failed")
        raise HTTPException(500, "No se pudo completar el análisis. Intenta otra imagen o reinicia la aplicación.") from exc
    finally:
        await file.close()
