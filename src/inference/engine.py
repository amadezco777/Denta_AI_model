"""Local, hash-verified research inference with the training preprocessing."""
import base64
import io
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from src.data.contract import sha256_file
from src.data.research import NAMES, PREPROCESSING, REFERENCE, prepare_image

MAX_BYTES = 12 * 1024 * 1024
MAX_PIXELS = 20_000_000


def decode_upload(payload):
    if not payload or len(payload) > MAX_BYTES:
        raise ValueError("El archivo debe contener entre 1 byte y 12 MB.")
    try:
        with Image.open(io.BytesIO(payload)) as source:
            if source.format not in {"JPEG", "PNG"}:
                raise ValueError("Solo se admiten imágenes JPEG y PNG.")
            if source.mode in {"I", "I;16", "I;16B", "I;16L", "F"}:
                raise ValueError("Esta versión solo admite imágenes de 8 bits por canal.")
            if getattr(source, "n_frames", 1) != 1:
                raise ValueError("Selecciona una imagen estática, sin múltiples fotogramas.")
            w, h = source.size
            if min(w, h) < 32 or w * h > MAX_PIXELS:
                raise ValueError("La imagen debe medir al menos 32 píxeles por lado y como máximo 20 megapíxeles.")
            source.verify()
        # Use the same OpenCV grayscale decoder as the frozen training view.
        decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if decoded is None:
            raise ValueError("No se pudo decodificar la imagen.")
        return decoded
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("El archivo no es una imagen válida y decodificable.") from exc


class ResearchDetector:
    def __init__(self, metadata_path, device="cpu"):
        self.metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        meta = self.metadata
        if meta["names"] != NAMES or meta["preprocessing"] != PREPROCESSING or meta["reference_status"] != REFERENCE:
            raise ValueError("Metadatos incompatibles con el detector de investigación.")
        if meta.get("nms_multi_label") is not True:
            raise ValueError("El modelo debe declarar la misma política NMS que la evaluación.")
        if meta.get("clinical_approval") is not False:
            raise ValueError("Esta versión exige declarar explícitamente su alcance sin aprobación clínica.")
        size=meta.get("imgsz")
        threshold=meta.get("threshold")
        if type(size) is not int or size<32 or size>2048 or size%32:
            raise ValueError("Resolución de inferencia inválida en los metadatos.")
        if type(threshold) not in (float,int) or not np.isfinite(threshold) or not .001<=threshold<=1:
            raise ValueError("Umbral operativo inválido en los metadatos.")
        self.weights = Path(meta["weights"])
        if sha256_file(self.weights) != meta["weights_sha256"]:
            raise ValueError("Los pesos del modelo no corresponden al registro congelado.")
        import torch
        from ultralytics import YOLO
        from src.inference.reference_predictor import ReferenceDetectionPredictor
        torch.set_num_threads(4)
        self.model = YOLO(str(self.weights))
        self.predictor_class = ReferenceDetectionPredictor
        if self.model.task != "detect" or list(self.model.names.values()) != NAMES:
            raise ValueError("El checkpoint no corresponde a las siete clases dentales.")
        self.device = device
        self.lock = threading.Lock()

    def predict(self, image, threshold=None):
        started = time.perf_counter()
        threshold = self.metadata["threshold"] if threshold is None else float(threshold)
        if not np.isfinite(threshold) or not .001 <= threshold <= 1:
            raise ValueError("El umbral debe estar entre 0,001 y 1.")
        prepared = prepare_image(image)
        prepared = cv2.cvtColor(prepared, cv2.COLOR_GRAY2BGR)
        with self.lock:
            results = self.model.predict(prepared, imgsz=self.metadata["imgsz"], conf=threshold,
                                         iou=.7, max_det=300, rect=False, device=self.device,
                                         verbose=False, save=False, half=False, predictor=self.predictor_class)[0]
        detections = [{"class_id": int(c), "class": NAMES[int(c)], "score": float(s),
                       "box_xyxy": list(map(float, b))}
                      for b, s, c in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.conf.cpu().numpy(), results.boxes.cls.cpu().numpy())]
        annotated = results.plot(labels=True, conf=True, line_width=2, pil=False)
        ok, encoded = cv2.imencode(".png", annotated)
        if not ok:
            raise RuntimeError("No se pudo codificar la visualización.")
        return {"model_version": self.metadata["version"], "weights_sha256": self.metadata["weights_sha256"],
                "reference_status": REFERENCE, "threshold": threshold, "image_shape": list(prepared.shape[:2]),
                "box_coordinate_policy": "inverse_native_affine_unclipped",
                "detections": detections, "elapsed_ms": (time.perf_counter() - started) * 1000,
                "image_png_base64": base64.b64encode(encoded).decode("ascii"),
                "notice": "Prototipo de investigación. Las puntuaciones no son probabilidades de enfermedad. La ausencia de detecciones no descarta patología."}
