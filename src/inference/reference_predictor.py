"""Use the installed official validation NMS for both research and the app.

Ultralytics' default predictor uses single-label NMS while its validator uses
multi-label NMS. Reuse the validator method so deployment and reported metrics
share the same policy. Ultralytics remains an AGPL/Enterprise dependency.
"""
import math
import cv2
import torch
import torchvision  # Native validation also imports this before choosing NMS.
from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import ops


class ReferenceDetectionPredictor(DetectionPredictor):
    def pre_transform(self, im):
        if self.imgsz[0] != self.imgsz[1]:
            raise ValueError("The research protocol requires a square model input")
        side = self.imgsz[0]
        letterbox = LetterBox((side, side), auto=False, scaleup=False)
        prepared = []
        for image in im:
            h, w = image.shape[:2]
            ratio = side / max(h, w)
            if ratio != 1:
                size = (min(math.ceil(w * ratio), side), min(math.ceil(h * ratio), side))
                image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
            prepared.append(letterbox(image=image))
        return prepared

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        self.nc = len(self.model.names)
        self.end2end = getattr(self.model, "end2end", False)
        outputs = DetectionValidator.postprocess(self, preds)
        boxes = [torch.cat((r["bboxes"], r["conf"][:, None], r["cls"][:, None]), dim=1) for r in outputs]
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
        return self.construct_results(boxes, img, orig_imgs)

    def construct_result(self, pred, img, orig_img, img_path):
        # Native validation evaluates in padded model coordinates. Invert its
        # exact ceil-based resize independently on both axes. Do not clip the
        # numeric proposal: clipping before IoU can change a match at a border.
        h, w = orig_img.shape[:2]
        side = img.shape[-1]
        ratio = side / max(h, w)
        rw, rh = min(math.ceil(w * ratio), side), min(math.ceil(h * ratio), side)
        left, top = round((side - rw) / 2 - .1), round((side - rh) / 2 - .1)
        pred[:, [0, 2]] = (pred[:, [0, 2]] - left) * (w / rw)
        pred[:, [1, 3]] = (pred[:, [1, 3]] - top) * (h / rh)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])
