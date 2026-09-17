#!/usr/bin/env python3

import cv2
import numpy as np
from stai_mpu import stai_mpu_network

# ================= CONFIG =================
IMG_SIZE = 416
CLASS_NAMES = ["phone", "cigarette"]

CLASS_THRESHOLDS = {
    0: 0.40,   # phone
    1: 0.60    # cigarette
}

NMS_IOU_THRESH = 0.45


# ================= LETTERBOX =================
def letterbox(img, new_size=416, color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(new_size / w, new_size / h)

    nw, nh = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (nw, nh))

    pad_w = new_size - nw
    pad_h = new_size - nh

    top = pad_h // 2
    left = pad_w // 2

    img_padded = cv2.copyMakeBorder(
        img_resized,
        top, pad_h - top,
        left, pad_w - left,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return img_padded, scale, left, top


# ================= YOLOv8 DETECTOR (NB / NPU) =================
class SmokingCallingDetectorYOLOv8:
    """
    DMS-compatible YOLOv8 NB (stai_mpu NPU) detector.
    Drop-in replacement for the TFLite version.

    OUTPUT FORMAT (MANDATORY):
        [x1, y1, x2, y2, score, class_id]
    """

    def __init__(self, model_path, iou=0.45, conf=0.40):

        print(f"[YOLOv8-NB] Loading model: {model_path}")

        # -------- LOAD NB MODEL --------
        self.model = stai_mpu_network(
            model_path=model_path,
            use_hw_acceleration=True
        )

        # -------- INPUT INFO --------
        input_info = self.model.get_input_infos()[0]
        self.input_shape = input_info.get_shape()   # (1, 3, 416, 416)
        self.input_dtype = input_info.get_dtype()   # float16

        print(f"[YOLOv8-NB] Input  shape={self.input_shape}, dtype={self.input_dtype}")

        # -------- OUTPUT INFO --------
        self.output_infos = self.model.get_output_infos()
        for i, info in enumerate(self.output_infos):
            print(f"[YOLOv8-NB] Output {i}: shape={info.get_shape()}, dtype={info.get_dtype()}")

        self.iou_thresh  = iou
        self.conf_thresh = conf

        print("[YOLOv8-NB] Ready.")


    # ---------- NORMALIZE OUTPUT SHAPE ----------
    def _normalize_output(self, output):
        """
        Ensures output shape is (N, features).
        Handles:
            (1, 6, N)
            (1, N, 6)
            (6, N)
            (N, 6)
        """
        if output.ndim == 3:
            output = output[0]

        # if features-first → transpose  (e.g. (6, 3549) → (3549, 6))
        if output.shape[0] <= 10:
            output = output.T

        return output  # (N, features)


    # ---------- INFERENCE ----------
    def inference(self, frame, mono=False):
        """
        Input:
            frame : BGR image (numpy uint8)
        Output:
            np.ndarray shape [N, 6]  →  x1, y1, x2, y2, score, class_id
            Empty array if no detections.
        """

        orig_h, orig_w = frame.shape[:2]

        # -------- PREPROCESS --------
        img_lb, scale, pad_x, pad_y = letterbox(frame, IMG_SIZE)
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        img_f = img_rgb.astype(np.float32) / 255.0

        # Layout: HWC → CHW
        img_chw = np.transpose(img_f, (2, 0, 1))           # (3, 416, 416)
        input_tensor = np.expand_dims(img_chw, axis=0)     # (1, 3, 416, 416)

        # Cast to float16 as required by the NB model
        input_tensor = input_tensor.astype(np.float16)

        # -------- INFERENCE --------
        self.model.set_input(0, input_tensor)
        self.model.run()

        # -------- READ OUTPUT --------
        # Output shape from model: (1, 6, 3549) float16
        output = self.model.get_output(0)

        # Cast to float32 for all downstream logic
        output = np.array(output, dtype=np.float32)

        output = self._normalize_output(output)   # → (3549, 6)

        print("Output shape:", output.shape)
        print("Sample row :", output[0])

        # -------- DECODE DETECTIONS --------
        results = []

        for det in output:
            x, y, w, h, score, cls = det

            if score < self.conf_thresh:
                continue

            # xywh → xyxy (still in letterboxed 416×416 space)
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            # Clamp to original image size
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            results.append([x1, y1, x2, y2, score, int(cls)])

        if not results:
            return np.array([])

        # -------- NMS --------
        boxes_nms  = [[r[0], r[1], r[2] - r[0], r[3] - r[1]] for r in results]
        scores_nms = [r[4] for r in results]

        indices = cv2.dnn.NMSBoxes(
            boxes_nms,
            scores_nms,
            score_threshold=0.0,
            nms_threshold=self.iou_thresh
        )

        if len(indices) == 0:
            return np.array([])

        final = [results[i] for i in indices.flatten()]

        return np.array(final)