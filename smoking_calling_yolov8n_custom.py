#!/usr/bin/env python3

"""
smoking_calling_yolov8n_custom.py
----------------------------------
Drop-in replacement for smoking_calling_yolov4.py

Same public interface:
    SmokingCallingDetector(model_path, iou=0.25, conf=0.55)
    .inference(frame, mono) -> np.ndarray  shape (N, 6)
                               each row: [x1, y1, x2, y2, confidence, class_id]

Supports two backends (auto-selected by file extension):
    .tflite  ->  TFLite runtime  (x86 / ARM Linux)
    .nb      ->  stai_mpu_network (STM32 HW accelerated)

COLOR NOTE:
    DMS.py GStreamer pipeline delivers RGB frames.
    DMS.py inference() does [..,::-1] to convert RGB → BGR
    before passing to this detector.
    So input_image here is BGR — we convert BGR→RGB for the model.

THRESHOLD NOTE:
    sigmoid(0) = 0.500 exactly — this is the noise floor.
    Any threshold at or below 0.505 lets background anchors through.
    Genuine phone detections sit at sigmoid ~0.67–0.72.
    Safe minimum is 0.55. Do NOT lower below 0.51.
    To get faster/more sensitive detection, lower CONFIRM_FRAMES
    in DMS.py instead of lowering the threshold here.
"""

import numpy as np
import cv2

# ── Backend imports (soft) ────────────────────────────────────────────────────
try:
    from stai_mpu import stai_mpu_network
    _HAS_STAI = True
except ImportError:
    _HAS_STAI = False

try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    except ImportError:
        Interpreter = None


# ──────────────────────────────────────────────────────────
#  Internal helpers
# ──────────────────────────────────────────────────────────

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def _letterbox(img, new_h, new_w, color=(114, 114, 114)):
    """Resize with preserved aspect ratio + symmetric padding."""
    h, w = img.shape[:2]
    scale = min(new_w / w, new_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = new_w - nw, new_h - nh
    top,  left   = pad_h // 2, pad_w // 2
    padded = cv2.copyMakeBorder(
        resized, top, pad_h - top, left, pad_w - left,
        cv2.BORDER_CONSTANT, value=color
    )
    return padded, scale, left, top


# ──────────────────────────────────────────────────────────
#  TFLite backend
# ──────────────────────────────────────────────────────────

class _TFLiteBackend:
    def __init__(self, model_path):

        interp = Interpreter(model_path=model_path)
        interp.allocate_tensors()
        self._interp = interp

        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]

        self._in_idx    = inp["index"]
        self._out_idx   = out["index"]
        self._in_scale,  self._in_zero  = inp["quantization"]
        self._out_scale, self._out_zero = out["quantization"]
        self.scores_are_logits = out["dtype"] in (np.int8, np.uint8)
        self.coords_are_normalized = out["dtype"] in (np.int8, np.uint8)

        shape = inp["shape"]            # (1, H, W, 3) or (1, 3, H, W)
        self.input_dtype = inp["dtype"]

        if len(shape) == 4 and shape[1] == 3:
            self.layout = "NCHW"
            self.input_height = int(shape[2])
            self.input_width  = int(shape[3])
        else:
            self.layout = "NHWC"
            self.input_height = int(shape[1])
            self.input_width  = int(shape[2])

        # print(f"[SmkCallYOLOv8] TFLite  input={shape}  layout={self.layout}  "
        #       f"in_scale={self._in_scale:.6f}  in_zero={self._in_zero}")
        # print(f"[SmkCallYOLOv8] TFLite  output={out['shape']}  "
        #       f"out_scale={self._out_scale:.6f}  out_zero={self._out_zero}")

    def infer(self, img_float):
        """
        img_float : (H, W, 3) float32 RGB in [0, 1]
        returns   : (6, N) float32
                    rows 0-3: cx, cy, w, h in MODEL pixel space (already decoded)
                    rows 4+:  class probabilities (already post-sigmoid)
        """
        if self.layout == "NCHW":
            x = np.transpose(img_float, (2, 0, 1))
        else:
            x = img_float

        if self.input_dtype == np.int8 or self.input_dtype == np.uint8:
            scale = self._in_scale if self._in_scale != 0.0 else 1.0 / 255.0
            zero = self._in_zero
            if self.input_dtype == np.int8:
                q = np.clip(x / scale + zero, -128, 127).astype(np.int8)
            else:
                q = np.clip(x / scale + zero, 0, 255).astype(np.uint8)
        else:
            q = x.astype(np.float32)

        self._interp.set_tensor(self._in_idx, q[None, ...])
        self._interp.invoke()

        raw_q = self._interp.get_tensor(self._out_idx)[0]   # (6, N)
        
        # Dequantize if output is quantized
        out_dtype = self._interp.get_output_details()[0]["dtype"]
        if out_dtype == np.int8 or out_dtype == np.uint8:
            scale = self._out_scale if self._out_scale != 0.0 else 1.0
            zero = self._out_zero
            return (raw_q.astype(np.float32) - zero) * scale
        else:
            return raw_q.astype(np.float32)


# ──────────────────────────────────────────────────────────
#  STM32 .nb backend
# ──────────────────────────────────────────────────────────

class _NBBackend:
    def __init__(self, model_path):
        if not _HAS_STAI:
            raise ImportError(
                "stai_mpu not available on this platform. "
                "Use a .tflite model instead.")
        print(f"[INFO] NN model used: {model_path}")
        self._model = stai_mpu_network(model_path=model_path, use_hw_acceleration=True)

        in_info = self._model.get_input_infos()[0]
        shape   = in_info.get_shape()       # (1, H, W, 3)

        self.input_height = shape[1]
        self.input_width  = shape[2]

        self._in_scale = in_info.get_scale()
        self._in_zero  = in_info.get_zero_point()

        out_info        = self._model.get_output_infos()[0]
        self._out_scale = out_info.get_scale()
        self._out_zero  = out_info.get_zero_point()
        self._n_outputs = len(self._model.get_output_infos())
        self.scores_are_logits = True
        self.coords_are_normalized = True

        # print(f"[SmkCallYOLOv8] NB  input={shape}  "
        #       f"in_scale={self._in_scale:.6f}  in_zero={self._in_zero}")
        # print(f"[SmkCallYOLOv8] NB  output={out_info.get_shape()}  "
        #       f"out_scale={self._out_scale:.6f}  out_zero={self._out_zero}")

    def infer(self, img_float):
        """
        img_float : (H, W, 3) float32 RGB in [0, 1]
        returns   : (6, N) float32 dequantized
        """
        q = np.clip(
            img_float / self._in_scale + self._in_zero,
            -128, 127
        ).astype(np.int8)

        self._model.set_input(0, q[None, ...])   # NHWC (1, H, W, 3)
        self._model.run()

        raw_q = self._model.get_output(0)        # int8 (1, 6, N)
        if raw_q.ndim == 3:
            raw_q = raw_q[0]                     # → (6, N)

        return (raw_q.astype(np.float32) - self._out_zero) * self._out_scale


# ──────────────────────────────────────────────────────────
#  Public class — drop-in for SmokingCallingDetector
# ──────────────────────────────────────────────────────────

class SmokingCallingDetector:
    """
    YOLOv8 INT8 smoking / calling detector.
    Drop-in replacement for the YOLOv4 version.
    Same constructor signature and same inference() return format.

    Threshold guide (sigmoid scale):
        0.500 = noise floor (sigmoid(0)) — NEVER use this
        0.510 = barely above noise — still noisy
        0.550 = safe minimum for phone (genuine ~0.67–0.72)
        0.600 = recommended for cigarette (more false positives)
        0.650 = strict — only very confident detections pass

    If DMS.py passes conf=0.3 (SMK_CALL_THRESHOLD), the hard floor
    clamps it to 0.55 automatically. Genuine detections still pass
    because they score 0.67+. Only noise (0.50–0.53) is rejected.
    """

    _MIN_PHONE_THRESH = 0.55
    _MIN_SMOKE_THRESH = 0.60

    def __init__(self, model_path, iou=0.25, conf=0.55):
        """
        Arguments:
            model_path : path to .tflite or .nb model file
            iou        : NMS IoU threshold
            conf       : confidence threshold hint from caller.
                         Clamped to safe minimums internally.
                         Use this to raise thresholds, not lower them.
        """
        self.nms_threshold = iou
        self.labels        = ["phone", "smoke"]

        # conf from DMS.py is SMK_CALL_THRESHOLD = 0.3 — too low for YOLOv8 sigmoid.
        # We clamp to safe minimums. If conf is already higher, we respect it.
        phone_thresh = max(conf, self._MIN_PHONE_THRESH)
        smoke_thresh = max(conf, self._MIN_SMOKE_THRESH)

        self._class_thresholds = {
            0: phone_thresh,   # phone
            1: smoke_thresh,   # cigarette / smoke
        }

        # Load backend
        if model_path.endswith(".nb"):
            self._backend = _NBBackend(model_path)
        else:
            self._backend = _TFLiteBackend(model_path)

        self._img_h     = self._backend.input_height   # 320
        self._img_w     = self._backend.input_width    # 320
        self._n_anchors = None
        self._arange    = None

        # print(f"[SmkCallYOLOv8] thresholds  phone={phone_thresh:.3f}  "
        #       f"smoke={smoke_thresh:.3f}")
        # print(f"[SmkCallYOLOv8] NMS IoU={iou}")

    # ──────────────────────────────────────────────────────
    def inference(self, input_image, mono):
        """
        Run YOLOv8 detection on a single frame.

        Arguments:
            input_image : np.ndarray (H, W, 3) BGR uint8
                          DMS.py passes BGR (it does [..,::-1] on the RGB frame)
            mono        : bool — not supported, kept for API compatibility

        Returns:
            np.ndarray shape (N, 6), each row: [x1, y1, x2, y2, confidence, class_id]
            Empty array np.array([]) when nothing detected.
        """
        if mono:
            print("[SmkCallYOLOv8] mono not supported")
            return np.array([])

        orig_h, orig_w = input_image.shape[:2]

        # ── Preprocess ──────────────────────────────────
        # DMS.py gives BGR → convert to RGB for the model
        img_lb, scale, pad_x, pad_y = _letterbox(input_image, self._img_h, self._img_w)
        img_float = img_lb.astype(np.float32) / 255.0   # [0, 1]

        # ── Inference ────────────────────────────────────
        output = self._backend.infer(img_float)          # (6, N) float32

        # Lazy-init anchor count from first real output
        if self._n_anchors is None:
            self._n_anchors = output.shape[1]            # 2100
            self._arange    = np.arange(self._n_anchors)

        # ── Decode (vectorized) ──────────────────────────
        # Check if coords are pixel-space or normalized:
        # Standard YOLOv8 TFLite exports decode coords internally → pixel space.
        # Only older/custom exports give normalized [0,1] coords.
        coords_are_pixels = (
            not self._backend.coords_are_normalized
            and output[0].max() > 2.0
        )

        if coords_are_pixels:
            # Already in model pixel space — just undo letterbox padding
            cx_orig = (output[0] - pad_x) / scale
            cy_orig = (output[1] - pad_y) / scale
            w_orig  =  output[2]           / scale
            h_orig  =  output[3]           / scale
        else:
            # Normalized [0,1] — multiply by model dims first
            cx_orig = (output[0] * self._img_w - pad_x) / scale
            cy_orig = (output[1] * self._img_h - pad_y) / scale
            w_orig  =  output[2] * self._img_w           / scale
            h_orig  =  output[3] * self._img_h           / scale

        # Class scores: INT8 YOLOv8 exports used on STM32/laptop store logits,
        # so they need sigmoid after dequantization. Some float exports already
        # store probabilities.
        raw_cls = output[4:4 + len(self.labels)]   # (num_classes, N)
        if self._backend.scores_are_logits:
            cls_sig = _sigmoid(raw_cls)
        else:
            scores_are_probs = (raw_cls.min() >= 0.0 and raw_cls.max() <= 1.0)
            cls_sig = raw_cls if scores_are_probs else _sigmoid(raw_cls)

        best_cls   = np.argmax(cls_sig, axis=0)       # (N,)
        best_score = cls_sig[best_cls, self._arange]  # (N,)

        # Per-class threshold
        thresh_arr = np.array([self._class_thresholds[c] for c in best_cls])
        keep       = best_score >= thresh_arr

        if not keep.any():
            return np.array([])

        # Box corners in original frame pixels
        x1s = np.clip((cx_orig[keep] - w_orig[keep] / 2).astype(int), 0, orig_w)
        y1s = np.clip((cy_orig[keep] - h_orig[keep] / 2).astype(int), 0, orig_h)
        x2s = np.clip((cx_orig[keep] + w_orig[keep] / 2).astype(int), 0, orig_w)
        y2s = np.clip((cy_orig[keep] + h_orig[keep] / 2).astype(int), 0, orig_h)

        bw    = x2s - x1s
        bh    = y2s - y1s
        valid = (bw > 4) & (bh > 4)

        if not valid.any():
            return np.array([])

        x1s = x1s[valid];  y1s = y1s[valid]
        x2s = x2s[valid];  y2s = y2s[valid]
        scs = best_score[keep][valid]
        cls = best_cls[keep][valid]

        # ── NMS ──────────────────────────────────────────
        boxes_xywh = [
            [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            for x1, y1, x2, y2 in zip(x1s, y1s, x2s, y2s)
        ]

        nms_out = cv2.dnn.NMSBoxes(
            boxes_xywh,
            scs.tolist(),
            score_threshold=0.0,
            nms_threshold=self.nms_threshold
        )

        if len(nms_out) == 0:
            return np.array([])

        idx = nms_out.flatten()

        # ── Output: [x1, y1, x2, y2, confidence, class_id] ──
        result = np.array([
            [
                float(x1s[i]),
                float(y1s[i]),
                float(x2s[i]),
                float(y2s[i]),
                float(scs[i]),
                float(cls[i]),
            ]
            for i in idx
        ], dtype=np.float32)

        return result
