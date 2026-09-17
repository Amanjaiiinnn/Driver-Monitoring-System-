"""
YOLOv8 INT8 Detector — Unified Backend (TFLite + STM32 .nb)
=============================================================
Supports two inference backends, auto-selected by model file extension:

  .tflite  →  TFLiteBackend  (Linux x86 / ARM)
  .nb      →  NBBackend      (STM32 stai_mpu with HW acceleration)

Both .nb and .tflite models are identical in quantization:
  Input : int8, scale=0.003922, zero=-128
  Output: int8, scale=0.005992, zero=-121
So NBBackend uses the exact same quant/dequant logic as TFLiteBackend.
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import cv2
import numpy as np
import time
import threading

Gst.init(None)

# ═══════════════════════════════════════════════════
#  CONFIG — only change MODEL_PATH to switch backends
# ═══════════════════════════════════════════════════

# .tflite → TFLite backend (development / x86)
# .nb     → STM32 HW accelerated backend
MODEL_PATH = "best_full_integer_quant.nb"

CAM_DEVICE = "/dev/video7"
CAM_WIDTH  = 640
CAM_HEIGHT = 480
CAM_FPS    = 30

IMG_SIZE    = 320
CLASS_NAMES = ["phone", "cigarette"]

# Thresholds (sigmoid scale — must stay > 0.505, noise floor is sigmoid(0)=0.500)
CLASS_THRESHOLDS = {
    0: 0.55,   # phone
    1: 0.60,   # cigarette
}
MIN_THRESHOLD  = 0.51   # hard clamp — prevents green screen no matter what is set above

CONFIRM_FRAMES = 1      # consecutive frames required to confirm a detection
NMS_IOU_THRESH = 0.40

DRAW_BOXES     = True   # False = headless embedded (no display)
PRINT_INTERVAL = 1.0    # seconds between terminal prints

BOX_COLOR  = (0, 255, 0)
TEXT_COLOR = (0,   0, 0)

# ═══════════════════════════════════════════════════


# ── Threshold safety clamp (run once at startup) ────
for _cid in list(CLASS_THRESHOLDS):
    if CLASS_THRESHOLDS[_cid] <= 0.505:
        print(f"[WARNING] {CLASS_NAMES[_cid]} threshold {CLASS_THRESHOLDS[_cid]:.3f} "
              f"<= sigmoid noise floor. Clamping to {MIN_THRESHOLD}.")
        CLASS_THRESHOLDS[_cid] = MIN_THRESHOLD


# ══════════════════════════════════════════════════
#  BACKEND: TFLite
# ══════════════════════════════════════════════════
class TFLiteBackend:
    def __init__(self, model_path):
        try:
            import tflite_runtime.interpreter as tflite
            Interpreter = tflite.Interpreter
            print("[BACKEND] tflite_runtime")
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
            print("[BACKEND] tensorflow.lite")

        interp = Interpreter(model_path=model_path)
        interp.allocate_tensors()
        self._interp = interp

        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]

        self._in_idx    = inp["index"]
        self._out_idx   = out["index"]
        self._in_scale, self._in_zero   = inp["quantization"]
        self._out_scale, self._out_zero = out["quantization"]

        shape = inp["shape"]            # (1, H, W, 3)
        self.input_height = int(shape[1])
        self.input_width  = int(shape[2])

        print(f"[BACKEND] Input  {shape}  "
              f"scale={self._in_scale:.6f}  zero={self._in_zero}")
        print(f"[BACKEND] Output {out['shape']}  "
              f"scale={self._out_scale:.6f}  zero={self._out_zero}")

    def infer(self, img_float):
        """
        img_float : (H, W, 3) float32 in [0, 1]
        returns   : (6, 2100) float32  dequantized
        """
        # quantize input: float [0,1] → int8
        q = np.clip(
            img_float / self._in_scale + self._in_zero,
            -128, 127
        ).astype(np.int8)

        self._interp.set_tensor(self._in_idx, q[None, ...])   # NHWC
        self._interp.invoke()

        raw_q = self._interp.get_tensor(self._out_idx)[0]     # int8 (6, 2100)

        # dequantize output: int8 → float
        return (raw_q.astype(np.float32) - self._out_zero) * self._out_scale


# ══════════════════════════════════════════════════
#  BACKEND: STM32 .nb  (stai_mpu_network)
# ══════════════════════════════════════════════════
class NBBackend:
    def __init__(self, model_path):
        from stai_mpu import stai_mpu_network
        print("[BACKEND] stai_mpu_network (.nb, HW acceleration)")

        self._model = stai_mpu_network(
            model_path=model_path,
            use_hw_acceleration=True
        )

        # ── Input info ──────────────────────────────
        in_info = self._model.get_input_infos()[0]
        shape   = in_info.get_shape()           # (1, 320, 320, 3)

        self.input_height = shape[1]
        self.input_width  = shape[2]

        self._in_scale = in_info.get_scale()        # 0.003922
        self._in_zero  = in_info.get_zero_point()   # -128

        # ── Output info ─────────────────────────────
        out_info = self._model.get_output_infos()[0]

        self._out_scale = out_info.get_scale()       # 0.005992
        self._out_zero  = out_info.get_zero_point()  # -121
        self._n_outputs = len(self._model.get_output_infos())

        print(f"[BACKEND] Input  {shape}  "
              f"scale={self._in_scale:.6f}  zero={self._in_zero}")
        print(f"[BACKEND] Output {out_info.get_shape()}  "
              f"scale={self._out_scale:.6f}  zero={self._out_zero}")

    def infer(self, img_float):
        """
        img_float : (H, W, 3) float32 in [0, 1]
        returns   : (6, 2100) float32  dequantized

        Identical quant/dequant math as TFLiteBackend —
        same scale/zero_point confirmed from model inspection.
        """
        # quantize input: float [0,1] → int8  (same formula as TFLite)
        q = np.clip(
            img_float / self._in_scale + self._in_zero,
            -128, 127
        ).astype(np.int8)

        self._model.set_input(0, q[None, ...])   # NHWC (1, 320, 320, 3)
        self._model.run()

        raw_q = self._model.get_output(0)        # int8, shape (1, 6, 2100)

        if raw_q.ndim == 3:
            raw_q = raw_q[0]                     # strip batch → (6, 2100)

        # dequantize output: int8 → float  (same formula as TFLite)
        return (raw_q.astype(np.float32) - self._out_zero) * self._out_scale


# ══════════════════════════════════════════════════
#  Backend factory
# ══════════════════════════════════════════════════
def load_backend(model_path):
    if model_path.endswith(".nb"):
        return NBBackend(model_path)
    return TFLiteBackend(model_path)


# ══════════════════════════════════════════════════
#  GStreamer Camera
# ══════════════════════════════════════════════════
class GstCamera:
    def __init__(self, device=CAM_DEVICE, width=CAM_WIDTH,
                 height=CAM_HEIGHT, fps=CAM_FPS):
        self._frame  = None
        self._lock   = threading.Lock()
        self._width  = width
        self._height = height

        self.pipeline = Gst.Pipeline()

        src = Gst.ElementFactory.make("v4l2src", "source")
        src.set_property("device", device)

        caps_in = Gst.ElementFactory.make("capsfilter", "caps_in")
        caps_in.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw,width={width},height={height},framerate={fps}/1"
            )
        )

        convert = Gst.ElementFactory.make("videoconvert", "convert")

        sink = Gst.ElementFactory.make("appsink", "sink")
        sink.set_property("emit-signals", True)
        sink.set_property("sync",         False)
        sink.set_property("max-buffers",  1)
        sink.set_property("drop",         True)
        sink.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw,format=RGB,width={width},height={height}"
            )
        )
        sink.connect("new-sample", self._on_frame)

        for el in (src, caps_in, convert, sink):
            self.pipeline.add(el)
        src.link(caps_in)
        caps_in.link(convert)
        convert.link(sink)

    def _on_frame(self, sink):
        sample = sink.emit("pull-sample")
        buf    = sample.get_buffer()
        frame  = np.ndarray(
            (self._height, self._width, 3),
            buffer=buf.extract_dup(0, buf.get_size()),
            dtype=np.uint8
        )
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        with self._lock:
            self._frame = frame
        return Gst.FlowReturn.OK

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def start(self):
        self.pipeline.set_state(Gst.State.PLAYING)
        print(f"[CAM] {CAM_DEVICE}  {CAM_WIDTH}x{CAM_HEIGHT}@{CAM_FPS}")

    def stop(self):
        self.pipeline.set_state(Gst.State.NULL)


# ══════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════
def letterbox(img, new_h, new_w, color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(new_w / w, new_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = new_w - nw, new_h - nh
    top, left = pad_h // 2, pad_w // 2
    padded = cv2.copyMakeBorder(
        resized, top, pad_h - top, left, pad_w - left,
        cv2.BORDER_CONSTANT, value=color
    )
    return padded, scale, left, top


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


# ══════════════════════════════════════════════════
#  Main inference loop
# ══════════════════════════════════════════════════
def main():
    backend = load_backend(MODEL_PATH)

    img_h = backend.input_height   # 320
    img_w = backend.input_width    # 320

    n_anchors  = None
    arange_all = None

    consecutive = {i: 0     for i in range(len(CLASS_NAMES))}
    confirmed   = {i: False for i in range(len(CLASS_NAMES))}
    num_classes = len(CLASS_NAMES)

    cam = GstCamera()
    cam.start()

    print("[INFO] Waiting for first frame...")
    while cam.get_frame() is None:
        time.sleep(0.05)
    print("[INFO] Camera ready.")

    last_print = time.time()

    try:
        while True:
            frame = cam.get_frame()
            if frame is None:
                continue

            orig   = frame
            orig_h, orig_w = orig.shape[:2]

            # ── Preprocess ────────────────────────────
            img_lb, scale, pad_x, pad_y = letterbox(frame, img_h, img_w)
            img_float = img_lb.astype(np.float32) / 255.0   # (H,W,3) in [0,1]

            # ── Inference ─────────────────────────────
            output = backend.infer(img_float)   # (6, N) float32 dequantized

            # Lazily init once we know anchor count from model output
            if arange_all is None:
                n_anchors  = output.shape[1]    # 2100
                arange_all = np.arange(n_anchors)
                print(f"[INFO] Anchors: {n_anchors}")

            # ── Decode (vectorized) ───────────────────
            # Coords are normalized 0–1 → convert to original frame pixels
            cx_orig = (output[0] * img_w - pad_x) / scale
            cy_orig = (output[1] * img_h - pad_y) / scale
            w_orig  =  output[2] * img_w           / scale
            h_orig  =  output[3] * img_h           / scale

            cls_sig    = sigmoid(output[4:4 + num_classes])   # (2, N)
            best_cls   = np.argmax(cls_sig, axis=0)           # (N,)
            best_score = cls_sig[best_cls, arange_all]        # (N,)

            thresh_arr = np.array([CLASS_THRESHOLDS[c] for c in best_cls])
            keep       = best_score >= thresh_arr

            boxes     = []
            scores    = []
            class_ids = []

            if keep.any():
                x1s = np.clip((cx_orig[keep] - w_orig[keep] / 2).astype(int), 0, orig_w)
                y1s = np.clip((cy_orig[keep] - h_orig[keep] / 2).astype(int), 0, orig_h)
                x2s = np.clip((cx_orig[keep] + w_orig[keep] / 2).astype(int), 0, orig_w)
                y2s = np.clip((cy_orig[keep] + h_orig[keep] / 2).astype(int), 0, orig_h)

                bw    = x2s - x1s
                bh    = y2s - y1s
                valid = (bw > 4) & (bh > 4)

                for x1, y1, bwi, bhi, s, cls in zip(
                    x1s[valid], y1s[valid], bw[valid], bh[valid],
                    best_score[keep][valid], best_cls[keep][valid]
                ):
                    boxes.append([int(x1), int(y1), int(bwi), int(bhi)])
                    scores.append(float(s))
                    class_ids.append(int(cls))

            # ── NMS ───────────────────────────────────
            final_indices = []
            if len(boxes) > 0:
                nms_out = cv2.dnn.NMSBoxes(
                    boxes, scores,
                    score_threshold=0.0,
                    nms_threshold=NMS_IOU_THRESH
                )
                if len(nms_out) > 0:
                    final_indices = nms_out.flatten().tolist()

            # ── Consecutive-frame gating ───────────────
            detected_this_frame = set(class_ids[i] for i in final_indices)
            for cls_id in range(num_classes):
                if cls_id in detected_this_frame:
                    consecutive[cls_id] += 1
                else:
                    consecutive[cls_id] = 0
                confirmed[cls_id] = consecutive[cls_id] >= CONFIRM_FRAMES

            # ── Print (throttled) ─────────────────────
            now = time.time()
            if now - last_print >= PRINT_INTERVAL:
                active  = [CLASS_NAMES[c] for c in range(num_classes) if confirmed[c]]
                streaks = "  |  ".join(
                    f"{CLASS_NAMES[c]}: {consecutive[c]}/{CONFIRM_FRAMES}"
                    for c in range(num_classes)
                )
                print(
                    f"[ALERT] {', '.join(a.upper() for a in active)}   ({streaks})"
                    if active else
                    f"[OK]    No distraction   ({streaks})"
                )
                last_print = now

            # ── Draw ──────────────────────────────────
            if DRAW_BOXES:
                draw = orig.copy()

                for i in final_indices:
                    x, y, bw, bh = boxes[i]
                    cls   = class_ids[i]
                    score = scores[i]
                    if not confirmed[cls]:
                        continue
                    label = f"{CLASS_NAMES[cls]} {score:.2f}"
                    cv2.rectangle(draw, (x, y), (x + bw, y + bh), BOX_COLOR, 2)
                    (tw, th), bl = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
                    )
                    cv2.rectangle(
                        draw,
                        (x, y - th - bl - 4), (x + tw + 4, y),
                        BOX_COLOR, -1
                    )
                    cv2.putText(
                        draw, label, (x + 2, y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1, cv2.LINE_AA
                    )

                # Streak overlay (top-left)
                for cls_id in range(num_classes):
                    color = (0, 255, 0) if confirmed[cls_id] else (0, 165, 255)
                    cv2.putText(
                        draw,
                        f"{CLASS_NAMES[cls_id]}: {consecutive[cls_id]}/{CONFIRM_FRAMES}",
                        (10, 25 + cls_id * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA
                    )

                cv2.imshow("YOLOv8", draw)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("[INFO] Stopped.")


if __name__ == "__main__":
    main()
