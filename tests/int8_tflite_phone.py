"""
INT8 TFLite YOLOv8 — GStreamer Camera + Embedded Optimized
===========================================================
Camera: GStreamer v4l2src pipeline (better for embedded than cv2.VideoCapture)
Model:  YOLOv8n INT8 TFLite — normalized coords, sigmoid scores
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import cv2
import numpy as np
import time
import threading

try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
    print("[INFO] Using tflite_runtime")
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    print("[INFO] Using tensorflow.lite")

Gst.init(None)

# ═══════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════

MODEL_PATH    = "best_full_integer_quant.tflite"

# Camera
CAM_DEVICE    = "/dev/video7"
CAM_WIDTH     = 640
CAM_HEIGHT    = 480
CAM_FPS       = 30

IMG_SIZE      = 320
CLASS_NAMES   = ["phone", "cigarette"]

# Thresholds — must be > 0.505 (sigmoid noise floor is 0.500)
CLASS_THRESHOLDS = {
    0: 0.55,   # phone
    1: 0.60,   # cigarette
}
MIN_THRESHOLD = 0.51   # hard safety clamp

# Consecutive-frame gate — lower this for faster response instead of lowering threshold
CONFIRM_FRAMES = 4

NMS_IOU_THRESH = 0.40

# Display — set False on headless embedded board
DRAW_BOXES     = True
PRINT_INTERVAL = 1.0   # seconds between terminal prints

BOX_COLOR  = (0, 255, 0)
TEXT_COLOR = (0,   0, 0)

# ═══════════════════════════════════════════════════


# ── Threshold safety clamp ──────────────────────────
for cls_id, thr in CLASS_THRESHOLDS.items():
    if thr <= 0.505:
        print(f"[WARNING] {CLASS_NAMES[cls_id]} threshold {thr:.3f} <= noise floor. "
              f"Clamping to {MIN_THRESHOLD}.")
        CLASS_THRESHOLDS[cls_id] = MIN_THRESHOLD


# ══════════════════════════════════════════════════
#  GStreamer Camera
#  Runs in its own thread via GStreamer callbacks.
#  Latest frame is always available via get_frame().
# ══════════════════════════════════════════════════
class GstCamera:
    def __init__(self, device=CAM_DEVICE, width=CAM_WIDTH, height=CAM_HEIGHT, fps=CAM_FPS):
        self._frame      = None
        self._lock       = threading.Lock()
        self._width      = width
        self._height     = height

        self.pipeline = Gst.Pipeline()

        src = Gst.ElementFactory.make("v4l2src", "source")
        src.set_property("device", device)

        caps_in = Gst.ElementFactory.make("capsfilter", "caps_in")
        caps_in.set_property(
            "caps",
            Gst.Caps.from_string(f"video/x-raw,width={width},height={height},framerate={fps}/1")
        )

        convert = Gst.ElementFactory.make("videoconvert", "convert")

        sink = Gst.ElementFactory.make("appsink", "sink")
        sink.set_property("emit-signals", True)
        sink.set_property("sync",         False)
        sink.set_property("max-buffers",  1)
        sink.set_property("drop",         True)
        sink.set_property(
            "caps",
            Gst.Caps.from_string(f"video/x-raw,format=RGB,width={width},height={height}")
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
        # GStreamer gives RGB — convert to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        with self._lock:
            self._frame = frame
        return Gst.FlowReturn.OK

    def get_frame(self):
        """Returns latest frame (BGR) or None if not ready yet."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def start(self):
        self.pipeline.set_state(Gst.State.PLAYING)
        print(f"[CAM] Started: {CAM_DEVICE}  {CAM_WIDTH}x{CAM_HEIGHT}@{CAM_FPS}")

    def stop(self):
        self.pipeline.set_state(Gst.State.NULL)
        print("[CAM] Stopped")


# ══════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════
def letterbox(img, new_size=320, color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(new_size / w, new_size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = new_size - nw, new_size - nh
    top, left = pad_h // 2, pad_w // 2
    padded = cv2.copyMakeBorder(
        resized, top, pad_h - top, left, pad_w - left,
        cv2.BORDER_CONSTANT, value=color
    )
    return padded, scale, left, top


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


# ══════════════════════════════════════════════════
#  Load model
# ══════════════════════════════════════════════════
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

in_scale,  in_zero  = input_details[0]["quantization"]
out_scale, out_zero = output_details[0]["quantization"]
n_anchors = output_details[0]["shape"][2]
arange_all = np.arange(n_anchors)

print(f"[MODEL] Input  {input_details[0]['shape']}  scale={in_scale:.6f}  zero={in_zero}")
print(f"[MODEL] Output {output_details[0]['shape']}  scale={out_scale:.6f}  zero={out_zero}")
print(f"[CONFIG] Thresholds : { {CLASS_NAMES[k]: v for k, v in CLASS_THRESHOLDS.items()} }")
print(f"[CONFIG] Confirm    : {CONFIRM_FRAMES} frames")

# ══════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════
def main():
    # Consecutive-frame state
    consecutive = {i: 0     for i in range(len(CLASS_NAMES))}
    confirmed   = {i: False for i in range(len(CLASS_NAMES))}

    cam = GstCamera()
    cam.start()

    # Wait for first frame
    print("[INFO] Waiting for first frame...")
    while cam.get_frame() is None:
        time.sleep(0.05)
    print("[INFO] Camera ready. Running inference.")

    last_print = time.time()
    num_classes = len(CLASS_NAMES)

    try:
        while True:
            frame = cam.get_frame()
            if frame is None:
                continue

            orig = frame
            orig_h, orig_w = orig.shape[:2]

            # ── Preprocess ────────────────────────────
            img_lb, scale, pad_x, pad_y = letterbox(frame, IMG_SIZE)
            img = img_lb.astype(np.float32) / 255.0
            img = np.clip(img / in_scale + in_zero, -128, 127).astype(np.int8)
            input_tensor = img[None, ...]   # (1, 320, 320, 3) NHWC

            # ── Inference ─────────────────────────────
            interpreter.set_tensor(input_details[0]["index"], input_tensor)
            interpreter.invoke()

            raw_q  = interpreter.get_tensor(output_details[0]["index"])[0]
            output = (raw_q.astype(np.float32) - out_zero) * out_scale

            # ── Decode (vectorized) ───────────────────
            cx_orig = (output[0] * IMG_SIZE - pad_x) / scale
            cy_orig = (output[1] * IMG_SIZE - pad_y) / scale
            w_orig  =  output[2] * IMG_SIZE           / scale
            h_orig  =  output[3] * IMG_SIZE           / scale

            cls_sig    = sigmoid(output[4:4 + num_classes])
            best_cls   = np.argmax(cls_sig, axis=0)
            best_score = cls_sig[best_cls, arange_all]

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

            # ── Print ─────────────────────────────────
            now = time.time()
            if now - last_print >= PRINT_INTERVAL:
                active  = [CLASS_NAMES[c] for c in range(num_classes) if confirmed[c]]
                streaks = "  |  ".join(
                    f"{CLASS_NAMES[c]}: {consecutive[c]}/{CONFIRM_FRAMES}"
                    for c in range(num_classes)
                )
                if active:
                    print(f"[ALERT] {', '.join(a.upper() for a in active)}   ({streaks})")
                else:
                    print(f"[OK]    No distraction   ({streaks})")
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
                    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                    cv2.rectangle(draw, (x, y - th - bl - 4), (x + tw + 4, y), BOX_COLOR, -1)
                    cv2.putText(draw, label, (x + 2, y - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)

                for cls_id in range(num_classes):
                    color = (0, 255, 0) if confirmed[cls_id] else (0, 165, 255)
                    cv2.putText(
                        draw,
                        f"{CLASS_NAMES[cls_id]}: {consecutive[cls_id]}/{CONFIRM_FRAMES}",
                        (10, 25 + cls_id * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA
                    )

                cv2.imshow("YOLOv8 INT8", draw)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Headless: still allow clean exit via keyboard (if terminal available)
                pass

    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("[INFO] Stopped.")


if __name__ == "__main__":
    main()
