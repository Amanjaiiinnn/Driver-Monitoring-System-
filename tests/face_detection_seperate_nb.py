#!/usr/bin/env python3

import cv2
import numpy as np
import time
import stai_mpu

MODEL_PATH     = "face_detection_ptq.nb"
INPUT_SIZE     = 128
CONF_THRESHOLD = 0.8
IOU_THRESHOLD  = 0.8

# -----------------------------
# Sigmoid
# -----------------------------
def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))

# -----------------------------
# Anchors (BlazeFace style)
# -----------------------------
def generate_anchors():
    anchors = []
    strides = [8, 16, 16, 16]

    for stride in strides:
        grid = INPUT_SIZE // stride
        for y in range(grid):
            for x in range(grid):
                cx = (x + 0.5) / grid
                cy = (y + 0.5) / grid
                anchors.append([cx, cy])
                anchors.append([cx, cy])

    return np.array(anchors, dtype=np.float32)

# -----------------------------
# Decode (CORRECT VERSION)
# -----------------------------
def decode_boxes(raw_boxes, anchors):
    scale = INPUT_SIZE

    # IMPORTANT: dx/dy are swapped in this model
    XY_SCALE = 0.8
    WH_SCALE = 0.9

    cx = anchors[:, 0] + (raw_boxes[:, 0] / scale) * XY_SCALE
    cy = anchors[:, 1] + (raw_boxes[:, 1] / scale) * XY_SCALE
    w  = (raw_boxes[:, 2] / scale) * WH_SCALE
    h  = (raw_boxes[:, 3] / scale) * WH_SCALE

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return np.stack([x1, y1, x2, y2], axis=1)

# -----------------------------
# NMS
# -----------------------------
def nms(boxes, scores, iou_thresh):
    if len(boxes) == 0:
        return [], []

    idx = np.argsort(scores)[::-1]
    keep = []

    while len(idx) > 0:
        i = idx[0]
        keep.append(i)

        if len(idx) == 1:
            break

        rest = idx[1:]

        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)

        area_i = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])
        area_r = (boxes[rest,2]-boxes[rest,0])*(boxes[rest,3]-boxes[rest,1])

        iou = inter / (area_i + area_r - inter + 1e-6)

        idx = rest[iou < iou_thresh]

    return boxes[keep], scores[keep]

# -----------------------------
# Preprocess
# -----------------------------
def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img = img.astype(np.float32)

    # THIS IS CRITICAL (don’t touch again)
    img = (img - 127.5) / 127.5

    return np.expand_dims(img, axis=0)

# -----------------------------
# Load model (clean)
# -----------------------------
print("[INFO] Loading model...")

net = stai_mpu.stai_mpu_network(
    model_path=MODEL_PATH,
    use_hw_acceleration=True
)

input_shape = net.get_input_infos()[0].get_shape()
print("[INFO] Input shape:", input_shape)

anchors = generate_anchors()
print("[INFO] Anchors:", anchors.shape)

# -----------------------------
# Camera
# -----------------------------
cap = cv2.VideoCapture(7)

if not cap.isOpened():
    raise RuntimeError("Camera not opened")

prev_box = None
alpha = 0.7

# -----------------------------
# Loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    blob = preprocess(frame)

    t0 = time.time()

    net.set_input(0, blob)
    net.run()

    t1 = time.time()

    # outputs
    out0 = net.get_output(0)
    out1 = net.get_output(1)

    if out0.shape[-1] == 1:
        raw_scores = out0
        raw_boxes  = out1
    else:
        raw_scores = out1
        raw_boxes  = out0

    scores = sigmoid(raw_scores[0]).reshape(-1)
    boxes  = decode_boxes(raw_boxes[0], anchors)

    # filter
    mask = scores > CONF_THRESHOLD
    boxes = boxes[mask]
    scores = scores[mask]

    # keep top 5 BEFORE NMS (important)
    if len(scores) > 0:
        top_idx = np.argsort(scores)[-5:]
        boxes = boxes[top_idx]
        scores = scores[top_idx]

    # NMS
    boxes, scores = nms(boxes, scores, IOU_THRESHOLD)

    if len(scores) > 0:
        best = np.argmax(scores)

        box = boxes[best]        # 🔥 THIS WAS MISSING
        score = scores[best]

        box = np.clip(box, 0, 1)

        # -----------------------------
        # 🔥 SMOOTHING (correct)
        # -----------------------------
        if prev_box is not None:
            box = alpha * prev_box + (1 - alpha) * box

        prev_box = box

        boxes = [box]
        scores = [score]

    # draw
    for box, sc in zip(boxes, scores):

        # convert to pixel coords
        x1 = box[0] * w
        y1 = box[1] * h
        x2 = box[2] * w
        y2 = box[3] * h

        bw = x2 - x1
        bh = y2 - y1

        # -----------------------------
        # 🔥 SMART FACE EXPANSION
        # -----------------------------
        cx = x1 + bw / 2
        cy = y1 + bh / 2

        # widen slightly
        bw *= 1.3

        # taller face box (important)
        bh *= 1.6

        # shift upward (THIS is the missing piece)
        cy -= 0.15 * bh   # ← critical fix

        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        x2 = int(cx + bw / 2)
        y2 = int(cy + bh / 2)

        # clamp
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        cv2.putText(frame, f"{sc:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    fps = 1 / (t1 - t0 + 1e-6)
    cv2.putText(frame, f"FPS:{fps:.1f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
