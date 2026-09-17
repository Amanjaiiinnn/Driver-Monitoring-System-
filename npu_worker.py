#!/usr/bin/env python3
"""
npu_worker.py  —  NPU inference subprocess for DMS.

Design
------
ONE request per frame, ONE response per frame.
The worker replicates the original dms.py single-process inference logic
exactly — detect → pick center face → square crop → landmark → eyes/mouth.
dms.py receives raw normalised boxes (for drawing) plus all computed results.

Protocol
--------
Parent → worker : struct(">I", len) + pickle(request)
Worker → parent : struct(">I", len) + pickle(result)
Shutdown        : parent sends b"QUIT" (no length prefix — checked first)
Ready signal    : worker sends struct(">I", 5) + b"READY" after models load

Request keys
------------
  frame        : bytes        — raw BGR uint8 frame
  shape        : (H, W, 3)   — frame shape
  run_face     : bool
  run_landmark : bool
  run_eye      : bool
  run_mouth    : bool
  run_smk      : bool

Result keys (all optional — only present when the relevant run_* was True
             and a face/result was actually found)
------------
  boxes        : ndarray (N,4) float  — raw normalised [0-1] detector output
  face_marks   : ndarray (468,3)      — landmarks in full-frame pixel coords
  mouth_ratio / head_down_ratio / mouth_face_ratio : float
  eyes         : {0: {ratio:float, iris:ndarray}, 1: {...}}
  smk          : raw yolo output
  error / tb   : on exception (result still sent so proxy doesn't hang)
"""
import sys
import os
import struct
import pickle
import math
import argparse
import numpy as np

sys.path.insert(0, "/home/root/Custom-DMS")

# ── Protect the binary protocol pipe BEFORE any import prints to stdout ───────
PROTO_FD = os.dup(1)       # save real stdout fd
os.dup2(2, 1)              # redirect fd-1 → stderr so model prints go to terminal


# ─────────────────────────────────────────────────────────────────────────────
#  Protocol helpers
# ─────────────────────────────────────────────────────────────────────────────
def _send(data: bytes) -> None:
    os.write(PROTO_FD, struct.pack(">I", len(data)))
    os.write(PROTO_FD, data)


def _recv() -> bytes | None:
    hdr = sys.stdin.buffer.read(4)
    if len(hdr) < 4:
        return None
    n = struct.unpack(">I", hdr)[0]
    return sys.stdin.buffer.read(n)


# ─────────────────────────────────────────────────────────────────────────────
#  Box helpers  — identical to GstWidget methods in the original dms.py
# ─────────────────────────────────────────────────────────────────────────────
def _transform_to_square(boxes: np.ndarray, scale: float = 1.0,
                          offset: tuple = (0, 0)) -> np.ndarray:
    """Vectorised square-crop — same formula as GstWidget.transform_to_square."""
    xmins, ymins, xmaxs, ymaxs = np.split(boxes, 4, axis=1)
    w        = xmaxs - xmins
    h        = ymaxs - ymins
    off_x    = offset[0] * w
    off_y    = offset[1] * h
    center_x = np.floor_divide(xmins + xmaxs, 2) + off_x
    center_y = np.floor_divide(ymins + ymaxs, 2) + off_y
    margin   = np.floor_divide(np.maximum(h, w) * scale, 2)
    return np.concatenate(
        (center_x - margin, center_y - margin,
         center_x + margin, center_y + margin), axis=1
    )


def _clip_boxes(boxes: np.ndarray,
                margins: tuple) -> np.ndarray:
    """Clip boxes to frame — same formula as GstWidget.clip_boxes."""
    left, top, right, bottom = margins
    boxes[:, 0] = np.maximum(boxes[:, 0], left)
    boxes[:, 1] = np.maximum(boxes[:, 1], top)
    boxes[:, 2] = np.minimum(boxes[:, 2], right)
    boxes[:, 3] = np.minimum(boxes[:, 3], bottom)
    return boxes


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    sys.stdout = sys.stderr   # all prints go to terminal, not the pipe

    # ── Parse model path args sent by NPUProxy ────────────────────────────────
    _NB = "/home/root/Custom-DMS/dms_nb_models"
    parser = argparse.ArgumentParser()
    parser.add_argument("--face-model",     default=f"{_NB}/face_detection_ptq.nb")
    parser.add_argument("--landmark-model", default=f"{_NB}/face_landmark_ptq.nb")
    parser.add_argument("--iris-model",     default=f"{_NB}/iris_landmark_ptq.nb")
    parser.add_argument("--smk-model",      default=f"{_NB}/yolov8n_smk_call_custom.nb")
    args = parser.parse_args()

    FACE_MODEL     = args.face_model
    LANDMARK_MODEL = args.landmark_model
    IRIS_MODEL     = args.iris_model
    SMK_CALL_MODEL = args.smk_model

    print(f"[WORKER] face     : {FACE_MODEL}", flush=True)
    print(f"[WORKER] landmark : {LANDMARK_MODEL}", flush=True)
    print(f"[WORKER] iris     : {IRIS_MODEL}", flush=True)
    print(f"[WORKER] smk/call : {SMK_CALL_MODEL}", flush=True)

    # Pin to core 1 (core 0 reserved for main DMS process)
    try:
        os.sched_setaffinity(0, {1})
    except Exception:
        pass

    # ── Load models ───────────────────────────────────────────────────────────
    try:
        from face_detection import FaceDetector
        from face_landmark  import FaceLandmark
        from eye            import Eye
        from mouth          import Mouth
        from smoking_calling_yolov8n_custom import SmokingCallingDetector

        print("[WORKER] Loading face detector…",  flush=True)
        face_detector = FaceDetector(FACE_MODEL, 0.5)
        print("[WORKER] Loading landmark…",        flush=True)
        face_landmark = FaceLandmark(LANDMARK_MODEL)
        print("[WORKER] Loading eye detector…",   flush=True)
        eye_detector  = Eye(IRIS_MODEL)
        print("[WORKER] Loading mouth…",           flush=True)
        mouth_model   = Mouth()
        print("[WORKER] Loading smk/call…",        flush=True)
        smk_detector  = SmokingCallingDetector(SMK_CALL_MODEL, conf=0.5)
        print("[WORKER] All models loaded.",       flush=True)

    except Exception as exc:
        import traceback
        print(f"[WORKER] FATAL during model load: {exc}", flush=True)
        traceback.print_exc()
        sys.exit(1)

    _send(b"READY")

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        raw = _recv()
        if raw is None or raw == b"QUIT":
            break

        result: dict = {}

        try:
            req   = pickle.loads(raw)
            frame = np.frombuffer(req["frame"], dtype=np.uint8).reshape(req["shape"])
            h, w  = frame.shape[:2]

            # ── Face detection ────────────────────────────────────────────────
            if req.get("run_face"):
                boxes = face_detector.detect(frame)

                if boxes is not None and len(boxes) > 0:
                    # Return raw normalised boxes so dms.py can draw them
                    result["boxes"] = boxes

                    # ── Pixel coords for internal landmark cropping ────────────
                    px = boxes.copy()
                    px[:, [0, 2]] *= w
                    px[:, [1, 3]] *= h

                    # ── Select centered face (same logic as original dms.py) ───
                    face_in_center     = 0
                    distance_to_center = math.hypot(w / 2, h / 2)
                    for i in range(len(px)):
                        x1, y1, x2, y2 = px[i]
                        dist = math.hypot((x2 + x1 - w) / 2, (y2 + y1 - h) / 2)
                        if dist < distance_to_center:
                            face_in_center     = i
                            distance_to_center = dist

                    # ── Square + clip (same formulas as original dms.py) ───────
                    sq   = _transform_to_square(px.copy(), scale=1.26, offset=(0, 0))
                    sq   = _clip_boxes(sq, (0, 0, w, h))
                    sq   = sq.astype(np.int32)

                    x1, y1, x2, y2 = sq[face_in_center]

                    # ── Landmark ──────────────────────────────────────────────
                    if req.get("run_landmark") and (x2 > x1) and (y2 > y1):
                        face_crop  = frame[y1:y2, x1:x2].copy()
                        face_marks = face_landmark.get_landmark(
                            face_crop, (x1, y1, x2, y2))
                        face_marks = np.array(face_marks, copy=True)
                        result["face_marks"] = face_marks

                        # ── Mouth ratios ──────────────────────────────────────
                        if req.get("run_mouth") and len(face_marks) == 468:
                            result["mouth_ratio"]      = float(mouth_model.yawning_ratio(face_marks))
                            result["head_down_ratio"]  = float(mouth_model.head_down_ratio(face_marks))
                            result["mouth_face_ratio"] = float(mouth_model.mouth_face_ratio(face_marks))

                        # ── Eyes ──────────────────────────────────────────────
                        if req.get("run_eye") and len(face_marks) == 468:
                            eyes = {}
                            for eye_id in [0, 1]:
                                ex1, ey1, ex2, ey2 = eye_detector.get_eye_roi(
                                    face_marks, eye_id)
                                # Clamp eye ROI to frame bounds
                                ex1 = max(0, min(int(ex1), w - 1))
                                ex2 = max(0, min(int(ex2), w))
                                ey1 = max(0, min(int(ey1), h - 1))
                                ey2 = max(0, min(int(ey2), h))
                                if ex2 > ex1 and ey2 > ey1:
                                    eye_crop = frame[ey1:ey2, ex1:ex2].copy()
                                    if eye_crop.size > 0:
                                        eye_marks, iris_marks = eye_detector.get_landmark(
                                            eye_crop, (ex1, ey1, ex2, ey2), eye_id)
                                        ratio = eye_detector.blinking_ratio(eye_marks, eye_id)
                                        eyes[eye_id] = {
                                            "ratio": float(ratio),
                                            "iris":  np.array(iris_marks, copy=True),
                                        }
                            if eyes:
                                result["eyes"] = eyes

                    # ── Smoking / Calling ─────────────────────────────────────────────
                    if req.get("run_smk"):
                        result["smk"] = smk_detector.inference(frame, False)

        except Exception as exc:
            import traceback
            result = {"error": str(exc), "tb": traceback.format_exc()}

        _send(pickle.dumps(result))


if __name__ == "__main__":
    main()