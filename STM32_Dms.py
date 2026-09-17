import os
import sys
import math
import time
import argparse
import numpy as np
try:
    import gi
    HAS_GI = True
except ImportError:
    HAS_GI = False

try:
    import cairo
    HAS_CAIRO = True
except ImportError:
    HAS_CAIRO = False

import cv2
from face_detection import FaceDetector
from face_landmark import FaceLandmark
from eye import Eye
from mouth import Mouth
# from smoking_calling_yolov4 import SmokingCallingDetector
# from smoking_calling_yolov8 import SmokingCallingDetectorYOLOv8
from smoking_calling_yolov8n_custom import SmokingCallingDetector

# Try GStreamer init
HAS_GST = False
try:
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
    Gst.init(None)
    HAS_GST = True
except Exception:
    pass

# Check stai_mpu availability
try:
    from stai_mpu import stai_mpu_network
    HAS_STAI = True
except ImportError:
    HAS_STAI = False

def translate_model_path(p):
    if not p: return p
    # Strip hardcoded Linux absolute path prefix on Windows PC
    if not HAS_STAI:
        p = p.replace("/home/root/Custom-DMS", "")
        p = p.replace("\\home\\root\\Custom-DMS", "")
        p = p.lstrip("/\\")
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        p = os.path.join(cur_dir, p)
    p = os.path.normpath(p)
    if not HAS_STAI and p.endswith(".nb"):
        p = p.replace(".nb", ".tflite").replace("dms_nb_models", "dms_tflite_models")
        p = p.replace("yolov8n_smk_call_custom", "yolov8n_smk_call")
    return p

# Default model paths — override via CLI (--face-model, --landmark-model, etc.)
# Accepts both .nb (NPU HW-accelerated) and .tflite (CPU) — backend auto-selected by extension
FACE_MODEL     = "dms_nb_models/face_detection_ptq.nb"
LANDMARK_MODEL = "dms_nb_models/face_landmark_ptq.nb"
IRIS_MODEL     = "dms_nb_models/iris_landmark_ptq.nb"
SMK_CALL_MODEL = "dms_nb_models/yolov8n_smk_call_custom.nb"

cur_path = os.path.dirname(os.path.abspath(__file__))

CAMERA_SRC = "/dev/video7"

# ═══════════════════════════════════════════════════════════
#  RESOLUTION CONFIG
# ═══════════════════════════════════════════════════════════
FRAME_WIDTH  = 640   # display resolution (Cairo draws at this size)
FRAME_HEIGHT = 480

# ML_WIDTH     = 320   # ML inference resolution (appsink receives this)
# ML_HEIGHT    = 240
 
# # Scale factors — used in draw() to map ML coords → display coords
# ML_SCALE_X = FRAME_WIDTH  / ML_WIDTH    # 2.0
# ML_SCALE_Y = FRAME_HEIGHT / ML_HEIGHT   # 2.0

# ═══════════════════════════════════════════════════════════
#  DRAWING CONFIG — all values relative to FRAME_WIDTH/HEIGHT
#  so changing display resolution automatically rescales UI
# ═══════════════════════════════════════════════════════════
DRAW_SMK_CALL_CORDS = False
DRAW_LANDMARKS      = False

# Alert panel (bottom-left)
_PANEL_W        = int(FRAME_WIDTH  * 0.16)   # ~102px at 640
_PANEL_H        = int(FRAME_HEIGHT * 0.2)    # ~96px at 480
_PANEL_X        = 0
_PANEL_Y        = FRAME_HEIGHT - _PANEL_H      # anchored to bottom
 
# Alert text layout
_FONT_SIZE      = max(8, int(FRAME_HEIGHT * 0.021))   # ~10px at 480
_LABEL_X        = int(FRAME_WIDTH  * 0.016)           # ~10px at 640
_STATUS_X       = int(FRAME_WIDTH  * 0.119)           # ~76px at 640
_ROW_START_Y    = _PANEL_Y + int(FRAME_HEIGHT * 0.042) # first row inside panel
_ROW_STEP       = int(FRAME_HEIGHT * 0.031)           # ~15px at 480
 
# Status bar (top)
_STATUS_BAR_H   = int(FRAME_HEIGHT * 0.083)           # ~40px at 480
_STATUS_FONT    = max(10, int(FRAME_HEIGHT * 0.042))  # ~20px at 480
_BRAND_X        = int(FRAME_WIDTH  * 0.859)           # ~550px at 640
_BRAND_Y        = int(FRAME_HEIGHT * 0.063)           # ~30px at 480
_STATUS_Y       = int(FRAME_HEIGHT * 0.063)           # ~30px at 480
 
# FPS display
_FPS_X          = int(FRAME_WIDTH  * 0.016)           # ~10px
_FPS_Y          = int(FRAME_HEIGHT * 0.146)           # ~70px at 480
_FPS_FONT       = max(10, int(FRAME_HEIGHT * 0.042))  # ~20px at 480

# ═══════════════════════════════════════════════════════════
#  SCORING / DETECTION THRESHOLDS
# ═══════════════════════════════════════════════════════════
BAD_FACE_PENALTY    = 0.01
NO_FACE_PENALTY     = 0.7
YAWN_PENALTY        = 7.0
DISTRACT_PENALTY    = 2.0
SLEEP_PENALTY       = 5.0
SMK_PENALTY         = 2.0
CALL_PENALTY        = 2.0
HEAD_DOWN_PENALTY   = 5.0
RESTORE_CREDIT      = -5.0
 
FACE_THRESHOLD      = 0.5
LEFT_EYE_THRESHOLD  = 0.35
RIGHT_EYE_THRESHOLD = 0.35
MOUTH_THRESHOLD     = 0.4
FACING_LEFT_THRESHOLD  = 0.5
FACING_RIGHT_THRESHOLD = 2
SMK_CALL_THRESHOLD  = 0.5
HEAD_DOWN_THRESHOLD = 0.78
 
LEFT_W  = 3
RIGHT_W = 3
LEFT_EYE_STATUS  = np.zeros(LEFT_W)
RIGHT_EYE_STATUS = np.zeros(RIGHT_W)
 
RIGHT_EYE_ID = 1
LEFT_EYE_ID  = 0
 
HEAD_W      = 7
HEAD_STATUS = [1] * HEAD_W
 
ENABLE_DISPLAY = True

FONT_SIZE = 10.0
DIFF_BETWEEN_TEXT = 15

ALERT_LABEL_X = 10
ALERT_STATUS_X = 100
ALERT_ROW_START_Y = 400

DEBUG_POINTS = {
    33:  "33",
    133: "133",
    159: "159",
    145: "145",
    362: "362",
    263: "263",
    386: "386",
    374: "374",
    78:  "78",
    308: "308",
    13: "13",
    14: "14",
    132: "132",
    361: "361"
}

def clamp_roi(x1, y1, x2, y2, w, h):
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    return x1, y1, x2, y2

if HAS_GST:
    Gst.init(None)
    
class DMSDemo:
    """The class to run the DMS Demo"""

    def __init__(self, device, width, height, fps):
        """
        Creates an instance of the DMSDemo

        Arguments:
        model_path -- the path to all models and image
        """

        self.inited = False
        self.distracted = False
        self.drowsy = False
        self.yawn = False
        self.smoking = False
        self.phone = False
        self.head_down = False
        self.last_phone_status = False
        self.last_smoke_status = False
        self.face_cords = []
        self.marks = []
        self.safe_value = 0.0
        self.smk_call_cords = []
        self.left_eye_closed = False
        self.right_eye_closed = False
        self.both_eyes_closed = False
        self.left_eye_ratio = 0.0
        self.right_eye_ratio = 0.0
        self.frame_count = 0
        self.debug_eyes_detection = False
        self.print_eye_status = False
        self._last_printed_status = None
        self.save_landmarks = False
        self.call_count = 0
        self.smoke_count = 0
        self.last_phone_detected_ts = 0
        self.phone_alrt_cooldown_ts = 0
        self.last_smoke_detected_ts = 0
        self.smoke_alrt_cooldown_ts = 0
        self.last_landmark_save_time = 0.0
        self.last_save_time = 0
        self.last_face_marks = []
        self.last_smk_call_cords = []

        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0

        self.setup_camera(device=device, width=width, height=height, fps=fps)

        self.run_face_detection          = True
        self.run_face_landmarks          = True
        self.run_eye_detection           = True
        self.run_yolo_smk_call_detection = True

        if self.run_face_detection:
            self.face_detector = FaceDetector(FACE_MODEL, FACE_THRESHOLD)
        
        if self.run_face_landmarks:
            self.face_landmark = FaceLandmark(LANDMARK_MODEL)
            self.mouth = Mouth()

        if self.run_eye_detection:
            self.eye_detector = Eye(IRIS_MODEL)

        if self.run_yolo_smk_call_detection:
            self.smoking_calling_detector = SmokingCallingDetector(SMK_CALL_MODEL,conf=SMK_CALL_THRESHOLD)
        
        # ---------------------------------------------------------
        # Mark system as fully initialized
        # ---------------------------------------------------------
        self.inited = True

    def setup_camera(self, device, width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=30):
        self.use_gst = HAS_GST
        if self.use_gst:
            try:
                if ENABLE_DISPLAY:
                    pipeline_str = f"""
                        v4l2src device={device} !
                        video/x-raw,width={width},height={height},framerate={fps}/1 !
                        videoconvert !
                        tee name=t

                        t. ! queue !
                        cairooverlay name=drawer !
                        autovideosink

                        t. ! queue !
                        videoconvert !
                        video/x-raw,format=BGR,width={width},height={height} !
                        appsink name=ml_sink emit-signals=true drop=true max-buffers=1
                        """
                else:
                    pipeline_str = f"""
                        v4l2src device={device} !
                        video/x-raw,width={width},height={height},framerate={fps}/1 !
                        videoconvert !
                        video/x-raw,format=BGR,width={width},height={height} !
                        appsink name=ml_sink emit-signals=true drop=true max-buffers=1
                    """

                self.pipeline = Gst.parse_launch(pipeline_str)

                drawer = self.pipeline.get_by_name("drawer")
                if drawer:
                    drawer.connect("draw", self.draw)

                ml_sink = self.pipeline.get_by_name("ml_sink")
                if ml_sink:
                    ml_sink.connect("new-sample", self.inference)

                self.pipeline.set_state(Gst.State.PLAYING)
            except Exception as e:
                print(f"[DMS] GStreamer initialization failed: {e}. Falling back to OpenCV.")
                self.use_gst = False

        if not self.use_gst:
            import threading
            self.cv_running = True
            self.cv_thread = threading.Thread(target=self._opencv_loop, args=(device, width, height, fps))
            self.cv_thread.daemon = True
            self.cv_thread.start()

    def stop_camera(self):
        if self.use_gst:
            self.pipeline.set_state(Gst.State.NULL)
        else:
            self.cv_running = False

    def save_frames(self, frame):
        os.makedirs("saved_frames", exist_ok=True)
        now = time.time()
        if now - self.last_save_time >= 1.0:
            cv2.imwrite(f"saved_frames/frame_{int(now)}.jpg", frame)
            self.last_save_time = now

    def inference(self, data):
        """
        Main inference callback triggered by GStreamer appsink.
        """
        sample = data.emit("pull-sample")
        if sample is None or not self.inited:
            return Gst.FlowReturn.OK if HAS_GST else None
        
        buffer = sample.get_buffer()
        caps   = sample.get_caps()

        ret, mem_buf = buffer.map(Gst.MapFlags.READ)
        if not ret:
            return Gst.FlowReturn.OK if HAS_GST else None

        height = caps.get_structure(0).get_value("height")
        width = caps.get_structure(0).get_value("width")

        frame = np.ndarray(shape=(height, width, 3), dtype=np.uint8, buffer=mem_buf.data)

        self.run_inference_core(frame, width, height)

        buffer.unmap(mem_buf)
        return Gst.FlowReturn.OK if HAS_GST else None

    def run_inference_core(self, frame, width, height):
        # Output containers for drawing/debug
        face_cords = []
        smk_call_cords = []
        mark_group = []

        # Default driver states (True = safe / normal)
        phone_detected    = False
        smoking_detected  = False
        distracted        = False
        yawn_detected     = False
        head_down_final   = False
        sleep_detected    = False

        self.frame_count += 1
        self.fps_counter += 1

        now = time.time()
        elapsed = now - self.fps_start_time

        if elapsed >= 1.0:
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = now

        # ── Per-frame model scheduling (staggered) ─────────
        run_landmark = (self.frame_count % 3 == 0) and self.run_face_landmarks
        run_eye      = (self.frame_count % 3 == 0) and self.run_eye_detection
        run_smk      = (self.frame_count % 3 == 1) and self.run_yolo_smk_call_detection

        # -------------------------------------------------
        # Face detection (normalized coordinates)
        # -------------------------------------------------
        if self.run_face_detection:
            boxes = self.face_detector.detect(frame)
            if boxes is None or len(boxes) == 0:
                boxes = []

        # -------------------------------------------------
        # If at least one face is detected
        # -------------------------------------------------
        if boxes is not None and len(boxes) > 0:

            # ---------------------------------------------
            # Smoking / Calling detection (YOLO)
            # ---------------------------------------------
            if run_smk:
                smk_call_cords, phone_detected, smoking_detected = self.smoke_call_detection(frame, phone_detected, smoking_detected)
                self.last_smk_call_cords = smk_call_cords
                self.last_phone_status = phone_detected
                self.last_smoke_status = smoking_detected
            else:
                smk_call_cords = self.last_smk_call_cords
                phone_detected = self.last_phone_status
                smoking_detected = self.last_smoke_status

            # Convert normalized face boxes → pixel coords
            for i in range(np.size(boxes, 0)):
                boxes[i][[0, 2]] *= width
                boxes[i][[1, 3]] *= height

            # Expand face boxes into squares (for landmarks)
            boxes = self.transform_to_square(boxes, scale=1.26, offset=(0, 0))

            # Clip boxes within image boundaries
            boxes, _ = self.clip_boxes(boxes, (0, 0, width, height))
            boxes = boxes.astype(np.int32)

            # Select face closest to image center
            face_in_center = 0
            distance_to_center = math.hypot(width / 2, height / 2)

            for i in range(np.size(boxes, 0)):
                x1, y1, x2, y2 = boxes[i]
                mid_to_center = math.hypot(
                    (x2 + x1 - width) / 2,
                    (y2 + y1 - height) / 2
                )
                if mid_to_center < distance_to_center:
                    face_in_center = i
                    distance_to_center = mid_to_center

            # Final selected face bounding box
            if len(boxes) > 0:
                x1, y1, x2, y2 = boxes[face_in_center]
                face_cords.append([x1, y1, x2, y2])
            else:
                face_cords = []

            # Face landmark inference
            if run_landmark:
                face_image = frame[y1:y2, x1:x2]
                face_marks = self.face_landmark.get_landmark(face_image, (x1, y1, x2, y2))
                face_marks = np.array(face_marks)
                mark_group.append(face_marks)
                self.last_face_marks = face_marks
            else:
                face_marks = self.last_face_marks

            if self.save_landmarks:
                self.save_face_landmarks(face_marks)

            # Yawn detection using mouth ratio
            if self.run_face_landmarks and len(face_marks) == 468:
                mouth_ratio = self.mouth.yawning_ratio(face_marks)
                yawn_detected = not (mouth_ratio <= MOUTH_THRESHOLD)

                head_down_ratio = self.mouth.head_down_ratio(face_marks)
                status = 1 if head_down_ratio > HEAD_DOWN_THRESHOLD else 0
                HEAD_STATUS[:-1] = HEAD_STATUS[1:]
                HEAD_STATUS[-1] = status
                head_down_avg = np.mean(HEAD_STATUS)
                head_down_final = head_down_avg > 0.5

                # Attention detection (head pose proxy)
                mouth_face_ratio = self.mouth.mouth_face_ratio(face_marks)
                if (
                    mouth_face_ratio < FACING_LEFT_THRESHOLD
                    or mouth_face_ratio > FACING_RIGHT_THRESHOLD
                ):
                    distracted = True
                else:
                    distracted = False

            # Left eye landmark & iris detection
            if run_eye:
                mark_group, sleep_detected = self.eye_landmark_detection(face_marks, frame, None, None, mark_group)
            else:
                sleep_detected = self.both_eyes_closed

        else:
            face_cords = []

        # ── Update shared state ────────────────────────────
        self.marks         = mark_group
        self.face_cords    = face_cords
        self.smk_call_cords = smk_call_cords
 
        self.distracted  = distracted
        self.drowsy      = sleep_detected
        self.yawn        = yawn_detected
        self.smoking     = smoking_detected
        self.phone       = phone_detected
        self.head_down   = head_down_final
 
        # ── Safety score ───────────────────────────────────
        if distracted:      self.safe_value = min(self.safe_value + DISTRACT_PENALTY,   100.0)
        if sleep_detected:  self.safe_value = min(self.safe_value + SLEEP_PENALTY,      100.0)
        if yawn_detected:   self.safe_value = min(self.safe_value + YAWN_PENALTY,       100.0)
        if smoking_detected:self.safe_value = min(self.safe_value + SMK_PENALTY,        100.0)
        if phone_detected:  self.safe_value = min(self.safe_value + CALL_PENALTY,       100.0)
        if head_down_final: self.safe_value = min(self.safe_value + HEAD_DOWN_PENALTY,  100.0)
        if not face_cords:  self.safe_value = min(self.safe_value + NO_FACE_PENALTY,    100.0)
 
        if (not distracted and not sleep_detected and not yawn_detected
                and not head_down_final and not smoking_detected
                and not phone_detected and face_cords):
            self.safe_value = max(self.safe_value + RESTORE_CREDIT, 0.0)

        if not ENABLE_DISPLAY:
            if self.frame_count % 20 == 0:
                print(f"""
            FPS: {self.current_fps:.2f}
            Driver: {"OK" if self.safe_value < 33 else "WARNING" if self.safe_value < 66 else "DANGER"}
            Distracted: {self.distracted}
            Drowsy: {self.drowsy}
            Yawn: {self.yawn}
            Head Down: {self.head_down}
            Smoking: {self.smoking}
            Phone: {self.phone}
            Risk: {self.safe_value:.2f}
            """)

    def _opencv_loop(self, device, width, height, fps):
        try:
            idx = int(''.join(filter(str.isdigit, device)))
        except ValueError:
            idx = 0

        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("[DMS] ERROR: OpenCV could not open any camera.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        print("[DMS] Running with OpenCV camera loop.")
        while self.cv_running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)

            # Run core inference
            self.run_inference_core(frame, width, height)

            # Render overlays directly onto frame
            self.draw_opencv(frame)

            cv2.imshow("DMS PC Test (OpenCV)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                self.cv_running = False
                break

        cap.release()
        cv2.destroyAllWindows()

    def draw_opencv(self, frame):
        h, w = frame.shape[:2]
        # Top status bar background
        cv2.rectangle(frame, (0, 0, w, 40), (0, 0, 0), -1)
        r = min(self.safe_value / 50.0, 1.0)
        g = min(1.0, (100.0 - self.safe_value) / 50.0)
        color = (0, int(g * 255), int(r * 255))

        status_text = "Driver OK" if self.safe_value < 33 else "Warning!" if self.safe_value < 66 else "DANGER!"
        if not self.face_cords:
            status_text = "Driver not found!"

        cv2.putText(frame, f"{status_text} ({self.safe_value:.1f}%)", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, "EDS-INDIA PC RUNNER", (w - 220, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Bottom Left Overlay Panel
        cv2.rectangle(frame, (10, h - 170), (180, h - 10), (0, 0, 0), -1)
        states = [
            ("Distracted", self.distracted),
            ("Drowsy",     self.drowsy),
            ("Yawn",       self.yawn),
            ("Head Down",  self.head_down),
            ("Smoking",    self.smoking),
            ("Phone",      self.phone),
        ]
        for idx, (label, val) in enumerate(states):
            val_text = "Yes" if val else "No"
            val_color = (0, 0, 255) if val else (0, 255, 0)
            y_pos = h - 145 + (idx * 22)
            cv2.putText(frame, f"{label}:", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(frame, val_text, (120, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, val_color, 2)

        # FPS indicator
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Draw face box
        if self.face_cords:
            x1, y1, x2, y2 = self.face_cords[0]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw landmarks/iris points
        if self.marks and DRAW_LANDMARKS:
            for m in self.marks:
                for pt in m:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, (0, 255, 255), -1)

        # Draw smoking/phone boxes
        if self.smk_call_cords and DRAW_SMK_CALL_CORDS:
            for c in self.smk_call_cords:
                cv2.rectangle(frame, (c[0], c[1]), (c[2], c[3]), (0, 165, 255), 2)

    def smoke_call_detection(self, frame, phone_detected, smoking_detected):

        smk_call_cords = []
        phone_seen = False
        cigrate_seen = False
        
        smk_call_result = self.smoking_calling_detector.inference(frame, False)

        if np.size(smk_call_result, 0) > 0:
            for i in range(np.size(smk_call_result, 0)):
                cls_id = int(smk_call_result[i][5])
                conf   = smk_call_result[i][4]
 
                if cls_id == 0 and conf > SMK_CALL_THRESHOLD:
                    phone_seen    = True
                    #phone_detected = True
 
                if cls_id == 1 and conf > SMK_CALL_THRESHOLD:
                    cigrate_seen     = True
                    #smoking_detected = True
 
                x1 = int(smk_call_result[i][0])
                y1 = int(smk_call_result[i][1])
                x2 = int(smk_call_result[i][2])
                y2 = int(smk_call_result[i][3])
                smk_call_cords.append([x1, y1, x2, y2])
 
            if phone_seen:
                self.call_count += 1
                phone_detected = True
            else:
                self.call_count = 0
 
            if cigrate_seen:
                self.smoke_count += 1
                smoking_detected = True
            else:
                self.smoke_count = 0
 
        return smk_call_cords, phone_detected, smoking_detected

    def eye_landmark_detection(self, face_marks, frame, buffer=None, mem_buf=None, mark_group=None):

        if face_marks is None or len(face_marks) < 468:
            return mark_group, self.both_eyes_closed

        for eye_id, eye_label in [(LEFT_EYE_ID, "left"), (RIGHT_EYE_ID, "right")]:
            x1, y1, x2, y2 = self.eye_detector.get_eye_roi(face_marks, eye_id)
            x1, y1, x2, y2 = clamp_roi(x1, y1, x2, y2, frame.shape[1], frame.shape[0])
            eye_img = frame[y1:y2, x1:x2]
 
            if eye_img is None or eye_img.size == 0:
                if buffer is not None and mem_buf is not None:
                    buffer.unmap(mem_buf)
                return mark_group, self.both_eyes_closed
 
            eye_marks, iris_marks = self.eye_detector.get_landmark(eye_img, (x1, y1, x2, y2), eye_id)
            mark_group.append(np.array(iris_marks))
 
            ratio = self.eye_detector.blinking_ratio(eye_marks, eye_id)
 
            if eye_id == LEFT_EYE_ID:
                LEFT_EYE_STATUS[:-1]  = LEFT_EYE_STATUS[1:]
                LEFT_EYE_STATUS[-1]   = 1 if ratio > LEFT_EYE_THRESHOLD  else 0
            else:
                RIGHT_EYE_STATUS[:-1] = RIGHT_EYE_STATUS[1:]
                RIGHT_EYE_STATUS[-1]  = 1 if ratio > RIGHT_EYE_THRESHOLD else 0
 
        self.left_eye_closed  = np.mean(LEFT_EYE_STATUS)  < 0.1
        self.right_eye_closed = np.mean(RIGHT_EYE_STATUS) < 0.1
        self.both_eyes_closed = self.left_eye_closed and self.right_eye_closed
 
        if self.debug_eyes_detection or self.print_eye_status:
            self.save_eyes_for_debug()
 
        return mark_group, self.both_eyes_closed

    def save_eyes_for_debug(self):
        if self.debug_eyes_detection:
            left_status = "CLOSED" if self.left_eye_closed else "OPEN"
            right_status = "CLOSED" if self.right_eye_closed else "OPEN"
            
            # Define directory path
            debug_eyes_dir = "/root/dms/dms-point-custom/debug/eyes/"
            
            # Create directory if it doesn't exist
            try:
                os.makedirs(debug_eyes_dir, exist_ok=True)
            except Exception as e:
                print(f"Error creating directory {debug_eyes_dir}: {e}")
            
            left_filename = os.path.join(debug_eyes_dir, f"left_eye_{left_status}_{self.left_eye_ratio:.3f}.jpg")
            right_filename = os.path.join(debug_eyes_dir, f"right_eye_{right_status}_{self.right_eye_ratio:.3f}.jpg")

            try:
                if left_eye_image.size > 0:
                    cv2.imwrite(left_filename, left_eye_image)
            except Exception as e:
                print("Error saving LEFT eye crop:", e)

            try:
                if right_eye_image.size > 0:
                    cv2.imwrite(right_filename, right_eye_image)
            except Exception as e:
                print("Error saving RIGHT eye crop:", e)

        # Print status every 30 frames (1 second at 30fps)
        if self.frame_count % 30 == 0 and self.print_eye_status:
            status = "✓" if not self.both_eyes_closed else "✗"
            print(f"Frame {self.frame_count}: {status} "
                f"Left: {'OPEN' if not self.left_eye_closed else 'CLOSED'} "
                f"({self.left_eye_ratio:.3f}), "
                f"Right: {'OPEN' if not self.right_eye_closed else 'CLOSED'} "
                f"({self.right_eye_ratio:.3f})")

    def save_face_landmarks(self, face_marks):
        import time

        # -------- FILE LOGGING (1 FPS) --------
        if face_marks is None or len(face_marks) != 468:
            return

        now = time.time()
        if now - self.last_landmark_save_time < 1.0:
            return

        self.last_landmark_save_time = now

        # freeze snapshot for this frame
        frozen = face_marks.copy()

        def log(line, file_obj=None):
            """Print to terminal + optionally write to file"""
            print(line)
            if file_obj:
                file_obj.write(line + "\n")

        with open("landmarks_main.txt", "a") as f:
            # -------- FRAME HEADER --------
            header = f"\n--- FRAME {now:.3f} ---"
            #log(header, f)

            # =====================================================
            # FULL LANDMARK DUMP (DISABLED)
            # Uncomment when needed
            # =====================================================
            # log("# ALL LANDMARKS (468)", f)
            # for i, (x, y) in enumerate(frozen):
            #     log(f"{i},{x},{y}", f)

            # =====================================================
            # MAIN / SEMANTIC POINTS (ACTIVE)
            # =====================================================
            #log("# MAIN POINTS (EYE / MOUTH / FACE)", f)

            MAIN_POINTS = {
                # Left Eye
                #33:  "LEFT_EYE_OUTER",
                #133: "LEFT_EYE_INNER",
                159: "LEFT_EYE_UPPER",
                145: "LEFT_EYE_LOWER",

                # Right Eye
                #362: "RIGHT_EYE_OUTER",
                #263: "RIGHT_EYE_INNER",
                386: "RIGHT_EYE_UPPER",
                374: "RIGHT_EYE_LOWER",

                # Mouth
                #78:  "MOUTH_LEFT",
                #308: "MOUTH_RIGHT",
                13:  "UPPER_LIP",
                14:  "LOWER_LIP",

                # Face direction
                132: "LEFT_CHEEK",
                361: "RIGHT_CHEEK",
            }

            for idx, name in MAIN_POINTS.items():
                x, y = frozen[idx]
                #log(f"{name} ({idx}): {x},{y}", f)

            f.flush()

    def transform_to_square(self, boxes, scale=1.0, offset=(0, 0)):
        """
        Convert rectangular bounding boxes into square boxes.

        This is typically used before landmark inference where
        models expect square face crops.

        Args:
            boxes   : NumPy array of shape (N, 4)
                    Each box is [xmin, ymin, xmax, ymax]
            scale   : Scaling factor applied to the square size
                    (>1.0 enlarges the box, <1.0 shrinks it)
            offset  : (x, y) fractional offset applied to the box center
                    Allows shifting the box relative to its center

        Returns:
            boxes   : NumPy array of square boxes [xmin, ymin, xmax, ymax]
        """

        # Split box coordinates into separate arrays
        xmins, ymins, xmaxs, ymaxs = np.split(boxes, 4, axis=1)

        # Compute width and height of each bounding box
        width = xmaxs - xmins
        height = ymaxs - ymins

        # -------------------------------------------------
        # Compute center offsets (relative to box size)
        # -------------------------------------------------
        offset_x = offset[0] * width
        offset_y = offset[1] * height

        # -------------------------------------------------
        # Compute center coordinates of each box
        # Using integer division to stay in pixel space
        # -------------------------------------------------
        center_x = np.floor_divide(xmins + xmaxs, 2) + offset_x
        center_y = np.floor_divide(ymins + ymaxs, 2) + offset_y

        # -------------------------------------------------
        # Convert rectangles into squares
        # - Take max(width, height) to preserve full face
        # - Apply scale factor
        # - margin = half side-length of square
        # -------------------------------------------------
        margin = np.floor_divide(
            np.maximum(height, width) * scale,
            2
        )

        # -------------------------------------------------
        # Reconstruct square bounding boxes
        # [xmin, ymin, xmax, ymax]
        # -------------------------------------------------
        boxes = np.concatenate(
            (
                center_x - margin,
                center_y - margin,
                center_x + margin,
                center_y + margin,
            ),
            axis=1,
        )

        return boxes

    def clip_boxes(self, boxes, margins):
        """
        Clip bounding boxes so they stay within safe image boundaries.

        This function ensures that no box coordinates fall outside
        the valid image region, preventing invalid array access
        during cropping or landmark inference.

        Args:
            boxes   : NumPy array of shape (N, 4)
                    Each box is [xmin, ymin, xmax, ymax]
            margins : Tuple of (left, top, right, bottom)
                    Defines the allowed coordinate limits

        Returns:
            boxes     : NumPy array of clipped bounding boxes
            clip_mark: Tuple of boolean arrays indicating which
                    sides were clipped for each box:
                    (top_clipped, left_clipped,
                        bottom_clipped, right_clipped)
        """

        # Unpack safe boundary limits
        left, top, right, bottom = margins

        # -------------------------------------------------
        # Identify which sides of each box exceed boundaries
        # -------------------------------------------------
        clip_mark = (
            boxes[:, 1] < top,     # top side clipped
            boxes[:, 0] < left,    # left side clipped
            boxes[:, 3] > bottom,  # bottom side clipped
            boxes[:, 2] > right,   # right side clipped
        )

        # -------------------------------------------------
        # Clamp box coordinates to valid range
        # -------------------------------------------------
        boxes[:, 1] = np.maximum(boxes[:, 1], top)      # ymin
        boxes[:, 0] = np.maximum(boxes[:, 0], left)     # xmin
        boxes[:, 3] = np.minimum(boxes[:, 3], bottom)   # ymax
        boxes[:, 2] = np.minimum(boxes[:, 2], right)    # xmax

        return boxes, clip_mark

    def increase_brightness(self, image):
        """Increase the brightness of input image, used for monochrome camera"""
        image = image.astype(np.uint16) * 4
        image[image > 255] = 255
        image = image.astype(np.uint8)
        return image

    def draw(self, overlay, context, timestamp, duration):
        """
        Cairo overlay callback.
        Draws DMS inference results (boxes, landmarks, alerts)
        on top of the live camera preview.
        """

        # Snapshot shared state once — prevents race condition with inference thread
        face_cords    = self.face_cords
        smk_call_cords = self.smk_call_cords
        marks          = self.marks

        # -------------------------------------------------
        # Font and drawing configuration
        # -------------------------------------------------
        context.select_font_face("Arial",cairo.FONT_SLANT_NORMAL,cairo.FONT_WEIGHT_BOLD)

        # Horizontal offset (used if display is letterboxed)
        offset = 0  # was 840 earlier, kept configurable

        # Default drawing color: green
        context.set_source_rgb(0, 1, 0)
        context.set_line_width(3)

        # -------------------------------------------------
        # Optional terminal logging (disabled)
        # -------------------------------------------------
        # if self.frame_count % 24 == 0:
        #     status = self.write_status()
        #     if status != self._last_printed_status:
        #         print(status)
        #         self._last_printed_status = status

        # -------------------------------------------------
        # Draw smoking / calling bounding boxes
        # -------------------------------------------------
        if smk_call_cords and DRAW_SMK_CALL_CORDS:
            for cords in smk_call_cords:
                context.rectangle(
                    (cords[0]) + offset,
                    (cords[1]),
                    (cords[2] - cords[0]),
                    (cords[3] - cords[1]),
                )
            context.stroke()

        # ── Landmarks ─────────────────────────────────────────
        if marks and DRAW_LANDMARKS:
            for m in marks:
                for mark in m:
                    mark = mark
                    mark[0] = mark[0] + offset
                    point = tuple(mark.astype(int))

                    # Draw each landmark as a small dot
                    context.arc(point[0], point[1], 1, 0, 1)
                    context.stroke()

        # if self.marks and DRAW_LANDMARKS:
        #     context.set_source_rgb(0, 1, 0)
        #     context.set_font_size(14)

        #     for m in self.marks:
        #         for idx, mark in enumerate(m):

        #             mark = mark * scale_x
        #             mark[0] = mark[0] + offset
        #             x, y = int(mark[0]), int(mark[1])

        #             # draw dot
        #             context.arc(x, y, 2, 0, 2 * math.pi)
        #             context.fill()

        #             # 🔥 draw index ONLY for debug points
        #             if idx in DEBUG_POINTS:
        #                 context.move_to(x + 4, y - 4)
        #                 context.show_text(str(idx))

        # ── Alert panel background ─────────────────────────────
        context.set_source_rgb(0, 0, 0)
        context.rectangle(_PANEL_X, _PANEL_Y, _PANEL_W, _PANEL_H)
        context.fill()

        # ── Alert labels ──────────────────────────────────────
        for row, label in enumerate(["Distracted:", "Drowsy:", "Yawn:",
                                      "Head Down:", "Smoking:", "Phone:"]):
            self.write_text(context, label,
                             _LABEL_X, _ROW_START_Y + row * _ROW_STEP)
 
        # ── FPS ───────────────────────────────────────────────
        context.save()
        context.set_source_rgb(0, 0, 1)
        context.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        context.set_font_size(_FPS_FONT)
        context.move_to(_FPS_X, _FPS_Y)
        context.show_text(f"FPS: {self.current_fps:.2f}")
        context.restore()

        # ── Alert statuses + face box ──────────────────────────
        if (isinstance(face_cords, list)
                and len(face_cords) > 0
                and len(face_cords[0]) == 4):
 
            for row, state in enumerate([self.distracted, self.drowsy, self.yawn,
                                          self.head_down, self.smoking, self.phone]):
                self.write_alert_status(context, state,
                                         _STATUS_X, _ROW_START_Y + row * _ROW_STEP)

            # Draw face bounding box
            context.set_source_rgb(0, 1, 0)
            context.rectangle(
                (face_cords[0][0]) + offset,
                (face_cords[0][1]),
                (face_cords[0][2] - face_cords[0][0]),
                (face_cords[0][3] - face_cords[0][1]),
            )
            context.stroke()

        else:
            for row in range(6):
                self.write_alert_status(context, None,
                                         _STATUS_X, _ROW_START_Y + row * _ROW_STEP)
 
        self.write_driver_status(context, face_cords)

    # def write_status(self):
    #     """
    #     Build multi-line terminal output for DMS status
    #     """

    #     # ---- DRIVER NOT FOUND ----
    #     if not self.face_cords:
    #         return (
    #             f"[DMS] Driver NOT FOUND (Risk={self.safe_value:.2f}%)\n"
    #         )

    #     distracted = self.write_text(self.distracted)
    #     drowsy     = self.write_text(self.drowsy)
    #     yawn       = self.write_text(self.yawn)
    #     smoking    = self.write_text(self.smoking)
    #     phone      = self.write_text(self.phone)

    #     # Risk level
    #     if self.safe_value < 33:
    #         level = "OK"
    #     elif self.safe_value < 66:
    #         level = "WARNING"
    #     else:
    #         level = "DANGER"

    #     return (
    #         f"[DMS] {level}  (Risk={self.safe_value:.2f}%)\n"
    #         f"Distracted ={distracted}\n"
    #         f"Drowsy     ={drowsy}\n"
    #         f"Yawn       ={yawn}\n"
    #         f"Smoking    ={smoking}\n"
    #         f"Phone      ={phone}\n"
    #     )

    # def write_text(self, yes):
    #     if yes is None:
    #         return "N/A"
    #     return "YES" if yes else "NO"

    def write_alert_status(self, context, state, y, x):
        """
        Draw the alert status text (Yes / No / N/A) on the display.

        Args:
            context : Cairo drawing context
            yes     : Boolean or None
                    - True  → alert active  (Yes, red)
                    - False → alert inactive (No, green)
                    - None  → status unknown / not applicable
            y       : X-coordinate for text placement
            x       : Y-coordinate for text placement
        """

        # Set font size for alert status text
        context.set_font_size(int(_FONT_SIZE))

        # Move drawing cursor to desired screen position
        context.move_to(y, x)

        # -------------------------------------------------
        # Unknown / not applicable state
        # -------------------------------------------------
        if state is None:
            context.set_source_rgb(1, 1, 1)   # white color
            context.show_text("N/A")

        # -------------------------------------------------
        # Alert active
        # -------------------------------------------------
        elif state:
            context.set_source_rgb(1, 0, 0)   # red color
            context.show_text("Yes")

        # -------------------------------------------------
        # Alert inactive
        # -------------------------------------------------
        else:
            context.set_source_rgb(0, 1, 0)   # green color
            context.show_text("No")
    
    def write_text(self, context, text, y, x):
        """
        Draw a plain text label on the display.

        Args:
            context : Cairo drawing context
            text    : String to be displayed
            y       : X-coordinate for text position
            x       : Y-coordinate for text position
        """

        # Set font size for text rendering
        context.set_font_size(int(_FONT_SIZE))

        # Move drawing cursor to specified position
        context.move_to(y, x)

        # Set text color to white
        context.set_source_rgb(1, 1, 1)

        # Render the text on screen
        context.show_text(text)

        return

    def write_driver_status(self, context, face_cords):
        """
        Draw the overall driver status banner at the top of the display.
        This includes:
        - Background panel
        - System label
        - Driver safety status with color-coded severity
        """

        # Status bar background
        context.set_source_rgb(0, 0, 0)
        context.rectangle(0, 0, FRAME_WIDTH, _STATUS_BAR_H)
        context.fill()
 
        self.write_text(context, "EDS-INDIA", _BRAND_X, _BRAND_Y)
 
        context.set_font_size(_STATUS_FONT)
        context.move_to(_LABEL_X, _STATUS_Y)
 
        r = min(self.safe_value / 50.0, 1.0)
        g = min(1.0, (100.0 - self.safe_value) / 50.0)
        context.set_source_rgb(r, g, 0)
 
        score = str(round(self.safe_value, 2))
        if face_cords:
            if self.safe_value < 33:
                context.show_text(f"Driver OK ({score}%)")
            elif self.safe_value < 66:
                context.show_text(f"Warning! ({score}%)")
            else:
                context.show_text(f"Danger! ({score}%)")
        else:
            context.show_text(f"Driver not found! ({score}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="STM32 DMS — USB camera. Pass .nb or .tflite models freely.")
    parser.add_argument("--device",          default=CAMERA_SRC)
    parser.add_argument("--width",           type=int, default=FRAME_WIDTH)
    parser.add_argument("--height",          type=int, default=FRAME_HEIGHT)
    parser.add_argument("--fps",             type=int, default=30)
    parser.add_argument("--face-model",      default=FACE_MODEL,
                        help="Face detection model (.nb or .tflite)")
    parser.add_argument("--landmark-model",  default=LANDMARK_MODEL,
                        help="Face landmark model (.nb or .tflite)")
    parser.add_argument("--iris-model",      default=IRIS_MODEL,
                        help="Iris/eye model (.nb or .tflite)")
    parser.add_argument("--smk-model",       default=SMK_CALL_MODEL,
                        help="Smoking/calling model (.nb or .tflite)")
    args = parser.parse_args()

    # Apply CLI model paths globally so DMSDemo picks them up
    FACE_MODEL     = translate_model_path(args.face_model)
    LANDMARK_MODEL = translate_model_path(args.landmark_model)
    IRIS_MODEL     = translate_model_path(args.iris_model)
    SMK_CALL_MODEL = translate_model_path(args.smk_model)

    window = DMSDemo(device=args.device, width=args.width, height=args.height, fps=args.fps)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        window.stop_camera()
        cv2.destroyAllWindows()
