import os
import sys
import math
import time
import signal
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
from smoking_calling_yolov8n_custom import SmokingCallingDetector

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

# Try to initialize GTK and GStreamer
HAS_GUI = False
try:
    gi.require_version("Gst", "1.0")
    gi.require_version("Gtk", "3.0")
    from gi.repository import Gtk, Gdk, GLib, Gst
    Gst.init(None)
    Gst.init_check(None)
    Gtk.init(None)
    Gtk.init_check(None)
    HAS_GUI = True
except Exception:
    pass

if not HAS_GUI:
    class MockClass:
        def __init__(self, *args, **kwargs):
            pass
    class MockModule:
        def __getattr__(self, name):
            return MockClass
    Gtk = MockModule()
    Gst = MockModule()
    Gdk = MockModule()
    GLib = MockModule()

# ═══════════════════════════════════════════════════════════
#  MODEL PATHS
# ═══════════════════════════════════════════════════════════
# Default model paths — override via CLI (--face-model, --landmark-model, etc.)
# Accepts both .nb (NPU HW-accelerated) and .tflite (CPU) — backend auto-selected by extension
FACE_MODEL     = "dms_nb_models/face_detection_ptq.nb"
LANDMARK_MODEL = "dms_nb_models/face_landmark_ptq.nb"
IRIS_MODEL     = "dms_nb_models/iris_landmark_ptq.nb"
SMK_CALL_MODEL = "dms_nb_models/yolov8n_smk_call_custom.nb"

# ═══════════════════════════════════════════════════════════
#  RESOLUTION CONFIG
# ═══════════════════════════════════════════════════════════
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# ═══════════════════════════════════════════════════════════
#  DRAWING CONFIG
# ═══════════════════════════════════════════════════════════
DRAW_SMK_CALL_CORDS = False
DRAW_LANDMARKS      = False

_PANEL_W       = int(FRAME_WIDTH  * 0.16)
_PANEL_H       = int(FRAME_HEIGHT * 0.2)
_PANEL_X       = 0
_PANEL_Y       = FRAME_HEIGHT - _PANEL_H

_FONT_SIZE     = max(8, int(FRAME_HEIGHT * 0.021))
_LABEL_X       = int(FRAME_WIDTH  * 0.016)
_STATUS_X      = int(FRAME_WIDTH  * 0.119)
_ROW_START_Y   = _PANEL_Y + int(FRAME_HEIGHT * 0.042)
_ROW_STEP      = int(FRAME_HEIGHT * 0.031)

_STATUS_BAR_H  = int(FRAME_HEIGHT * 0.083)
_STATUS_FONT   = max(10, int(FRAME_HEIGHT * 0.042))
_BRAND_X       = int(FRAME_WIDTH  * 0.859)
_BRAND_Y       = int(FRAME_HEIGHT * 0.063)
_STATUS_Y      = int(FRAME_HEIGHT * 0.063)

_FPS_X         = int(FRAME_WIDTH  * 0.016)
_FPS_Y         = int(FRAME_HEIGHT * 0.146)
_FPS_FONT      = max(10, int(FRAME_HEIGHT * 0.042))

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

FACE_THRESHOLD         = 0.5
LEFT_EYE_THRESHOLD     = 0.35
RIGHT_EYE_THRESHOLD    = 0.35
MOUTH_THRESHOLD        = 0.4
FACING_LEFT_THRESHOLD  = 0.5
FACING_RIGHT_THRESHOLD = 2
SMK_CALL_THRESHOLD     = 0.5
HEAD_DOWN_THRESHOLD    = 0.78

LEFT_W  = 3
RIGHT_W = 3
LEFT_EYE_STATUS  = np.zeros(LEFT_W)
RIGHT_EYE_STATUS = np.zeros(RIGHT_W)

RIGHT_EYE_ID = 1
LEFT_EYE_ID  = 0

HEAD_W      = 7
HEAD_STATUS = [1] * HEAD_W

DEBUG_POINTS = {
    33: "33", 133: "133", 159: "159", 145: "145",
    362: "362", 263: "263", 386: "386", 374: "374",
    78: "78", 308: "308", 13: "13", 14: "14",
    132: "132", 361: "361"
}

def clamp_roi(x1, y1, x2, y2, w, h):
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    return x1, y1, x2, y2


# ═══════════════════════════════════════════════════════════════════════════════
#  GSTREAMER WIDGET  — Gtk.Box that owns the pipeline
#  Pipeline built in _on_realize() so GTK is ready before GStreamer starts.
#  All DMS inference logic from STM32_Dms.py lives here unchanged.
# ═══════════════════════════════════════════════════════════════════════════════
class GstWidget(Gtk.Box):

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.connect("realize", self._on_realize)

        # ── shared inference state ────────────────────────────────────────
        self.inited              = False
        self.distracted          = False
        self.drowsy              = False
        self.yawn                = False
        self.smoking             = False
        self.phone               = False
        self.head_down           = False
        self.last_phone_status   = False
        self.last_smoke_status   = False
        self.face_cords          = []
        self.marks               = []
        self.safe_value          = 0.0
        self.smk_call_cords      = []
        self.left_eye_closed     = False
        self.right_eye_closed    = False
        self.both_eyes_closed    = False
        self.left_eye_ratio      = 0.0
        self.right_eye_ratio     = 0.0
        self.frame_count         = 0
        self.debug_eyes_detection= False
        self.print_eye_status    = False
        self._last_printed_status= None
        self.save_landmarks      = False
        self.call_count          = 0
        self.smoke_count         = 0
        self.last_phone_detected_ts  = 0
        self.phone_alrt_cooldown_ts  = 0
        self.last_smoke_detected_ts  = 0
        self.smoke_alrt_cooldown_ts  = 0
        self.last_landmark_save_time = 0.0
        self.last_save_time      = 0
        self.last_face_marks     = []
        self.last_smk_call_cords = []

        self.fps_counter    = 0
        self.fps_start_time = time.time()
        self.current_fps    = 0.0
        self.display_fps    = 0.0

        self.run_face_detection          = True
        self.run_face_landmarks          = True
        self.run_eye_detection           = True
        self.run_yolo_smk_call_detection = True

    # ── realize: load models then build pipeline ──────────────────────────────
    def _on_realize(self, widget):
        self._load_models()
        self._build_pipeline()
        self.pipeline.set_state(Gst.State.PLAYING)
        print("[DMS] Pipeline started.")

    def _load_models(self):

        print("[DMS] Loading models...")

        if self.run_face_detection:
            self.face_detector = FaceDetector(FACE_MODEL, FACE_THRESHOLD)

        if self.run_face_landmarks:
            self.face_landmark = FaceLandmark(LANDMARK_MODEL)
            self.mouth         = Mouth()

        if self.run_eye_detection:
            self.eye_detector  = Eye(IRIS_MODEL)

        if self.run_yolo_smk_call_detection:
            self.smoking_calling_detector = SmokingCallingDetector(SMK_CALL_MODEL, conf=SMK_CALL_THRESHOLD)
        
        self.inited = True
        print("[DMS] Models loaded.")

    def _build_pipeline(self):
        """
        Dual-pad libcamerasrc pipeline — identical structure to csi_dms.py:

        src  (view-finder)  → videorate → queue [RGB16] → cairooverlay
                                                         → videoconvert → fpsdisplaysink → gtkwaylandsink
        src_%u (still-cap)  → videorate → queue [BGR]   → appsink → inference()
        """
        w   = self.app.frame_width
        h   = self.app.frame_height
        fps = self.app.framerate

        self.pipeline = Gst.Pipeline.new("dms-pipeline")

        # ── source ────────────────────────────────────────────────────────
        src = Gst.ElementFactory.make("libcamerasrc", "libcamera")
        if not src:
            raise RuntimeError("libcamerasrc not available")

        # ── display branch caps ───────────────────────────────────────────
        caps_disp = Gst.Caps.from_string(
            f"video/x-raw,width={w},height={h},format=RGB16,framerate={fps}/1"
        )

        # ── AI branch caps ────────────────────────────────────────────────
        caps_nn = Gst.Caps.from_string(
            f"video/x-raw,width={w},height={h},format=BGR,framerate={fps}/1"
        )

        # ── display branch elements ───────────────────────────────────────
        vrate_disp = Gst.ElementFactory.make("videorate",    "vrate_disp")
        queue_disp = Gst.ElementFactory.make("queue",        "queue_disp")
        queue_disp.set_property("leaky",           2)
        queue_disp.set_property("max-size-buffers", 1)

        self.cairooverlay = Gst.ElementFactory.make("cairooverlay", "overlay")
        if not self.cairooverlay:
            raise RuntimeError("cairooverlay not found")
        self.cairooverlay.connect("draw", self.draw)

        vconv_disp = Gst.ElementFactory.make("videoconvert", "vconv_disp")

        # ── gtkwaylandsink packed into THIS Gtk.Box (self) ────────────────
        gtkwaylandsink = Gst.ElementFactory.make("gtkwaylandsink", "sink_disp")
        if not gtkwaylandsink:
            raise RuntimeError("gtkwaylandsink not found")
        self.pack_start(gtkwaylandsink.props.widget, True, True, 0)
        gtkwaylandsink.props.widget.show()

        fps_sink = Gst.ElementFactory.make("fpsdisplaysink", "fps_sink")
        fps_sink.set_property("signal-fps-measurements", True)
        fps_sink.set_property("fps-update-interval",     2000)
        fps_sink.set_property("text-overlay",            False)
        fps_sink.set_property("video-sink",              gtkwaylandsink)
        fps_sink.connect("fps-measurements",             self._on_fps)

        # ── AI branch elements ────────────────────────────────────────────
        vrate_nn = Gst.ElementFactory.make("videorate", "vrate_nn")
        queue_nn = Gst.ElementFactory.make("queue",     "queue_nn")
        queue_nn.set_property("leaky",           2)
        queue_nn.set_property("max-size-buffers", 1)

        self.appsink = Gst.ElementFactory.make("appsink", "appsink")
        self.appsink.set_property("emit-signals",       True)
        self.appsink.set_property("sync",               False)
        self.appsink.set_property("max-buffers",        1)
        self.appsink.set_property("drop",               True)
        self.appsink.set_property("enable-last-sample", False)
        self.appsink.connect("new-sample", self.inference)

        # ── add all elements ──────────────────────────────────────────────
        for el in [src,
                   vrate_disp, queue_disp, self.cairooverlay, vconv_disp, fps_sink,
                   vrate_nn,   queue_nn,   self.appsink]:
            self.pipeline.add(el)

        # ── link display branch ───────────────────────────────────────────
        src.link(vrate_disp)
        vrate_disp.link(queue_disp)
        queue_disp.link_filtered(self.cairooverlay, caps_disp)
        self.cairooverlay.link(vconv_disp)
        vconv_disp.link(fps_sink)

        static_src = src.get_static_pad("src")
        static_src.set_property("stream-role", 3)   # view-finder

        # ── link AI branch ────────────────────────────────────────────────
        pad_tmpl   = src.get_pad_template("src_%u")
        src_pad_nn = src.request_pad(pad_tmpl, None, None)
        src_pad_nn.set_property("stream-role", 1)   # still-capture
        src_pad_nn.link(vrate_nn.get_static_pad("sink"))
        vrate_nn.link(queue_nn)
        queue_nn.link_filtered(self.appsink, caps_nn)

        # ── bus ───────────────────────────────────────────────────────────
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error",         self._on_error)
        bus.connect("message::eos",           self._on_eos)
        bus.connect("message::state-changed", self._on_state_changed)

    # ── GStreamer bus callbacks ────────────────────────────────────────────────
    def _on_fps(self, sink, fps, droprate, avgfps):
        self.display_fps = fps

    def _on_eos(self, bus, msg):
        print("[GST] EOS")

    def _on_error(self, bus, msg):
        err, dbg = msg.parse_error()
        print(f"[GST] Error: {err}\n      {dbg}")

    def _on_state_changed(self, bus, msg):
        old, new, _ = msg.parse_state_changed()
        if old == Gst.State.NULL and new == Gst.State.READY:
            Gst.debug_bin_to_dot_file(
                self.pipeline, Gst.DebugGraphDetails.ALL, "dms_pipeline"
            )

    def stop_camera(self):
        self.pipeline.set_state(Gst.State.NULL)

    def save_frames(self, frame):
        os.makedirs("saved_frames", exist_ok=True)
        now = time.time()
        if now - self.last_save_time >= 1.0:
            cv2.imwrite(f"saved_frames/frame_{int(now)}.jpg", frame)
            self.last_save_time = now

    # ═════════════════════════════════════════════════════════════════════════
    #  INFERENCE  — exactly as STM32_Dms.py
    # ═════════════════════════════════════════════════════════════════════════
    def inference(self, data):
        """
        Main inference callback triggered by GStreamer appsink.
        Runs all DMS models (face, eyes, mouth, smoking, calling)
        on a single video frame and updates driver state.

        example boxes:
        1st:[[0.26253396 0.48752266 0.71962506 0.94461375]]
        2nd:[[168.02173 234.01088 460.56003 453.4146 ]]
        3rd:[[130. 159. 498. 527.]]
        4th:[[130 159 498 480]]
        """

        #print("frame came")
        # Pull the latest frame from appsink
        sample = self.appsink.emit("pull-sample")

        # Output containers for drawing/debug
        face_cords     = []
        smk_call_cords = []
        mark_group     = []

        # Default driver states (True = safe / normal)
        phone_detected    = False
        smoking_detected  = False
        distracted        = False
        yawn_detected     = False
        head_down_final   = False
        sleep_detected    = False

        # Drop frame if pipeline is not ready
        if sample is None or not self.inited:
            return Gst.FlowReturn.OK
        
        # Frame counter for debugging / profiling
        self.frame_count += 1
        self.fps_counter += 1

        now     = time.time()
        elapsed = now - self.fps_start_time
        if elapsed >= 1.0:
            self.current_fps    = self.fps_counter / elapsed
            self.fps_counter    = 0
            self.fps_start_time = now

        # -------------------------------------------------
        # Extract raw image buffer from GStreamer sample
        # -------------------------------------------------
        buffer = sample.get_buffer()
        caps   = sample.get_caps()

        ret, mem_buf = buffer.map(Gst.MapFlags.READ)

        if not ret:
            return Gst.FlowReturn.OK

        height = caps.get_structure(0).get_value("height")
        width  = caps.get_structure(0).get_value("width")

        # Convert raw buffer into NumPy array (BGR → RGB)
        # frame = np.ndarray(
        #     shape=(height, width, 3),
        #     dtype=np.uint8,
        #     buffer=mem_buf.data
        # )[..., ::-1]

        frame = np.ndarray(shape=(height, width, 3), dtype=np.uint8, buffer=mem_buf.data)

        #self.save_frames(frame)

        # ── Per-frame model scheduling (staggered) ─────────
        # Face detection runs every frame (cheap, needed for all others)
        run_landmark = (self.frame_count % 3 == 0) and self.run_face_landmarks
        run_eye      = (self.frame_count % 3 == 0) and self.run_eye_detection
        run_smk      = (self.frame_count % 3 == 1) and self.run_yolo_smk_call_detection

        # -------------------------------------------------
        # Face detection (normalized coordinates)
        # -------------------------------------------------
        if self.run_face_detection:
            boxes = self.face_detector.detect(frame)

            # normalize shape
            if boxes is None or len(boxes) == 0:
                boxes = []

        #print(f"1st:{boxes}")

        # Reset per-frame outputs
        face_cords     = []
        smk_call_cords = []
        mark_group     = []

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
                self.last_phone_status   = phone_detected
                self.last_smoke_status   = smoking_detected
            else:
                smk_call_cords   = self.last_smk_call_cords
                phone_detected   = self.last_phone_status
                smoking_detected = self.last_smoke_status

            # ---------------------------------------------
            # Convert normalized face boxes → pixel coords
            # ---------------------------------------------
            for i in range(np.size(boxes, 0)):
                boxes[i][[0, 2]] *= width
                boxes[i][[1, 3]] *= height

            #print(f"2nd:{boxes}")

            # Expand face boxes into squares (for landmarks)
            boxes = self.transform_to_square(boxes, scale=1.26, offset=(0, 0))

            #print(f"3rd:{boxes}")

            # Clip boxes within image boundaries
            boxes, _ = self.clip_boxes(boxes, (0, 0, width, height))
            boxes = boxes.astype(np.int32)

            #print(f"4th:{boxes}")

            # ---------------------------------------------
            # Select face closest to image center
            # (Assume driver is centered)
            # ---------------------------------------------
            face_in_center     = 0
            distance_to_center = math.hypot(width / 2, height / 2)
            for i in range(np.size(boxes, 0)):
                x1, y1, x2, y2 = boxes[i]
                mid_to_center = math.hypot(
                    (x2 + x1 - width) / 2,
                    (y2 + y1 - height) / 2
                )
                if mid_to_center < distance_to_center:
                    face_in_center     = i
                    distance_to_center = mid_to_center

            # Final selected face bounding box
            if len(boxes) > 0:
                x1, y1, x2, y2 = boxes[face_in_center]
                face_cords.append([x1, y1, x2, y2])
            else:
                face_cords = []

            # ---------------------------------------------
            # Face landmark inference
            # ---------------------------------------------
            if run_landmark:
                face_image = frame[y1:y2, x1:x2]
                face_marks = self.face_landmark.get_landmark(face_image, (x1, y1, x2, y2))

                face_marks = np.array(face_marks)
                mark_group.append(face_marks)
                self.last_face_marks = face_marks
            else:
                face_marks = self.last_face_marks

            #print("Landmark count:", len(face_marks))

            #-------For testing the face points------------
            if self.save_landmarks:
                self.save_face_landmarks(face_marks)
            # ---------------------------------------------

            # ---------------------------------------------
            # Yawn detection using mouth ratio
            # ---------------------------------------------
            if self.run_face_landmarks and len(face_marks) == 468:
                mouth_ratio   = self.mouth.yawning_ratio(face_marks)
                yawn_detected = not (mouth_ratio <= MOUTH_THRESHOLD)

                head_down_ratio = self.mouth.head_down_ratio(face_marks)
                status = 1 if head_down_ratio > HEAD_DOWN_THRESHOLD else 0
                HEAD_STATUS[:-1] = HEAD_STATUS[1:]
                HEAD_STATUS[-1]  = status
                head_down_avg    = np.mean(HEAD_STATUS)
                head_down_final  = head_down_avg > 0.5
                #print(f"ratio={head_down_ratio:.3f}, status={status}, avg={head_down_avg:.3f}, final={head_down_final}")

                # ---------------------------------------------
                # Attention detection (head pose proxy)
                # ---------------------------------------------
                mouth_face_ratio = self.mouth.mouth_face_ratio(face_marks)
                #print(f"mouth face Ratio {mouth_face_ratio}")
                if (
                    mouth_face_ratio < FACING_LEFT_THRESHOLD
                    or mouth_face_ratio > FACING_RIGHT_THRESHOLD
                ):
                    distracted = True
                else:
                    distracted = False

            # ---------------------------------------------
            # Left eye landmark & iris detection
            # ---------------------------------------------
            if run_eye:
                mark_group, sleep_detected = self.eye_landmark_detection(face_marks, frame, buffer, mem_buf, mark_group)
            else:
                sleep_detected = self.both_eyes_closed

        else:
            # No face detected
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
 
        buffer.unmap(mem_buf)

        return Gst.FlowReturn.OK

    # ── model sub-routines — exactly as STM32_Dms.py ─────────────────────────
    def smoke_call_detection(self, frame, phone_detected, smoking_detected):
        smk_call_cords = []
        phone_seen     = False
        cigrate_seen   = False

        smk_call_result = self.smoking_calling_detector.inference(frame, False)

        if np.size(smk_call_result, 0) > 0:
            for i in range(np.size(smk_call_result, 0)):
                cls_id = int(smk_call_result[i][5])
                conf   = smk_call_result[i][4]

                if cls_id == 0 and conf > SMK_CALL_THRESHOLD:
                    phone_seen   = True
                    #phone_detected = True

                if cls_id == 1 and conf > SMK_CALL_THRESHOLD:
                    cigrate_seen = True
                    #smoking_detected = True

                x1 = int(smk_call_result[i][0])
                y1 = int(smk_call_result[i][1])
                x2 = int(smk_call_result[i][2])
                y2 = int(smk_call_result[i][3])
                smk_call_cords.append([x1, y1, x2, y2])

            if phone_seen:
                self.call_count += 1
                phone_detected   = True
            else:
                self.call_count  = 0

            if cigrate_seen:
                self.smoke_count  += 1
                smoking_detected   = True
            else:
                self.smoke_count   = 0

        return smk_call_cords, phone_detected, smoking_detected

    def eye_landmark_detection(self, face_marks, frame, buffer, mem_buf, mark_group):
        if face_marks is None or len(face_marks) < 468:
            return mark_group, self.both_eyes_closed

        for eye_id, eye_label in [(LEFT_EYE_ID, "left"), (RIGHT_EYE_ID, "right")]:
            x1, y1, x2, y2 = self.eye_detector.get_eye_roi(face_marks, eye_id)
            x1, y1, x2, y2 = clamp_roi(x1, y1, x2, y2, frame.shape[1], frame.shape[0])
            eye_img = frame[y1:y2, x1:x2]

            if eye_img is None or eye_img.size == 0:
                buffer.unmap(mem_buf)
                return mark_group, self.both_eyes_closed

            eye_marks, iris_marks = self.eye_detector.get_landmark(
                eye_img, (x1, y1, x2, y2), eye_id
            )
            mark_group.append(np.array(iris_marks))

            ratio = self.eye_detector.blinking_ratio(eye_marks, eye_id)

            if eye_id == LEFT_EYE_ID:
                LEFT_EYE_STATUS[:-1] = LEFT_EYE_STATUS[1:]
                LEFT_EYE_STATUS[-1]  = 1 if ratio > LEFT_EYE_THRESHOLD  else 0
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
            left_status  = "CLOSED" if self.left_eye_closed  else "OPEN"
            right_status = "CLOSED" if self.right_eye_closed else "OPEN"
            debug_eyes_dir = "/root/dms/dms-point-custom/debug/eyes/"
            try:
                os.makedirs(debug_eyes_dir, exist_ok=True)
            except Exception as e:
                print(f"Error creating directory {debug_eyes_dir}: {e}")

        if self.frame_count % 30 == 0 and self.print_eye_status:
            status = "✓" if not self.both_eyes_closed else "✗"
            print(f"Frame {self.frame_count}: {status} "
                  f"Left: {'OPEN' if not self.left_eye_closed else 'CLOSED'} "
                  f"({self.left_eye_ratio:.3f}), "
                  f"Right: {'OPEN' if not self.right_eye_closed else 'CLOSED'} "
                  f"({self.right_eye_ratio:.3f})")

    def save_face_landmarks(self, face_marks):

        if face_marks is None or len(face_marks) != 468:
            return
        
        now = time.time()
        if now - self.last_landmark_save_time < 1.0:
            return
        
        self.last_landmark_save_time = now

        # freeze snapshot for this frame
        frozen = face_marks.copy()
        with open("landmarks_main.txt", "a") as f:
            MAIN_POINTS = {
                159: "LEFT_EYE_UPPER",  145: "LEFT_EYE_LOWER",
                386: "RIGHT_EYE_UPPER", 374: "RIGHT_EYE_LOWER",
                13:  "UPPER_LIP",       14:  "LOWER_LIP",
                132: "LEFT_CHEEK",      361: "RIGHT_CHEEK",
            }
            for idx, name in MAIN_POINTS.items():
                x, y = frozen[idx]
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
        margin = np.floor_divide(np.maximum(height, width) * scale,2)

        # -------------------------------------------------
        # Reconstruct square bounding boxes
        # [xmin, ymin, xmax, ymax]
        # -------------------------------------------------
        return np.concatenate(
            (center_x - margin, center_y - margin,
             center_x + margin, center_y + margin), axis=1
        )

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

    # ═════════════════════════════════════════════════════════════════════════
    #  CAIRO DRAW  — exactly as STM32_Dms.py
    # ═════════════════════════════════════════════════════════════════════════
    def draw(self, overlay, context, timestamp, duration):
        """
        Cairo overlay callback.
        Draws DMS inference results (boxes, landmarks, alerts)
        on top of the live camera preview.
        """

        # Snapshot shared state once — prevents race condition with inference thread
        face_cords     = self.face_cords
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


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW  — exactly as csi_dms.py
# ═══════════════════════════════════════════════════════════════════════════════
class MainWindow(Gtk.Window):

    def __init__(self, app):
        Gtk.Window.__init__(self)
        self.app = app
        self.set_decorated(False)
        self.maximize()
        self.set_position(Gtk.WindowPosition.CENTER)
        self.connect("destroy",        Gtk.main_quit)
        self.connect("key-press-event", self._on_key)
        # GstWidget IS the only child — same as csi_dms.py
        self.add(self.app.gst_widget)

    def _on_key(self, widget, event):
        if event.keyval == Gdk.KEY_Escape:
            Gtk.main_quit()


# ═══════════════════════════════════════════════════════════════════════════════
#  APPLICATION  — same pattern as csi_dms.py, minus check_camera / modetest
# ═══════════════════════════════════════════════════════════════════════════════
class Application:
    def __init__(self, args):
        self.frame_width  = args.frame_width
        self.frame_height = args.frame_height
        self.framerate    = args.framerate
        # Pass model paths into the widget via global update
        global FACE_MODEL, LANDMARK_MODEL, IRIS_MODEL, SMK_CALL_MODEL
        FACE_MODEL     = args.face_model
        LANDMARK_MODEL = args.landmark_model
        IRIS_MODEL     = args.iris_model
        SMK_CALL_MODEL = args.smk_model
        self.gst_widget  = GstWidget(self)
        self.main_window = MainWindow(self)

    def run(self):
        self.main_window.show_all()
        self.main_window.connect("delete-event", Gtk.main_quit)
        Gtk.main()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser(
        description="STM32MP2x DMS — CSI camera. Pass .nb or .tflite models freely.")
    parser.add_argument("--frame_width",     type=int, default=640)
    parser.add_argument("--frame_height",    type=int, default=480)
    parser.add_argument("--framerate",       type=int, default=30)
    parser.add_argument("--face-model",      default=FACE_MODEL,
                        help="Face detection model (.nb or .tflite)")
    parser.add_argument("--landmark-model",  default=LANDMARK_MODEL,
                        help="Face landmark model (.nb or .tflite)")
    parser.add_argument("--iris-model",      default=IRIS_MODEL,
                        help="Iris/eye model (.nb or .tflite)")
    parser.add_argument("--smk-model",       default=SMK_CALL_MODEL,
                        help="Smoking/calling model (.nb or .tflite)")
    args = parser.parse_args()

    try:
        FACE_MODEL     = translate_model_path(args.face_model)
        LANDMARK_MODEL = translate_model_path(args.landmark_model)
        IRIS_MODEL     = translate_model_path(args.iris_model)
        SMK_CALL_MODEL = translate_model_path(args.smk_model)

        if HAS_GUI:
            app = Application(args)
            app.run()
        else:
            print("[DMS] GTK/GStreamer not available. Running in PC OpenCV fallback mode...")
            import STM32_Dms
            STM32_Dms.FACE_MODEL = FACE_MODEL
            STM32_Dms.LANDMARK_MODEL = LANDMARK_MODEL
            STM32_Dms.IRIS_MODEL = IRIS_MODEL
            STM32_Dms.SMK_CALL_MODEL = SMK_CALL_MODEL

            demo = STM32_Dms.DMSDemo(device="0", width=args.frame_width, height=args.frame_height, fps=args.framerate)
            while getattr(demo, "cv_running", False):
                time.sleep(0.5)
    except Exception as e:
        print(f"[DMS] Fatal: {e}")
        import traceback; traceback.print_exc()

    print("[DMS] Exited cleanly.")
    os._exit(0)