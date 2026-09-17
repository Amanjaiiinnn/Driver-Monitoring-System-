import os
import sys
import math
import time
import gzip
import gc
import signal
import hashlib
import pickle
import tempfile
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
import concurrent.futures as cf

try:
    import face_recognition as _fr
    HAS_FR = True
except ImportError:
    HAS_FR = False

# ── Custom-DMS on path so all helper modules are found ───────────────────────
sys.path.insert(0, "/home/root/Custom-DMS")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from npu_proxy import NPUProxy

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


def check_camera_status(frame):
    """
    Checks if the frame is black or blurry.
    Returns: "BLACK", "BLUR", or "OK"
    """
    if frame is None or frame.size == 0:
        return "BLACK"
        
    # Check if frame is grayscale or color
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # 1. Black check (average brightness < 15.0)
    brightness = np.mean(gray)
    if brightness < 15.0:
        return "BLACK"

    # 2. Blur check (Laplacian variance on a central patch < BLUR_SKIP_THRESHOLD)
    h, w = gray.shape[:2]
    cx, cy = w // 2, h // 2
    # Ensure crop boundaries are within frame
    x1 = max(0, cx - 80)
    x2 = min(w, cx + 80)
    y1 = max(0, cy - 60)
    y2 = min(h, cy + 60)

    if (x2 - x1) > 10 and (y2 - y1) > 10:
        patch = gray[y1:y2, x1:x2]
        blur_val = cv2.Laplacian(patch, cv2.CV_64F).var()
        if blur_val < BLUR_SKIP_THRESHOLD:
            return "BLUR"

    return "OK"

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
#  FACE RECOGNITION CONFIG
# ═══════════════════════════════════════════════════════════
RUN_FACE_RECOG = HAS_FR

SAMPLE_DIR   = "/home/root/face-recognition/haarcascade/sample"
CACHE_FILE   = "/home/root/face-recognition/haarcascade/face_cache.pkl.gz"
CASCADE_PATH = "/home/root/face-recognition/haarcascade/haarcascade_frontalface_default.xml"
DETECTED_DIR = "/home/root/face-recognition/haarcascade/detected_faces"
TOLERANCE    = 0.50

RECOG_RETRY_INTERVAL  = 0.0
RECOG_VERIFY_INTERVAL = 120.0
RETRY_COUNT = 5
NO_FACE_RESET_AFTER   = 30

DETECTING  = "DETECTING"
IDENTIFIED = "IDENTIFIED"
UNKNOWN    = "UNKNOWN"

# ═══════════════════════════════════════════════════════════
#  RESOLUTION
# ═══════════════════════════════════════════════════════════
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# ═══════════════════════════════════════════════════════════
#  DRAWING
# ═══════════════════════════════════════════════════════════
DRAW_SMK_CALL_CORDS = False
DRAW_LANDMARKS      = False

_PANEL_W     = int(FRAME_WIDTH  * 0.16)
_PANEL_H     = int(FRAME_HEIGHT * 0.2)
_PANEL_X     = 0
_PANEL_Y     = FRAME_HEIGHT - _PANEL_H

_FONT_SIZE   = max(8,  int(FRAME_HEIGHT * 0.021))
_LABEL_X     = int(FRAME_WIDTH  * 0.016)
_STATUS_X    = int(FRAME_WIDTH  * 0.119)
_ROW_START_Y = _PANEL_Y + int(FRAME_HEIGHT * 0.042)
_ROW_STEP    = int(FRAME_HEIGHT * 0.031)

_STATUS_BAR_H = int(FRAME_HEIGHT * 0.083)
_STATUS_FONT  = max(10, int(FRAME_HEIGHT * 0.042))
_BRAND_X      = int(FRAME_WIDTH  * 0.859)
_BRAND_Y      = int(FRAME_HEIGHT * 0.063)
_STATUS_Y     = int(FRAME_HEIGHT * 0.063)

_FPS_X    = int(FRAME_WIDTH  * 0.016)
_FPS_Y    = int(FRAME_HEIGHT * 0.146)
_FPS_FONT = max(10, int(FRAME_HEIGHT * 0.042))

# ═══════════════════════════════════════════════════════════
#  THRESHOLDS / PENALTIES
# ═══════════════════════════════════════════════════════════
NO_FACE_PENALTY     = 0.7
YAWN_PENALTY        = 7.0
DISTRACT_PENALTY    = 2.0
SLEEP_PENALTY       = 5.0
SMK_PENALTY         = 2.0
CALL_PENALTY        = 2.0
HEAD_DOWN_PENALTY   = 5.0
RESTORE_CREDIT      = -5.0

FACE_THRESHOLD         = 0.5
LEFT_EYE_THRESHOLD     = 0.40 # eye blink ratio
RIGHT_EYE_THRESHOLD    = 0.40
MOUTH_THRESHOLD        = 0.4 # yawn ratio
FACING_LEFT_THRESHOLD  = 0.5
FACING_RIGHT_THRESHOLD = 2
SMK_CALL_THRESHOLD     = 0.25  # TFLite outputs raw probabilities (not logits)
HEAD_DOWN_THRESHOLD    = 0.58   # ratio of nose position in face — >0.58 = head tilted down
HEAD_DOWN_FRAME_AVG = 0.5

# Sliding-window eye / head state  (module-level so they persist across frames)
LEFT_EYE_STATUS  = np.zeros(2)
RIGHT_EYE_STATUS = np.zeros(2)
HEAD_STATUS      = [0] * 5   # init to 0 = not head-down
YAWN_STATUS      = [0] * 3

# ── Motion / blur config ──────────────────────────────────────────────────────
# Laplacian variance below this = blurry/motion frame → skip NPU inference.
# Tune: lower = only skip extreme blur, higher = skip mild motion too.
# Recommended range: 30 (lenient) – 80 (aggressive skip)
BLUR_SKIP_THRESHOLD = 50.0

# ═══════════════════════════════════════════════════════════
#  DLIB WORKER (face recognition, runs in separate process)
# ═══════════════════════════════════════════════════════════
_worker_encodings = None
_worker_names     = None

# Target size for face crop sent to dlib ResNet.
# dlib internally resizes to 150×150 anyway — sending exactly that
# avoids a redundant resize step inside the library.
_RECOG_CROP_SIZE  = 150


def _worker_init(cache_file: str) -> None:
    global _worker_encodings, _worker_names
    # ── Pin to core 0. npu_worker.py owns core 1 — don't compete with it ──
    try:
        os.sched_setaffinity(0, {0})
    except Exception:
        pass

    # ── Load encodings from cache ──────────────────────────────────────────
    try:
        with gzip.open(cache_file, "rb") as f:
            data = pickle.load(f)
        _worker_encodings = data.get("encodings", [])
        _worker_names     = data.get("names",     [])
        print(f"[RECOG WORKER] Ready — {len(_worker_names)} encoding(s)", flush=True)
    except Exception as exc:
        print(f"[RECOG WORKER] DB load failed: {exc}", flush=True)
        _worker_encodings = []
        _worker_names     = []


def _worker_encode(crop_bytes: bytes, shape: tuple, tolerance: float):

    # Reconstruct crop — already BGR from camera
    crop = np.frombuffer(crop_bytes, dtype=np.uint8).reshape(shape)

    # Resize to exactly 150×150 — dlib ResNet target size.
    # Avoids redundant internal resize and keeps encoding time consistent.
    crop_resized = cv2.resize(crop, (_RECOG_CROP_SIZE, _RECOG_CROP_SIZE),
                                  interpolation=cv2.INTER_LINEAR)

    # Convert BGR → RGB (dlib expects RGB)
    rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    sz  = _RECOG_CROP_SIZE

    t0   = time.time()
    # known_face_locations=(top, right, bottom, left) in CSS order
    encs = _fr.face_encodings(rgb,
                               known_face_locations=[(0, sz, sz, 0)],
                               num_jitters=1)
                               #model="small")   # "small" is 5x faster than default
    t_ms = (time.time() - t0) * 1000

    if not encs or not _worker_encodings:
        return "Unknown", 1.0, [], t_ms

    dists = _fr.face_distance(_worker_encodings, encs[0])
    idx   = int(np.argmin(dists))
    name  = _worker_names[idx] if dists[idx] < tolerance else "Unknown"
    top3  = [(_worker_names[t], float(dists[t]))
              for t in np.argsort(dists)[:3]]
    return name, float(dists[idx]), top3, t_ms


# ═══════════════════════════════════════════════════════════
#  FACE DATABASE
# ═══════════════════════════════════════════════════════════
def _atomic_gzip_write(path: str, obj) -> None:
    fd, tmp = tempfile.mkstemp(prefix=".tmp_facecache_",
                               dir=os.path.dirname(path) or ".")
    os.close(fd)
    with gzip.open(tmp, "wb", compresslevel=5) as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _atomic_gzip_read(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return default


def _file_blake2b(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


class FaceDB:
    def __init__(self, cache_file=CACHE_FILE, sample_dir=SAMPLE_DIR,
                 rebuild=False):
        self.cache_file = cache_file
        self.sample_dir = sample_dir

        self.cascade = cv2.CascadeClassifier(CASCADE_PATH)
        if self.cascade.empty():
            print(f"[DB] WARNING: cannot load '{CASCADE_PATH}'")

        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(DETECTED_DIR,   exist_ok=True)

        if rebuild and os.path.exists(cache_file):
            os.remove(cache_file)
            print("[DB] Cache cleared — rebuilding…")

        data           = _atomic_gzip_read(
            cache_file, {"encodings": [], "names": [], "hashes": {}})
        self.encodings = data.get("encodings", [])
        self.names     = data.get("names",     [])
        self.hashes    = data.get("hashes",    {})
        print(f"[DB] Loaded {len(self.names)} cached encoding(s)")
        self._sync()

    def _name_from_file(self, fname: str) -> str:
        raw = os.path.splitext(fname)[0]
        if "_" in raw and raw.rsplit("_", 1)[-1].isdigit():
            return raw.rsplit("_", 1)[0].capitalize()
        return raw.capitalize()

    def _haar_crop(self, img):
        if self.cascade.empty():
            return img
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        return img[y:y + h, x:x + w]

    def _sync(self) -> None:
        files = [
            (os.path.join(self.sample_dir, f), self._name_from_file(f))
            for f in sorted(os.listdir(self.sample_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        file_keys    = {f"{n}::{p}" for p, n in files}
        removed_keys = set(self.hashes.keys()) - file_keys
        if removed_keys:
            rn   = {k.split("::")[0] for k in removed_keys}
            keep = [i for i in range(len(self.names))
                    if self.names[i] not in rn]
            self.encodings = [self.encodings[i] for i in keep]
            self.names     = [self.names[i]     for i in keep]
            for k in removed_keys:
                self.hashes.pop(k, None)
            _atomic_gzip_write(self.cache_file,
                {"encodings": self.encodings, "names": self.names,
                 "hashes": self.hashes})

        updated = False
        for path, name in files:
            key = f"{name}::{path}"
            h   = _file_blake2b(path)
            if self.hashes.get(key) == h:
                continue

            print(f"[DB] Encoding: {name}  ({os.path.basename(path)})")
            img = cv2.imread(path)
            if img is None:
                continue
            crop = self._haar_crop(img)
            if crop is None:
                print(f"[DB] WARN: no face in {path}")
                continue

            rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            ch, cw = rgb.shape[:2]
            encs   = face_recognition.face_encodings(
                rgb, known_face_locations=[(0, cw, ch, 0)], num_jitters=1)
            if not encs:
                print(f"[DB] WARN: dlib failed on {path}")
                continue

            if name in self.names:
                self.encodings[self.names.index(name)] = encs[0]
            else:
                self.names.append(name)
                self.encodings.append(encs[0])

            self.hashes[key] = h
            updated = True
            print(f"[DB]   ✔ {name}")

        if updated:
            _atomic_gzip_write(self.cache_file,
                {"encodings": self.encodings, "names": self.names,
                 "hashes": self.hashes})
            print("[DB] Cache saved ✅")

        print(f"[DB] Ready — {len(set(self.names))} person(s), "
              f"{len(self.names)} encoding(s): {sorted(set(self.names))}")


# ═══════════════════════════════════════════════════════════
#  FACE RECOGNITION TRACK
# ═══════════════════════════════════════════════════════════
class FaceTrack:
    def __init__(self):
        self.state       = DETECTING
        self.name        = ""
        self.retry_count = 0
        self.pending     = False

    def reset(self):
        self.state       = DETECTING
        self.name        = ""
        self.retry_count = 0
        self.pending     = False

    @property
    def label(self) -> str:
        if self.state == DETECTING:
            return (f"Verifying… ({self.retry_count}/{RETRY_COUNT})"
                    if self.retry_count > 0 else "Identifying…")
        return self.name if self.state == IDENTIFIED else "Unknown"

    @property
    def label_color(self) -> tuple:
        if self.state == DETECTING:  return (1.0, 0.55, 0.0)
        if self.state == IDENTIFIED: return (0.0, 1.0, 0.0)
        return (1.0, 0.0, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  GSTREAMER WIDGET
# ═══════════════════════════════════════════════════════════════════════════════
class GstWidget(Gtk.Box):

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.connect("realize", self._on_realize)

        # ── DMS state ─────────────────────────────────────────────────────────
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
        self.frame_count         = 0
        self.call_count          = 0
        self.smoke_count         = 0
        self.last_face_marks     = []
        self.last_smk_call_cords = []
        self._last_boxes         = None   # last valid face detection result
        self.camera_status       = "OK"

        self.fps_counter    = 0
        self.fps_start_time = time.time()
        self.current_fps    = 0.0

        self.run_face_detection          = True
        self.run_face_landmarks          = True
        self.run_eye_detection           = True
        self.run_yolo_smk_call_detection = True

        # ── Face recognition ──────────────────────────────────────────────────
        self.face_track       = FaceTrack()
        self._active_job      = False
        self._executor        = None
        self._no_face_frames  = 0
        self._last_recog_time = 0.0
        self._face_stable_count = 0

        # ── NPU proxy ─────────────────────────────────────────────────────────
        self.npu = None

    # ─────────────────────────────────────────────────────────────────────────
    def _on_realize(self, widget):
        self._load_models()
        if RUN_FACE_RECOG:
            self._start_recog_executor()
        self._build_pipeline()
        self.pipeline.set_state(Gst.State.PLAYING)
        print("[DMS] Pipeline started.")

    def _load_models(self):
        print("[DMS] Starting NPU worker…")
        self.npu = NPUProxy(
            face_model     = self.app.face_model,
            landmark_model = self.app.landmark_model,
            iris_model     = self.app.iris_model,
            smk_model      = self.app.smk_model,
        )
        if RUN_FACE_RECOG:
            self.face_db = FaceDB()
        self.inited = True
        print("[DMS] Ready.")

    def _start_recog_executor(self):
        print("[RECOG] Starting dlib worker (core 0)…")
        self._executor = cf.ProcessPoolExecutor(
            max_workers=1,
            initializer=_worker_init,
            initargs=(CACHE_FILE,),
        )
        # Warm up with realistic crop size so dlib JIT-compiles its kernels
        # before the first real face appears — otherwise first recognition
        # takes an extra ~1-2s for JIT overhead on ARM.
        dummy = np.zeros((_RECOG_CROP_SIZE, _RECOG_CROP_SIZE, 3), dtype=np.uint8)
        self._executor.submit(
            _worker_encode, dummy.tobytes(), dummy.shape, TOLERANCE
        ).result()
        print("[RECOG] Dlib worker ready ✅")

    def _build_pipeline(self):
        w   = self.app.frame_width
        h   = self.app.frame_height
        fps = self.app.framerate

        self.pipeline = Gst.Pipeline.new("dms-pipeline")

        src = Gst.ElementFactory.make("libcamerasrc", "libcamera")
        if not src:
            raise RuntimeError("libcamerasrc not available")

        caps_disp = Gst.Caps.from_string(
            f"video/x-raw,width={w},height={h},format=RGB16,framerate={fps}/1")
        caps_nn = Gst.Caps.from_string(
            f"video/x-raw,width={w},height={h},format=BGR,framerate={fps}/1")

        vrate_disp = Gst.ElementFactory.make("videorate",    "vrate_disp")
        queue_disp = Gst.ElementFactory.make("queue",        "queue_disp")
        queue_disp.set_property("leaky",            2)
        queue_disp.set_property("max-size-buffers", 1)

        self.cairooverlay = Gst.ElementFactory.make("cairooverlay", "overlay")
        if not self.cairooverlay:
            raise RuntimeError("cairooverlay not found")
        self.cairooverlay.connect("draw", self.draw)

        vconv_disp = Gst.ElementFactory.make("videoconvert", "vconv_disp")

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

        vrate_nn = Gst.ElementFactory.make("videorate", "vrate_nn")
        queue_nn = Gst.ElementFactory.make("queue",     "queue_nn")
        queue_nn.set_property("leaky",            2)
        queue_nn.set_property("max-size-buffers", 1)

        self.appsink = Gst.ElementFactory.make("appsink", "appsink")
        self.appsink.set_property("emit-signals",       True)
        self.appsink.set_property("sync",               False)
        self.appsink.set_property("max-buffers",        1)
        self.appsink.set_property("drop",               True)
        self.appsink.set_property("enable-last-sample", False)
        self.appsink.connect("new-sample", self.inference)

        for el in [src,
                   vrate_disp, queue_disp, self.cairooverlay,
                   vconv_disp, fps_sink,
                   vrate_nn,   queue_nn,   self.appsink]:
            self.pipeline.add(el)

        src.link(vrate_disp)
        vrate_disp.link(queue_disp)
        queue_disp.link_filtered(self.cairooverlay, caps_disp)
        self.cairooverlay.link(vconv_disp)
        vconv_disp.link(fps_sink)

        static_src = src.get_static_pad("src")
        static_src.set_property("stream-role", 3)

        pad_tmpl   = src.get_pad_template("src_%u")
        src_pad_nn = src.request_pad(pad_tmpl, None, None)
        src_pad_nn.set_property("stream-role", 1)
        src_pad_nn.link(vrate_nn.get_static_pad("sink"))
        vrate_nn.link(queue_nn)
        queue_nn.link_filtered(self.appsink, caps_nn)

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error",         self._on_error)
        bus.connect("message::eos",           self._on_eos)
        bus.connect("message::state-changed", self._on_state_changed)

    # ── Bus callbacks ─────────────────────────────────────────────────────────
    def _on_fps(self, sink, fps, droprate, avgfps):
        pass   # fpsdisplaysink signal — we compute FPS ourselves

    def _on_eos(self, bus, msg):
        print("[GST] EOS")

    def _on_error(self, bus, msg):
        err, dbg = msg.parse_error()
        print(f"[GST] Error: {err}\n      {dbg}")

    def _on_state_changed(self, bus, msg):
        old, new, _ = msg.parse_state_changed()
        if old == Gst.State.NULL and new == Gst.State.READY:
            Gst.debug_bin_to_dot_file(
                self.pipeline, Gst.DebugGraphDetails.ALL, "dms_pipeline")

    def stop_camera(self):
        self.pipeline.set_state(Gst.State.NULL)
        if self.npu:
            self.npu.shutdown()
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

    # ═════════════════════════════════════════════════════════════════════════
    #  FACE RECOGNITION
    # ═════════════════════════════════════════════════════════════════════════
    def _submit_recog_job(self, crop_bgr: np.ndarray) -> None:
        if self._active_job or crop_bgr is None or crop_bgr.size == 0:
            return
        self._active_job        = True
        self.face_track.pending = True
        future = self._executor.submit(
            _worker_encode, crop_bgr.tobytes(), crop_bgr.shape, TOLERANCE)
        future.add_done_callback(self._on_recog_result)

    def _on_recog_result(self, future) -> None:
        self._active_job        = False
        self.face_track.pending = False

        if future.cancelled():
            return
        try:
            name, score, top3, t_ms = future.result()
        except Exception as exc:
            print(f"[RECOG] Embedding failed: {exc}")
            return

        print(f"[RECOG] Result [{t_ms:.0f}ms]: {name}")

        if self.face_track.state == IDENTIFIED:
            if name != self.face_track.name:
                print(f"[RECOG] Changed {self.face_track.name}→{name} — reset")
                self.face_track.reset()
            else:
                print(f"[RECOG] Re-verified: {name}")
        else:
            if name != "Unknown":
                self.face_track.state = IDENTIFIED
                self.face_track.name  = name
            else:
                self.face_track.retry_count += 1
                if self.face_track.retry_count < RETRY_COUNT:
                    self.face_track.state = DETECTING
                else:
                    self.face_track.state = UNKNOWN

    # ═════════════════════════════════════════════════════════════════════════
    #  BLUR / MOTION DETECTION
    # ═════════════════════════════════════════════════════════════════════════
    # @staticmethod
    # def _is_blurry(frame: np.ndarray) -> bool:
    #     """Return True if frame is too blurry/motion-affected to be useful.
    #     Uses Laplacian variance — fast single-channel op on a downscaled patch."""
    #     # Use centre crop (160x120) — faster than full frame, face is usually central
    #     h, w  = frame.shape[:2]
    #     cx, cy = w // 2, h // 2
    #     patch  = frame[cy - 60:cy + 60, cx - 80:cx + 80]
    #     gray   = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    #     return cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_SKIP_THRESHOLD

    # ═════════════════════════════════════════════════════════════════════════
    #  INFERENCE  — one NPU call per frame, all results in one round-trip
    # ═════════════════════════════════════════════════════════════════════════
    def inference(self, _data) -> Gst.FlowReturn:
        sample = self.appsink.emit("pull-sample")
        if sample is None or not self.inited:
            return Gst.FlowReturn.OK

        # ── Extract frame ──────────────────────────────────────────────────
        buf    = sample.get_buffer()
        caps   = sample.get_caps()
        height = caps.get_structure(0).get_value("height")
        width  = caps.get_structure(0).get_value("width")
        raw    = buf.extract_dup(0, buf.get_size())
        sample = None
        buf    = None

        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            (height, width, 3)).copy()
        raw   = None

        self.frame_count += 1
        self.fps_counter += 1

        now     = time.time()
        elapsed = now - self.fps_start_time
        if elapsed >= 1.0:
            self.current_fps    = self.fps_counter / elapsed
            self.fps_counter    = 0
            self.fps_start_time = now

        # ── Check camera visibility status (Black or Blur) ────────────────
        status = check_camera_status(frame)
        self.camera_status = status

        if status != "OK":
            # Clear face coordinates and landmarks
            self.face_cords     = []
            self.smk_call_cords = []
            self.marks          = []
            self._last_boxes    = None
            
            # Reset active alert states to prevent false triggers during black/blur
            self.distracted     = False
            self.drowsy         = False
            self.yawn           = False
            self.smoking        = False
            self.phone          = False
            self.head_down      = False
            
            # Repaint GUI frame with camera status warning
            self.queue_draw()
            return Gst.FlowReturn.OK

        # ── Decide which sub-tasks run this frame ──────────────────────────
        # Fix 1: Face detection throttled to every 2nd frame.
        # Landmark / eye every 3rd frame.  Smk/call every 3rd frame (offset).
        run_face     = (self.frame_count % 1 == 0) and self.run_face_detection
        run_landmark = (self.frame_count % 3 == 0) and self.run_face_landmarks
        run_eye      = (self.frame_count % 3 == 0) and self.run_eye_detection
        run_smk      = (self.frame_count % 3 == 1) and self.run_yolo_smk_call_detection

        # ── Single NPU request — worker does everything in one shot ────────
        req = {
            "frame":        frame.tobytes(),
            "shape":        frame.shape,
            "run_face":     run_face,
            "run_landmark": run_landmark,
            "run_eye":      run_eye,
            "run_mouth":    run_landmark,
            "run_smk":      run_smk,
        }
        result = self.npu.infer(req)

        # ── Carry over last-known alert states (fix: they must not reset to
        #    False on every non-landmark frame as local vars would do) ───────
        distracted       = self.distracted
        yawn_detected    = self.yawn
        head_down_final  = self.head_down
        sleep_detected   = self.drowsy
        phone_detected   = self.last_phone_status
        smoking_detected = self.last_smoke_status

        face_cords     = []
        smk_call_cords = []
        mark_group     = []

        # ── Unpack boxes ───────────────────────────────────────────────────
        # Fix 1: If face detection skipped this frame, reuse last valid boxes
        if run_face:
            boxes = result.get("boxes")
            if boxes is not None and len(boxes) > 0:
                self._last_boxes = boxes   # save for next non-detect frame
        else:
            boxes = self._last_boxes       # carry over last known boxes

        if boxes is not None and len(boxes) > 0:
            self._no_face_frames = 0

            # ── Smoking / Calling (run on smk frames) ─────────────────────
            if run_smk and "smk" in result:
                smk_call_cords, phone_detected, smoking_detected = \
                    self._parse_smk(result["smk"], False, False)
                self.last_smk_call_cords = smk_call_cords
                self.last_phone_status   = phone_detected
                self.last_smoke_status   = smoking_detected
            else:
                smk_call_cords   = self.last_smk_call_cords
                phone_detected   = self.last_phone_status
                smoking_detected = self.last_smoke_status

            # ── Pixel coords for face rectangle drawing ────────────────────
            # (boxes are raw normalised from detector; scale to pixels here)
            px = boxes.copy()
            px[:, [0, 2]] *= width
            px[:, [1, 3]] *= height

            # ── Save tight boxes for face recognition ──────────────────────
            if RUN_FACE_RECOG:
                tight_px = px.copy().astype(np.int32)
                tight_px[:, 0] = np.clip(tight_px[:, 0], 0, width)
                tight_px[:, 1] = np.clip(tight_px[:, 1], 0, height)
                tight_px[:, 2] = np.clip(tight_px[:, 2], 0, width)
                tight_px[:, 3] = np.clip(tight_px[:, 3], 0, height)

            # ── Square + clip for drawing the face box ─────────────────────
            sq = self.transform_to_square(px, scale=1.26, offset=(0, 0))
            sq, _ = self.clip_boxes(sq, (0, 0, width, height))
            sq = sq.astype(np.int32)

            # ── Select centered face ───────────────────────────────────────
            face_in_center     = 0
            distance_to_center = math.hypot(width / 2, height / 2)
            for i in range(len(sq)):
                x1, y1, x2, y2 = sq[i]
                d = math.hypot((x2 + x1 - width) / 2,
                               (y2 + y1 - height) / 2)
                if d < distance_to_center:
                    face_in_center     = i
                    distance_to_center = d

            x1, y1, x2, y2 = sq[face_in_center]
            face_cords.append([x1, y1, x2, y2])

            # ── Landmark results ───────────────────────────────────────────
            if run_landmark and "face_marks" in result:
                face_marks = result["face_marks"]
                mark_group.append(face_marks)
                self.last_face_marks = face_marks

                mouth_ratio      = result.get("mouth_ratio", 0.0)
                YAWN_STATUS[:-1] = YAWN_STATUS[1:]
                YAWN_STATUS[-1]  = 1 if mouth_ratio > MOUTH_THRESHOLD else 0
                yawn_detected    = np.mean(YAWN_STATUS) > 0.5

                head_down_ratio  = result.get("head_down_ratio", 1.0)
                status           = 1 if head_down_ratio > HEAD_DOWN_THRESHOLD else 0
                HEAD_STATUS[:-1] = HEAD_STATUS[1:]
                HEAD_STATUS[-1]  = status
                head_down_final  = np.mean(HEAD_STATUS) > HEAD_DOWN_FRAME_AVG

                # ── TUNING PRINT — remove once thresholds are set ─────────
                # print(
                #     f"[HEAD] ratio={head_down_ratio:.3f}(th={HEAD_DOWN_THRESHOLD}) | "
                #     f"win={float(np.mean(HEAD_STATUS)):.2f} | "
                #     f"HEAD_DOWN={head_down_final}",
                #     flush=True
                # )

                mouth_face_ratio = result.get("mouth_face_ratio", 1.0)
                distracted       = not (FACING_LEFT_THRESHOLD
                                        <= mouth_face_ratio
                                        <= FACING_RIGHT_THRESHOLD)

            elif len(self.last_face_marks) > 0:
                # Non-landmark frame — keep last marks for drawing
                mark_group.append(np.array(self.last_face_marks))

            # ── Eye results ────────────────────────────────────────────────
            if run_eye and "eyes" in result:
                eyes = result["eyes"]
                for eye_id, status_arr, threshold in [
                    (0, LEFT_EYE_STATUS,  LEFT_EYE_THRESHOLD),
                    (1, RIGHT_EYE_STATUS, RIGHT_EYE_THRESHOLD),
                ]:
                    if eye_id in eyes:
                        ratio           = eyes[eye_id]["ratio"]
                        status_arr[:-1] = status_arr[1:]
                        status_arr[-1]  = 1 if ratio > threshold else 0
                        iris = eyes[eye_id].get("iris")
                        if iris is not None:
                            mark_group.append(iris)

                self.left_eye_closed  = np.mean(LEFT_EYE_STATUS)  < 0.1
                self.right_eye_closed = np.mean(RIGHT_EYE_STATUS) < 0.1
                self.both_eyes_closed = (self.left_eye_closed
                                         and self.right_eye_closed)
                sleep_detected        = self.both_eyes_closed

                # ── TUNING PRINT — remove once thresholds are set ─────────
                #l_ratio = eyes[0]["ratio"] if 0 in eyes else -1.0
                #r_ratio = eyes[1]["ratio"] if 1 in eyes else -1.0
                #l_mean  = float(np.mean(LEFT_EYE_STATUS))
                #r_mean  = float(np.mean(RIGHT_EYE_STATUS))
                #print(
                #    f"[EYE] L_ratio={l_ratio:.3f}(th={LEFT_EYE_THRESHOLD}) "
                #    f"R_ratio={r_ratio:.3f}(th={RIGHT_EYE_THRESHOLD}) | "
                #    f"L_win={l_mean:.2f} R_win={r_mean:.2f} | "
                #    f"L_closed={self.left_eye_closed} "
                #    f"R_closed={self.right_eye_closed} "
                #    f"DROWSY={sleep_detected}",
                #    flush=True
                #)

            # ── Face recognition ───────────────────────────────────────────
            if RUN_FACE_RECOG and not self._active_job:
                rx1, ry1, rx2, ry2 = tight_px[face_in_center]
                do_recog = False
                if self.face_track.state != IDENTIFIED:
                    do_recog = (now - self._last_recog_time) >= RECOG_RETRY_INTERVAL
                else:
                    do_recog = (now - self._last_recog_time) >= RECOG_VERIFY_INTERVAL
                if do_recog:
                    self._face_stable_count += 1
                    if self._face_stable_count >= 3:
                        self._face_stable_count = 0
                        crop_bgr = frame[ry1:ry2, rx1:rx2].copy()
                        if crop_bgr.size > 0:
                            self._last_recog_time = now
                            self._submit_recog_job(crop_bgr)
                else:
                    self._face_stable_count = 0

        else:
            # ── No face detected ───────────────────────────────────────────
            self._last_boxes     = None   # clear — face is gone
            self._no_face_frames += 1
            if self._no_face_frames >= NO_FACE_RESET_AFTER:
                self.face_track.reset()
                self._no_face_frames     = 0
                self.last_face_marks     = []
                self.last_smk_call_cords = []
                self.last_phone_status   = False
                self.last_smoke_status   = False
                # Also reset alert states when face has been gone a while
                distracted       = False
                yawn_detected    = False
                head_down_final  = False
                sleep_detected   = False
                phone_detected   = False
                smoking_detected = False

        # ── Publish state (read by Cairo draw callback) ────────────────────
        self.marks          = mark_group
        self.face_cords     = face_cords
        self.smk_call_cords = smk_call_cords
        self.distracted     = distracted
        self.drowsy         = sleep_detected
        self.yawn           = yawn_detected
        self.smoking        = smoking_detected
        self.phone          = phone_detected
        self.head_down      = head_down_final

        # ── Safety score ───────────────────────────────────────────────────
        if distracted:       self.safe_value = min(self.safe_value + DISTRACT_PENALTY,  100.0)
        if sleep_detected:   self.safe_value = min(self.safe_value + SLEEP_PENALTY,     100.0)
        if yawn_detected:    self.safe_value = min(self.safe_value + YAWN_PENALTY,      100.0)
        if smoking_detected: self.safe_value = min(self.safe_value + SMK_PENALTY,       100.0)
        if phone_detected:   self.safe_value = min(self.safe_value + CALL_PENALTY,      100.0)
        if head_down_final:  self.safe_value = min(self.safe_value + HEAD_DOWN_PENALTY, 100.0)
        if not face_cords:   self.safe_value = min(self.safe_value + NO_FACE_PENALTY,   100.0)

        if (not distracted and not sleep_detected and not yawn_detected
                and not head_down_final and not smoking_detected
                and not phone_detected and face_cords):
            self.safe_value = max(self.safe_value + RESTORE_CREDIT, 0.0)

        if self.frame_count % 300 == 0:
            gc.collect()

        return Gst.FlowReturn.OK

    # ── Smk/call parser ───────────────────────────────────────────────────────
    def _parse_smk(self, smk_result, phone_detected: bool,
                   smoking_detected: bool) -> tuple:
        smk_call_cords = []
        phone_seen     = False
        cigrate_seen   = False

        if smk_result is not None and np.size(smk_result, 0) > 0:
            for i in range(np.size(smk_result, 0)):
                cls_id = int(smk_result[i][5])
                conf   = smk_result[i][4]
                if cls_id == 0 and conf > SMK_CALL_THRESHOLD:
                    phone_seen   = True
                if cls_id == 1 and conf > SMK_CALL_THRESHOLD:
                    cigrate_seen = True
                smk_call_cords.append([
                    int(smk_result[i][0]), int(smk_result[i][1]),
                    int(smk_result[i][2]), int(smk_result[i][3]),
                ])

            if phone_seen:
                self.call_count += 1
                phone_detected   = True
            else:
                self.call_count  = 0
                phone_detected   = False

            if cigrate_seen:
                self.smoke_count += 1
                smoking_detected  = True
            else:
                self.smoke_count  = 0
                smoking_detected  = False

        return smk_call_cords, phone_detected, smoking_detected

    # ── Box helpers (identical to original dms.py) ────────────────────────────
    def transform_to_square(self, boxes: np.ndarray, scale: float = 1.0,
                             offset: tuple = (0, 0)) -> np.ndarray:
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
             center_x + margin, center_y + margin), axis=1)

    def clip_boxes(self, boxes: np.ndarray,
                   margins: tuple) -> tuple:
        left, top, right, bottom = margins
        clip_mark = (boxes[:, 1] < top,  boxes[:, 0] < left,
                     boxes[:, 3] > bottom, boxes[:, 2] > right)
        boxes[:, 0] = np.maximum(boxes[:, 0], left)
        boxes[:, 1] = np.maximum(boxes[:, 1], top)
        boxes[:, 2] = np.minimum(boxes[:, 2], right)
        boxes[:, 3] = np.minimum(boxes[:, 3], bottom)
        return boxes, clip_mark

    # ═════════════════════════════════════════════════════════════════════════
    #  CAIRO DRAW
    # ═════════════════════════════════════════════════════════════════════════
    def draw(self, overlay, context, timestamp, duration):
        face_cords     = self.face_cords
        smk_call_cords = self.smk_call_cords
        marks          = self.marks

        context.select_font_face("Arial", cairo.FONT_SLANT_NORMAL,
                                  cairo.FONT_WEIGHT_BOLD)
        context.set_line_width(3)

        # ── Optional overlays ──────────────────────────────────────────────
        if smk_call_cords and DRAW_SMK_CALL_CORDS:
            context.set_source_rgb(1, 0.5, 0)
            for c in smk_call_cords:
                context.rectangle(c[0], c[1], c[2] - c[0], c[3] - c[1])
            context.stroke()

        if marks and DRAW_LANDMARKS:
            context.set_source_rgb(0, 1, 0)
            for m in marks:
                for pt in m:
                    px, py = int(pt[0]), int(pt[1])
                    context.arc(px, py, 1, 0, 1)
                    context.stroke()

        # ── Alert panel (bottom-left) ──────────────────────────────────────
        context.set_source_rgb(0, 0, 0)
        context.rectangle(_PANEL_X, _PANEL_Y, _PANEL_W, _PANEL_H)
        context.fill()

        labels = ["Distracted:", "Drowsy:", "Yawn:",
                  "Head Down:", "Smoking:", "Phone:"]
        for row, label in enumerate(labels):
            self.write_text(context, label,
                             _LABEL_X, _ROW_START_Y + row * _ROW_STEP)

        # ── FPS counter ────────────────────────────────────────────────────
        context.save()
        context.set_source_rgb(0, 0, 1)
        context.select_font_face("Sans", cairo.FONT_SLANT_NORMAL,
                                  cairo.FONT_WEIGHT_BOLD)
        context.set_font_size(_FPS_FONT)
        context.move_to(_FPS_X, _FPS_Y)
        context.show_text(f"FPS: {self.current_fps:.1f}")
        context.restore()

        # ── Camera Status (below FPS) ──────────────────────────────────────
        context.save()
        if self.camera_status == "OK":
            context.set_source_rgb(0, 1, 0)      # Green
        elif self.camera_status == "BLUR":
            context.set_source_rgb(1, 0.55, 0)   # Orange
        else:
            context.set_source_rgb(1, 0, 0)      # Red (BLACK)
        context.select_font_face("Sans", cairo.FONT_SLANT_NORMAL,
                                  cairo.FONT_WEIGHT_BOLD)
        context.set_font_size(int(_FPS_FONT * 0.8))
        context.move_to(_FPS_X, _FPS_Y + 25)
        context.show_text(f"Camera: {self.camera_status}")
        context.restore()

        # ── Alert statuses + face box ──────────────────────────────────────
        has_face = (isinstance(face_cords, list)
                    and len(face_cords) > 0
                    and len(face_cords[0]) == 4)

        if has_face:
            states = [self.distracted, self.drowsy, self.yawn,
                      self.head_down, self.smoking, self.phone]
            for row, state in enumerate(states):
                self.write_alert_status(
                    context, state, _STATUS_X, _ROW_START_Y + row * _ROW_STEP)

            r, g, b = self.face_track.label_color if RUN_FACE_RECOG else (0, 1, 0)
            context.set_source_rgb(r, g, b)
            context.set_line_width(3)
            x1, y1, x2, y2 = face_cords[0]
            context.rectangle(x1, y1, x2 - x1, y2 - y1)
            context.stroke()

            if RUN_FACE_RECOG:
                lbl = self.face_track.label
                context.set_font_size(16)
                te  = context.text_extents(lbl)
                pad = 4
                context.set_source_rgba(0, 0, 0, 0.70)
                context.rectangle(x1, y1 - te.height - pad * 2,
                                   te.width + pad * 2, te.height + pad * 2)
                context.fill()
                context.set_source_rgb(r, g, b)
                context.move_to(x1 + pad, y1 - pad)
                context.show_text(lbl)
        else:
            for row in range(6):
                self.write_alert_status(
                    context, None, _STATUS_X, _ROW_START_Y + row * _ROW_STEP)

        self.write_driver_status(context, face_cords)

    def write_alert_status(self, context, state, x: int, y: int) -> None:
        context.set_font_size(int(_FONT_SIZE))
        context.move_to(x, y)
        if state is None:
            context.set_source_rgb(1, 1, 1); context.show_text("N/A")
        elif state:
            context.set_source_rgb(1, 0, 0); context.show_text("Yes")
        else:
            context.set_source_rgb(0, 1, 0); context.show_text("No")

    def write_text(self, context, text: str, x: int, y: int) -> None:
        context.set_font_size(int(_FONT_SIZE))
        context.move_to(x, y)
        context.set_source_rgb(1, 1, 1)
        context.show_text(text)

    def write_driver_status(self, context, face_cords) -> None:
        context.set_source_rgb(0, 0, 0)
        context.rectangle(0, 0, FRAME_WIDTH, _STATUS_BAR_H)
        context.fill()

        self.write_text(context, "EDS-INDIA", _BRAND_X, _BRAND_Y)

        context.set_font_size(_STATUS_FONT)
        context.move_to(_LABEL_X, _STATUS_Y)

        r = min(self.safe_value / 50.0, 1.0)
        g = min(1.0, (100.0 - self.safe_value) / 50.0)
        context.set_source_rgb(r, g, 0)

        score = f"{self.safe_value:.2f}"
        if face_cords:
            if   self.safe_value < 33: context.show_text(f"Driver OK ({score}%)")
            elif self.safe_value < 66: context.show_text(f"Warning! ({score}%)")
            else:                      context.show_text(f"Danger! ({score}%)")
        else:
            context.show_text(f"Driver not found! ({score}%)")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════
class MainWindow(Gtk.Window):
    def __init__(self, app):
        Gtk.Window.__init__(self)
        self.app = app
        self.set_decorated(False)
        self.maximize()
        self.set_position(Gtk.WindowPosition.CENTER)
        self.connect("destroy",         Gtk.main_quit)
        self.connect("key-press-event", self._on_key)
        self.add(self.app.gst_widget)

    def _on_key(self, widget, event):
        if event.keyval == Gdk.KEY_Escape:
            Gtk.main_quit()


# ═══════════════════════════════════════════════════════════════════════════════
#  APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════
class Application:
    def __init__(self, args):
        self.frame_width  = args.frame_width
        self.frame_height = args.frame_height
        self.framerate    = args.framerate
        # Store model paths so GstWidget can read them via self.app
        self.face_model     = args.face_model
        self.landmark_model = args.landmark_model
        self.iris_model     = args.iris_model
        self.smk_model      = args.smk_model
        self.gst_widget   = GstWidget(self)
        self.main_window  = MainWindow(self)

    def run(self):
        self.main_window.show_all()
        self.main_window.connect("delete-event", Gtk.main_quit)
        Gtk.main()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def run_pc_opencv_loop(args):
    import cv2
    face_model     = translate_model_path(args.face_model)
    landmark_model = translate_model_path(args.landmark_model)
    iris_model     = translate_model_path(args.iris_model)
    smk_model      = translate_model_path(args.smk_model)

    print("[DMS] Starting NPU worker proxy on PC...")
    npu = NPUProxy(
        face_model     = face_model,
        landmark_model = landmark_model,
        iris_model     = iris_model,
        smk_model      = smk_model
    )

    video_source = args.video if getattr(args, "video", "") else 0
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        source_name = f"video file '{video_source}'" if video_source != 0 else "webcam"
        print(f"[DMS] ERROR: OpenCV could not open {source_name}.")
        return

    # Frame state variables
    frame_count = 0
    fps_start_time = time.time()
    current_fps = 0.0

    safe_value = 0.0
    distracted = False
    drowsy = False
    yawn = False
    smoking = False
    phone = False
    head_down = False
    both_eyes_closed = False

    last_face_marks = []
    last_smk_call_cords = []
    last_phone_status = False
    last_smoke_status = False
    last_boxes = None

    def _transform_to_square(boxes, scale=1.0, offset=(0, 0)):
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
             center_x + margin, center_y + margin), axis=1)

    def _clip_boxes(boxes, margins):
        left, top, right, bottom = margins
        boxes[:, 0] = np.maximum(boxes[:, 0], left)
        boxes[:, 1] = np.maximum(boxes[:, 1], top)
        boxes[:, 2] = np.minimum(boxes[:, 2], right)
        boxes[:, 3] = np.minimum(boxes[:, 3], bottom)
        return boxes

    print("\n" + "="*60)
    print("PC DMS Runner (with Subprocess NPU Proxy) is active.")
    print("Press 'q' or 'ESC' to exit.")
    print("="*60 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            is_webcam = False
            try:
                int(video_source)
                is_webcam = True
            except ValueError:
                is_webcam = False

            if not is_webcam and video_source != "":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                time.sleep(0.01)
                continue

        is_webcam = False
        try:
            int(video_source)
            is_webcam = True
        except ValueError:
            is_webcam = False

        if is_webcam:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        frame_count += 1

        # Calculate FPS
        now = time.time()
        elapsed = now - fps_start_time
        if elapsed >= 1.0:
            current_fps = frame_count / elapsed
            frame_count = 0
            fps_start_time = now

        # Check camera visibility status (Black or Blur)
        camera_status = check_camera_status(frame)

        if camera_status != "OK":
            # Clear coordinates and landmarks
            face_cords     = []
            mark_group     = []
            smk_call_cords = []
            last_boxes     = None
            
            # Reset active alert states to prevent false alerts during black/blur
            distracted       = False
            drowsy           = False
            yawn             = False
            smoking_detected = False
            phone_detected   = False
            head_down        = False
            both_eyes_closed = False
        else:
            run_face     = True
            run_landmark = (frame_count % 3 == 0)
            run_eye      = (frame_count % 3 == 0)
            run_smk      = (frame_count % 3 == 1)

            req = {
                "frame":        frame.tobytes(),
                "shape":        frame.shape,
                "run_face":     run_face,
                "run_landmark": run_landmark,
                "run_eye":      run_eye,
                "run_mouth":    run_landmark,
                "run_smk":      run_smk,
            }
            
            result = npu.infer(req)

            # Carry over last states
            phone_detected    = last_phone_status
            smoking_detected  = last_smoke_status
            smk_call_cords    = last_smk_call_cords

            if run_face:
                boxes = result.get("boxes")
                if boxes is not None and len(boxes) > 0:
                    last_boxes = boxes
            else:
                boxes = last_boxes

            face_cords = []
            mark_group = []

            if boxes is not None and len(boxes) > 0:
                # Parse smoking/calling
                if run_smk and "smk" in result:
                    smk_res = result["smk"]
                    phone_seen = False
                    cigrate_seen = False
                    smk_call_cords = []
                    if smk_res is not None and len(smk_res) > 0:
                        for i in range(len(smk_res)):
                            cls_id = int(smk_res[i][5])
                            conf = smk_res[i][4]
                            if cls_id == 0 and conf > SMK_CALL_THRESHOLD:
                                phone_seen = True
                            if cls_id == 1 and conf > SMK_CALL_THRESHOLD:
                                cigrate_seen = True
                            smk_call_cords.append([
                                int(smk_res[i][0]), int(smk_res[i][1]),
                                int(smk_res[i][2]), int(smk_res[i][3])
                            ])
                    phone_detected = phone_seen
                    smoking_detected = cigrate_seen
                    last_phone_status = phone_detected
                    last_smoke_status = smoking_detected
                    last_smk_call_cords = smk_call_cords
                else:
                    phone_detected = last_phone_status
                    smoking_detected = last_smoke_status
                    smk_call_cords = last_smk_call_cords

                # Convert face boxes
                px = boxes.copy()
                px[:, [0, 2]] *= w
                px[:, [1, 3]] *= h

                # Square + clip
                sq = _transform_to_square(px, scale=1.26, offset=(0, 0))
                sq = _clip_boxes(sq, (0, 0, w, h)).astype(np.int32)

                # Center face
                face_in_center = 0
                distance_to_center = math.hypot(w / 2, h / 2)
                for i in range(len(sq)):
                    x1, y1, x2, y2 = sq[i]
                    d = math.hypot((x2 + x1 - w) / 2, (y2 + y1 - h) / 2)
                    if d < distance_to_center:
                        face_in_center = i
                        distance_to_center = d

                x1, y1, x2, y2 = sq[face_in_center]
                face_cords.append([x1, y1, x2, y2])

                # Landmarks
                if run_landmark and "face_marks" in result:
                    face_marks = result["face_marks"]
                    mark_group.append(face_marks)
                    last_face_marks = face_marks
                    
                    # Ratios
                    mouth_ratio = result.get("mouth_ratio", 0.0)
                    YAWN_STATUS[:-1] = YAWN_STATUS[1:]
                    YAWN_STATUS[-1]  = 1 if mouth_ratio > MOUTH_THRESHOLD else 0
                    yawn = np.mean(YAWN_STATUS) > 0.5

                    head_down_ratio = result.get("head_down_ratio", 1.0)
                    status = 1 if head_down_ratio > HEAD_DOWN_THRESHOLD else 0
                    HEAD_STATUS[:-1] = HEAD_STATUS[1:]
                    HEAD_STATUS[-1]  = status
                    head_down = np.mean(HEAD_STATUS) > HEAD_DOWN_FRAME_AVG

                    mouth_face_ratio = result.get("mouth_face_ratio", 1.0)
                    distracted = not (FACING_LEFT_THRESHOLD <= mouth_face_ratio <= FACING_RIGHT_THRESHOLD)
                elif len(last_face_marks) > 0:
                    mark_group.append(np.array(last_face_marks))

                # Eyes
                if run_eye and "eyes" in result:
                    eyes = result["eyes"]
                    for eye_id, status_arr, threshold in [
                        (0, LEFT_EYE_STATUS,  LEFT_EYE_THRESHOLD),
                        (1, RIGHT_EYE_STATUS, RIGHT_EYE_THRESHOLD),
                    ]:
                        if eye_id in eyes:
                            ratio = eyes[eye_id]["ratio"]
                            status_arr[:-1] = status_arr[1:]
                            status_arr[-1]  = 1 if ratio > threshold else 0
                            iris = eyes[eye_id].get("iris")
                            if iris is not None:
                                mark_group.append(iris)

                    left_eye_closed  = np.mean(LEFT_EYE_STATUS)  < 0.1
                    right_eye_closed = np.mean(RIGHT_EYE_STATUS) < 0.1
                    both_eyes_closed = left_eye_closed and right_eye_closed
                    drowsy = both_eyes_closed
                else:
                    drowsy = both_eyes_closed
            else:
                # Face lost — reset tracking but preserve alert states until face returns
                last_boxes = None
                last_face_marks = []
                last_smk_call_cords = []
                last_phone_status = False
                last_smoke_status = False
                phone_detected = False
                smoking_detected = False
                both_eyes_closed = False
                drowsy = False

        # Scoring
        if distracted:       safe_value = min(safe_value + DISTRACT_PENALTY,  100.0)
        if drowsy:           safe_value = min(safe_value + SLEEP_PENALTY,     100.0)
        if yawn:             safe_value = min(safe_value + YAWN_PENALTY,      100.0)
        if smoking_detected: safe_value = min(safe_value + SMK_PENALTY,       100.0)
        if phone_detected:   safe_value = min(safe_value + CALL_PENALTY,      100.0)
        if head_down:        safe_value = min(safe_value + HEAD_DOWN_PENALTY, 100.0)
        if not face_cords:   safe_value = min(safe_value + NO_FACE_PENALTY,   100.0)

        if (not distracted and not drowsy and not yawn
                and not head_down and not smoking_detected
                and not phone_detected and face_cords):
            safe_value = max(safe_value + RESTORE_CREDIT, 0.0)

        # Render OpenCV annotations
        cv2.rectangle(frame, (0, 0, w, 40), (0, 0, 0), -1)
        r_c = min(safe_value / 50.0, 1.0)
        g_c = min(1.0, (100.0 - safe_value) / 50.0)
        color = (0, int(g_c * 255), int(r_c * 255))

        status_text = "Driver OK" if safe_value < 33 else "Warning!" if safe_value < 66 else "DANGER!"
        if not face_cords:
            status_text = "Driver not found!"

        cv2.putText(frame, f"{status_text} ({safe_value:.1f}%)", (10, 26), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, "EDS-INDIA PC PROXY RUNNER", (w - 280, 26), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Panel
        cv2.rectangle(frame, (10, h - 170), (180, h - 10), (0, 0, 0), -1)
        states = [
            ("Distracted", distracted),
            ("Drowsy",     drowsy),
            ("Yawn",       yawn),
            ("Head Down",  head_down),
            ("Smoking",    smoking_detected),
            ("Phone",      phone_detected),
        ]
        for idx, (label, val) in enumerate(states):
            val_text = "Yes" if val else "No"
            val_color = (0, 0, 255) if val else (0, 255, 0)
            y_pos = h - 145 + (idx * 22)
            cv2.putText(frame, f"{label}:", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(frame, val_text, (120, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, val_color, 2)

        # FPS
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Camera Status
        status_color = (0, 255, 0) if camera_status == "OK" else (0, 140, 255) if camera_status == "BLUR" else (0, 0, 255)
        cv2.putText(frame, f"Camera Status: {camera_status}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

        # Bounding box
        if face_cords:
            x1, y1, x2, y2 = face_cords[0]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Landmarks
        if DRAW_LANDMARKS:
            for group in mark_group:
                for pt in group:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, (0, 255, 255), -1)

        # YOLO boxes
        if DRAW_SMK_CALL_CORDS:
            for c in smk_call_cords:
                cv2.rectangle(frame, (c[0], c[1]), (c[2], c[3]), (0, 165, 255), 2)

        cv2.imshow("DMS PC Proxy Runner (TFLite)", frame)
        # Delay dynamically based on video frame rate (prevent fast-forwarding on file inputs)
        delay_ms = 1
        is_webcam = False
        try:
            int(video_source)
            is_webcam = True
        except ValueError:
            is_webcam = False

        if not is_webcam and video_source != "":
            fps_val = cap.get(cv2.CAP_PROP_FPS)
            if fps_val > 0:
                delay_ms = max(1, int(1000 / fps_val))
            else:
                delay_ms = int(1000 / args.framerate)

        key = cv2.waitKey(delay_ms) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    npu.shutdown()
    print("[DMS] Exited safely.")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    _NB   = "/home/root/Custom-DMS/dms_nb_models"
    _TF   = "/home/root/Custom-DMS/dms_tflite_models"

    parser = argparse.ArgumentParser(
        description="STM32MP2x DMS — CSI camera. Pass .nb or .tflite models freely.")
    parser.add_argument("--frame_width",     type=int, default=640)
    parser.add_argument("--frame_height",    type=int, default=480)
    parser.add_argument("--framerate",       type=int, default=30)
    parser.add_argument("--video",           default="",
                        help="Path to input video file (e.g. cameratest1.mp4). If omitted, uses live webcam.")
    parser.add_argument("--rebuild",         action="store_true",
                        help="Force rebuild of face encoding cache")
    parser.add_argument("--face-model",      default=f"{_NB}/face_detection_ptq.nb",
                        help="Face detection model (.nb or .tflite)")
    parser.add_argument("--landmark-model",  default=f"{_NB}/face_landmark_ptq.nb",
                        help="Face landmark model (.nb or .tflite)")
    parser.add_argument("--iris-model",      default=f"{_NB}/iris_landmark_ptq.nb",
                        help="Iris/eye model (.nb or .tflite)")
    parser.add_argument("--smk-model",       default=f"{_NB}/yolov8n_smk_call_custom.nb",
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
            run_pc_opencv_loop(args)
    except Exception as exc:
        print(f"[DMS] Fatal: {exc}")
        import traceback; traceback.print_exc()

    print("[DMS] Exited cleanly.")
    os._exit(0)