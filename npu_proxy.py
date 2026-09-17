#!/usr/bin/env python3
"""
npu_proxy.py  —  Manages the NPU worker subprocess.

Restarts the worker every RESTART_CALLS inferences to free driver-level
memory leaked by libstai_mpu_ovx.so (confirmed ~47 MB per 500 calls,
unfixable from Python — driver bug).

The restart is transparent to callers: infer() always returns a dict.
An empty dict is returned on the restart frame (same as a dropped frame).
"""
import sys
import os
import subprocess
import struct
import pickle

# Check stai_mpu availability
try:
    from stai_mpu import stai_mpu_network
    HAS_STAI = True
except ImportError:
    HAS_STAI = False

cur_dir = os.path.dirname(os.path.abspath(__file__))
WORKER_SCRIPT = os.path.join(cur_dir, "npu_worker.py")

# Restart every N inferences.
# At ~15 inference-frames/sec the worker leaks ~47 MB per 500 calls.
# 5000 calls ≈ 5 min runtime before restart; tune lower if needed.
RESTART_CALLS = 1000


class NPUProxy:

    def __init__(self, face_model=None, landmark_model=None,
                 iris_model=None, smk_model=None) -> None:
        
        # Helper to translate path based on platform
        def translate_model_path(p):
            if not p: return p
            if not HAS_STAI:
                p = p.replace("/home/root/Custom-DMS", "")
                p = p.replace("\\home\\root\\Custom-DMS", "")
                p = p.lstrip("/\\")
                p = os.path.join(cur_dir, p)
            p = os.path.normpath(p)
            if not HAS_STAI and p.endswith(".nb"):
                p = p.replace(".nb", ".tflite").replace("dms_nb_models", "dms_tflite_models")
                p = p.replace("yolov8n_smk_call_custom", "yolov8n_smk_call")
            return p

        # Resolve directories dynamically
        _NB = os.path.join(cur_dir, "dms_nb_models")
        _TF = os.path.join(cur_dir, "dms_tflite_models")
        _DIR = _NB if HAS_STAI else _TF
        _EXT = "nb" if HAS_STAI else "tflite"
        _SMK = "yolov8n_smk_call_custom.nb" if HAS_STAI else "yolov8n_smk_call.tflite"

        self._face_model     = translate_model_path(face_model)     or os.path.join(_DIR, f"face_detection_ptq.{_EXT}")
        self._landmark_model = translate_model_path(landmark_model) or os.path.join(_DIR, f"face_landmark_ptq.{_EXT}")
        self._iris_model     = translate_model_path(iris_model)     or os.path.join(_DIR, f"iris_landmark_ptq.{_EXT}")
        self._smk_model      = translate_model_path(smk_model)      or os.path.join(_DIR, _SMK)
        
        self._proc  = None
        self._calls = 0
        self._spawn()

    # ── Spawn / restart ───────────────────────────────────────────────────────
    def _spawn(self) -> None:
        # Cleanly terminate old worker if still alive
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._send_raw(b"QUIT")
            except Exception:
                pass
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()

        self._proc = subprocess.Popen(
            [
                sys.executable, "-u", WORKER_SCRIPT,
                "--face-model",     self._face_model,
                "--landmark-model", self._landmark_model,
                "--iris-model",     self._iris_model,
                "--smk-model",      self._smk_model,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,   # worker prints visible in terminal
            bufsize=0,           # unbuffered pipe — critical for latency
        )

        # Block until worker signals it is ready
        msg = self._recv_raw()
        if msg != b"READY":
            raise RuntimeError(
                f"[NPU] Worker failed to start, got: {msg!r}")

        self._calls = 0
        print(f"[NPU] Worker ready (PID {self._proc.pid})", flush=True)

    # ── Public API ────────────────────────────────────────────────────────────
    def infer(self, request: dict) -> dict:
        """Send a request dict, return a result dict (empty on error/restart)."""
        # Restart if call limit hit or worker died unexpectedly
        if self._calls >= RESTART_CALLS or self._proc.poll() is not None:
            print(f"[NPU] Restarting worker after {self._calls} calls…",
                  flush=True)
            self._spawn()
            # Skip this frame so the caller gets a clean empty result
            # rather than stale data from the previous worker lifetime.
            return {}

        try:
            self._send_raw(pickle.dumps(request))
            raw = self._recv_raw()
            self._calls += 1

            if raw is None:
                print("[NPU] Worker closed pipe — restarting", flush=True)
                self._spawn()
                return {}

            result = pickle.loads(raw)

            if "error" in result:
                print(f"[NPU] Worker error: {result['error']}", flush=True)
                if result.get("tb"):
                    print(result["tb"], flush=True)
                # Don't restart on a single inference error — may be transient
                return {}

            return result

        except Exception as exc:
            print(f"[NPU] Comm error: {exc} — restarting", flush=True)
            self._spawn()
            return {}

    def shutdown(self) -> None:
        """Cleanly shut down the worker process."""
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._send_raw(b"QUIT")
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()

    # ── Low-level pipe I/O ────────────────────────────────────────────────────
    def _send_raw(self, data: bytes) -> None:
        self._proc.stdin.write(struct.pack(">I", len(data)))
        self._proc.stdin.write(data)
        self._proc.stdin.flush()

    def _recv_raw(self) -> bytes | None:
        hdr = self._proc.stdout.read(4)
        if len(hdr) < 4:
            return None
        n = struct.unpack(">I", hdr)[0]
        return self._proc.stdout.read(n)