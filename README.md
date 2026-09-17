# Custom DMS — Driver Monitoring System

Custom DMS watches a driver through a single camera and flags drowsiness, yawning, distraction, head-down posture, phone use and smoking in real time. The alerts feed a rolling risk score that is drawn on the live video.

The same code runs in two places:

- **STM32MP2 board**: `.nb` models on the NPU through ST's `stai_mpu` runtime, displayed with GStreamer and GTK.
- **Windows laptop**: `.tflite` models on the CPU through TensorFlow Lite, displayed in an OpenCV window from a webcam or a video file.

Every model wrapper picks its backend from the file extension (`.nb` → NPU, `.tflite` → CPU), so the detection logic is shared between the two.

## Contents

- [What it detects](#what-it-detects)
- [Which script to run](#which-script-to-run)
- [Quick start on Windows](#quick-start-on-windows)
- [Running on the STM32MP2 board](#running-on-the-stm32mp2-board)
- [Command-line options](#command-line-options)
- [How it works](#how-it-works)
- [Thresholds and tuning](#thresholds-and-tuning)
- [Worker-process runner](#worker-process-runner)
- [Driver identification](#driver-identification)
- [Models](#models)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Known issues](#known-issues)

---

## What it detects

| Alert | How it is decided |
|---|---|
| **Drowsy** | Both eyes read as closed in every recent eye sample. Eye openness is eye height ÷ eye width, from the iris-landmark model. |
| **Yawn** | Mouth height ÷ mouth width goes above a threshold. |
| **Distracted** | The head is turned: the distance from the upper lip to the left cheek ÷ the distance to the right cheek falls outside 0.5–2.0. |
| **Head down** | The nose sits low in the face (measured between forehead and chin) for most of a short window. |
| **Phone** | The phone/smoking detector reports class 0 (phone). |
| **Smoking** | The phone/smoking detector reports class 1 (smoke). |
| **Driver not found** | No face in the frame. |

### Risk score

The score starts at 0, is updated on every processed frame, and always stays between 0 and 100.

| Condition in this frame | Score change |
|---|---|
| Distracted | +2 |
| Drowsy | +5 |
| Yawn | +7 |
| Smoking | +2 |
| Phone | +2 |
| Head down | +5 |
| No face | +0.7 |
| Face found and no alert active | −5 |

| Score | Status shown |
|---|---|
| Under 33 | Driver OK |
| 33 to under 66 | Warning! |
| 66 and above | Danger! (`DANGER!` in the OpenCV windows) |
| Any score, no face | Driver not found! |

Several alerts can add up in the same frame. Once the driver is back to normal, the score falls by 5 per frame.

---

## Which script to run

| Script | Runs on | Camera | Where inference runs | Use it for |
|---|---|---|---|---|
| `pc_dms_runner.py` | Windows | Webcam 0 | Main process, TFLite | Trying the system on a laptop |
| `dms-npu-restart.py` | Board, with a Windows fallback | Board: CSI camera (`libcamerasrc`). Windows: webcam 0, or a video file with `--video` | Child process `npu_worker.py`, restarted every 1000 calls | Long runs on the board, and testing with a video file on Windows. Adds a black/blurry camera check, and driver identification on the board |
| `STM32_Dms-csi.py` | Board, with a Windows fallback | Board: CSI camera (`libcamerasrc`). Windows: webcam 0 | Main process | CSI camera without the worker process |
| `STM32_Dms.py` | Board, with a Windows fallback | Board: USB camera (`/dev/video7`). Windows: the webcam numbered by the digits in `--device`, or webcam 0 if that fails | Main process | USB camera on the board |

When they start, the board scripts look for their display stack: GTK 3 and GStreamer through PyGObject for `dms-npu-restart.py` and `STM32_Dms-csi.py`, and GStreamer for `STM32_Dms.py`. A normal Windows Python install has neither, and has no `stai_mpu`, so on Windows these scripts open an OpenCV window and load the `.tflite` versions of the `.nb` models. `STM32_Dms-csi.py` does this by running the OpenCV loop from `STM32_Dms.py`.

---

## Quick start on Windows

### Requirements

- Python 3.10 or newer, in a version that TensorFlow supports
- `tensorflow`, `opencv-python`, `numpy`
- A webcam or a video file

Known-good setup: Windows 11, Python 3.12.4, TensorFlow 2.19.0, OpenCV 4.13.0, NumPy 2.1.3.

```powershell
pip install tensorflow opencv-python numpy
```

- Install `opencv-python`, not `opencv-python-headless`. The headless build cannot open windows.
- You don't need `stai_mpu`, which only exists on the board, or `tflite-runtime`, which isn't available for Windows. The code falls back to the TFLite interpreter built into TensorFlow.

### Run with the webcam

```powershell
cd path\to\Custom-DMS
python pc_dms_runner.py
```

Start it **from inside the `Custom-DMS` folder**, because this script loads its models with relative paths (`dms_tflite_models/...`).

A window titled **DMS PC Runner (TFLite)** opens with a mirrored webcam view. The script asks the camera for 640×480 at 30 fps. Press **q** or **Esc** to quit.

It uses these models:

- `dms_tflite_models/face_detection_ptq.tflite`
- `dms_tflite_models/face_landmark_ptq.tflite`
- `dms_tflite_models/iris_landmark_ptq.tflite`
- `dms_tflite_models/best_full_integer_quant.tflite`: the custom phone/smoking model, identical to `tests/yolov8n_smk_call_custom.tflite`

To use a different camera, change `cv2.VideoCapture(0)` in `main()` to another index.

### Run on a video file

```powershell
python dms-npu-restart.py --video path\to\drive.mp4
```

- A window titled **DMS PC Proxy Runner (TFLite)** opens. The video loops, played at up to its own frame rate. Leave out `--video` to use webcam 0, shown mirrored.
- Inference runs in a separate worker process, the same way it does on the board.
- Each frame is checked first. If it is black (average brightness below 15) or blurry (Laplacian variance below 50 in a 160×120 patch at the centre), inference is skipped, all alerts are cleared and the driver shows as not found. The result is shown as **Camera Status: OK / BLUR / BLACK**.
- Model paths are resolved relative to the script, so this one can be started from any folder.
- Press **q** or **Esc** to quit.

On Windows this script swaps the board's custom phone/smoking model for `yolov8n_smk_call.tflite`. To use the custom model instead:

```powershell
python dms-npu-restart.py --video path\to\drive.mp4 --smk-model dms_tflite_models\best_full_integer_quant.tflite
```

### What you see

- **Top bar**: driver status and score, turning from green to red as the score rises.
- **FPS** below the top bar. `dms-npu-restart.py` shows the camera status under it.
- **Face box** in green.
- **Bottom-left panel**: Distracted, Drowsy, Yawn, Head Down, Smoking and Phone, each **Yes** (red) or **No** (green).
- `pc_dms_runner.py` also draws phone and cigarette boxes in orange, and landmark dots when `DRAW_LANDMARKS = True`. In the other scripts both overlays are off by default (`DRAW_LANDMARKS`, `DRAW_SMK_CALL_CORDS`).

---

## Running on the STM32MP2 board

### Board requirements

- An STM32MP2 board (the models are compiled for `stm32mp25`) running OpenSTLinux with ST's X-LINUX-AI, which provides the `stai_mpu` Python API.
- GStreamer plugins: `libcamerasrc` (CSI camera), `v4l2src` (USB camera), `cairooverlay`, `gtkwaylandsink`, `fpsdisplaysink` and `autovideosink`.
- Python packages: PyGObject (`gi`) with GTK 3 and GStreamer, `pycairo`, OpenCV (`cv2`) and NumPy.
- Optional: `face_recognition` (dlib), for [driver identification](#driver-identification).

### Deploy

Copy the project to **`/home/root/Custom-DMS`** on the board. `dms-npu-restart.py` and `npu_worker.py` use this path for their default models and imports. From PowerShell, in the folder that contains `Custom-DMS`:

```powershell
scp -r Custom-DMS root@<board-ip>:/home/root/
```

### Run

```bash
cd /home/root/Custom-DMS

# CSI camera, inference in a restartable worker process (best for long runs)
python3 dms-npu-restart.py

# CSI camera, inference in the main process
python3 STM32_Dms-csi.py

# USB camera
python3 STM32_Dms.py --device /dev/video7
```

- `dms-npu-restart.py` and `STM32_Dms-csi.py` open a borderless, maximised window. Press **Esc** to quit.
- `STM32_Dms.py` draws on the GStreamer video sink. Press **Ctrl+C** in the terminal to stop it.
- `STM32_Dms.py` and `STM32_Dms-csi.py` load models from relative paths (`dms_nb_models/...`), so run them from the project folder.
- To run `STM32_Dms.py` without a display, set `ENABLE_DISPLAY = False`. The status is then printed to the terminal every 20 frames.
- On the board, the alert panel shows **N/A** while no face is found.

---

## Command-line options

### `dms-npu-restart.py`

| Option | Default | Description |
|---|---|---|
| `--frame_width` | `640` | Camera width (board) |
| `--frame_height` | `480` | Camera height (board) |
| `--framerate` | `30` | Camera frame rate (board). On Windows it also paces a video whose frame rate can't be read |
| `--video` | none | Windows only: play this video file instead of the webcam |
| `--face-model` | `/home/root/Custom-DMS/dms_nb_models/face_detection_ptq.nb` | Face detector |
| `--landmark-model` | `/home/root/Custom-DMS/dms_nb_models/face_landmark_ptq.nb` | Face landmarks |
| `--iris-model` | `/home/root/Custom-DMS/dms_nb_models/iris_landmark_ptq.nb` | Eye and iris landmarks |
| `--smk-model` | `/home/root/Custom-DMS/dms_nb_models/yolov8n_smk_call_custom.nb` | Phone/smoking detector |
| `--rebuild` | off | Meant to rebuild the face-recognition cache. It currently has no effect (see [Known issues](#known-issues)) |

### `STM32_Dms-csi.py`

| Option | Default |
|---|---|
| `--frame_width` | `640` |
| `--frame_height` | `480` |
| `--framerate` | `30` |
| `--face-model` | `dms_nb_models/face_detection_ptq.nb` |
| `--landmark-model` | `dms_nb_models/face_landmark_ptq.nb` |
| `--iris-model` | `dms_nb_models/iris_landmark_ptq.nb` |
| `--smk-model` | `dms_nb_models/yolov8n_smk_call_custom.nb` |

### `STM32_Dms.py`

| Option | Default | Description |
|---|---|---|
| `--device` | `/dev/video7` | Camera device. On Windows, its digits pick the webcam index, with webcam 0 as the fallback |
| `--width` | `640` | Camera width |
| `--height` | `480` | Camera height |
| `--fps` | `30` | Camera frame rate (board) |
| `--face-model`, `--landmark-model`, `--iris-model`, `--smk-model` | Same as `STM32_Dms-csi.py` | Models |

`pc_dms_runner.py` takes no options. Its settings are constants at the top of the file.

Every model option accepts either `.nb` or `.tflite`. On Windows, where `stai_mpu` is missing, the board scripts rewrite each model path:

1. Remove `/home/root/Custom-DMS` and resolve the rest relative to the script's folder.
2. For `.nb` paths only: change `.nb` to `.tflite`, `dms_nb_models` to `dms_tflite_models`, and `yolov8n_smk_call_custom` to `yolov8n_smk_call`.

Because of step 1, the board scripts can be started from any folder on Windows. A `.tflite` path you pass is resolved against the script folder but never renamed.

---

## How it works

### Per-frame pipeline

```
camera frame
 ├─ face detector ─► face closest to the image centre ─► square crop
 │    └─ face landmarks ─► mouth ratios ─► Yawn, Distracted, Head down
 │         └─ eye regions ─► iris landmarks ─► eye openness ─► Drowsy
 ├─ phone/smoking detector on the full frame (only while a face is found) ─► Phone, Smoking
 └─ alerts ─► risk score ─► overlay
```

| Model | Input | Runs on |
|---|---|---|
| Face detector | 128×128 | Every frame |
| Face landmarks (468 points) | 192×192 | Every 3rd frame |
| Iris landmarks (71 eye-contour + 5 iris points) | 64×64 | The same frames as the face landmarks |
| Phone/smoking detector (YOLOv8n) | 320×320 (custom model) or 416×416 | Every 3rd frame, one frame after the face landmarks |

- The square crop is 1.26 × the longer side of the face box.
- When a model skips a frame, its last result is reused, so every frame is still drawn.
- In `dms-npu-restart.py`, all models for a frame run in the worker from a single request.

### Measurements

Point numbers are MediaPipe face-mesh indices (`mouth.py`) or eye-contour indices from the iris model (`eye.py`).

| Measurement | Points | Formula |
|---|---|---|
| Eye openness | Eye contour 12 and 4 (top, bottom), 0 and 8 (corners) | Eye height ÷ eye width |
| Yawn ratio | 13 and 14 (inner lips), 78 and 308 (mouth corners) | Mouth height ÷ mouth width |
| Mouth–face ratio | 13 (upper lip), 132 (left cheek), 361 (right cheek) | Distance 13→132 ÷ distance 361→13 |
| Head-down ratio | 10 (forehead), 1 (nose tip), 152 (chin) | (nose y − forehead y) ÷ (chin y − forehead y) |

The eye regions are square crops centred between points 33 and 133 (one eye) and 362 and 263 (the other). Each side is twice the width of the eye, and the second eye is mirrored before it goes to the iris model.

---

## Thresholds and tuning

Each runner keeps its own constants at the top of its file, and the values are not the same everywhere:

| Setting | `pc_dms_runner.py` | `STM32_Dms.py`, `STM32_Dms-csi.py` | `dms-npu-restart.py` |
|---|---|---|---|
| Face detection score | 0.5 | 0.5 | 0.5 |
| An eye counts as open when openness is above | 0.35 | 0.35 | 0.40 |
| Eye samples kept (drowsy when both eyes are closed in all of them) | 3 | 3 | 2 |
| Yawn when the mouth ratio is above | 0.28, single frame | 0.4, single frame | 0.4, in 2 of the last 3 samples |
| Head down when the ratio is above | 0.58 | 0.78 | 0.58 |
| Head-down window (majority vote) | 5 frames | 7 frames | 5 samples |
| Distracted when the mouth–face ratio is outside | 0.5–2.0 | 0.5–2.0 | 0.5–2.0 |
| Phone confidence | 0.55, and near the face | 0.55 | 0.55 |
| Smoking confidence | 0.60, and near the face | 0.60 | 0.60 |

The constants to edit are `LEFT_EYE_THRESHOLD`, `RIGHT_EYE_THRESHOLD`, `MOUTH_THRESHOLD`, `HEAD_DOWN_THRESHOLD`, `FACING_LEFT_THRESHOLD`, `FACING_RIGHT_THRESHOLD`, `FACE_THRESHOLD`, the `*_PENALTY` values and `RESTORE_CREDIT`. Window lengths come from the size of `LEFT_EYE_STATUS`, `RIGHT_EYE_STATUS`, `HEAD_STATUS` and `YAWN_STATUS` (`LEFT_W`, `RIGHT_W` and `HEAD_W` in the STM32 scripts).

In `dms-npu-restart.py`, the face detector and the phone/smoking detector are created inside `npu_worker.py` with a threshold of 0.5, so change those two there. `FACE_THRESHOLD` in `dms-npu-restart.py` is not used.

**Phone and smoking thresholds.** `SmokingCallingDetector` never uses a confidence below 0.55 for phones or 0.60 for smoking, whatever a runner passes in. For the INT8 models, class scores go through a sigmoid, so 0.50 means no signal at all; the detector's notes put real phone detections at about 0.67–0.72.

- To make detection stricter in one runner, raise `PHONE_THRESHOLD` / `SMOKE_THRESHOLD` in `pc_dms_runner.py`, or `SMK_CALL_THRESHOLD` in the other scripts. Values below the floors have no effect.
- To make it stricter everywhere, raise `_MIN_PHONE_THRESH` and `_MIN_SMOKE_THRESH` in `smoking_calling_yolov8n_custom.py`. Do not go below 0.51.

**Near the face** (`pc_dms_runner.py` only). A phone or cigarette box only counts if its centre falls inside the face box widened by 1.25 face widths on each side, 0.75 face heights above and 1.75 face heights below.

**Black/blur check** (`dms-npu-restart.py` only). `BLUR_SKIP_THRESHOLD = 50.0`. The code's notes suggest a range from 30 (skip only heavy blur) to 80 (also skip mild motion blur).

**Tuning tips**

- Drowsy triggers while the eyes are open: lower the eye threshold, for example from 0.35 to 0.30.
- Yawns are missed, or trigger while talking: adjust `MOUTH_THRESHOLD` (0.28 is sensitive, 0.4 is strict).
- Head down triggers in a normal posture: raise `HEAD_DOWN_THRESHOLD`. The right value depends on camera height.
- Distracted triggers when the camera is mounted off to one side: widen the 0.5–2.0 range.

---

## Worker-process runner

`dms-npu-restart.py` runs every model in a separate process. According to the notes in `npu_proxy.py`, the NPU driver library (`libstai_mpu_ovx.so`) leaks about 47 MB per 500 inference calls and the leak cannot be fixed from Python, so the worker is restarted regularly to give that memory back. On Windows the same worker runs the TFLite models.

```
dms-npu-restart.py      camera, drawing, alert logic, risk score
   │  one request per frame: raw frame + which models to run
   ▼
npu_proxy.NPUProxy      starts, feeds and restarts the worker
   │  length-prefixed pickle over the worker's stdin / stdout
   ▼
npu_worker.py           face → landmarks → eyes and mouth → phone/smoking
   │  face boxes, 468 landmarks, eye ratios and iris points,
   │  mouth ratios, phone/smoking boxes
   ▼
back to dms-npu-restart.py
```

- The worker restarts after `RESTART_CALLS = 1000` calls (set in `npu_proxy.py`), or as soon as it has died or the pipe breaks. The frame that triggers a restart gets no result.
- A single inference error is logged and skipped, without a restart.
- The worker sends its log output to stderr, so the pipe carries only results.
- On the board, the worker is pinned to CPU core 1, and the face-recognition process to core 0.
- On the board, the script also runs `gc.collect()` every 300 frames.

---

## Driver identification

On the board, `dms-npu-restart.py` identifies the driver when the `face_recognition` package is installed. Without it the feature stays off. The Windows mode doesn't use it.

> **Note:** encoding a driver photo currently fails with a `NameError`, and so does rebuilding the cache. See [Known issues](#known-issues) for the one-line fix.

### Setup

1. Put driver photos (`.jpg`, `.jpeg` or `.png`) in `/home/root/face-recognition/haarcascade/sample/`. The file name becomes the driver's name: `alex.jpg` and `alex_1.jpg` both become "Alex", because a trailing `_number` is removed and the name is capitalised.
2. Place the Haar cascade at `/home/root/face-recognition/haarcascade/haarcascade_frontalface_default.xml`. It crops the face out of each sample photo; without it, the whole photo is used.
3. Encodings are cached in `face_cache.pkl.gz` in the same folder. A photo is encoded again when its contents change, and its name is removed from the cache when the file is deleted. To rebuild everything, delete the cache file.

Only one encoding is kept per name, so several photos of the same person replace each other instead of adding up.

### How it runs

- While a face is visible and no job is running, every 3rd frame sends the detected face box, resized to 150×150, to a separate process that encodes it with dlib.
- A match needs a face distance below 0.50.
- After 5 failed attempts the label shows Unknown, but attempts continue, and a later match still identifies the driver.
- An identified driver is checked again every 120 seconds. If the check returns a different result, identification starts over.
- After 30 frames without a face, identification starts over.
- The label above the face box, and the box itself, are orange while identifying ("Identifying…" or "Verifying… (n/5)"), green with the driver's name once matched, and red for "Unknown".

---

## Models

| Model | Task | Input | Output | Used by |
|---|---|---|---|---|
| `face_detection_ptq` (`.tflite`, `.nb`) | Face detection, BlazeFace-style, 896 anchors | 128×128×3, float | 896 scores; 896 × 16 box and keypoint values | All runners |
| `face_landmark_ptq` (`.tflite`, `.nb`) | 468 face landmarks | 192×192×3, float | Face score; 1404 values (468 × x, y, z) | All runners |
| `iris_landmark_ptq` (`.tflite`, `.nb`) | 71 eye-contour and 5 iris points | 64×64×3, float | 213 values; 15 values | All runners |
| `best_full_integer_quant.tflite` and `yolov8n_smk_call_custom.nb` | Custom YOLOv8n: phone (0), smoking (1) | 320×320×3, INT8 | 6 × 2100, INT8 | `pc_dms_runner.py` (`.tflite`); the board scripts on the board (`.nb`) |
| `yolov8n_smk_call` (`.tflite`, `.nb`) | Non-custom YOLOv8n phone/smoking model | 1×3×416×416, float (channels first) | 6 × 3549, float | The board scripts on Windows, by default |
| `yolov4_tiny_smk_call` (`.tflite`, `.nb`) | Legacy YOLOv4-tiny phone/smoking model | 416×416×3, float | 2535 × 2 scores; 2535 × 4 boxes | `smoking_calling_yolov4.py` only |

`.tflite` files are in `dms_tflite_models/` and `.nb` files are in `dms_nb_models/`. The shapes above are those of the `.tflite` files; the `.nb` files are the NPU builds.

### Converting a model for the board

`model_convertion.ipynb` records the ST Edge AI Core 2.2 command used to compile a TFLite model for the board:

```bash
stedgeai generate --model yolov8n_int8.tflite --target stm32mp25 --type tflite
```

- Replace `yolov8n_int8.tflite` with the model you want to convert. The result is written to `stm32ai_output/<model>.nb`.
- Copy the `.nb` file into `dms_nb_models/` on the board and select it with the matching `--*-model` option.
- `stedgeai analyze` doesn't support STM32MP2 yet.
- The notebook's first cell imports `torch`, `ultralytics` and `onnx`. Its last cells check an ONNX face-embedding model (`mobilefacenet`) that isn't part of this project.

### Inspecting a model

`inspect_model.py` prints a TFLite model's input and output shapes, data types and quantisation, then runs the model once on random float data and prints output statistics. Set the path in `m` at the top of the file and run it from the `Custom-DMS` folder:

```powershell
python inspect_model.py
```

The random-data run only works for models with float inputs. With an INT8-input model such as `best_full_integer_quant.tflite`, it stops with a type error after printing the details.

---

## Project structure

```
Custom-DMS/
├── pc_dms_runner.py                   Windows runner: webcam, TFLite, single process
├── dms-npu-restart.py                 Board runner with a restartable worker process; Windows mode with --video
├── npu_proxy.py                       Starts, feeds and restarts npu_worker.py
├── npu_worker.py                      Worker process that runs every model for a frame
├── STM32_Dms-csi.py                   Board runner: CSI camera, models in the main process
├── STM32_Dms.py                       Board runner: USB camera, models in the main process; OpenCV fallback
├── face_detection.py                  FaceDetector (.nb and .tflite)
├── face_landmark.py                   FaceLandmark: 468 points (.nb and .tflite)
├── eye.py                             Eye regions, iris landmarks, eye openness
├── mouth.py                           Yawn, mouth–face and head-down ratios
├── smoking_calling_yolov8n_custom.py  Phone/smoking detector used by every runner
├── smoking_calling_yolov8.py          Older YOLOv8 NPU detector (416×416, float16); unused, board only
├── smoking_calling_yolov4.py          Older YOLOv4-tiny detector; unused, board only
├── camera_usb_gst.py                  Board USB-camera test: GStreamer capture shown with OpenCV
├── face_detection_seperate_nb.py      Standalone NPU face-detection test (OpenCV camera 7); board only
├── inspect_model.py                   Prints a TFLite model's inputs and outputs
├── model_convertion.ipynb             ST Edge AI conversion command and ONNX checks
├── camera_usb_gst1_6.6.26.zip         Zip snapshot of the scripts and notebook (no models or tests)
├── dms_nb_models/                     NPU models for the board (.nb)
├── dms_tflite_models/                 CPU models for Windows (.tflite)
└── tests/                             Experiments and older versions
```

### `tests/`

| File | What it is |
|---|---|
| `int8_tflite_phone.py` | Standalone phone/cigarette detector using the custom INT8 TFLite model and a GStreamer USB camera (`/dev/video7`). A detection must hold for 4 consecutive frames |
| `int8_nb_phone.py` | The same detector for `.nb` or `.tflite` models, with the backend picked from the extension. A detection needs 1 frame |
| `STM32_Dms_old.py` | Earlier version of `STM32_Dms.py` |
| `face_detection_seperate_nb.py` | Copy of the script in the project root |
| `yolov8n_smk_call_custom.tflite`, `yolov8n_smk_call_custom.nb` | Copies of the custom phone/smoking model |
| `Readme.txt` | Note that this folder holds the DMS test scripts |

Both phone scripts need GStreamer, so they run on the board. They load `best_full_integer_quant.*` from the current folder, so set `MODEL_PATH` before running them.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ValueError: Could not open 'dms_tflite_models/...'` when starting `pc_dms_runner.py` | Start it from inside the `Custom-DMS` folder. |
| The webcam doesn't open (`Could not open webcam`), or the window stays black | Close other apps that are using the camera. In Windows Settings → Privacy & security → Camera, allow desktop apps to use the camera. For a different camera, change the index in `cv2.VideoCapture(0)`. |
| `Neither tflite_runtime nor tensorflow is installed` | Run `pip install tensorflow`. Ignore the message's `tflite-runtime` hint; that package isn't available for Windows. |
| `stai_mpu not available on this platform. Use a .tflite model instead.` | A `.nb` model was loaded without `stai_mpu`. On Windows, use the `.tflite` file from `dms_tflite_models/`. |
| `The function is not implemented` from `cv2.imshow` | The headless OpenCV build is installed. Run `pip uninstall opencv-python-headless`, then `pip install opencv-python`. |
| `dms-npu-restart.py` shows **Camera Status: BLUR** and detects nothing | The centre of the image is blurry or has little detail, so inference is skipped. Lower `BLUR_SKIP_THRESHOLD`. |
| `STM32_Dms.py` on Windows keeps running after its window closes | Press Ctrl+C in the terminal. |
| Board: `libcamerasrc not available`, `cairooverlay not found` or `gtkwaylandsink not found` | The board image is missing that GStreamer element. |
| Memory keeps growing on the board | Use `dms-npu-restart.py`. If memory still climbs too far between restarts, lower `RESTART_CALLS` in `npu_proxy.py`. |
| Low frame rate on Windows | Everything runs on the CPU. Close other heavy apps; the landmark, eye and phone models already run on only one frame in three. |

---

## Known issues

1. **`--rebuild` does nothing.** `dms-npu-restart.py` parses the flag but never passes it to `FaceDB`. Deleting `face_cache.pkl.gz` rebuilds the cache instead, once issue 2 is fixed.
2. **Encoding a driver photo crashes.** `FaceDB._sync()` (`dms-npu-restart.py`, line 389) calls `face_recognition.face_encodings(...)`, but the module is imported as `_fr`. A new or changed photo, or a deleted cache, raises `NameError` while the models load, and the video never starts. This only affects boards with `face_recognition` installed. Fix: change the call to `_fr.face_encodings(...)`.
3. **One photo per driver.** Photos that map to the same name replace each other's encoding rather than adding to it.
4. **Alerts can stay on after the face disappears** in `dms-npu-restart.py`. On the board, every alert keeps its last value for up to 30 frames without a face. In the Windows window, Distracted, Yawn and Head Down keep theirs until the face comes back. Their penalties keep adding to the score in the meantime.
5. **`STM32_Dms.py` on Windows** keeps running after **q** or **Esc** closes the window. Use Ctrl+C. The Windows fallback of `STM32_Dms-csi.py` exits normally.
6. **`STM32_Dms.py` starts its OpenCV camera loop before the models load.** The loop doesn't wait for `inited`, so if the first frame arrives while the models are still loading, the loop stops with an `AttributeError` and no window appears. This also applies to the Windows fallback of `STM32_Dms-csi.py`. Press Ctrl+C and start again.
7. **Head Down reads Yes right after start-up** in `STM32_Dms.py` and `STM32_Dms-csi.py`. `HEAD_STATUS` starts as all ones, so the first 3 head-pose updates after the first face is seen count as head down and add their penalty.
8. **A different phone/smoking model on Windows.** The board scripts switch to `yolov8n_smk_call.tflite` instead of the custom model. Pass `--smk-model dms_tflite_models\best_full_integer_quant.tflite` to match the board.
9. **Face boxes differ slightly between Windows and the board.** For `.nb` models, `face_detection.py` enlarges the detected box, smooths it across frames and keeps only the best face. For `.tflite` models it returns every face that survives non-maximum suppression.
10. **Eye debugging doesn't work.** With `debug_eyes_detection = True`, `STM32_Dms.py` tries to save eye images that don't exist inside `save_eyes_for_debug()` (the error is caught and printed), and `STM32_Dms-csi.py` only creates the folder. With `print_eye_status = True`, both print eye ratios that are never updated, so they always read 0.000.
11. **Stale comment.** `npu_proxy.py` describes restarting after 5000 calls, but the actual value is `RESTART_CALLS = 1000`.
12. **Hard-coded test paths.** `face_detection_seperate_nb.py` expects `face_detection_ptq.nb` in the current folder, but the file lives in `dms_nb_models/`. `tests/int8_nb_phone.py` expects `best_full_integer_quant.nb`, which is not in the project; the compiled custom model is `yolov8n_smk_call_custom.nb`.
