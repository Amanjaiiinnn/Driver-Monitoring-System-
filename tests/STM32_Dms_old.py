import os
import sys
import math
import time
import argparse
import numpy as np
import gi
import cairo
import cv2
from face_detection import FaceDetector
from face_landmark import FaceLandmark
from eye import Eye
from mouth import Mouth
# from smoking_calling_yolov4 import SmokingCallingDetector
# from smoking_calling_yolov8 import SmokingCallingDetectorYOLOv8
from smoking_calling_yolov8n_custom import SmokingCallingDetector

FACE_MODEL     = "face_detection_ptq.nb" #tflite"
LANDMARK_MODEL = "face_landmark_ptq.nb" #tflite"
IRIS_MODEL     = "iris_landmark_ptq.nb" #tflite"
# SMK_CALL_MODEL = "yolov4_tiny_smk_call.nb" #tflite"
# SMK_CALL_MODEL = "yolov8n_smk_call.nb" #tflite"
SMK_CALL_MODEL = "yolov8n_smk_call_custom.nb" #tflite"

gi.require_version("Gst", "1.0")
from gi.repository import Gst

cur_path = os.path.dirname(os.path.abspath(__file__))

DRAW_SMK_CALL_CORDS = False
""" To enable drawing for smk/call detection box """

DRAW_LANDMARKS = False
""" To enable drawing for face landmarks
(this will slow down the drawing process a lot, just for debug) """

FRAME_WIDTH = 640
""" The frame width of image from gstreamer pipeline to ml_sink """

FRAME_HEIGHT = 480
""" The frame height of image from gstreamer pipeline to ml_sink """

BAD_FACE_PENALTY = 0.01
""" % to remove for far away face """

NO_FACE_PENALTY = 0.7
""" % to remove for no faces in frame """

YAWN_PENALTY = 7.0
""" % to remove for yawning """

DISTRACT_PENALTY = 2.0
""" % to remove for looking away """

SLEEP_PENALTY = 5.0
""" % to remove for sleeping """

SMK_PENALTY = 2.0
""" % to remove for smoking """

CALL_PENALTY = 2.0
""" % to remove for calling """

HEAD_DOWN_PENALTY = 5.0
""" % to remove for head down """

RESTORE_CREDIT = -5.0
""" % to restore for doing everything right """

FACE_THRESHOLD = 0.5
""" The threshold value for face detection """

LEFT_EYE_THRESHOLD = 0.3
""" if the left_eye ratio is greater then this value, then left eye will be
    considered as open, otherwise be considered as closed. """

RIGHT_EYE_THRESHOLD = 0.3
""" if the right_eye ratio is greater then this value, then right eye will be
    considered as open, otherwise be considered as closed. """

MOUTH_THRESHOLD = 0.4
""" if the mouth ratio is greater then this value, then mouth will be
    considered as open, otherwise be considered as closed. """

FACING_LEFT_THRESHOLD = 0.5
""" if the mouth_face_ratio is less then this value then face will be
    considered as turning left """

FACING_RIGHT_THRESHOLD = 2
""" if the mouth_face_ratio is greater then this value, then face will be
    considered as turning right"""

SMK_CALL_THRESHOLD = 0.5
""" The threshold value for smoking/calling detection """

LEFT_W = 3
""" The filter window size for left eye """

RIGHT_W = 3
""" The filter window size for right eye """

LEFT_EYE_STATUS = np.zeros(LEFT_W)
""" Array to filter out left eye blinking. In the array, 0 means eye closed, 1 means eye open """

RIGHT_EYE_STATUS = np.zeros(RIGHT_W)
""" Array to filter out right eye blinking. In the array, 0 means eye closed, 1 means eye open """

RIGHT_EYE_ID = 1
LEFT_EYE_ID = 0

FONT_SIZE = 10.0
DIFF_BETWEEN_TEXT = 15

ALERT_LABEL_X = 10
ALERT_STATUS_X = 100
ALERT_ROW_START_Y = 400
""" --- Hysteresis thresholds --- """
HEAD_DOWN_THRESHOLD = 0.78

HEAD_W = 7
HEAD_STATUS = [1] * HEAD_W   # 1 = head up, 0 = head down

ENABLE_DISPLAY = True

DEFAULT_BECKEND = "NPU"
DEFAULT_PLATFORM = "STM32"#"i.MX93" #"PC"
DEFAULT_PATH = "/root/dms/downloads/" #"/home/aman/Learning/Python_Coding/dms/downloads"

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

        face_model     = FACE_MODEL
        landmark_model = LANDMARK_MODEL
        iris_model     = IRIS_MODEL
        smk_call_model = SMK_CALL_MODEL

        self.run_face_detection = True
        self.run_face_landmarks = True
        self.run_eye_detection = True
        self.run_yolo_smk_call_detection = True

        if self.run_face_detection:
            self.face_detector = FaceDetector(face_model, FACE_THRESHOLD)
        
        if self.run_face_landmarks:
            self.face_landmark = FaceLandmark(landmark_model)

            self.mouth = Mouth()

        if self.run_eye_detection:
            self.eye_detector = Eye(iris_model)

        # if self.run_yolo_smk_call_detection:
        #     self.smoking_calling_detector = SmokingCallingDetector(smk_call_model, conf=SMK_CALL_THRESHOLD)

        if self.run_yolo_smk_call_detection:
            try:
                self.smoking_calling_detector = SmokingCallingDetector(
                    smk_call_model,
                    conf=SMK_CALL_THRESHOLD
                )
            except NameError:
                print("SmokingCallingDetector not found, using YOLOv8 version instead.")
                self.smoking_calling_detector = SmokingCallingDetectorYOLOv8(
                    smk_call_model,
                    conf=SMK_CALL_THRESHOLD
                )
        
        # ---------------------------------------------------------
        # Mark system as fully initialized
        # ---------------------------------------------------------
        self.inited = True

    def setup_camera(self, device="/dev/video7", width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=30):
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
                video/x-raw,format=RGB,width={width},height={height} !
                appsink name=ml_sink emit-signals=true drop=true max-buffers=1
            """
        self.pipeline = Gst.parse_launch(pipeline_str)

        # ---------------------------------------------------------
        # Overlay callback for drawing results on display
        # ---------------------------------------------------------
        drawer = self.pipeline.get_by_name("drawer")
        if drawer:
            drawer.connect("draw", self.draw)

        # ---------------------------------------------------------
        # ML inference callback (called per frame)
        # ---------------------------------------------------------
        ml_sink = self.pipeline.get_by_name("ml_sink")
        ml_sink.connect("new-sample", self.inference)

        # Start camera 
        self.pipeline.set_state(Gst.State.PLAYING)
    
    def stop_camera(self):
        self.pipeline.set_state(Gst.State.NULL)

    def save_frames(self, frame):
        save_folder = "saved_frames"
        os.makedirs(save_folder, exist_ok=True)

        now = time.time()

        if now - self.last_save_time >= 1.0:
            timestamp = int(now)
            filename = os.path.join(save_folder, f"frame_{timestamp}.jpg")

            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")

            self.last_save_time = now

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
        frame = data.emit("pull-sample")

        # Output containers for drawing/debug
        face_cords = []
        smk_call_cords = []
        mark_group = []

        # Default driver states (True = safe / normal)
        phone_detected = False
        smoking_detected = False
        distracted = False
        yawn_detected = False
        head_down_final = False
        sleep_detected = False

        # Drop frame if pipeline is not ready
        if frame is None:
            return 0
        if self.inited is False:
            return 0

        # Frame counter for debugging / profiling
        self.frame_count += 1
        self.fps_counter += 1

        now = time.time()
        elapsed = now - self.fps_start_time

        if elapsed >= 1.0:
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = now

        # -------------------------------------------------
        # Extract raw image buffer from GStreamer sample
        # -------------------------------------------------
        buffer = frame.get_buffer()
        caps = frame.get_caps()

        ret, mem_buf = buffer.map(Gst.MapFlags.READ)

        if not ret:
            return Gst.FlowReturn.OK

        height = caps.get_structure(0).get_value("height")
        width = caps.get_structure(0).get_value("width")

        # Convert raw buffer into NumPy array (BGR → RGB)
        # frame = np.ndarray(
        #     shape=(height, width, 3),
        #     dtype=np.uint8,
        #     buffer=mem_buf.data
        # )[..., ::-1]

        frame = np.ndarray(shape=(height, width, 3), dtype=np.uint8, buffer=mem_buf.data)

        #self.save_frames(frame)

        if self.frame_count % 3 == 0 and self.run_face_landmarks:
            run_landmark = True
        else:
            run_landmark = False

        if self.frame_count % 3 == 1 and self.run_eye_detection:
            run_eye = True
        else:
            run_eye = False

        if self.frame_count % 3 == 2 and self.run_yolo_smk_call_detection:
            run_smk = True
        else:
            run_smk = False

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
        face_cords = []
        smk_call_cords = []
        mark_group = []

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
                

            # ---------------------------------------------
            # Convert normalized face boxes → pixel coords
            # ---------------------------------------------
            for i in range(np.size(boxes, 0)):
                boxes[i][[0, 2]] *= width
                boxes[i][[1, 3]] *= height

            #print(f"2nd:{boxes}")

            # Expand face boxes into squares (for landmarks)
            boxes = self.transform_to_square(
                boxes, scale=1.26, offset=(0, 0)
            )

            #print(f"3rd:{boxes}")

            # Clip boxes within image boundaries
            boxes, _ = self.clip_boxes(
                boxes, (0, 0, width, height)
            )
            boxes = boxes.astype(np.int32)

            #print(f"4th:{boxes}")

            # ---------------------------------------------
            # Select face closest to image center
            # (Assume driver is centered)
            # ---------------------------------------------
            face_in_center = 0
            distance_to_center = math.hypot(
                width / 2, height / 2
            )

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

            # ---------------------------------------------
            # Face landmark inference
            # ---------------------------------------------
            if run_landmark:
                face_image = frame[y1:y2, x1:x2]
                face_marks = self.face_landmark.get_landmark(
                    face_image, (x1, y1, x2, y2)
                )

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
                mouth_ratio = self.mouth.yawning_ratio(face_marks)
                yawn_detected = not (mouth_ratio <= MOUTH_THRESHOLD)

                head_down_ratio = self.mouth.head_down_ratio(face_marks)

                status = 1 if head_down_ratio > HEAD_DOWN_THRESHOLD else 0

                HEAD_STATUS[:-1] = HEAD_STATUS[1:]
                HEAD_STATUS[-1] = status

                head_down_avg = np.mean(HEAD_STATUS)
                head_down_final = head_down_avg > 0.5
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

        # -------------------------------------------------
        # Update shared state for drawing & alerts
        # -------------------------------------------------
        self.marks = mark_group
        self.face_cords = face_cords
        self.smk_call_cords = smk_call_cords

        self.distracted = distracted
        self.drowsy = sleep_detected
        self.yawn = yawn_detected
        self.smoking = smoking_detected
        self.phone = phone_detected
        self.head_down = head_down_final

        # -------------------------------------------------
        # Driver safety score accumulation
        # -------------------------------------------------
        if distracted:
            self.safe_value = min(self.safe_value + DISTRACT_PENALTY, 100.0)
        if sleep_detected:
            self.safe_value = min(self.safe_value + SLEEP_PENALTY, 100.0)
        if yawn_detected:
            self.safe_value = min(self.safe_value + YAWN_PENALTY, 100.0)
        if smoking_detected:
            self.safe_value = min(self.safe_value + SMK_PENALTY, 100.0)
        if phone_detected:
            self.safe_value = min(self.safe_value + CALL_PENALTY, 100.0)
        if head_down_final:
            self.safe_value = min(self.safe_value + HEAD_DOWN_PENALTY, 100.0)
        if not face_cords:
            self.safe_value = min(self.safe_value + NO_FACE_PENALTY, 100.0)

        # # Restore safety score when driver is fully attentive
        if not distracted and not sleep_detected and not yawn_detected and not head_down_final and not smoking_detected and not phone_detected and face_cords:
            self.safe_value = max(self.safe_value + RESTORE_CREDIT, 0.0)

        # Release GStreamer buffer
        buffer.unmap(mem_buf)

        if not ENABLE_DISPLAY:
            if self.frame_count % 5 == 0:
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

        return Gst.FlowReturn.OK

    def smoke_call_detection(self, frame, phone_detected, smoking_detected):

        smk_call_cords = []
        now = time.time()
        phone_seen = False
        cigrate_seen = False

        # Resize frame for faster YOLO
        #small_frame = cv2.resize(frame, (224, 224))
        
        smk_call_result = self.smoking_calling_detector.inference(frame, False)

        if np.size(smk_call_result, 0) > 0:
            #print(f"smk_call_result:-{smk_call_result}")
            for i in range(np.size(smk_call_result, 0)):

                # Class 0 → phone call
                if int(smk_call_result[i][5]) == 0 and smk_call_result[i][4] > SMK_CALL_THRESHOLD: #0.85:
                    phone_seen = True
                    # phone_detected = True

                # Class 1 → smoking
                if int(smk_call_result[i][5]) == 1 and smk_call_result[i][4] > SMK_CALL_THRESHOLD: #0.85:    
                    cigrate_seen = True
                    # smoking_detected = True

                # Bounding box extraction
                x1 = int(smk_call_result[i][0])
                y1 = int(smk_call_result[i][1])
                x2 = int(smk_call_result[i][2])
                y2 = int(smk_call_result[i][3])

                smk_call_cords.append([x1, y1, x2, y2])
                
            # ------------------------------------
            # Update counter & timestamp for Phone
            # ------------------------------------
            #if now - self.phone_alrt_cooldown_ts > 2.0:
            if phone_seen:
                self.call_count += 1
                phone_detected = True
                #print("call_count:", self.call_count)
            else:
                self.call_count = 0   # 🔥 THIS LINE IS CRITICAL
            
            # -------------------------------
            # Decide CALL state
            # -------------------------------
            # Confirm call after enough frames
            # if self.call_count >= 5 and not phone_detected:
            #     #print("phone detected true")
            #     self.phone_alrt_cooldown_ts = now
            #     self.last_phone_detected_ts = now   # 🔥 MUST
            #     self.call_count = 0
            #     phone_detected = True
            #     self.last_phone_status = phone_detected
            # elif self.last_phone_status:
            #     phone_detected = True

            # # Release call if phone gone for > 1 sec
            # if self.last_phone_status and (now - self.last_phone_detected_ts) > 1.0:
            #     #print("Resetting call_count due to timeout")
            #     self.call_count = 0
            #     phone_detected = False
            #     self.last_phone_status = False

            # ------------------------------------
            # Update counter & timestamp for smoke
            # ------------------------------------
            # if now - self.smoke_alrt_cooldown_ts > 2.0:
            if cigrate_seen:
                self.smoke_count += 1
                smoking_detected = True
                #print("smoke_count:", self.smoke_count)
            else:
                self.smoke_count = 0   # 🔥 THIS LINE IS CRITICAL
            
            # -------------------------------
            # Decide Smoke state
            # -------------------------------
            # Confirm smoke after enough frames
            # if self.smoke_count >= 5 and not smoking_detected:
            #     # print("Smoke detected true")
            #     self.smoke_alrt_cooldown_ts = now
            #     self.last_smoke_detected_ts = now   # 🔥 MUST
            #     self.smoke_count = 0
            #     smoking_detected = True
            #     self.last_smoke_status = smoking_detected
            # elif self.last_smoke_status:
            #     smoking_detected = True

            # # Release smoke if phone gone for > 1 sec
            # if self.last_smoke_status and (now - self.last_smoke_detected_ts) > 1.0:
            #     print("Resetting smoke_count due to timeout")
            #     self.smoke_count = 0
            #     smoking_detected = False
            #     self.last_smoke_status = False
        
        return smk_call_cords, phone_detected, smoking_detected

    def eye_landmark_detection(self, face_marks, frame, buffer, mem_buf, mark_group):
        x1, y1, x2, y2 = self.eye_detector.get_eye_roi(
            face_marks, LEFT_EYE_ID
        )

        # 🔥 CLAMP HERE
        x1, y1, x2, y2 = clamp_roi(
            x1, y1, x2, y2,
            frame.shape[1],  # width
            frame.shape[0],  # height
        )
        left_eye_image = frame[y1:y2, x1:x2]

        if left_eye_image is None or left_eye_image.size == 0:
            # Skip this frame safely
            buffer.unmap(mem_buf)
            return mark_group, self.both_eyes_closed

        left_eye_marks, left_iris_marks = self.eye_detector.get_landmark(
            left_eye_image, (x1, y1, x2, y2), LEFT_EYE_ID
        )
        mark_group.append(np.array(left_iris_marks))

        # ---------------------------------------------
        # Right eye landmark & iris detection
        # ---------------------------------------------
        x1, y1, x2, y2 = self.eye_detector.get_eye_roi(
            face_marks, RIGHT_EYE_ID
        )

        # 🔥 CLAMP HERE
        x1, y1, x2, y2 = clamp_roi(
            x1, y1, x2, y2,
            frame.shape[1],  # width
            frame.shape[0],  # height
        )
        right_eye_image = frame[y1:y2, x1:x2]

        if right_eye_image is None or right_eye_image.size == 0:
            # Skip this frame safely
            buffer.unmap(mem_buf)
            return mark_group, self.both_eyes_closed

        right_eye_marks, right_iris_marks = self.eye_detector.get_landmark(
            right_eye_image, (x1, y1, x2, y2), RIGHT_EYE_ID
        )
        mark_group.append(np.array(right_iris_marks))

        # ---------------------------------------------
        # Eye blink ratio calculation
        # ---------------------------------------------
        left_eye_ratio = self.eye_detector.blinking_ratio(
            left_eye_marks, 0
        )
        right_eye_ratio = self.eye_detector.blinking_ratio(
            right_eye_marks, 1
        )

        # ---------------------------------------------
        # Temporal filtering (LEFT eye)
        # ---------------------------------------------
        for i in range(LEFT_W - 1):
            LEFT_EYE_STATUS[i] = LEFT_EYE_STATUS[i + 1]

        LEFT_EYE_STATUS[LEFT_W - 1] = (
            1 if left_eye_ratio > LEFT_EYE_THRESHOLD else 0
        )

        # ---------------------------------------------
        # Temporal filtering (RIGHT eye)
        # ---------------------------------------------
        for i in range(RIGHT_W - 1):
            RIGHT_EYE_STATUS[i] = RIGHT_EYE_STATUS[i + 1]

        RIGHT_EYE_STATUS[RIGHT_W - 1] = (
            1 if right_eye_ratio > RIGHT_EYE_THRESHOLD else 0
        )

        # ---------------------------------------------
        # Final eye closure decision
        # ---------------------------------------------
        left_eye_avg = np.mean(LEFT_EYE_STATUS)
        right_eye_avg = np.mean(RIGHT_EYE_STATUS)

        self.left_eye_closed = left_eye_avg < 0.1
        self.right_eye_closed = right_eye_avg < 0.1
        # if self.frame_count % 15 == 0:
        #     print(f"left eye ratio:- {left_eye_avg}, right eye ratio:- {right_eye_avg}")
        self.both_eyes_closed = (
            self.left_eye_closed and self.right_eye_closed
        )
        # if self.both_eyes_closed:
        #     print(f"Eyes Status:-{self.both_eyes_closed}")

        # Debug / logging hooks
        if self.debug_eyes_detection or self.print_eye_status:
            self.save_eyes_for_debug()

        # Sleep detection based on both eyes
        sleep_detected = self.both_eyes_closed

        return mark_group, sleep_detected

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

        # -------------------------------------------------
        # Font and drawing configuration
        # -------------------------------------------------
        context.select_font_face(
            "Arial",
            cairo.FONT_SLANT_NORMAL,
            cairo.FONT_WEIGHT_BOLD
        )

        # Scale factor to map ML resolution → display resolution
        # scale_x = 640.0 / FRAME_WIDTH
        # scale_y = 480.0 / FRAME_HEIGHT

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
        if self.smk_call_cords and DRAW_SMK_CALL_CORDS:
            for cords in self.smk_call_cords:
                context.rectangle(
                    (cords[0]) + offset,
                    (cords[1]),
                    (cords[2] - cords[0]),
                    (cords[3] - cords[1]),
                )
            context.stroke()

        # -------------------------------------------------
        # Draw face / eye / iris landmarks
        # -------------------------------------------------
        if self.marks and DRAW_LANDMARKS:
            for m in self.marks:
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

        # -------------------------------------------------
        # ALERT PANEL BACKGROUND
        # -------------------------------------------------
        panel_x = 0
        panel_y = 380
        panel_width  = 150
        panel_height = 160

        # Draw black background rectangle
        context.set_source_rgb(0, 0, 0)
        context.rectangle(panel_x, panel_y, panel_width, panel_height)
        context.fill()

        # -------------------------------------------------
        # Alert labels
        # -------------------------------------------------
        self.write_text(context, "Distracted:", ALERT_LABEL_X, ALERT_ROW_START_Y)
        self.write_text(context, "Drowsy:",     ALERT_LABEL_X, ALERT_ROW_START_Y + (DIFF_BETWEEN_TEXT * 1))
        self.write_text(context, "Yawn:",       ALERT_LABEL_X, ALERT_ROW_START_Y + (DIFF_BETWEEN_TEXT * 2))
        self.write_text(context, "Head Down:",  ALERT_LABEL_X, ALERT_ROW_START_Y + (DIFF_BETWEEN_TEXT * 3))
        self.write_text(context, "Smoking:",    ALERT_LABEL_X, ALERT_ROW_START_Y + (DIFF_BETWEEN_TEXT * 4))
        self.write_text(context, "Phone:",      ALERT_LABEL_X, ALERT_ROW_START_Y + (DIFF_BETWEEN_TEXT * 5))

        # ---------------------------------------------
        # FPS display
        # ---------------------------------------------
        context.save()
        context.set_source_rgb(0, 0, 1)
        context.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        context.set_font_size(20)

        context.move_to(10, 70)
        context.show_text(f"FPS: {self.current_fps:.2f}")

        context.restore()

        # -------------------------------------------------
        # Alert status + face bounding box
        # -------------------------------------------------
        #if isinstance(self.face_cords, list) and len(self.face_cords) > 0: # if self.face_cords:
        if (isinstance(self.face_cords, list)
            and len(self.face_cords) > 0
            and len(self.face_cords[0]) == 4):
            # Draw alert states (ON / OFF)
            self.write_alert_status(context, self.distracted, ALERT_STATUS_X, ALERT_ROW_START_Y)
            self.write_alert_status(context, self.drowsy,     ALERT_STATUS_X, ALERT_ROW_START_Y + (DIFF_BETWEEN_TEXT * 1))
            self.write_alert_status(context, self.yawn,       ALERT_STATUS_X, ALERT_ROW_START_Y + (DIFF_BETWEEN_TEXT * 2))
            self.write_alert_status(context, self.head_down,  ALERT_STATUS_X, ALERT_ROW_START_Y + (DIFF_BETWEEN_TEXT * 3))
            self.write_alert_status(context, self.smoking,    ALERT_STATUS_X, ALERT_ROW_START_Y + (DIFF_BETWEEN_TEXT * 4))
            self.write_alert_status(context, self.phone,      ALERT_STATUS_X, ALERT_ROW_START_Y + (DIFF_BETWEEN_TEXT * 5))

            # Draw face bounding box
            context.set_source_rgb(0, 1, 0)
            context.rectangle(
                (self.face_cords[0][0]) + offset,
                (self.face_cords[0][1]),
                (self.face_cords[0][2] - self.face_cords[0][0]),
                (self.face_cords[0][3] - self.face_cords[0][1]),
            )
            context.stroke()

        else:
            # No face detected → show neutral / unknown state
            self.write_alert_status(context, None, ALERT_STATUS_X, ALERT_ROW_START_Y)
            self.write_alert_status(context, None, ALERT_STATUS_X, ALERT_ROW_START_Y + (DIFF_BETWEEN_TEXT * 1))
            self.write_alert_status(context, None, ALERT_STATUS_X, ALERT_ROW_START_Y + (DIFF_BETWEEN_TEXT * 2))
            self.write_alert_status(context, None, ALERT_STATUS_X, ALERT_ROW_START_Y + (DIFF_BETWEEN_TEXT * 3))
            self.write_alert_status(context, None, ALERT_STATUS_X, ALERT_ROW_START_Y + (DIFF_BETWEEN_TEXT * 4))
            self.write_alert_status(context, None, ALERT_STATUS_X, ALERT_ROW_START_Y + (DIFF_BETWEEN_TEXT * 5))

        # -------------------------------------------------
        # Overall driver status (safe / warning / danger)
        # -------------------------------------------------
        self.write_driver_status(context)

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

    def write_alert_status(self, context, yes, y, x):
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
        context.set_font_size(int(FONT_SIZE))

        # Move drawing cursor to desired screen position
        context.move_to(y, x)

        # -------------------------------------------------
        # Unknown / not applicable state
        # -------------------------------------------------
        if yes is None:
            context.set_source_rgb(1, 1, 1)   # white color
            context.show_text("N/A")
            return

        # -------------------------------------------------
        # Alert active
        # -------------------------------------------------
        if yes:
            context.set_source_rgb(1, 0, 0)   # red color
            context.show_text("Yes")

        # -------------------------------------------------
        # Alert inactive
        # -------------------------------------------------
        else:
            context.set_source_rgb(0, 1, 0)   # green color
            context.show_text("No")

        return
    
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
        context.set_font_size(int(FONT_SIZE))

        # Move drawing cursor to specified position
        context.move_to(y, x)

        # Set text color to white
        context.set_source_rgb(1, 1, 1)

        # Render the text on screen
        context.show_text(text)

        return

    def write_driver_status(self, context):
        """
        Draw the overall driver status banner at the top of the display.
        This includes:
        - Background panel
        - System label
        - Driver safety status with color-coded severity
        """

        # ---------------------------
        # Driver Status PANEL BACKGROUND
        # ---------------------------
        panel_x = 0
        panel_y = 0
        panel_width  = 640
        panel_height = 40

        # Draw black background for status bar
        context.set_source_rgb(0, 0, 0)
        context.rectangle(panel_x, panel_y, panel_width, panel_height)
        context.fill()

        # Display system / project label on the right side
        self.write_text(context, "EDS-INDIA", 550, 30)

        # ---------------------------
        # Driver status text styling
        # ---------------------------
        context.set_font_size(int(FONT_SIZE + 10))
        context.move_to(10, 30)

        # Compute color based on safety score
        # Green → safe, Red → danger
        r = min(self.safe_value / 50.0, 1.0)
        g = min(1.0, (100.0 - self.safe_value) / 50.0)
        b = 0

        context.set_source_rgb(r, g, b)

        # ---------------------------
        # Driver status message
        # ---------------------------
        if self.face_cords:
            if self.safe_value < 33:
                context.show_text(
                    "Driver OK (" + str(round(self.safe_value, 2)) + "%)"
                )
            elif self.safe_value < 66:
                context.show_text(
                    "Warning! (" + str(round(self.safe_value, 2)) + "%)"
                )
            else:
                context.show_text(
                    "Danger! (" + str(round(self.safe_value, 2)) + "%)"
                )
        else:
            # No face detected in current frame
            context.show_text(
                "Driver not found! (" + str(round(self.safe_value, 2)) + "%)"
            )


if __name__ == "__main__":
    window = DMSDemo(device="/dev/video7", width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=30)

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        window.stop_camera()
        cv2.destroyAllWindows()