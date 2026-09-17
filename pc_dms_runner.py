#!/usr/bin/env python3
"""
pc_dms_runner.py  —  Run Driver Monitoring System (DMS) on your PC (Windows/Mac/Linux)
using your Webcam and OpenCV (no GStreamer or GTK required).

Usage:
    python pc_dms_runner.py
"""

import os
import sys
import time
import math
import cv2
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from face_detection import FaceDetector
from face_landmark import FaceLandmark
from eye import Eye
from mouth import Mouth
from smoking_calling_yolov8n_custom import SmokingCallingDetector

# ═══════════════════════════════════════════════════════════
#  MODEL PATHS (TFLite CPU fallback)
# ═══════════════════════════════════════════════════════════
MODEL_DIR = "dms_tflite_models"
FACE_MODEL     = f"{MODEL_DIR}/face_detection_ptq.tflite"
LANDMARK_MODEL = f"{MODEL_DIR}/face_landmark_ptq.tflite"
IRIS_MODEL     = f"{MODEL_DIR}/iris_landmark_ptq.tflite"
SMK_CALL_MODEL = f"{MODEL_DIR}/best_full_integer_quant.tflite"

# ═══════════════════════════════════════════════════════════
#  THRESHOLDS & CONFIG
# ═══════════════════════════════════════════════════════════
FACE_THRESHOLD      = 0.5
LEFT_EYE_THRESHOLD  = 0.35
RIGHT_EYE_THRESHOLD = 0.35
MOUTH_THRESHOLD     = 0.28
FACING_LEFT_THRESHOLD  = 0.5
FACING_RIGHT_THRESHOLD = 2.0
PHONE_THRESHOLD     = 0.55
SMOKE_THRESHOLD     = 0.30
SMK_CALL_THRESHOLD  = min(PHONE_THRESHOLD, SMOKE_THRESHOLD)
HEAD_DOWN_THRESHOLD = 0.58
DRAW_LANDMARKS      = False

# Scoring / Penalties (same as STM32_Dms.py)
DISTRACT_PENALTY    = 2.0
SLEEP_PENALTY       = 5.0
YAWN_PENALTY        = 7.0
SMK_PENALTY         = 2.0
CALL_PENALTY        = 2.0
HEAD_DOWN_PENALTY   = 5.0
NO_FACE_PENALTY     = 0.7
RESTORE_CREDIT      = -5.0

# Sliding-window eye & head state
LEFT_EYE_STATUS  = np.zeros(3)
RIGHT_EYE_STATUS = np.zeros(3)
HEAD_STATUS      = [0] * 5

class PCDMSApp:
    def __init__(self):
        print("[INFO] Initializing DMS Models on PC...")
        self.face_detector = FaceDetector(FACE_MODEL, FACE_THRESHOLD)
        self.face_landmark = FaceLandmark(LANDMARK_MODEL)
        self.eye_detector  = Eye(IRIS_MODEL)
        self.mouth         = Mouth()
        self.smoking_detector = SmokingCallingDetector(SMK_CALL_MODEL, iou=0.40, conf=SMK_CALL_THRESHOLD)

        # App state
        self.safe_value = 0.0
        self.distracted = False
        self.drowsy = False
        self.yawn = False
        self.smoking = False
        self.phone = False
        self.head_down = False
        self.both_eyes_closed = False

        # Last values for carry-over state
        self.last_face_marks = []
        self.last_smk_call_cords = []
        self.last_phone_status = False
        self.last_smoke_status = False
        self.last_distracted_status = False
        self.last_yawn_status = False
        self.last_head_down_status = False

        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0

    def process_frame(self, frame):
        self.frame_count += 1

        # Calculate FPS
        now = time.time()
        elapsed = now - self.fps_start_time
        if elapsed >= 1.0:
            self.current_fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start_time = now

        h, w = frame.shape[:2]

        # Reset frame states
        distracted = self.last_distracted_status
        yawn_detected = self.last_yawn_status
        head_down_final = self.last_head_down_status
        sleep_detected = False
        phone_detected = self.last_phone_status
        smoking_detected = self.last_smoke_status
        smk_call_cords = self.last_smk_call_cords

        # Run schedule (similar to STM32_Dms.py)
        run_landmark = (self.frame_count % 3 == 0)
        run_eye      = (self.frame_count % 3 == 0)
        run_smk      = (self.frame_count % 3 == 1)

        # 1. Face detection
        boxes = self.face_detector.detect(frame)
        face_cords = []
        mark_group = []

        if boxes is not None and len(boxes) > 0:
            # Scale coordinates
            for i in range(np.size(boxes, 0)):
                boxes[i][[0, 2]] *= w
                boxes[i][[1, 3]] *= h

            # Make square and clip
            boxes = self.transform_to_square(boxes, scale=1.26)
            boxes = self.clip_boxes(boxes, (0, 0, w, h)).astype(np.int32)

            # Select center face
            face_in_center = 0
            distance_to_center = math.hypot(w / 2, h / 2)
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                d = math.hypot((x2 + x1 - w) / 2, (y2 + y1 - h) / 2)
                if d < distance_to_center:
                    face_in_center = i
                    distance_to_center = d

            x1, y1, x2, y2 = boxes[face_in_center]
            face_cords.append([x1, y1, x2, y2])

            # 2. Landmarks
            face_marks = self.last_face_marks
            if run_landmark:
                face_image = frame[y1:y2, x1:x2]
                if face_image.size > 0:
                    face_marks = self.face_landmark.get_landmark(face_image, (x1, y1, x2, y2))
                    face_marks = np.array(face_marks)
                    self.last_face_marks = face_marks

            # 3. Yawn + Head Down + Attention
            if len(face_marks) == 468:
                mark_group.append(face_marks)

                # Yawn
                mouth_ratio = self.mouth.yawning_ratio(face_marks)
                yawn_detected = mouth_ratio > MOUTH_THRESHOLD

                # Head down
                head_down_ratio = self.mouth.head_down_ratio(face_marks)
                status = 1 if head_down_ratio > HEAD_DOWN_THRESHOLD else 0
                HEAD_STATUS[:-1] = HEAD_STATUS[1:]
                HEAD_STATUS[-1] = status
                head_down_final = np.mean(HEAD_STATUS) > 0.5

                # Distraction
                mouth_face_ratio = self.mouth.mouth_face_ratio(face_marks)
                distracted = not (FACING_LEFT_THRESHOLD <= mouth_face_ratio <= FACING_RIGHT_THRESHOLD)
                self.last_distracted_status = distracted
                self.last_yawn_status = yawn_detected
                self.last_head_down_status = head_down_final

            # 4. Eyes & Blinking
            if run_eye and len(face_marks) == 468:
                for eye_id in [0, 1]:
                    ex1, ey1, ex2, ey2 = self.eye_detector.get_eye_roi(face_marks, eye_id)
                    ex1 = max(0, min(int(ex1), w - 1))
                    ex2 = max(0, min(int(ex2), w))
                    ey1 = max(0, min(int(ey1), h - 1))
                    ey2 = max(0, min(int(ey2), h))

                    if ex2 > ex1 and ey2 > ey1:
                        eye_crop = frame[ey1:ey2, ex1:ex2]
                        if eye_crop.size > 0:
                            eye_marks, iris_marks = self.eye_detector.get_landmark(eye_crop, (ex1, ey1, ex2, ey2), eye_id)
                            mark_group.append(np.array(iris_marks))
                            
                            ratio = self.eye_detector.blinking_ratio(eye_marks, eye_id)
                            if eye_id == 0:
                                LEFT_EYE_STATUS[:-1] = LEFT_EYE_STATUS[1:]
                                LEFT_EYE_STATUS[-1] = 1 if ratio > LEFT_EYE_THRESHOLD else 0
                            else:
                                RIGHT_EYE_STATUS[:-1] = RIGHT_EYE_STATUS[1:]
                                RIGHT_EYE_STATUS[-1] = 1 if ratio > RIGHT_EYE_THRESHOLD else 0

                left_eye_closed = np.mean(LEFT_EYE_STATUS) < 0.1
                right_eye_closed = np.mean(RIGHT_EYE_STATUS) < 0.1
                self.both_eyes_closed = left_eye_closed and right_eye_closed
                sleep_detected = self.both_eyes_closed
            else:
                sleep_detected = self.both_eyes_closed

            # 5. Smoking / Calling (YOLOv8)
            if run_smk:
                smk_call_result = self.smoking_detector.inference(frame, mono=False)
                phone_seen = False
                cigrate_seen = False
                smk_call_cords = []

                if len(smk_call_result) > 0:
                    for i in range(len(smk_call_result)):
                        cls_id = int(smk_call_result[i][5])
                        conf = smk_call_result[i][4]
                        if not self.is_near_driver_face(smk_call_result[i][:4], face_cords[0], w, h):
                            continue

                        if cls_id == 0 and conf > PHONE_THRESHOLD:
                            phone_seen = True
                        if cls_id == 1 and conf > SMOKE_THRESHOLD:
                            cigrate_seen = True
                        
                        smk_call_cords.append([
                            int(smk_call_result[i][0]), int(smk_call_result[i][1]),
                            int(smk_call_result[i][2]), int(smk_call_result[i][3])
                        ])
                
                phone_detected = phone_seen
                smoking_detected = cigrate_seen
                self.last_phone_status = phone_detected
                self.last_smoke_status = smoking_detected
                self.last_smk_call_cords = smk_call_cords
        else:
            # Reset states when face is lost
            self.last_face_marks = []
            self.last_smk_call_cords = []
            self.last_phone_status = False
            self.last_smoke_status = False
            self.last_distracted_status = False
            self.last_yawn_status = False
            self.last_head_down_status = False
            self.both_eyes_closed = False
            distracted = False
            yawn_detected = False
            head_down_final = False

        # Update scoring / Penalties
        self.distracted = distracted
        self.drowsy = sleep_detected
        self.yawn = yawn_detected
        self.smoking = smoking_detected
        self.phone = phone_detected
        self.head_down = head_down_final

        if distracted:      self.safe_value = min(self.safe_value + DISTRACT_PENALTY, 100.0)
        if sleep_detected:  self.safe_value = min(self.safe_value + SLEEP_PENALTY,    100.0)
        if yawn_detected:   self.safe_value = min(self.safe_value + YAWN_PENALTY,     100.0)
        if smoking_detected:self.safe_value = min(self.safe_value + SMK_PENALTY,      100.0)
        if phone_detected:  self.safe_value = min(self.safe_value + CALL_PENALTY,     100.0)
        if head_down_final: self.safe_value = min(self.safe_value + HEAD_DOWN_PENALTY, 100.0)
        if not face_cords:  self.safe_value = min(self.safe_value + NO_FACE_PENALTY,   100.0)

        if (not distracted and not sleep_detected and not yawn_detected
                and not head_down_final and not smoking_detected
                and not phone_detected and face_cords):
            self.safe_value = max(self.safe_value + RESTORE_CREDIT, 0.0)

        # ─── DRAW ON FRAME ───
        # Draw face box
        if len(face_cords) > 0:
            x1, y1, x2, y2 = face_cords[0]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw landmarks/iris points
        if DRAW_LANDMARKS:
            for group in mark_group:
                for pt in group:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, (0, 255, 255), -1)

        # Draw smoking/phone boxes
        for c in smk_call_cords:
            cv2.rectangle(frame, (c[0], c[1]), (c[2], c[3]), (0, 165, 255), 2)

        # Top status bar
        cv2.rectangle(frame, (0, 0, w, 40), (0, 0, 0), -1)
        r = min(self.safe_value / 50.0, 1.0)
        g = min(1.0, (100.0 - self.safe_value) / 50.0)
        color = (0, int(g * 255), int(r * 255))
        
        status_text = "Driver OK" if self.safe_value < 33 else "Warning!" if self.safe_value < 66 else "DANGER!"
        if not face_cords:
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

        return frame

    def transform_to_square(self, boxes, scale=1.0):
        xmins, ymins, xmaxs, ymaxs = np.split(boxes, 4, axis=1)
        w = xmaxs - xmins
        h = ymaxs - ymins
        center_x = np.floor_divide(xmins + xmaxs, 2)
        center_y = np.floor_divide(ymins + ymaxs, 2)
        margin = np.floor_divide(np.maximum(h, w) * scale, 2)
        return np.concatenate((center_x - margin, center_y - margin, center_x + margin, center_y + margin), axis=1)

    def clip_boxes(self, boxes, margins):
        left, top, right, bottom = margins
        boxes[:, 0] = np.maximum(boxes[:, 0], left)
        boxes[:, 1] = np.maximum(boxes[:, 1], top)
        boxes[:, 2] = np.minimum(boxes[:, 2], right)
        boxes[:, 3] = np.minimum(boxes[:, 3], bottom)
        return boxes

    def is_near_driver_face(self, det_box, face_box, frame_w, frame_h):
        dx1, dy1, dx2, dy2 = [float(v) for v in det_box]
        fx1, fy1, fx2, fy2 = [float(v) for v in face_box]

        face_w = fx2 - fx1
        face_h = fy2 - fy1
        if face_w <= 0 or face_h <= 0:
            return False

        roi_x1 = max(0.0, fx1 - 1.25 * face_w)
        roi_y1 = max(0.0, fy1 - 0.75 * face_h)
        roi_x2 = min(float(frame_w), fx2 + 1.25 * face_w)
        roi_y2 = min(float(frame_h), fy2 + 1.75 * face_h)

        cx = (dx1 + dx2) / 2.0
        cy = (dy1 + dy2) / 2.0
        return roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2

def main():
    app = PCDMSApp()
    
    # Open webcam (device 0 is usually default laptop camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("\n" + "="*50)
    print("DMS Webcam Runner active. Press 'q' or 'ESC' to exit.")
    print("="*50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally for a natural mirror view
        frame = cv2.flip(frame, 1)

        # Process and draw overlays
        processed_frame = app.process_frame(frame)

        # Display window
        cv2.imshow("DMS PC Runner (TFLite)", processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or Q key to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Web camera closed. Exited safely.")

if __name__ == "__main__":
    main()
