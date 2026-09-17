#!/usr/bin/env python3

"""
Copyright © 2021 Patrick Levin
Copyright 2022-2024 NXP

SPDX-License-Identifier: MIT
Original Source: https://github.com/patlevin/face-detection-tflite

This script define class of face detection used in DMS demo
"""
import time
import numpy as np
import cv2

# ── Backend imports (soft) ─────────────────────────────────────────────────────
# stai_mpu  : STM32 NPU driver — only available on target hardware
# tflite    : works on any platform (Windows, Linux, ARM)
try:
    from stai_mpu import stai_mpu_network
    _HAS_STAI = True
except ImportError:
    _HAS_STAI = False

try:
    import tflite_runtime.interpreter as tflite
    _Interpreter = tflite.Interpreter
except ImportError:
    try:
        import tensorflow as tf
        _Interpreter = tf.lite.Interpreter
    except ImportError:
        _Interpreter = None

# score limit is 100 in mediapipe and leads to overflows with IEEE 754 floats
# this lower limit is safe for use with the sigmoid functions and float32
RAW_SCORE_LIMIT = 80

# NMS similarity threshold
INPUT_SIZE     = 128
CONF_THRESHOLD = 0.8
IOU_THRESHOLD  = 0.8
MIN_SUPPRESSION_THRESHOLD = 0.5


def sigmoid(x):
    """Apply the sigmoid function on the input x"""
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))


class FaceDetector:
    """The class to do face dectcion"""

    def __init__(self, model_path, threshold):
        """
        Creates an instance of the face detector

        Arguments:
        model_path -- the path to the model
        inf_device -- the inference device, CPU or NPU
        platform -- the plaform that running this demo
        threshold -- the threshold for confidence scores
        """
        self.threshold = threshold
        self.use_nb_model = False

        # Store basic parameters
        self._model_file = model_path
        print(f"[INFO] NN model used: {self._model_file}")

        # Load and initialize the model (with or without hardware acceleration)
        if self._model_file.endswith(".nb"):
            if not _HAS_STAI:
                raise ImportError(
                    "stai_mpu not available on this platform. "
                    "Use a .tflite model instead.")
            self.use_nb_model = True
            self.stai_mpu_model = stai_mpu_network(model_path=self._model_file, use_hw_acceleration=True)

            input_info = self.stai_mpu_model.get_input_infos()[0]
            self.input_shape = input_info.get_shape()

            # Optional debug (do once)
            #print("NB INPUT SHAPE:", self.input_shape)

            output_infos = self.stai_mpu_model.get_output_infos()
            # for i, info in enumerate(output_infos):
            #     print(f"NB OUTPUT {i}: shape={info.get_shape()}, dtype={info.get_dtype()}")

            self.anchors = self._generate_anchors()
            self.prev_box = None
            self.alpha = 0.5

            # pre-allocate reusable buffers — avoids new allocation every frame
            self._resized_buf = np.zeros(
                (self.input_shape[1], self.input_shape[2], 3), dtype=np.uint8)
            self._input_buf = np.zeros(
                (1, self.input_shape[1], self.input_shape[2], 3), dtype=np.float32)

        elif self._model_file.endswith(".tflite"):
            if _Interpreter is None:
                raise ImportError(
                    "Neither tflite_runtime nor tensorflow is installed. "
                    "Run: pip install tflite-runtime")
            # inf_device is CPU
            self.interpreter = _Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()

            # model warm up
            time_start = time.time()
            self.interpreter.invoke()
            time_end = time.time()
            print("face detection model warm up time:")
            print((time_end - time_start) * 1000, " ms")

            self.input_index = self.interpreter.get_input_details()[0]["index"]
            self.input_shape = self.interpreter.get_input_details()[0]["shape"]
            self.bbox_index = self.interpreter.get_output_details()[1]["index"]
            self.score_index = self.interpreter.get_output_details()[0]["index"]

            # (reference: modules/face_detection/face_detection_short_range_common.pbtxt)
            self.ssd_opts = {
                "num_layers": 4,
                "input_size_height": 128,
                "input_size_width": 128,
                "anchor_offset_x": 0.5,
                "anchor_offset_y": 0.5,
                "strides": [8, 16, 16, 16],
                "interpolated_scale_aspect_ratio": 1.0,
            }

            self.anchors = self._ssd_generate_anchors(self.ssd_opts)

    def _pre_processing(self, input_data):
        """Preprocessing the input_data for the model"""
        input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)

        if self.use_nb_model:
            # resize into pre-allocated buffer — no new allocation
            cv2.resize(input_data,
                       (self.input_shape[2], self.input_shape[1]),
                       dst=self._resized_buf)

            # normalize into pre-allocated input buffer in-place — no new allocation
            # THIS IS CRITICAL (don't touch again)
            np.multiply(self._resized_buf, 1.0 / 127.5,
                        out=self._input_buf[0], casting='unsafe')
            self._input_buf[0] -= 1.0

            return self._input_buf
        else:
            input_data = cv2.resize(input_data, self.input_shape[1:3]).astype(np.float32)
            input_data = (input_data[np.newaxis, :, :, :] - 128) / 128.0
            return input_data

    def detect(self, img):
        """Detect the face from img and return the bounding box"""
        input_data = self._pre_processing(img)

        if self.use_nb_model:
            self.stai_mpu_model.set_input(0, input_data)
            self.stai_mpu_model.run()

            out0 = self.stai_mpu_model.get_output(0)
            out1 = self.stai_mpu_model.get_output(1)

            if out0.shape[-1] == 1:
                raw_scores = out0
                raw_boxes  = out1
            else:
                raw_scores = out1
                raw_boxes  = out0

            #scores = 1.0 / (1.0 + np.exp(-np.clip(raw_scores[0], -50, 50)))
            scores = sigmoid(raw_scores[0]).reshape(-1)
            boxes  = self._decode_boxes(raw_boxes[0],self.anchors)

            # release output buffers now — all data copied into scores and boxes
            raw_scores = None
            raw_boxes  = None
            out0       = None
            out1       = None

            # 🔥 match old TFLite behavior

            bw = boxes[:, 2] - boxes[:, 0]
            bh = boxes[:, 3] - boxes[:, 1]

            cx = boxes[:, 0] + bw / 2
            cy = boxes[:, 1] + bh / 2

            bw *= 1.3
            bh *= 1.6
            cy -= 0.15 * bh

            boxes[:, 0] = cx - bw / 2
            boxes[:, 1] = cy - bh / 2
            boxes[:, 2] = cx + bw / 2
            boxes[:, 3] = cy + bh / 2

            # keep normalized range
            boxes = np.clip(boxes, 0, 1)

            # filter
            mask = scores > self.threshold
            boxes = boxes[mask]
            scores = scores[mask]

            # top-k
            if len(scores) > 0:
                #top_idx = np.argsort(scores)[-5:]
                k = min(5, len(scores))
                top_idx = np.argpartition(scores, -k)[-k:]
                boxes = boxes[top_idx]
                scores = scores[top_idx]

            # NMS (use your existing or keep simple)
            if len(scores) > 0:
                best = np.argmax(scores)
                box = boxes[best]
                score = scores[best]

                #box = np.clip(box, 0, 1)

                # 🔥 smoothing
                if self.prev_box is not None:
                    box = self.alpha * self.prev_box + (1 - self.alpha) * box

                self.prev_box = box

                return np.array([box])

            # no face detected — clear prev_box so it doesn't hold stale data
            self.prev_box = None
            return np.array([])
        else:
            self.interpreter.set_tensor(self.input_index, input_data)
            self.interpreter.invoke()
            raw_boxes = self.interpreter.get_tensor(self.bbox_index)
            raw_scores = self.interpreter.get_tensor(self.score_index)

            boxes = self._decode_boxes(raw_boxes,anchors=None)
            scores = self._get_sigmoid_scores(raw_scores)

            score_above_threshold = scores > self.threshold
            filtered_boxes = boxes[np.argwhere(score_above_threshold)[:, 1], :]
            filtered_scores = scores[score_above_threshold]

            output_boxes = np.array(
                self._non_maximum_suppression(
                    filtered_boxes, filtered_scores, MIN_SUPPRESSION_THRESHOLD
                )
            )

            return output_boxes

    def _overlap_similarity(self, box1, box2):
        """Return intersection-over-union similarity of two bounding boxes"""
        if box1 is None or box2 is None:
            return 0
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        x3_min = max(x1_min, x2_min)
        x3_max = min(x1_max, x2_max)
        y3_min = max(y1_min, y2_min)
        y3_max = min(y1_max, y2_max)
        intersect_area = (x3_max - x3_min) * (y3_max - y3_min)
        denominator = box1_area + box2_area - intersect_area
        return intersect_area / denominator if denominator > 0.0 else 0.0

    def _non_maximum_suppression(self, boxes, scores, min_suppression_threshold):
        """Return only the most significant detections"""
        candidates_list = []
        for i in range(np.size(boxes, 0)):
            candidates_list.append((boxes[i], scores[i]))
        candidates_list = sorted(candidates_list, key=lambda x: x[1], reverse=True)
        kept_list = []
        for sorted_boxes, sorted_scores in candidates_list:
            suppressed = False
            for kept in kept_list:
                similarity = self._overlap_similarity(kept, sorted_boxes)
                if similarity > min_suppression_threshold:
                    suppressed = True
                    break
            if not suppressed:
                kept_list.append(sorted_boxes)
        return kept_list

    def nms(self, boxes, scores, iou_thresh):
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

    def _decode_boxes(self, raw_boxes, anchors):
        """
        Simplified version of
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
        """

        if self.use_nb_model:
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
        else:
            # width == height so scale is the same across the board
            scale = self.input_shape[1]
            num_points = raw_boxes.shape[-1] // 2
            # scale all values (applies to positions, width, and height alike)
            boxes = raw_boxes.reshape(-1, num_points, 2) / scale
            # adjust center coordinates and key points to anchor positions
            boxes[:, 0] += self.anchors
            for i in range(2, num_points):
                boxes[:, i] += self.anchors
            # convert x_center, y_center, w, h to xmin, ymin, xmax, ymax
            center = np.array(boxes[:, 0])
            half_size = boxes[:, 1] / 2
            boxes[:, 0] = center - half_size
            boxes[:, 1] = center + half_size

            # only need boxes xmin, ymin, xmax, ymax
            boxes = boxes[:, 0:2, :].reshape(-1, 4)
            return boxes

    def _generate_anchors(self):
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
    
    def _get_sigmoid_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        """
        Extracted loop from ProcessCPU (line 327) in
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
        """
        # just a single class ("face"), which simplifies this a lot
        # 1) thresholding; adjusted from 100 to 80, since sigmoid of [-]100
        #    causes overflow with IEEE single precision floats (max ~10e38)
        raw_scores[raw_scores < -RAW_SCORE_LIMIT] = -RAW_SCORE_LIMIT
        raw_scores[raw_scores > RAW_SCORE_LIMIT] = RAW_SCORE_LIMIT
        # 2) apply sigmoid function on clipped confidence scores
        return sigmoid(raw_scores)

    def _ssd_generate_anchors(self, opts: dict) -> np.ndarray:
        """
        This is a trimmed down version of the C++ code; all irrelevant parts
        have been removed.
        (reference: mediapipe/calculators/tflite/ssd_anchors_calculator.cc)
        """
        layer_id = 0
        num_layers = opts["num_layers"]
        strides = opts["strides"]
        assert len(strides) == num_layers
        input_height = opts["input_size_height"]
        input_width = opts["input_size_width"]
        anchor_offset_x = opts["anchor_offset_x"]
        anchor_offset_y = opts["anchor_offset_y"]
        interpolated_scale_aspect_ratio = opts["interpolated_scale_aspect_ratio"]
        anchors = []
        while layer_id < num_layers:
            last_same_stride_layer = layer_id
            repeats = 0
            while (
                last_same_stride_layer < num_layers
                and strides[last_same_stride_layer] == strides[layer_id]
            ):
                last_same_stride_layer += 1
                # aspect_ratios are added twice per iteration
                repeats += 2 if interpolated_scale_aspect_ratio == 1.0 else 1
            stride = strides[layer_id]
            feature_map_height = input_height // stride
            feature_map_width = input_width // stride
            for y in range(feature_map_height):
                y_center = (y + anchor_offset_y) / feature_map_height
                for x in range(feature_map_width):
                    x_center = (x + anchor_offset_x) / feature_map_width
                    for _ in range(repeats):
                        anchors.append((x_center, y_center))
            layer_id = last_same_stride_layer
        return np.array(anchors, dtype=np.float32)
