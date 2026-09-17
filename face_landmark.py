#!/usr/bin/env python3

"""
Copyright 2022-2024 NXP
SPDX-License-Identifier: BSD-3-Clause

This script define class of face landmark used in DMS demo
"""
import time
import numpy as np
import cv2

# ── Backend imports (soft) ─────────────────────────────────────────────────────
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


class FaceLandmark:
    """The class to get face landmark"""

    def __init__(self, model_path):
        """
        Creates an instance of the face landmark class

        Arguments:
        model_path -- the path to the model
        inf_device -- the inference device, CPU or NPU
        platform -- the plaform that running this demo
        """
        # Store basic parameters
        self._model_file = model_path
        self.use_nb_model = False
        print(f"[INFO] NN model used: {self._model_file}")

        if self._model_file.endswith(".nb"):
            if not _HAS_STAI:
                raise ImportError(
                    "stai_mpu not available on this platform. "
                    "Use a .tflite model instead.")
            self.use_nb_model = True
            self.stai_mpu_model = stai_mpu_network(model_path=self._model_file, use_hw_acceleration=True)

            # INPUT
            input_info = self.stai_mpu_model.get_input_infos()[0]
            self.input_shape = input_info.get_shape()
            self.input_dtype = input_info.get_dtype()

            # print("NB INPUT SHAPE:", self.input_shape)
            # print("NB INPUT DTYPE:", self.input_dtype)

            # OUTPUT
            self.output_infos = self.stai_mpu_model.get_output_infos()

            # for i, info in enumerate(self.output_infos):
            #     print(f"NB OUTPUT {i}: shape={info.get_shape()}, dtype={info.get_dtype()}")

            for i, info in enumerate(self.output_infos):
                shape = info.get_shape()
                if shape[-1] == 1404:
                    self.landmark_index = i
                else:
                    self.score_index = i
        elif self._model_file.endswith(".tflite"):
            if _Interpreter is None:
                raise ImportError(
                    "Neither tflite_runtime nor tensorflow is installed. "
                    "Run: pip install tflite-runtime")
            self.interpreter = _Interpreter(model_path=model_path)

            self.interpreter.allocate_tensors()

            # model warm up
            time_start = time.time()
            self.interpreter.invoke()
            time_end = time.time()
            print("face landmark model warm up time:")
            print((time_end - time_start) * 1000, " ms")

            self.input_index = self.interpreter.get_input_details()[0]["index"]
            self.input_shape = self.interpreter.get_input_details()[0]["shape"]
            self.landmark_index = self.interpreter.get_output_details()[1]["index"]
            self.score_index = self.interpreter.get_output_details()[0]["index"]

    def _pre_processing(self, input_data):
        """Preprocessing the input_data for the model"""
        input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
        input_data = cv2.resize(input_data, self.input_shape[1:3]).astype(np.float32)
        input_data = (input_data[np.newaxis, :, :, :] - 128) / 128.0
        #input_data = input_data.astype(self.input_dtype)
        return input_data

    def get_landmark(self, img, roi):
        """Get the face landmarks from img, return a list of all landmarks' position"""
        input_data = self._pre_processing(img)

        if self.use_nb_model:
            # set input
            self.stai_mpu_model.set_input(0, input_data)

            # run inference
            self.stai_mpu_model.run()

            # get output
            raw_landmarks = self.stai_mpu_model.get_output(self.landmark_index)[0]
            #print("RAW OUTPUT SHAPE:", raw_landmarks.shape)
        else:
            self.interpreter.set_tensor(self.input_index, input_data)
            self.interpreter.invoke()
            raw_landmarks = self.interpreter.get_tensor(self.landmark_index)[0]

        raw_landmarks = raw_landmarks.astype(np.float32)
        raw_landmarks = np.reshape(raw_landmarks, (-1, 3))
        #print("RESHAPED:", raw_landmarks.shape)

        height, width = self.input_shape[1:3]
        xmin, ymin, xmax, ymax = roi
        roi_width = xmax - xmin
        roi_height = ymax - ymin

        #print("Min:", raw_landmarks.min(), "Max:", raw_landmarks.max())

        output_landmarks = []
        #for i in range(np.size(raw_landmarks, 0)):
        for i in range(len(raw_landmarks)):
            x = int((raw_landmarks[i][0] / width) * roi_width + xmin)
            y = int((raw_landmarks[i][1] / height) * roi_height + ymin)
            output_landmarks.append([x, y])

        return output_landmarks
