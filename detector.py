# detector.py
import time
from collections import deque

import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

# indexes for the eye landmarks in the 68-point model
LEFT_EYE_IDX = list(range(36, 42))
RIGHT_EYE_IDX = list(range(42, 48))


def eye_aspect_ratio(eye):
    # eye: array of 6 (x,y) points
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    if C == 0:
        return 0.0
    ear = (A + B) / (2.0 * C)
    return ear


class DrowsinessDetector:
    """
    Wraps dlib face detector + 68-landmark predictor, calculates EAR,
    maintains temporal queue for sequence detection and triggers drowsiness alarm.
    """

    def __init__(
        self,
        predictor_path="models/shape_predictor_68_face_landmarks.dat",
        ear_threshold=0.20,
        consecutive_frames=15,
        seq_length=30,
    ):
        # dlib detectors
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

        # parameters
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames

        # state
        self.counter = 0
        self.alarm_on = False

        # for sequence (e.g., feeding to LSTM)
        self.seq_length = seq_length
        self.ear_deque = deque(maxlen=seq_length)
        # store latest values to display
        self.last_ear = 0.0

    def detect_and_annotate(self, frame):
        """
        Given a BGR frame, returns:
         - annotated_frame (BGR)
         - ear_value (float)
         - drowsy (bool)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        drowsy = False
        ear_val = None

        if len(rects) > 0:
            rect = rects[0]  # choose first face
            shape = self.predictor(gray, rect)
            coords = np.zeros((68, 2), dtype="int")
            for i in range(68):
                coords[i] = (shape.part(i).x, shape.part(i).y)

            leftEye = coords[LEFT_EYE_IDX]
            rightEye = coords[RIGHT_EYE_IDX]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            self.last_ear = ear
            ear_val = ear
            self.ear_deque.append(ear)

            # annotate eyes
            for (x, y) in np.concatenate((leftEye, rightEye)):
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # annotate EAR
            cv2.putText(
                frame,
                f"EAR: {ear:.3f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            # simple consecutive-frame thresholding
            if ear < self.ear_threshold:
                self.counter += 1
            else:
                self.counter = 0
                self.alarm_on = False

            if self.counter >= self.consecutive_frames:
                drowsy = True
                self.alarm_on = True
                cv2.putText(
                    frame,
                    "DROWSINESS ALERT!",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            # draw face rectangle
            x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
        else:
            # no face -> reset counter
            self.counter = 0
            self.alarm_on = False

        return frame, ear_val, drowsy

    def get_sequence(self):
        """
        Returns a length seq_length numpy array padded with last EAR if needed.
        """
        if len(self.ear_deque) < self.seq_length:
            # pad front with last value
            pad_len = self.seq_length - len(self.ear_deque)
            pad = [self.last_ear] * pad_len
            arr = np.array(pad + list(self.ear_deque), dtype=np.float32)
        else:
            arr = np.array(self.ear_deque, dtype=np.float32)
        return arr.reshape(1, self.seq_length, 1)  # shape for LSTM/1D conv
