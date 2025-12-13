# camera.py
import cv2
import threading
import time

from detector import DrowsinessDetector

class VideoCamera:
    def __init__(self, src=0, predictor_path="models/shape_predictor_68_face_landmarks.dat"):
        # open webcam
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam. Check camera index or permissions.")
        self.detector = DrowsinessDetector(predictor_path=predictor_path)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        # start background thread
        t = threading.Thread(target=self._capture_loop, daemon=True)
        t.start()

        # track current drowsiness
        self.drowsy = False
        self.ear = 0.0

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            annotated, ear_val, drowsy = self.detector.detect_and_annotate(frame)
            with self.lock:
                self.frame = annotated.copy()
                self.drowsy = drowsy
                self.ear = ear_val if ear_val is not None else 0.0
            time.sleep(0.01)  # small sleep to yield

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            ret, jpeg = cv2.imencode('.jpg', self.frame)
            return jpeg.tobytes()

    def get_status(self):
        with self.lock:
            return {"drowsy": self.drowsy, "ear": float(self.ear)}

    def release(self):
        self.running = False
        time.sleep(0.2)
        self.cap.release()
