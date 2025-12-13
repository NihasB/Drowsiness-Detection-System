# app.py
import io
import json
import threading
import time
from flask import Flask, render_template, Response, jsonify

from camera import VideoCamera

# optional import for keras model
try:
    from tensorflow.keras.models import load_model
    HAS_KERAS = True
except Exception:
    HAS_KERAS = False

CAM_INDEX = 0
PREDICTOR_PATH = "models/shape_predictor_68_face_landmarks.dat"
EAR_SEQ_MODEL = "models/ear_seq_model.h5"

app = Flask(__name__)
camera = VideoCamera(src=CAM_INDEX, predictor_path=PREDICTOR_PATH)

# load keras model if available
ear_seq_model = None
if HAS_KERAS:
    try:
        from tensorflow.keras.models import load_model
        ear_seq_model = load_model(EAR_SEQ_MODEL)
        app.logger.info("Loaded ear_seq_model.h5")
    except Exception as e:
        app.logger.info("No Keras model loaded: %s" % e)


def gen_frames():
    """Yield frames in multipart format for MJPEG streaming."""
    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    # return JSON with latest EAR & drowsy boolean
    st = camera.get_status()
    if ear_seq_model is not None:
        seq = camera.detector.get_sequence()
        # model predicts probability of drowsy
        try:
            prob = float(ear_seq_model.predict(seq, verbose=0)[0][0])
        except Exception:
            prob = 0.0
        st["drowsy_prob_model"] = prob
        # you can combine model + simple threshold if desired
        st["drowsy_model"] = prob > 0.5
    return jsonify(st)


@app.route("/shutdown")
def shutdown():
    camera.release()
    func = request.environ.get('werkzeug.server.shutdown')
    if func:
        func()
    return "Server shutting down..."


if __name__ == "__main__":
    try:
        app.run(debug=True, threaded=True)
    finally:
        camera.release()
