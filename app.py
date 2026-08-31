import cv2
import threading
import time

from flask import Flask, Response, jsonify, send_from_directory
from ultralytics import YOLO


# ==========================================
# FLASK
# ==========================================

app = Flask(__name__)


# ==========================================
# CCTV PANTAU SEMAR
# ==========================================

CCTV_URL = "https://livepantau.semarangkota.go.id/0b44f5b2-780d-4f86-9ecd-801041b8925a/video1_stream.m3u8"


# ==========================================
# YOLO
# ==========================================

model = YOLO("yolo11n.pt")


# ==========================================
# DATA GLOBAL
# ==========================================

output_frame = None

current_count = 0
average_count = 0
crowd_status = "SEPI"

lock = threading.Lock()


# ==========================================
# CCTV + YOLO
# ==========================================

def process_cctv():

    global output_frame
    global current_count
    global average_count
    global crowd_status

    cap = cv2.VideoCapture(CCTV_URL)

    # Buffer kecil supaya tidak terlalu tertinggal
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():

        print("GAGAL membuka CCTV")

        return

    print("CCTV TERHUBUNG!")
    print("YOLO WEB SERVER AKTIF!")

    history = []

    while True:

        ret, frame = cap.read()

        if not ret:

            print("CCTV terputus, mencoba reconnect...")

            cap.release()

            time.sleep(2)

            cap = cv2.VideoCapture(CCTV_URL)

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            continue


        # ==================================
        # YOLO DETECTION
        # ==================================

        results = model.predict(
            frame,
            classes=[0],       # hanya PERSON
            conf=0.05,         # lebih sensitif
            imgsz=1280,
            verbose=False
        )


        # ==================================
        # COUNT PERSON
        # ==================================

        person_count = 0


        for result in results:

            if result.boxes is None:
                continue


            for box in result.boxes:

                person_count += 1


                # Koordinat bounding box

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )


                confidence = float(box.conf[0])


                # ==================================
                # BOUNDING BOX
                # ==================================

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )


                # ==================================
                # LABEL
                # ==================================

                cv2.putText(
                    frame,
                    f"PERSON {confidence:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )


        # ==================================
        # UPDATE COUNT
        # ==================================

        current_count = person_count


        # ==================================
        # HISTORY
        # ==================================

        history.append(current_count)


        # Simpan 30 frame terakhir

        if len(history) > 30:

            history.pop(0)


        average_count = (
            sum(history) / len(history)
        )


        # ==================================
        # STATUS ANTREAN
        # ==================================

        if average_count <= 5:

            crowd_status = "SEPI"

        elif average_count <= 15:

            crowd_status = "NORMAL"

        elif average_count <= 30:

            crowd_status = "RAMAI"

        else:

            crowd_status = "SANGAT RAMAI"


        # ==================================
        # TULIS INFORMASI KE VIDEO
        # ==================================

        cv2.putText(
            frame,
            f"ORANG: {current_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            3
        )


        cv2.putText(
            frame,
            f"STATUS: {crowd_status}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            3
        )


        # ==================================
        # ENCODE JPEG
        # ==================================

        success, encoded = cv2.imencode(
            ".jpg",
            frame
        )

        if not success:
            continue


        with lock:

            output_frame = encoded.tobytes()


# ==========================================
# VIDEO STREAM
# ==========================================

def generate_video():

    global output_frame

    while True:

        with lock:

            frame = output_frame

        if frame is None:

            time.sleep(0.05)

            continue


        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )


# ==========================================
# BERANDA
# ==========================================

@app.route("/")
def index():

    return send_from_directory(
        ".",
        "index.html"
    )


# ==========================================
# HALAMAN MONITORING CCTV
# ==========================================

@app.route("/monitoring")
def monitoring():

    return send_from_directory(
        ".",
        "monitoring.html"
    )


# ==========================================
# LIVE VIDEO
# ==========================================

@app.route("/video")
def video():

    return Response(
        generate_video(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ==========================================
# API STATUS
# ==========================================

@app.route("/api/status")
def status():

    return jsonify({

        "person_count": current_count,

        "average": round(
            average_count,
            1
        ),

        "status": crowd_status

    })


# ==========================================
# START SERVER
# ==========================================

import os

if __name__ == "__main__":

    thread = threading.Thread(
        target=process_cctv,
        daemon=True
    )

    thread.start()

    print("SAMSAT AI CCTV SERVER AKTIF")

    port = int(os.environ.get("PORT", "8080"))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True
    )