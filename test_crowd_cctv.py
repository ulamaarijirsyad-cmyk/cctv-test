import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# ==========================================
# URL CCTV PANTAU SEMAR
# ==========================================
url = "https://livepantau.semarangkota.go.id/75722eec-3065-4f02-b1a3-c6ec259dc52b/video1_stream.m3u8"

# ==========================================
# LOAD YOLO
# ==========================================
model = YOLO("yolo11n.pt")

# ==========================================
# BUKA CCTV
# ==========================================
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("GAGAL membuka CCTV")
    exit()

print("CCTV TERHUBUNG!")
print("YOLO + TRACKING + CROWD MONITORING AKTIF!")
print()
print("Klik 4 titik untuk menentukan AREA.")
print("Tekan R untuk menggambar ulang.")
print("Tekan Q untuk keluar.")
print()

# ==========================================
# ROI
# ==========================================
roi_points = []
roi_polygon = None


# ==========================================
# HISTORY JUMLAH ORANG
# ==========================================
count_history = deque(maxlen=30)


# ==========================================
# MOUSE
# ==========================================
def mouse_callback(event, x, y, flags, param):

    global roi_points
    global roi_polygon

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(roi_points) < 4:

            roi_points.append((x, y))

            print(
                f"Titik {len(roi_points)}: "
                f"X={x}, Y={y}"
            )

            if len(roi_points) == 4:

                roi_polygon = np.array(
                    roi_points,
                    dtype=np.int32
                )

                print()
                print("ROI AKTIF!")
                print()


# ==========================================
# WINDOW
# ==========================================
window_name = "AI CCTV - Crowd Monitoring"

cv2.namedWindow(window_name)
cv2.setMouseCallback(
    window_name,
    mouse_callback
)


# ==========================================
# LOOP
# ==========================================
while True:

    ret, frame = cap.read()

    if not ret:
        print("Gagal mengambil frame")
        break

    # ======================================
    # GAMBAR TITIK
    # ======================================
    for i, point in enumerate(roi_points):

        cv2.circle(
            frame,
            point,
            7,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            str(i + 1),
            (point[0] + 10, point[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    # ======================================
    # GARIS SEMENTARA
    # ======================================
    if len(roi_points) >= 2:

        for i in range(len(roi_points) - 1):

            cv2.line(
                frame,
                roi_points[i],
                roi_points[i + 1],
                (255, 0, 0),
                2
            )

    # ======================================
    # JIKA ROI SUDAH DIBUAT
    # ======================================
    if roi_polygon is not None:

        # Gambar ROI
        cv2.polylines(
            frame,
            [roi_polygon],
            True,
            (255, 0, 0),
            3
        )

        # ==================================
        # YOLO TRACKING
        # ==================================
        results = model.track(
            frame,
            persist=True,
            classes=[0],
            tracker="bytetrack.yaml",
            conf=0.20,
            imgsz=1280,
            verbose=False
        )

        current_ids = set()

        # ==================================
        # PROSES DETEKSI
        # ==================================
        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                if box.id is None:
                    continue

                track_id = int(box.id[0])

                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # ==================================
                # TITIK KAKI PERSON
                # ==================================
                center_x = int((x1 + x2) / 2)
                bottom_y = int(y2)

                person_point = (
                    center_x,
                    bottom_y
                )

                # ==================================
                # CEK ROI
                # ==================================
                inside = cv2.pointPolygonTest(
                    roi_polygon,
                    person_point,
                    False
                )

                # ==================================
                # PERSON DALAM AREA
                # ==================================
                if inside >= 0:

                    current_ids.add(track_id)

                    # Bounding box
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # Titik kaki
                    cv2.circle(
                        frame,
                        person_point,
                        5,
                        (0, 255, 0),
                        -1
                    )

                    # ID
                    cv2.putText(
                        frame,
                        f"ID {track_id}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

                # ==================================
                # PERSON DI LUAR ROI
                # ==================================
                else:

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (100, 100, 100),
                        1
                    )

        # ==================================
        # JUMLAH SAAT INI
        # ==================================
        current_count = len(current_ids)

        # Masukkan ke history
        count_history.append(current_count)

        # ==================================
        # RATA-RATA
        # ==================================
        if len(count_history) > 0:

            average_count = sum(
                count_history
            ) / len(count_history)

        else:

            average_count = 0

        # ==================================
        # STATUS KERAMAIAN
        # ==================================
        if average_count <= 5:

            status = "SEPI"

        elif average_count <= 15:

            status = "NORMAL"

        elif average_count <= 30:

            status = "RAMAI"

        else:

            status = "SANGAT RAMAI"

        # ==================================
        # TAMPILKAN DATA
        # ==================================
        cv2.putText(
            frame,
            f"ORANG SAAT INI : {current_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"RATA-RATA : {average_count:.1f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"STATUS : {status}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )

    # ======================================
    # BELUM ADA ROI
    # ======================================
    else:

        cv2.putText(
            frame,
            f"BUAT ROI ({len(roi_points)}/4)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            3
        )

        cv2.putText(
            frame,
            "Klik 4 titik area",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    # ======================================
    # TAMPILKAN
    # ======================================
    cv2.imshow(
        window_name,
        frame
    )

    # ======================================
    # KEYBOARD
    # ======================================
    key = cv2.waitKey(1) & 0xFF

    # Q = keluar
    if key == ord("q"):
        break

    # R = reset ROI + history
    if key == ord("r"):

        roi_points = []
        roi_polygon = None

        count_history.clear()

        print()
        print("ROI DI-RESET!")
        print("Klik 4 titik lagi.")
        print()


# ==========================================
# SELESAI
# ==========================================
cap.release()
cv2.destroyAllWindows()

print()
print("Program selesai.")