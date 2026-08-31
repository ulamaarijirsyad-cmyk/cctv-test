import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================
# URL CCTV PANTAU SEMAR
# ==========================================
url = "https://livepantau.semarangkota.go.id/75722eec-3065-4f02-b1a3-c6ec259dc52b/video1_stream.m3u8"

# ==========================================
# LOAD MODEL YOLO
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
print()
print("========================================")
print("       AI CCTV + ROI MONITORING")
print("========================================")
print()
print("CARA MENGGUNAKAN:")
print("1. Klik 4 titik pada area yang ingin dihitung")
print("2. Setelah 4 titik, ROI akan aktif")
print("3. Tekan R untuk menggambar ulang")
print("4. Tekan Q untuk keluar")
print()

# ==========================================
# VARIABLE ROI
# ==========================================
roi_points = []
roi_polygon = None


# ==========================================
# FUNGSI MOUSE
# ==========================================
def mouse_callback(event, x, y, flags, param):

    global roi_points
    global roi_polygon

    # Klik kiri
    if event == cv2.EVENT_LBUTTONDOWN:

        # Jangan tambah lebih dari 4 titik
        if len(roi_points) < 4:

            roi_points.append((x, y))

            print(
                f"Titik {len(roi_points)}: "
                f"X={x}, Y={y}"
            )

            # Kalau sudah 4 titik
            if len(roi_points) == 4:

                roi_polygon = np.array(
                    roi_points,
                    dtype=np.int32
                )

                print()
                print("ROI AKTIF!")
                print("Area sekarang sedang dihitung.")
                print()


# ==========================================
# BUAT WINDOW
# ==========================================
window_name = "AI CCTV - ROI Monitoring"

cv2.namedWindow(window_name)

cv2.setMouseCallback(
    window_name,
    mouse_callback
)


# ==========================================
# LOOP CCTV
# ==========================================
while True:

    ret, frame = cap.read()

    if not ret:
        print("Gagal mengambil frame")
        break

    # ======================================
    # GAMBAR TITIK ROI
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
    # GAMBAR GARIS ROI SEMENTARA
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
    # ROI SUDAH LENGKAP
    # ======================================
    if roi_polygon is not None:

        # Gambar polygon
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

                # Harus punya ID
                if box.id is None:
                    continue

                track_id = int(box.id[0])

                # Confidence
                confidence = float(box.conf[0])

                # Koordinat bounding box
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # ==================================
                # TITIK TENGAH BAWAH PERSON
                # ==================================
                center_x = int((x1 + x2) / 2)
                center_y = int(y2)

                person_point = (
                    center_x,
                    center_y
                )

                # ==================================
                # CEK APAKAH PERSON ADA DI ROI
                # ==================================
                inside = cv2.pointPolygonTest(
                    roi_polygon,
                    person_point,
                    False
                )

                # ==================================
                # HANYA HITUNG PERSON DI DALAM ROI
                # ==================================
                if inside >= 0:

                    current_ids.add(track_id)

                    # Kotak person
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # Titik kaki / titik tengah bawah
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
                        f"ID {track_id} {confidence:.2f}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

                else:

                    # Person di luar ROI
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (100, 100, 100),
                        1
                    )

        # ==================================
        # JUMLAH PERSON DI DALAM ROI
        # ==================================
        person_count = len(current_ids)

        # ==================================
        # STATUS KERAMAIAN SEMENTARA
        # ==================================
        if person_count <= 5:
            status = "SEPI"

        elif person_count <= 15:
            status = "NORMAL"

        elif person_count <= 30:
            status = "RAMAI"

        else:
            status = "SANGAT RAMAI"

        # ==================================
        # INFO DI LAYAR
        # ==================================
        cv2.putText(
            frame,
            f"AREA PERSON: {person_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 255),
            3
        )

        cv2.putText(
            frame,
            f"STATUS: {status}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            3
        )

    else:

        # ==================================
        # PETUNJUK MEMBUAT ROI
        # ==================================
        cv2.putText(
            frame,
            f"PILIH 4 TITIK ROI ({len(roi_points)}/4)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            3
        )

        cv2.putText(
            frame,
            "Klik 4 titik area yang ingin dihitung",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    # ======================================
    # TAMPILKAN CCTV
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

    # R = reset ROI
    if key == ord("r"):

        roi_points = []
        roi_polygon = None

        print()
        print("ROI DI-RESET!")
        print("Silakan klik 4 titik lagi.")
        print()


# ==========================================
# SELESAI
# ==========================================
cap.release()

cv2.destroyAllWindows()

print()
print("Program selesai.")