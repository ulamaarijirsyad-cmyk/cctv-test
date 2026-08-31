import cv2
from ultralytics import YOLO

# ==========================================
# CCTV PANTAU SEMAR
# ==========================================
url = "https://livepantau.semarangkota.go.id/75722eec-3065-4f02-b1a3-c6ec259dc52b/video1_stream.m3u8"

# ==========================================
# LOAD MODEL
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
print("YOLO + TRACKING AKTIF!")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Gagal mengambil frame")
        break

    # ==========================================
    # YOLO TRACKING
    # ==========================================
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

    # ==========================================
    # PROSES HASIL
    # ==========================================
    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            # Hanya proses yang punya tracking ID
            if box.id is None:
                continue

            track_id = int(box.id[0])

            current_ids.add(track_id)

            # Confidence
            confidence = float(box.conf[0])

            # Koordinat
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # ==================================
            # KOTAK
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
                f"ID {track_id} {confidence:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    # ==========================================
    # JUMLAH ORANG YANG SEDANG TER-TRACK
    # ==========================================
    person_count = len(current_ids)

    cv2.putText(
        frame,
        f"TRACKED PERSON: {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        3
    )

    # ==========================================
    # TAMPILKAN CCTV
    # ==========================================
    cv2.imshow(
        "AI CCTV - Pantau Semar",
        frame
    )

    # Q = keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()