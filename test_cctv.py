import cv2

url = "https://livepantau.semarangkota.go.id/75722eec-3065-4f02-b1a3-c6ec259dc52b/video1_stream.m3u8"

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("GAGAL membuka CCTV")
    exit()

print("CCTV BERHASIL TERHUBUNG!")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Gagal mengambil frame")
        break

    cv2.imshow("Pantau Semar CCTV", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()