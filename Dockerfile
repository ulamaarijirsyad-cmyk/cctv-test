FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Paksa gunakan OpenCV headless
RUN pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless || true \
    && pip install --no-cache-dir opencv-python-headless

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]