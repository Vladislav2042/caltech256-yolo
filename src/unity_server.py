# yolo_server_unity.py — принимает кадры от Unity, делает YOLO, отдаёт JSON
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime

app = Flask (__name__)

# Загружаем твою обученную модель
model = YOLO("runs/detect/caltech256_ultra_fast3/weights/best.pt")
print("Модель YOLO загружена — готов к приёму кадров от Unity!")

@app.route('/')
def index():
    return "<h1>YOLO-сервер для Unity работает!</h1>Отправляй кадры на /upload"

@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Получаем JPEG от Unity
        file = request.files['image']
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # YOLO детекция
        results = model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            cls_name = results.names[int(box.cls)]

            if conf > 0.4:
                detections.append({
                    "class": cls_name,
                    "confidence": round(conf, 3),
                    "center_x": (x1 + x2) // 2,
                    "center_y": (y1 + y2) // 2,
                    "width": x2 - x1,
                    "height": y2 - y1
                })

        return jsonify({
            "status": "ok",
            "objects": detections,
            "count": len(detections),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("ЗАПУСКАЮ YOLO-СЕРВЕР ДЛЯ UNITY")
    print("Unity должен слать кадры на: http://localhost:5000/upload")
    print("Получать ответ можно с того же адреса")
    app.run(host="127.0.0.1", port=5000, threaded=True)