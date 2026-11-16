import requests
import cv2
import numpy as np
import imutils
from ultralytics import YOLO
import os
import time
from datetime import datetime

# URL камеры (подставьте адрес своей IP-камеры)
url = "http://192.168.1.100:8080/shot.jpg"

# ЗАГРУЖАЕМ ВАШУ ОБУЧЕННУЮ МОДЕЛЬ CALTECH256
model_path = "runs/detect/caltech256_ultra_fast3/weights/best.pt"
model = YOLO(model_path)

# Папки для сохранения
SAVE_DIR = "detected_objects"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Счетчики для статистики
detection_stats = {}


def save_object_image(frame, x1, y1, x2, y2, confidence, class_name):
    """Сохраняет обнаруженный объект"""
    try:
        # Проверяем границы
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        # Извлекаем объект
        obj_img = frame[y1:y2, x1:x2]

        # Пропускаем слишком маленькие детекции
        if obj_img.size == 0 or obj_img.shape[0] < 20 or obj_img.shape[1] < 20:
            return None

        # Создаем имя файла
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{SAVE_DIR}/{class_name}_{timestamp}_{confidence:.2f}.jpg"

        # Сохраняем изображение
        cv2.imwrite(filename, obj_img)
        print(f"💾 Saved: {filename}")

        # Обновляем статистику
        if class_name in detection_stats:
            detection_stats[class_name] += 1
        else:
            detection_stats[class_name] = 1

        return filename

    except Exception as e:
        print(f"❌ Error saving {class_name} image: {e}")
        return None


def get_color_for_class(class_id):
    """Возвращает цвет для разных классов"""
    colors = [
        (0, 255, 0),  # green
        (255, 0, 0),  # blue
        (0, 255, 255),  # yellow
        (255, 0, 255),  # magenta
        (0, 0, 255),  # red
        (255, 255, 0),  # cyan
        (128, 0, 128),  # purple
        (0, 128, 128),  # teal
        (128, 128, 0),  # olive
        (128, 0, 0)  # maroon
    ]
    return colors[class_id % len(colors)]


def process_frame(frame):
    """Обрабатывает кадр с помощью обученной модели Caltech256"""
    results = model(frame, verbose=False)
    detected_objects = []

    for result in results:
        for box in result.boxes:
            # Получаем информацию о детекции
            class_id = int(box.cls.item())
            confidence = box.conf.item()
            class_name = model.names[class_id]

            # Используем все классы (не только "person")
            if confidence > 0.3:  # Более низкий порог для тестирования
                # Используем xyxy координаты (уже готовые к рисованию)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                detected_objects.append({
                    'coords': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'class_name': class_name,
                    'class_id': class_id
                })

                # Получаем цвет для класса
                color = get_color_for_class(class_id)

                # Рисуем рамку
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Добавляем текст с классом и уверенностью
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame, detected_objects


def display_stats(frame, objects, auto_save, total_saved):
    """Отображает статистику на кадре"""
    stats = [
        f"Objects: {len(objects)}",
        f"Total saved: {total_saved}",
        f"Auto-save: {'ON' if auto_save else 'OFF'}",
        f"Classes: {len(detection_stats)}",
        "Press 's' to save detections",
        "Press 'a' to toggle auto-save",
        "Press 'c' to show classes",
        "Press 'ESC' to exit"
    ]

    for i, text in enumerate(stats):
        y_position = 30 + i * 25
        # Черная обводка для лучшей читаемости
        cv2.putText(frame, text, (10, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        # Белый текст
        cv2.putText(frame, text, (10, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def show_classes_info():
    """Показывает информацию о обнаруженных классах"""
    print("\n📋 Обнаруженные классы:")
    if detection_stats:
        for class_name, count in sorted(detection_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {class_name}: {count}")
    else:
        print("  Пока не обнаружено объектов")
    print()


# Основная функция
def main():
    auto_save = False
    total_saved = 0

    print("🚀 Caltech256 Object Detection запущен!")
    print(f"📦 Модель: {model_path}")
    print(f"🎯 Доступно классов: {len(model.names)}")
    print("\nУправление:")
    print("  's' - сохранить текущие детекции")
    print("  'a' - вкл/выкл автосохранение")
    print("  'c' - показать статистику классов")
    print("  'ESC' - выход")

    while True:
        try:
            # Получаем изображение с URL
            img_resp = requests.get(url, timeout=5)
            img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
            frame = cv2.imdecode(img_arr, -1)

            # Проверяем, успешно ли получен кадр
            if frame is None:
                print("❌ Error: Failed to decode image from URL")
                time.sleep(1)
                continue

            # Изменяем размер изображения
            frame = imutils.resize(frame, width=1000)

            # Обрабатываем кадр с помощью ВАШЕЙ ОБУЧЕННОЙ МОДЕЛИ
            processed_frame, objects = process_frame(frame)

            # Авто-сохранение
            if auto_save and objects:
                for obj in objects:
                    x1, y1, x2, y2 = obj['coords']
                    if save_object_image(frame, x1, y1, x2, y2,
                                         obj['confidence'],
                                         obj['class_name']):
                        total_saved += 1

            # Отображаем статистику
            display_stats(processed_frame, objects, auto_save, total_saved)

            # Отображаем кадр
            cv2.imshow(f"Caltech256 Detection - {len(model.names)} classes", processed_frame)

            # Обработка нажатий клавиш
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Esc для выхода
                break
            elif key == ord('s'):  # 's' для сохранения
                for obj in objects:
                    x1, y1, x2, y2 = obj['coords']
                    if save_object_image(frame, x1, y1, x2, y2,
                                         obj['confidence'],
                                         obj['class_name']):
                        total_saved += 1
            elif key == ord('a'):  # 'a' для переключения авто-сохранения
                auto_save = not auto_save
                print(f"🔧 Auto-save: {'ON' if auto_save else 'OFF'}")
            elif key == ord('c'):  # 'c' для показа классов
                show_classes_info()

        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching image from URL: {e}")
            time.sleep(1)
            continue
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            continue

    cv2.destroyAllWindows()

    # Финальная статистика
    print(f"\n🎉 Program finished!")
    print(f"📊 Total objects saved: {total_saved}")
    print(f"🎯 Unique classes detected: {len(detection_stats)}")
    if detection_stats:
        print("🏆 Top detected classes:")
        for class_name, count in sorted(detection_stats.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {class_name}: {count}")


if __name__ == "__main__":
    main()