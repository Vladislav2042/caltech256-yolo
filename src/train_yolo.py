import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
import torch


def train_caltech256_full():
    print(" ЗАПУСК ОБУЧЕНИЯ НА ВСЕХ ДАННЫХ CALTECH256")
    print("=" * 50)

    # Проверяем датасет
    dataset_path = "caltech256_yolo_full/caltech256.yaml"
    if not os.path.exists(dataset_path):
        print(" Датасет не найден! Сначала запустите prepare_caltech256_correct.py")
        return

    # Проверяем GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"⚙️ Используется: {device}")
    if device == 'cuda':
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

    # Загружаем модель (рекомендую yolov8m для 257 классов)
    print(" Загрузка модели YOLOv8m...")
    model = YOLO('yolov8m.pt')

    # Настройки обучения
    config = {
        'data': dataset_path,
        'epochs': 50,  # Можно увеличить до 100 если есть время
        'imgsz': 640,
        'batch': 8 if device == 'cuda' else 2,
        'device': device,
        'workers': 4 if device == 'cuda' else 0,
        'patience': 10,
        'save': True,
        'name': 'caltech256_257_classes'
    }

    print("️ Настройки обучения:")
    for key, value in config.items():
        print(f"   {key}: {value}")

    print(f"\n Обучение на 257 классах...")
    print("   Это займет значительное время!")

    try:
        print(" НАЧАЛО ОБУЧЕНИЯ...")
        results = model.train(**config)

        print(" ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        model_path = f"runs/detect/{config['name']}/weights/best.pt"
        print(f" Модель сохранена: {model_path}")

    except Exception as e:
        print(f" Ошибка обучения: {e}")


if __name__ == "__main__":
    train_caltech256_full()
