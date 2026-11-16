# ultra_fast.py
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

from ultralytics import YOLO


def train_ultra_fast():
    print("⚡ УЛЬТРА-БЫСТРОЕ ОБУЧЕНИЕ")

    # Используем существующий датасет
    dataset_path = "caltech256_yolo_full/caltech256.yaml"

    # СУПЕР-ОПТИМИЗИРОВАННЫЕ НАСТРОЙКИ
    config = {
        'data': dataset_path,
        'epochs': 20,  # Всего 20 эпох
        'imgsz': 224,  # Минимальное разрешение
        'batch': 8,  # Большой batch
        'device': 'cpu',
        'workers': 0,
        'patience': 3,  # Очень ранняя остановка
        'save': True,
        'cache': False,
        'name': 'caltech256_ultra_fast',
        'optimizer': 'SGD',
        'lr0': 0.1,  # Высокая скорость обучения
        'momentum': 0.9,
    }

    print(f"⚙️  СУПЕР-БЫСТРЫЕ НАСТРОЙКИ:")
    print(f"   🖼️  Размер: {config['imgsz']}px")
    print(f"   📦 Batch: {config['batch']}")
    print(f"   🔁 Эпохи: {config['epochs']}")
    print("   ⏱️  Ожидаемое время: 6-12 часов")

    model = YOLO('yolov8n.pt')
    results = model.train(**config)
    print("✅ Ультра-быстрое обучение завершено!")


if __name__ == "__main__":
    train_ultra_fast()