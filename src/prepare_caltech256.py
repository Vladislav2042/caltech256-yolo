# prepare_caltech256_correct.py
import os
import shutil
import yaml
from sklearn.model_selection import train_test_split
import cv2


def prepare_caltech256_for_yolo():
    """
    Преобразует Caltech256 в YOLO формат
    """
    # Правильный путь к данным
    caltech_path = "256_ObjectCategories/256_ObjectCategories"
    output_path = "caltech256_yolo_full"

    print(" Подготовка Caltech256 для YOLO...")
    print(f"Источник: {caltech_path}")
    print(f" Выход: {output_path}")

    # Создаем структуру папок
    directories = [
        'images/train',
        'images/val',
        'labels/train',
        'labels/val'
    ]

    for directory in directories:
        dir_path = os.path.join(output_path, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f" Создана: {dir_path}")

    # Собираем все изображения
    images_data = []
    class_names = []

    print("📊 Сбор информации о классах...")

    # Проходим по всем папкам классов
    for class_dir in sorted(os.listdir(caltech_path)):
        class_path = os.path.join(caltech_path, class_dir)

        if os.path.isdir(class_path) and not class_dir.startswith('.'):
            # Извлекаем номер и название класса (формат: "001.ak47")
            parts = class_dir.split('.')
            if len(parts) >= 2:
                class_num = parts[0]
                class_name = '.'.join(parts[1:])  # Объединяем оставшиеся части
            else:
                class_num = class_dir
                class_name = class_dir

            class_names.append(class_name)
            class_id = len(class_names) - 1  # YOLO классы с 0

            # Ищем изображения в папке класса
            image_files = []
            for file in os.listdir(class_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(file)

            if image_files:
                print(f"  {class_id:3d}: {class_name} ({len(image_files)} изображений)")

                for img_file in image_files:
                    img_path = os.path.join(class_path, img_file)
                    # Создаем уникальное имя файла
                    new_filename = f"{class_id:03d}_{img_file}"

                    images_data.append({
                        'path': img_path,
                        'class_id': class_id,
                        'class_name': class_name,
                        'filename': new_filename
                    })

    print(f"\n Всего: {len(images_data)} изображений, {len(class_names)} классов")

    if not images_data:
        print(" Не найдено изображений!")
        return 0

    # Разделяем на train/val (80/20)
    print("📊 Разделение на train/val...")
    train_data, val_data = train_test_split(images_data, test_size=0.2, random_state=42)

    def create_yolo_annotation(img_path, class_id, output_label_path):
        """Создает YOLO аннотацию (объект на все изображение)"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                return False

            # Создаем bounding box на все изображение
            x_center = 0.5
            y_center = 0.5
            width = 0.95
            height = 0.95

            with open(output_label_path, 'w') as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            return True

        except Exception as e:
            print(f" Ошибка создания аннотации: {e}")
            return False

    # Обрабатываем тренировочные данные
    print("\n Копирование тренировочных данных...")
    train_count = 0
    for i, data in enumerate(train_data):
        if i % 500 == 0 and i > 0:
            print(f"  Обработано {i}/{len(train_data)}")

        new_img_path = os.path.join(output_path, 'images/train', data['filename'])
        label_path = os.path.join(output_path, 'labels/train',
                                  os.path.splitext(data['filename'])[0] + '.txt')

        try:
            shutil.copy2(data['path'], new_img_path)
            if create_yolo_annotation(data['path'], data['class_id'], label_path):
                train_count += 1
        except Exception as e:
            print(f" Ошибка копирования {data['path']}: {e}")

    # Обрабатываем валидационные данные
    print("\n Копирование валидационных данных...")
    val_count = 0
    for i, data in enumerate(val_data):
        if i % 500 == 0 and i > 0:
            print(f"  Обработано {i}/{len(val_data)}")

        new_img_path = os.path.join(output_path, 'images/val', data['filename'])
        label_path = os.path.join(output_path, 'labels/val',
                                  os.path.splitext(data['filename'])[0] + '.txt')

        try:
            shutil.copy2(data['path'], new_img_path)
            if create_yolo_annotation(data['path'], data['class_id'], label_path):
                val_count += 1
        except Exception as e:
            print(f"Ошибка копирования {data['path']}: {e}")

    # Создаем data.yaml файл
    yaml_data = {
        'path': os.path.abspath(output_path),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(class_names)},
        'nc': len(class_names)
    }

    yaml_path = os.path.join(output_path, 'caltech256.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print(f"\n ПОДГОТОВКА ЗАВЕРШЕНА!")
    print("=" * 50)
    print(f" Статистика:")
    print(f"    Классов: {len(class_names)}")
    print(f"    Train изображений: {train_count}")
    print(f"    Val изображений: {val_count}")
    print(f"   Конфиг файл: {yaml_path}")
    print(f"   Выходная папка: {output_path}")

    # Сохраняем список классов в отдельный файл
    with open(os.path.join(output_path, 'classes.txt'), 'w') as f:
        for class_id, class_name in enumerate(class_names):
            f.write(f"{class_id}: {class_name}\n")

    return len(class_names)


if __name__ == "__main__":
    num_classes = prepare_caltech256_for_yolo()

    print(f"\n ДАТАСЕТ ГОТОВ К ОБУЧЕНИЮ!")
    print(f" Запустите: python train_caltech256_full.py")