from graphviz import Digraph

def create_scientific_clear_scheme():
    g = Digraph('DroneScientificClearScheme', format='png')
    g.attr(rankdir='TB', size='15,15', ratio='fill')  # Большой квадрат для читаемости и большого шрифта
    g.attr('node', shape='box', style='filled,rounded', fontname='Times-Bold', fontsize='20', penwidth='3')
    g.attr('edge', color='navy', fontname='Times-Roman', fontsize='18', arrowsize='1.8')

    # Научные цвета и значки (понятные описания)
    g.node('start', '▶ Начало выполнения программы', fillcolor='lightyellow', fontcolor='black')
    g.node('init', '⚙ Инициализация\n• Импорт библиотек\n• Подключение к камере дрона\n• Загрузка модели YOLOv8\n(датасет Caltech-256: 256 классов)', fillcolor='lightblue')
    g.node('loop', '🔁 Основной цикл обработки', fillcolor='lightgreen')
    g.node('frame', '📷 Захват кадра\n• Рендеринг изображения с камеры\n• Кодирование в JPEG для передачи', fillcolor='wheat')
    g.node('process', '🔍 Инференс модели\n• Детекция объектов YOLOv8\n• Фильтрация по уверенности (>0.4)\n• Извлечение данных:\n  - Класс объекта (256 категорий)\n  - Уровень уверенности\n  - Координаты центра\n  - Размеры bounding box', fillcolor='lightsalmon')
    g.node('hud', '🖥 Отображение в HUD\n• Количество объектов\n• Список: класс (confidence %)\n• Обновление в реальном времени', fillcolor='lightpink')
    g.node('keys', '⌨ Ввод пользователя\n• Y: принудительная обработка кадра\n• Esc: завершение программы', fillcolor='lightgray')
    g.node('save', '💾 Сохранение данных\n• Вырезка объектов\n• Файлы по классам (опционально)', fillcolor='lavender')
    g.node('end', '⏹ Завершение программы', fillcolor='lightyellow', fontcolor='black')

    g.edge('start', 'init', label='Запуск')
    g.edge('init', 'loop', label='Вход в цикл')
    g.edge('loop', 'frame', label='Захват кадра')
    g.edge('frame', 'process', label='Инференс на сервере')
    g.edge('process', 'hud', label='Возврат результатов')
    g.edge('hud', 'keys', label='Ожидание ввода')
    g.edge('keys', 'save', label='Сохранение')
    g.edge('save', 'loop', style='dashed', label='Следующая итерация')
    g.edge('keys', 'end', label='Выход')

    g.render('drone_scheme_scientific_clear_caltech256', cleanup=True, view=True)  # Создаст PNG и откроет

if __name__ == "__main__":
    create_scientific_clear_scheme()