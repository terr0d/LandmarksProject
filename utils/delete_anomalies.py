import os
from pathlib import Path

class DatasetFinalizer:
    """
    Класс для финального удаления оригинальных фотографий,
    которые были подтверждены в папках _candidates_for_deletion.
    """
    
    def __init__(self, data_dir="data", candidates_folder_name="_candidates_for_deletion"):
        self.data_dir = Path(data_dir)
        self.candidates_folder_name = candidates_folder_name

    def get_cities(self):
        cities = []
        for item in self.data_dir.iterdir():
            if item.is_dir() and not item.name.startswith('_'):
                cities.append(item)
        return sorted(cities)

    def parse_original_filename(self, candidate_filename):
        parts = candidate_filename.split('_')
        original_name_parts = parts[:2]
        return f"{'_'.join(original_name_parts)}.jpg"

    def delete_original_file(self, candidate_path, dry_run=True):
        # Извлекаем оригинальное имя файла
        original_filename = self.parse_original_filename(candidate_path.name)
        
        # Путь к оригинальному файлу на два уровня выше
        original_path = candidate_path.parent.parent / original_filename
        
        if original_path.exists():
            if dry_run:
                print(f"[DRY RUN] Готов к удалению: {original_path}")
            else:
                try:
                    os.remove(original_path)
                    print(f"[DELETED] {original_path}")
                except OSError as e:
                    print(f"Ошибка при удалении {original_path}: {e}")
        else:
            print(f"Оригинальный файл не найден: {original_path}")

    def process_city(self, city_path, dry_run=True):
        print(f"\nОбработка города: {city_path.name}")
        candidates_dir = city_path / self.candidates_folder_name
        
        candidate_files = list(candidates_dir.glob('*'))
        
        deleted_count = 0
        for candidate_file in candidate_files:
            if candidate_file.is_file():
                self.delete_original_file(candidate_file, dry_run)
                deleted_count += 1
        
        return deleted_count

    def run(self, dry_run=True):
        cities = self.get_cities()

        total_deleted = 0
        for city_path in cities:
            total_deleted += self.process_city(city_path, dry_run)

        print("Обработка завершена.")
        if dry_run:
            print(f"Всего файлов к удалению: {total_deleted}")
        else:
            print(f"Всего файлов удалено: {total_deleted}")


finalizer = DatasetFinalizer()
finalizer.run(dry_run=False)