import shutil
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from transformers import CLIPProcessor, CLIPModel


@dataclass
class PhotoCandidate:
    original_path: Path
    confidence: float
    reason: str


class LandmarkDatasetCleaner:
    
    def __init__(self, data_dir="data", model_name="openai/clip-vit-base-patch32", device="cuda", batch_size=32):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.device = device
        
        print(f"Загрузка модели {model_name}...")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        
        self.candidates_folder_name = "_candidates_for_deletion"
    
    def get_cities(self):
        cities = []
        for item in self.data_dir.iterdir():
            if item.is_dir() and not item.name.startswith('_'):
                cities.append(item)
        return sorted(cities)
    
    def parse_filename(self, filepath):
        stem = filepath.stem
        parts = stem.rsplit('_', 1)
        
        return parts[0], int(parts[1])

    
    def get_photos_by_landmark(self, city_path):
        photos_by_landmark = defaultdict(list)
        
        for photo_path in city_path.glob("*.jpg"):
            if photo_path.is_file():
                landmark_name, _ = self.parse_filename(photo_path)
                photos_by_landmark[landmark_name].append(photo_path)
        
        return dict(photos_by_landmark)
    
    @torch.no_grad()
    def extract_embeddings(self, image_paths):
        embeddings = []
        
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            images = []
            
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(img)
                except Exception as e:
                    print(f"  Ошибка загрузки {path}: {e}")

            
            # Обрабатываем батч
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            image_features = self.model.get_image_features(**inputs)
            # Нормализуем
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            embeddings.append(image_features.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def find_outliers(self, embeddings, image_paths, eps=0.2, min_samples=5, distance_threshold=0.3):
        """
        Находит выбросы с помощью DBSCAN.
        
        Args:
            embeddings: CLIP эмбеддинги
            image_paths: пути к изображениям
            eps: параметр DBSCAN - максимальное расстояние между точками в кластере
            min_samples: минимальное количество точек для формирования кластера
            distance_threshold: порог расстояния от центра основного кластера для пометки как выброс
        """
        if len(embeddings) < min_samples:
            # Слишком мало фото для кластеризации
            return []
        
        candidates = []
        
        # Вычисляем матрицу косинусных расстояний
        distances = cosine_distances(embeddings)
        
        # Кластеризация DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        labels = clustering.fit_predict(distances)
        
        # Находим основной кластер (самый большой)
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        
        if len(unique_labels) == 0:
            # Все точки - шум, помечаем все как подозрительное
            centroid = embeddings.mean(axis=0)
            for idx, path in enumerate(image_paths):
                dist = cosine_distances([embeddings[idx]], [centroid])[0, 0]
                candidates.append(PhotoCandidate(
                    original_path=path,
                    confidence=min(dist / distance_threshold, 1.0),
                    reason="no_clear_cluster"
                ))
            return candidates
        
        main_cluster_label = unique_labels[np.argmax(counts)]
        main_cluster_mask = labels == main_cluster_label
        
        # Центроид основного кластера
        main_centroid = embeddings[main_cluster_mask].mean(axis=0)
        
        # Анализируем каждое фото
        for idx, (path, label) in enumerate(zip(image_paths, labels)):
            dist_to_main = cosine_distances([embeddings[idx]], [main_centroid])[0, 0]
            
            if label == -1:
                # Шум по DBSCAN - высокая уверенность
                confidence = min(0.7 + 0.3 * (dist_to_main / distance_threshold), 1.0)
                candidates.append(PhotoCandidate(
                    original_path=path,
                    confidence=confidence,
                    reason="noise_point"
                ))
            elif label != main_cluster_label:
                # Другой кластер - средняя уверенность
                cluster_size = counts[np.where(unique_labels == label)[0][0]]
                size_factor = 1.0 - (cluster_size / len(image_paths))
                confidence = min(0.5 + 0.3 * size_factor, 0.9)
                candidates.append(PhotoCandidate(
                    original_path=path,
                    confidence=confidence,
                    reason=f"small_cluster_{cluster_size}"
                ))
            elif dist_to_main > distance_threshold:
                # В основном кластере, но далеко от центра
                confidence = min((dist_to_main - distance_threshold) / distance_threshold, 0.6)
                candidates.append(PhotoCandidate(
                    original_path=path,
                    confidence=confidence,
                    reason="far_from_center"
                ))
        
        return candidates
    
    def create_candidates_folder(self, city_path, candidates):
        candidates_dir = city_path / self.candidates_folder_name
        candidates_dir.mkdir(exist_ok=True)
        
        for candidate in candidates:
            # Формируем новое имя: originalname_confidence_reason.jpg
            confidence_str = f"{candidate.confidence:.2f}"
            
            # Разделяем имя файла и расширение
            original_stem = candidate.original_path.stem
            original_suffix = candidate.original_path.suffix
            
            new_name = f"{original_stem}_{confidence_str}_{candidate.reason}{original_suffix}"
            new_path = candidates_dir / new_name
            
            # Копируем (не перемещаем!) файл
            shutil.copy2(candidate.original_path, new_path)
    
    def process_city(self, city_path):
        print(f"Обработка города: {city_path.name}")
        
        # Группируем фото по достопримечательностям
        photos_by_landmark = self.get_photos_by_landmark(city_path)
        print(f"Найдено достопримечательностей: {len(photos_by_landmark)}")
        
        all_candidates = []
        processed_landmarks = 0
        
        for landmark_name, photo_paths in photos_by_landmark.items():
            print(f"Анализ достопримечательности '{landmark_name}' ({len(photo_paths)} фото)")
            
            # Извлекаем эмбеддинги
            embeddings = self.extract_embeddings(photo_paths)
            
            # Ищем выбросы
            candidates = self.find_outliers(embeddings, photo_paths)
            if candidates:
                print(f"Найдено кандидатов на удаление: {len(candidates)}")
            all_candidates.extend(candidates)
            processed_landmarks += 1
        
        # Создаём папку с кандидатами
        if all_candidates:
            self.create_candidates_folder(city_path, all_candidates)
            print(f"\nКандидатов на удаление для города '{city_path.name}': {len(all_candidates)}")
        else:
            print(f"\nДля города '{city_path.name}' кандидатов на удаление не найдено.")

        return {
            "city": city_path.name,
            "landmarks": processed_landmarks,
            "candidates": len(all_candidates)
        }

    def run(self):
        cities = self.get_cities()

        print(f"Найдено городов для обработки: {len(cities)}")
        total_stats = {"cities_processed": 0, "total_landmarks": 0, "total_candidates": 0}

        for city_path in cities:
            stats = self.process_city(city_path)
            total_stats["cities_processed"] += 1
            total_stats["total_landmarks"] += stats["landmarks"]
            total_stats["total_candidates"] += stats["candidates"]

        print("Обработка завершена")
        print(f"Всего обработано городов: {total_stats['cities_processed']}")
        print(f"Всего проанализировано достопримечательностей: {total_stats['total_landmarks']}")
        print(f"Всего найдено кандидатов на удаление: {total_stats['total_candidates']}")


cleaner = LandmarkDatasetCleaner()
cleaner.run()