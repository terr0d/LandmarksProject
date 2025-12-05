import os
import numpy as np
import pandas as pd
import torch
import faiss
from PIL import Image
from sentence_transformers import SentenceTransformer

INDEX_DIR = 'vector_index'
IMG_MODEL_NAME = 'clip-ViT-B-32'
TEXT_MODEL_NAME = 'sentence-transformers/clip-ViT-B-32-multilingual-v1'

class SearchEngine:
    def __init__(self, index_dir):
        print("Инициализация поискового движка...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"Загрузка модели для изображений: {IMG_MODEL_NAME}")
        self.img_model = SentenceTransformer(IMG_MODEL_NAME, device=device)
        
        print(f"Загрузка модели для текста: {TEXT_MODEL_NAME}")
        self.text_model = SentenceTransformer(TEXT_MODEL_NAME, device=device)
        print("Модели загружены.")

        vectors_path = os.path.join(index_dir, 'image_vectors.npy')
        metadata_path = os.path.join(index_dir, 'metadata.csv')

        self.image_vectors = np.load(vectors_path)
        self.metadata_df = pd.read_csv(metadata_path)
        print(f"Загружено {len(self.image_vectors)} векторов и {len(self.metadata_df)} записей метаданных.")

        self.index = self._build_faiss_index(self.image_vectors)
        print("Поисковый индекс Faiss готов.")

    def _build_faiss_index(self, vectors):
        dimension = vectors.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)
        return index

    def _get_text_vector(self, text):
        return self.text_model.encode(text, convert_to_numpy=True, normalize_embeddings=True).astype('float32')

    def _get_image_vector(self, image_input):
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("Неподдерживаемый тип входных данных. Ожидается путь к файлу (str) или PIL Image.")

        return self.img_model.encode(image, convert_to_numpy=True, normalize_embeddings=True).astype('float32')

    def search_by_text(self, query, k=5):
        print(f"\nПоиск по запросу: '{query}'")
        query_vector = self._get_text_vector(query)
        query_vector = query_vector.reshape(1, -1)
        
        scores, indices = self.index.search(query_vector, k)
        
        results = []
        for i in range(k):
            idx = indices[0][i]
            score = scores[0][i]
            if idx != -1:
                result = self.metadata_df.iloc[idx].to_dict()
                result['score'] = score
                results.append(result)
        
        return results

    def search_by_image(self, image_path, k=20):
        print("\nПоиск похожих на изображение")
        query_vector = self._get_image_vector(image_path)
        query_vector = query_vector.reshape(1, -1)
        
        scores, indices = self.index.search(query_vector, k)
        
        results = []
        for i in range(k):
            idx = indices[0][i]
            score = scores[0][i]
            if idx != -1:
                result = self.metadata_df.iloc[idx].to_dict()
                result['score'] = score
                results.append(result)
        
        return results

    def get_top_names_and_kinds(self, search_results, top_n=5):
        name_scores = {}
        kind_scores = {}

        for res in search_results:
            score = float(res.get('score', 0.0))
            name = res.get('name')
            if name:
                name_scores[name] = name_scores.get(name, 0.0) + score

            kind_str = res.get('kind')
            if pd.notna(kind_str):
                for k in kind_str.split(','):
                    k = k.strip()
                    if not k:
                        continue
                    kind_scores[k] = kind_scores.get(k, 0.0) + score

        top_names = [n for n, _ in sorted(name_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]]
        top_kinds = [k for k, _ in sorted(kind_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]]

        return top_names, top_kinds