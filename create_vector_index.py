import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

img_model = SentenceTransformer('clip-ViT-B-32', device='cuda')

def get_image_vector(image_path, model):
    try:
        image = Image.open(image_path).convert("RGB")
        embedding = model.encode(image, convert_to_tensor=True, normalize_embeddings=True)
        return embedding.cpu().numpy()
    except Exception as e:
        print(f"Ошибка при обработке изображения {image_path}: {e}")
        return None

def create_vector_index(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    image_vectors = []
    metadata = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Векторизация изображений"):
        vector = get_image_vector(row['image_path'], img_model)
        
        if vector is not None:
            image_vectors.append(vector)
            metadata.append({
                'image_path': row['image_path'],
                'name': row['Name'],
                'kind': row['Kind'],
                'city': row['City'],
                'OSM': row['OSM'],
                'WikiData': row['WikiData']
            })

    image_vectors_np = np.array(image_vectors).astype('float32')
    
    vectors_path = os.path.join(output_dir, 'image_vectors.npy')
    metadata_path = os.path.join(output_dir, 'metadata.csv')

    np.save(vectors_path, image_vectors_np)
    
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(metadata_path, index=False)


create_vector_index('data/mapped_dataset.csv', 'vector_index')