import os
import re
import pandas as pd
from thefuzz import process
from thefuzz import fuzz

def normalize_string(s):
    s = s.lower()
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\d+', '', s)
    s = s.strip()
    return s

def extract_poi_name_from_filename(filename):
    base_name = os.path.splitext(filename)[0]
    poi_name = base_name.rsplit('_', 1)[0]
    return poi_name

def map_images_to_pois(data_dir):
    all_matched_data = []
    
    city_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for city_name in city_folders:
        
        image_dir = os.path.join(data_dir, city_name)
        csv_path = os.path.join(data_dir, f"{city_name}_places.csv")

        try:
            df_places = pd.read_csv(csv_path)
            original_place_names = df_places['Name'].tolist()
            normalized_place_names = [normalize_string(name) for name in original_place_names]
        except Exception as e:
            print(f"Ошибка при чтении CSV файла {csv_path}: {e}. Пропускаем.")
            continue
            
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        matched_count = 0

        for image_file in image_files:
            poi_name_from_image = extract_poi_name_from_filename(image_file)
            normalized_poi_name = normalize_string(poi_name_from_image)
            
            best_match_normalized, score = process.extractOne(
                normalized_poi_name, 
                normalized_place_names,
                scorer=fuzz.token_set_ratio
            )

            if score >= 85:
                match_index = normalized_place_names.index(best_match_normalized)
                
                poi_data = df_places.iloc[match_index].to_dict()
                
                poi_data['image_path'] = os.path.join(image_dir, image_file)
                
                all_matched_data.append(poi_data)
                matched_count += 1

    return pd.DataFrame(all_matched_data)

final_df = map_images_to_pois('data')
final_df['Name'] = final_df['Name'].str.replace(r'^№\d+\s*', '', regex=True).str.strip()

output_path = os.path.join('data', 'mapped_dataset.csv')
final_df.to_csv(output_path, index=False)
