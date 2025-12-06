import pandas as pd
import base64
from pathlib import Path
from tqdm import tqdm

def build_image_data_lookup(source_dir):
    image_lookup = {}
    source_files = list(source_dir.glob('*_images.csv'))

    if not source_files:
        raise FileNotFoundError(f"В папке {source_dir} не найдено файлов с именем '*_images.csv'")

    for csv_file in source_files:
        city_name = csv_file.stem.replace('_images', '')
        if city_name.lower() == 'yaroslavl':
            city_name = 'Yaroslavl'
        try:
            df = pd.read_csv(csv_file)
            name_counter = {}
            
            for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Обработка {csv_file.name}", leave=True):
                name = row['name']
                name_counter[name] = name_counter.get(name, 0) + 1
                current_count = name_counter[name]
                
                try:
                    image_base64 = row['image']
                except KeyError:
                    image_base64 = row['img']

                lookup_key = (city_name, name, current_count)
                image_lookup[lookup_key] = image_base64

        except Exception as e:
            print(f"Ошибка при обработке файла {csv_file}: {e}")
            
    return image_lookup

def create_dataset(metadata_csv_path, lookup_dict, output_dir):
    df_metadata = pd.read_csv(metadata_csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for _, row in tqdm(df_metadata.iterrows(), total=df_metadata.shape[0], desc="Сохранение изображений"):
        image_path = Path(row['image_path'])
        
        city_name = image_path.parent.name
        file_stem = image_path.stem
        poi_name, image_number_str = file_stem.rsplit('_', 1)
        image_number = int(image_number_str)
        
        lookup_key = (city_name, poi_name, image_number)
        
        if lookup_key in lookup_dict:
            image_base64 = lookup_dict[lookup_key]
            
            target_folder = output_dir / city_name
            target_folder.mkdir(parents=True, exist_ok=True)
            
            try:
                image_data = base64.b64decode(image_base64)
                with open(image_path, 'wb') as f:
                    f.write(image_data)
            except Exception as e:
                print(f"Ошибка при декодировании/сохранении {image_path}: {e}")
        else:
            print(f"Не найдены данные для изображения: {image_path}")



image_lookup_dict = build_image_data_lookup(Path('data'))
        
create_dataset(Path('vector_index/metadata.csv'), image_lookup_dict, Path('data'))