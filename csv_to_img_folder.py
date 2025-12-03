import os
import pandas as pd
import base64
from pathlib import Path

def decode_and_save_images(csv_path):
    city_name = os.path.basename(csv_path).replace('_images.csv', '')
    
    output_dir = Path(f'data/{city_name}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    
    name_counter = {}
    
    for _, row in df.iterrows():
        name = row['name']
        try:
            image_base64 = row['image']
        except:
            image_base64 = row['img']
        
        name_counter[name] = name_counter.get(name, 0) + 1
        current_count = name_counter[name]
        
        file_name = f"{name}_{current_count}.jpg"
        file_path = output_dir / file_name
        
        try:
            image_data = base64.b64decode(image_base64)
            
            with open(file_path, 'wb') as f:
                f.write(image_data)
            
            print(f"Сохранено: {file_path}")
            
        except Exception as e:
            print(f"Ошибка при обработке {file_path}: {str(e)}")

data_dir = Path('data')
    
for csv_file in data_dir.glob('*_images.csv'):
    print(f"\nОбработка файла: {csv_file}")
    decode_and_save_images(csv_file)