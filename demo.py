import gradio as gr
import os
from search_engine import SearchEngine

engine = SearchEngine('vector_index')

def search_by_image_interface(image):
    # Находим похожие изображения
    similar_images = engine.search_by_image(image, k=20)
    
    # Агрегируем результаты
    top_names, top_kinds = engine.get_top_names_and_kinds(similar_images, top_n=5)
    
    # Форматируем вывод для текстовых полей
    names_str = "\n".join([f"{i+1}. {name}" for i, name in enumerate(top_names)])
    kinds_str = "\n".join([f"{i+1}. {kind}" for i, kind in enumerate(top_kinds)])
    
    return names_str, kinds_str

def search_by_text_interface(query):
    search_results = engine.search_by_text(query, k=5)
    
    # Gradio Gallery ожидает список кортежей: (путь_к_изображению, подпись)
    gallery_items = []
    for res in search_results:
        image_path = res['image_path']
        caption = f"{res['name']} ({res['city']})"
        
        if os.path.exists(image_path):
            gallery_items.append((image_path, caption))
        else:
            gallery_items.append(('src/placeholder.jpg', caption))
        
    return gallery_items

with gr.Blocks(title="Landmark Search Demo") as demo:
    gr.Markdown(
        """
        # Демо
        
        Поиск достопримечательностей по изображению и тексту
        """
    )
    
    with gr.Tabs():
        with gr.TabItem("img2text"):
            gr.Markdown("Загрузите изображение достопримечательности, чтобы получить топ-5 наиболее вероятных названий и категорий")
            with gr.Row():
                with gr.Column(scale=3):
                    image_input = gr.Image(type="pil", label="Ваше изображение")
                    image_search_btn = gr.Button("Найти", variant="primary")
                with gr.Column(scale=2):
                    names_output = gr.Textbox(label="Топ-5 названий", lines=5, interactive=False)
                    kinds_output = gr.Textbox(label="Топ-5 категорий", lines=5, interactive=False)
            
            image_search_btn.click(
                fn=search_by_image_interface,
                inputs=image_input,
                outputs=[names_output, kinds_output]
            )

        with gr.TabItem("text2img"):
            gr.Markdown("Введите текстовый запрос, чтобы найти топ-5 наиболее релевантных изображений.")
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        label="Текстовый запрос", 
                        lines=1
                    )
                with gr.Column():
                    text_search_btn = gr.Button("Найти", variant="primary")
            
            gallery_output = gr.Gallery(
                label="Результаты поиска", 
                show_label=True, 
                elem_id="gallery", 
                columns=3, 
                rows=1, 
                object_fit="cover", 
                height="auto"
            )

            gr.Examples(
                examples=[
                    ["статуя с мужчинами"],
                    ["деревянный дом"],
                    ["Гоголь"],
                    ["a red brick fortress"],
                    ["a person sitting on a bench"]
                ],
                inputs=text_input
            )

            text_search_btn.click(
                fn=search_by_text_interface,
                inputs=text_input,
                outputs=gallery_output
            )

if __name__ == "__main__":
    if engine:
        demo.launch()
    else:
        print("Приложение не может быть запущено из-за ошибок при инициализации.")