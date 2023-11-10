import streamlit as st
import pandas as pd 
import torch
from PIL import Image
from io import BytesIO
import requests
import faiss


from transformers import AutoTokenizer, AutoModel
import numpy as np
st.set_page_config(layout="wide")

@st.cache_resource()
def load_model():
    model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    return model , tokenizer

model, tokenizer = load_model()

@st.cache_data()
def load_data():
    df = pd.read_csv('Dataset/recipesdataset.csv')
    with open('Dataset/embeddingsrecipes.txt', 'r') as file:
        embeddings_list = [list(map(float, line.split())) for line in file.readlines()]
    index = faiss.read_index('Dataset/faissrecipes.index')
    return df, embeddings_list, index

df, embeddings_list, index = load_data()

def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()

col3, col4 = st.columns([5,1])

with col3:
    text = st.text_input('Введите ваше предпочтение для рекомендации')
with col4:
    num = st.number_input('Количество блюд', step=1, value=1)
    button = st.button('Отправить запрос')


if text and button:
    decode_text = embed_bert_cls(text, model, tokenizer)  # Получение вектора для введенного текста
    k = num 
    D, I = index.search(decode_text.reshape(1, -1), k)
    
    top_similar_indices = I[0]
    top_similar_annotations = [df['annotation'].iloc[i] for i in top_similar_indices]
    top_similar_images = [df['image_url'].iloc[i] for i in top_similar_indices]
    images = [Image.open(BytesIO(requests.get(url).content)) for url in top_similar_images]
    top_similar_title = [df['title'].iloc[i] for i in top_similar_indices]
    top_similar_url = [df['page_url'].iloc[i] for i in top_similar_indices]
    top_cosine_similarities = [1 - d / 2 for d in D[0]]  # Преобразование расстояний в косинусное сходство

# Отображение изображений и названий
    for similarity, image, annotation, title, url in zip(top_cosine_similarities, images, top_similar_annotations, top_similar_title, top_similar_url):
        col1, col2 = st.columns([3, 4]) 
        with col1:
            st.image(image, width=300)
        with col2:
            st.write(f"***Название:*** {title}")
            st.write(f"***Описание:*** {annotation}")
            similarity = float(similarity)
            st.write(f"***Cosine Similarity : {round(similarity, 3)}***")
            st.write(f"***Ссылка на блюдо : {url}***")

        st.markdown(
        "<hr style='border: 2px solid #000; margin-top: 10px; margin-bottom: 10px;'>",
        unsafe_allow_html=True
    )