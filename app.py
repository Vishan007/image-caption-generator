import streamlit as st
import numpy as np
import pickle

from utils import *
from tensorflow.keras.models import load_model


vgg_model = load_model(r'models\vgg_model.h5')

with open(r'models\tokenizer.pkl' , 'rb') as t:
    tokenizer = pickle.load(t)


original_title = '<p style="font-family:Helvetica; text-align: center; color:orange; font-size: 70px;">Image Caption Generator</p>'
st.markdown(original_title, unsafe_allow_html=True)

image_file=st.sidebar.file_uploader('Choose a Image' , type=['png', 'jpg','jpeg','jfif'])

if image_file is not None:
    img = open_image(image_file)
    st.image(img ,width=500)
    if st.button('Generate Caption'):
        feature_array = preprocess_image(img_path=image_file)
        generated_caption =predict_caption(model=vgg_model,image=feature_array,tokenizer=tokenizer,max_length=35)
        st.text(generated_caption)
        


