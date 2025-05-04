import streamlit as st
from utils import load_model, preprocess_audio
import torch

# Load model once
MODEL_PATH = "../results/best_model.pth"
model = load_model(MODEL_PATH)
CLASSES = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square',
           'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram']

st.title("🎧 Urban Sound Classification")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    input_tensor = preprocess_audio("temp.wav")
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = CLASSES[output.argmax(dim=1).item()]

    st.success(f"Predicted Class: **{predicted_class}**")
