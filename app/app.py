import streamlit as st
# Set page config must be the first Streamlit command
st.set_page_config(page_title="Urban Sound Classification", page_icon="🎧")

import torch
import os
import sys
import matplotlib.pyplot as plt

# Import local modules
try:
    from utils import load_model, preprocess_audio
except Exception as e:
    st.error(f"Error importing modules: {str(e)}")
    st.write("This may be due to missing packages. Try installing requirements:")
    st.code("pip install torchvision torchaudio librosa numpy")
    st.stop()  # Stop execution if imports fail

# Debug section
with st.expander("Debug Information"):
    st.write("Python version:", sys.version)
    st.write("PyTorch version:", torch.__version__)
    st.write("Current directory:", os.getcwd())
    st.write("Files in current directory:", os.listdir("."))
    
    # Check if utils.py exists
    if os.path.exists("utils.py"):
        st.write("utils.py exists in current directory")

# Set the path to your model file
MODEL_PATH = "../results/best_model.pth"  # Change to your actual model path

# Try to load the model with error handling
try:
    model = load_model(MODEL_PATH)
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.info(f"Make sure the model file exists at: {os.path.abspath(MODEL_PATH)}")
    model_loaded = False

# Define class names
CLASSES = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square',
           'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram']

# Create the UI
st.title("🎧 Urban Sound Classification")
st.write("Upload an audio file to classify the urban environment.")

# Add file uploader widget
uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None:
    # Display audio player
    st.audio(uploaded_file, format='audio/wav')
    
    # Save uploaded file temporarily
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())
    
    if model_loaded:
        with st.spinner('Analyzing audio...'):
            # Process the audio and make prediction
            try:
                input_tensor = preprocess_audio("temp.wav")
                
                # Add debug info about the tensor
                with st.expander("Debug: Input Tensor Info"):
                    st.write(f"Tensor shape: {input_tensor.shape}")
                    st.write(f"Tensor type: {input_tensor.dtype}")
                    st.write(f"Tensor min: {input_tensor.min().item():.4f}, max: {input_tensor.max().item():.4f}")
                    st.write(f"Tensor mean: {input_tensor.mean().item():.4f}, std: {input_tensor.std().item():.4f}")
                    
                    # Display the spectrogram
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.imshow(input_tensor[0, 0].numpy(), aspect='auto', origin='lower', cmap='inferno')
                    ax.set_title("Input Mel Spectrogram")
                    ax.set_ylabel("Mel Frequency Bin")
                    ax.set_xlabel("Time Frame")
                    st.pyplot(fig)
                
                # Ensure model is in evaluation mode
                model.eval()
                
                with torch.no_grad():
                    # Get raw logits
                    output = model(input_tensor)
                    
                    # Show raw model output
                    with st.expander("Debug: Model Output"):
                        st.write("Raw model output (logits):")
                        for i, (cls, val) in enumerate(zip(CLASSES, output[0].tolist())):
                            st.write(f"{cls}: {val:.4f}")
                    
                    # Calculate softmax probabilities
                    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                    predicted_class_idx = output.argmax(dim=1).item()
                    predicted_class = CLASSES[predicted_class_idx]
                    confidence = probabilities[predicted_class_idx].item() * 100
                
                # Show prediction with confidence
                st.success(f"Predicted Environment: **{predicted_class.title()}**")
                st.info(f"Confidence: {confidence:.2f}%")
                
                # Display top 3 predictions as a bar chart
                top_indices = torch.topk(probabilities, 3).indices.tolist()
                top_probs = torch.topk(probabilities, 3).values.tolist()
                top_classes = [CLASSES[i].title() for i in top_indices]
                
                chart_data = {
                    'Environment': top_classes,
                    'Confidence (%)': [p*100 for p in top_probs]
                }
                
                st.write("### Top Predictions")
                st.bar_chart(chart_data, x='Environment', y='Confidence (%)')
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.info("Debug info: Please check if the audio format is compatible.")
    else:
        st.warning("Model not loaded. Please check the model path and try again.")

# Add information about the app
with st.expander("About this app"):
    st.write("""
    This app uses a deep learning model to classify urban sounds into different environments.
    The model was trained on the Urban Sound dataset and can recognize 10 different urban environments.
    
    Upload a WAV audio file to get started!
    """)