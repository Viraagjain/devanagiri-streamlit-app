import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import cv2

# Load the pre-trained model
@st.cache_resource
def load_devanagari_model():
    try:
        model = tf.keras.models.load_model("devanagiri.h5", compile=False)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def extract_full_label(filename):
    """Extracts 'character_4_gha' from 'character_4_gha_correct.jpg'"""
    base = os.path.basename(filename)
    return '_'.join(base.split('_')[:3])  # Joins first three parts

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Devanagari Character Checker", page_icon="📝", layout="centered")
st.title("📖 Devanagari Character Checker")

# Load model
model = load_devanagari_model()
if model is None:
    st.stop()

# Image Upload
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Character", use_column_width=True)
        
        if st.button("Check Character"):
            with st.spinner("Checking..."):
                # Convert to numpy array and BGR format
                img_array = np.array(image)
                if img_array.shape[-1] == 4:  # Remove alpha channel
                    img_array = img_array[..., :3]
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # EXACT PREDICTION STEPS
                resized_word = cv2.resize(img_array, (32, 32))
                predictions = model.predict(resized_word.reshape(1, 32, 32, 3)/255)
                predicted_class = np.argmax(predictions)
                
                # Extract and display labels
                actual_full_label = extract_full_label(uploaded_file.name)  # 'character_4_gha'
                actual_char_part = uploaded_file.name.split('_')[2]  # 'gha'
                
                st.write(f"Filename indicates: {actual_full_label}")
                st.write(f"Model predicts class: {predicted_class}")
                st.write(f"Confidence: {np.max(predictions)*100:.2f}%")
                
                # Compare just the character part (e.g., 'gha')
                if str(actual_char_part) == str(predicted_class):
                    st.success("✅ Correct - The character matches the expected label")
                else:
                    st.error(f"❌ Incorrect - Expected {actual_char_part}, got {predicted_class}")
    
    except Exception as e:
        st.error(f"Error processing image: {e}")