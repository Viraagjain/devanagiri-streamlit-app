import streamlit as st
from PIL import Image
import numpy as np
import os
from dotenv import load_dotenv
import cv2
import google.generativeai as genai

load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY")
# Devanagari character mapping
DEVANAGARI_MAP = {
    "character_28_la": "ल",
    "character_15_adna": "ण",
    'character_33_ha': "ह",
    'character_3_ga': "ग",
    'character_22_pha': "फ",
    'character_31_petchiryakha': "ष",
    'character_29_waw': "व",
    'character_34_chhya': "क्ष",
    'character_17_tha': "थ",
    'character_21_pa': "प",
    'character_7_chha': "छ",
    'digit_8': "८",
    'character_27_ra': "र",
    'character_35_tra': "त्र",
    'character_18_da': "द",
    'character_20_na': "न",
    'character_2_kha': "ख",
    'character_16_tabala': "त",
    'character_12_thaa': "ठ",
    'digit_3': "३",
    'character_5_kna': "ङ",
    'character_24_bha': "भ",
    'character_6_cha': "च",
    'character_13_daa': "ड",
    'digit_1': "१",
    'digit_0': "०",
    'digit_5': "५",
    'character_1_ka': "क",
    'digit_4': "४",
    'character_4_gha': "घ",
    'digit_2': "२",
    'character_36_gya': "ज्ञ",
    'character_19_dha': "ध",
    'character_30_motosaw': "श",
    'character_14_dhaa': "द",
    'character_8_ja': "ज",
    'digit_9': "९",
    'character_10_yna': "ञ",
    'character_9_jha': "झ",
    'character_26_yaw': "य",
    'character_25_ma': "म",
    'character_32_patalosaw': "स",
    'character_23_ba': "ब",
    'digit_7': "७",
    'character_11_taamatar': "ट",
    'digit_6': "६"
}

# Reverse mapping to go from Devanagari character to label
REVERSE_MAP = {v: k for k, v in DEVANAGARI_MAP.items()}

# Configure Gemini API
genai.configure(api_key=API_KEY)

def preprocess_image(image):
    """Simple preprocessing pipeline for display"""
    img_array = np.array(image)
    
    # Handle RGBA images
    if img_array.ndim == 3 and img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    
    return img_array

def extract_base_label(filename):
    """Extracts base label from filename (e.g., 'character_32_ka' from 'character_32_ka_correct.jpg')"""
    base = os.path.basename(filename)
    return '_'.join(base.split('_')[:3])

def analyze_image_with_gemini(image):
    """Use Gemini to analyze the image and identify the Devanagari character"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = """Identify this Devanagari character precisely. 
    Respond ONLY with the exact matching Unicode character from:
    [क,ख,ग,घ,ङ,च,छ,ज,झ,ञ,ट,ठ,ड,ढ,ण,त,थ,द,ध,न,प,फ,ब,भ,म,य,र,ल,व,श,ष,स,ह,क्ष,त्र,ज्ञ,०,१,२,३,४,५,६,७,८,९]
    No explanations. Key distinctions:
    - 'क' vs 'फ' (right curve)
    - 'त' vs 'ट' (horizontal bar)
    - 'न' vs 'ण' (top curve)
    - Digits: ०-९"""
    
    try:
        response = model.generate_content([prompt, image])
        char = response.text.strip()
        return char if char in REVERSE_MAP else None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

st.set_page_config(page_title="Devanagari Character Recognizer", layout="centered")
st.title("Devanagari Character Recognizer")

uploaded_file = st.file_uploader("Upload character image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Recognize Character"):
        with st.spinner("Analyzing ..."):
            # Get prediction from Gemini
            predicted_char = analyze_image_with_gemini(image)
            
            if predicted_char:
                # Find the corresponding label
                predicted_label = REVERSE_MAP.get(predicted_char, "Unknown")
                
                st.subheader("Results")
                st.write(f"Predicted label: **{predicted_label}**")
                
                # Extract and show base label from filename if available
                base_label = extract_base_label(uploaded_file.name)
                if base_label:
                    expected_char = DEVANAGARI_MAP.get(base_label, "Unknown")
                    st.write(f"Actual Label: {base_label}")
                    if base_label == predicted_label:
                        st.success("✅ Correctly written")
                    else:
                        st.warning("⚠️ Inaccuracy detected")
            else:
                st.error("Could not predict")