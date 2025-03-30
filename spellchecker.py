import streamlit as st
import google.generativeai as genai
from PIL import Image
from spellchecker import SpellChecker

# Configure Gemini API (replace with your actual API key)
genai.configure(api_key='AIzaSyAhBAwIVkv7mb0OXYfVTl1W58HXmblUIi4')

# Set page config
st.set_page_config(
    page_title="Devanagari Spell Checker",
    page_icon="📝",
    layout="wide"
)

# Main app
st.title("📝 Devanagari Spell Checker")
st.markdown("Upload an image containing Devanagari text to check for spelling mistakes")

def extract_text_from_image(image):
    """Extract Devanagari text from image using Gemini"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash') 
        img = Image.open(image)
        
