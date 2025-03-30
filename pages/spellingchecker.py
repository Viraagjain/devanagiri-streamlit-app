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
        
        response = model.generate_content([
            "Extract all Devanagari text from this image exactly as it appears. "
            "Preserve the original formatting and include any errors. "
            "Return only the text without commentary.",
            img
        ])
        
        return response.text
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return None

def check_spelling(text, language='en'):
    """Check spelling using spellchecker library"""
    try:
        spell = SpellChecker(language=language)
        words = text.split()
        mistakes = []
        
        for i, word in enumerate(words):
            clean_word = word.strip('।,!?.()[]{}"\'')
            if clean_word and not spell.known([clean_word]):
                suggestions = spell.candidates(clean_word)
                mistakes.append({
                    'word': word,
                    'position': i,
                    'suggestions': list(suggestions)[:3] if suggestions else []
                })
        return mistakes
    except Exception as e:
        st.error(f"Error checking spelling: {str(e)}")
        return None

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image file", 
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(uploaded_file, use_column_width=True)
    
    with col2:
        st.subheader("Results")
        
        with st.spinner("Extracting text and checking spelling..."):
            # Extract text
            extracted_text = extract_text_from_image(uploaded_file)
            
            if extracted_text:
                st.markdown("**Extracted Text:**")
                st.text_area("", extracted_text, height=150, label_visibility="collapsed")
                
                # Check spelling
                mistakes = check_spelling(extracted_text, language='en')
                
                if mistakes:
                    st.error(f"Found {len(mistakes)} potential spelling mistakes:")
                    
                    # Create a table for mistakes
                    mistake_data = []
                    for mistake in mistakes:
                        mistake_data.append({
                            "Word": mistake['word'],
                            "Position": mistake['position'],
                            "Suggestions": ", ".join(mistake['suggestions']) if mistake['suggestions'] else "No suggestions"
                        })
                    
                    st.table(mistake_data)
                else:
                    st.success("No spelling mistakes found!")
            else:
                st.warning("No text could be extracted from the image")