import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY=os.getenv("GENAI_API_KEY")
# Configure Gemini API (replace with your actual API key)
genai.configure(api_key=API_KEY)

# Set page config
st.set_page_config(
    page_title="Devanagari Spell Checker",
    page_icon="📝",
    layout="wide"
)

# Main app
st.title("📝 Hindi Spell Checker")
st.markdown("Upload an image containing Devanagari text to check for spelling mistakes")

# Initialize corpus as empty list
corpus = []

def loadCorpus():
    """Function to load the dictionary/corpus and store it in a global list"""
    global corpus
    try:
        with open('hindi_corpus.txt', encoding='utf-8') as file:
            for word in file:
                word = word.strip()
                corpus.append(word)
    except FileNotFoundError:
        st.error("Error: hindi_corpus.txt file not found. Please make sure it exists in the same directory.")
    except Exception as e:
        st.error(f"Error loading corpus: {str(e)}")

def getLevenshteinDistance(s, t):
    """Calculate Levenshtein distance between two strings"""
    rows = len(s)+1
    cols = len(t)+1
    dist = [[0 for x in range(cols)] for x in range(rows)]

    for i in range(1, rows):
        dist[i][0] = i

    for i in range(1, cols):
        dist[0][i] = i
        
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0
            else:
                cost = 1
            dist[row][col] = min(dist[row-1][col] + 1,      # deletion
                                 dist[row][col-1] + 1,      # insertion
                                 dist[row-1][col-1] + cost) # substitution

    return dist[row][col]

def getCorrectWord(word):
    """Find the closest matching word from corpus using Levenshtein distance"""
    if not corpus:
        return word  # Return original word if corpus isn't loaded
    
    min_dis = 100
    correct_word = word  # Default to original word if no close match found
    
    for s in corpus:
        cur_dis = getLevenshteinDistance(s, word)
        if min_dis > cur_dis:
            min_dis = cur_dis
            correct_word = s
            if min_dis == 0:  # Early exit if perfect match found
                break
    return correct_word

def check_spelling_hindi(text):
    """Check spelling of Hindi text using corpus and Levenshtein distance"""
    mistakes = []
    if not corpus:
        return mistakes  # Return empty list if corpus isn't loaded
    
    lines = text.split('\n')
    for line_num, line in enumerate(lines, 1):
        words = line.strip().split()
        for word_num, word in enumerate(words, 1):
            if word not in corpus:
                corrected = getCorrectWord(word)
                if corrected != word:  # Only report if correction is different
                    mistakes.append({
                        'word': word,
                        'line': line_num,
                        'position': word_num,
                        'correction': corrected
                    })
    return mistakes

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

# Load corpus when the app starts
loadCorpus()

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
                mistakes = check_spelling_hindi(extracted_text)
                
                if mistakes:
                    st.error(f"Found {len(mistakes)} potential spelling mistakes:")
                    
                    # Create a table for mistakes
                    mistake_data = []
                    for mistake in mistakes:
                        mistake_data.append({
                            "Word": mistake['word'],
                            "Line": mistake['line'],
                            "Position": mistake['position'],
                            "Suggested Correction": mistake['correction']
                        })
                    
                    st.table(mistake_data)
                else:
                    st.success("No spelling mistakes found!")
            else:
                st.warning("No text could be extracted from the image")