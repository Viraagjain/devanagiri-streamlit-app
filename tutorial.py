import streamlit as st

st.set_page_config(
    page_title="Devanagari Learning Tool",
    page_icon=":scroll:",
    layout="wide"
)

st.title("Welcome to the Devanagari Learning Tool")
st.write("""
This application helps you learn and practice Devanagari script (used for Hindi, Marathi, and other languages) 
through three specialized tools. Below you'll find instructions for each tool.
""")

st.divider()

# Character Recognizer Section
st.header("1. Character Recognizer")
with st.expander("How to use this tool"):
    st.write("""
    **Purpose**: Verify if you're writing Devanagari characters correctly by comparing against standard forms.
    
    **Steps**:
    1. Click on the 'Character Recognizer' page in the sidebar
    2. Upload an image of a single Devanagari character
    3. The tool will:
       - Identify which character you wrote
       - Compare it to the expected character (if filename matches our format)
       - Tell you if it's correctly written
    
    **Tips**:
    - Best for single characters (not words)
    - Use clear, centered images
    - Filename should follow format: character_[number]_[name].jpg (e.g., character_1_ka.jpg)
    """)


# Inaccuracy Detector Section
st.header("2. Inaccuracy Detector")
with st.expander("How to use this tool"):
    st.write("""
    **Purpose**: Find potential mistakes in Devanagari text by comparing OCR results with model predictions.
    
    **Steps**:
    1. Click on the 'Inaccuracy Detector' page in the sidebar
    2. Upload an image containing Devanagari text
    3. The tool will:
       - Extract text using OCR
       - Analyze each character with a recognition model
       - Flag characters where OCR and model predictions disagree
       - Provide a confidence score for each detection
    
    **Tips**:
    - Works best with clear, horizontal text
    - Good for checking handwritten or printed materials
    - Helps identify problematic characters in longer text
    """)


# Spell Checker Section
st.header("3. Spell Checker")
with st.expander("How to use this tool"):
    st.write("""
    **Purpose**: Check spelling of Hindi words in Devanagari script.
    
    **Steps**:
    1. Click on the 'Spell Checker' page in the sidebar
    2. Upload an image containing Devanagari text
    3. The tool will:
       - Extract the text
       - Compare each word against a Hindi dictionary
       - Suggest corrections for misspelled words
       - Show line and position of each potential error
    
    **Tips**:
    - Works with multi-line text
    - Uses Levenshtein distance for smart suggestions
    - Best for checking complete sentences/paragraphs
    """)

st.divider()

st.subheader("Getting Started")
st.write("""
Select any of the tools from the sidebar to begin. Each tool is designed to help with different aspects 
of Devanagari script learning and verification.

For best results:
- Use well-lit images
- Ensure characters/text are clearly visible
- For handwriting, try to write neatly
- The tools work best with standard Devanagari forms
""")

st.success("Ready to begin? Choose a tool from the sidebar!")