import easyocr

# Initialize the EasyOCR reader for Marathi and Hindi
reader = easyocr.Reader(['mr', 'hi'])  # 'mr' for Marathi, 'hi' for Hindi

# Function to extract text from an image
def extract_text(image_path):
    # Read the image
    result = reader.readtext(image_path)
    
    # Extract and concatenate the text
    extracted_text = " ".join([text for (_, text, _) in result])
    return extracted_text