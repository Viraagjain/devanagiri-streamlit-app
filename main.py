import streamlit as st
import easyocr
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

# Load the pre-trained model
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Initialize EasyOCR reader
def initialize_easyocr():
    reader = easyocr.Reader(['mr', 'hi'])  # 'mr' for Marathi, 'hi' for Hindi
    return reader

# Preprocess the image to extract characters
def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    # Apply thresholding to get binary image
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    # Find contours of characters
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours from left to right
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    # Extract characters
    characters = []
    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        char = gray[y:y+h, x:x+w]
        char = cv2.resize(char, (32, 32))  # Resize to match model input size
        char = char / 255.0  # Normalize
        characters.append(char)
    return characters

# Predict characters using the model
def predict_characters(model, characters):
    predictions = []
    for char in characters:
        char = np.expand_dims(char, axis=-1)  # Add channel dimension
        char = np.expand_dims(char, axis=0)   # Add batch dimension
        pred = model.predict(char)
        pred_label = np.argmax(pred, axis=1)[0]  # Get predicted class
        confidence = np.max(pred)  # Get confidence score
        predictions.append((pred_label, confidence))
    return predictions

# Compare OCR output with model predictions to find inaccuracies
def find_inaccuracies(ocr_text, model_predictions, confidence_threshold=0.8):
    inaccuracies = []
    for i, (char, (pred_label, confidence)) in enumerate(zip(ocr_text, model_predictions)):
        if confidence < confidence_threshold:
            inaccuracies.append({
                "character": char,
                "position": i,
                "predicted_label": pred_label,
                "confidence": confidence
            })
    return inaccuracies

# Generate a report for inaccuracies
def generate_report(inaccuracies):
    report = "Inaccuracy Report:\n"
    for inaccuracy in inaccuracies:
        report += (
            f"Character: {inaccuracy['character']}, "
            f"Position: {inaccuracy['position']}, "
            f"Predicted Label: {inaccuracy['predicted_label']}, "
            f"Confidence: {inaccuracy['confidence']:.2f}\n"
        )
    return report

# Streamlit app
def main():
    st.title("Devanagari Text Inaccuracy Detection")
    st.write("Upload an image (JPG/PNG) to detect inaccuracies in Devanagari text.")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Save the image to a temporary file
        image_path = "temp_image.jpg"
        image.save(image_path)

        # Initialize EasyOCR
        reader = initialize_easyocr()

        # Extract text using EasyOCR
        ocr_result = reader.readtext(image_path)
        ocr_text = " ".join([text for (_, text, _) in ocr_result])
        st.write("Extracted Text:")
        st.write(ocr_text)

        # Load the Devanagari model
        model_path = "devanagiri.h5"  # Update this path
        model = load_model(model_path)

        # Preprocess the image to extract characters
        characters = preprocess_image(image)

        # Predict characters using the model
        model_predictions = predict_characters(model, characters)

        # Find inaccuracies
        inaccuracies = find_inaccuracies(ocr_text, model_predictions)

        # Generate and display the report
        if inaccuracies:
            report = generate_report(inaccuracies)
            st.write("Inaccuracies Detected:")
            st.text(report)
        else:
            st.write("No inaccuracies detected.")


main()