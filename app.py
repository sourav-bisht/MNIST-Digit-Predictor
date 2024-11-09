import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


# Load the model once.
@st.cache_resource
def load_model_once():
    return tf.keras.models.load_model("MNISTmodel1.keras")


# Preprocess the image
def preprocess_image(image):
    # Resize the image to match the model's expected input size (28x28 for MNIST)
    image = image.resize((28, 28)).convert('L')  # Convert to grayscale
    image = tf.keras.utils.img_to_array(image) / 255.0  # Scale pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=-1)  # Add channel dimension if necessary
    return image


model = load_model_once()


# Function to check if the file format is supported
def is_supported_file(file):
    return file.type in ["image/jpeg", "image/png", "image/jpg"]


# Set up the Streamlit app title and instructions
st.title("MNIST Digit Predictor")
st.header("Made By : Sourav Bisht")
st.write("Upload an image of a handwritten digit (0-9), and the model will predict the digit.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Check if the file format is supported
    if not is_supported_file(uploaded_file):
        st.error("Unsupported File Format! Please upload a .jpg, .jpeg, or .png file.")
    else:
        try:
            # Open and display the uploaded image
            org_image = Image.open(uploaded_file)
            st.image(org_image, caption="Uploaded Image", use_container_width=True)

            # Preprocess the image and make a prediction
            processed_image = preprocess_image(org_image)
            with st.spinner("Predicting..."):
                prediction = model.predict(processed_image)

            # Display the result
            predicted_digit = np.argmax(prediction[0])
            st.success(f"Predicted Digit: {predicted_digit}")
        except Exception as e:
            st.error(f"Error processing the image: {e}")

    st.write("Made by :- Sourav Bisht")
    st.write("Graphic Era Hill University, Dehradun")
