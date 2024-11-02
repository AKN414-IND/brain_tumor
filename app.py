import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Load the pre-trained model
model = load_model('Brain.h5')

# Define the classes (these should correspond to the model's output classes)
class_names = ['Glioma', 'Meningioma', 'Pituitary Tumor', 'No Tumor']

# Function to load and preprocess the image
def load_and_preprocess_image(image):
    img = load_img(image, target_size=(224, 224))  # Resize the image to the size expected by the model
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img /= 255.0  # Normalize to [0, 1] range
    return img

# Streamlit app
st.title("Brain Tumor Classification")

# File uploader
uploaded_file = st.file_uploader("Upload an MRI scan image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image and predict
    img = load_and_preprocess_image(uploaded_file)
    prediction = model.predict(img)
    prediction_class = np.argmax(prediction, axis=1)

    # Output the class with the highest probability
    st.write(f"Prediction: **{class_names[prediction_class[0]]}**")
