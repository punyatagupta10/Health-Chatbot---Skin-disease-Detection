import streamlit as st
import numpy as np
# import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# ✅ Load the trained model
@st.cache_resource
def load_saved_model():
    return load_model("model.h5")

model = load_saved_model()

# ✅ Define Class Labels (Updated to match dataset)
categories = [
    'Atopic Dermatitis', 'Basal Cell Carcinoma', 'Benign Keratosis-like Lesions (BKL)', 
    'Eczema', 'Melanocytic Nevi', 'Melanoma', 'Psoriasis and Lichen Planus', 
    'Seborrheic Keratoses', 'Tinea Ringworm Candidiasis', 'Warts Molluscum'
]

# ✅ Streamlit UI
st.title("Skin Disease Classification App")
st.write("Upload an image, and the model will classify it.")

# ✅ Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # ✅ Image Preprocessing
    img = load_img(uploaded_file, target_size=(125, 125))  # Resize to match model input
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand dims for batch

    # ✅ Model Prediction
    pred = model.predict(img_array)
    predicted_class_index = np.argmax(pred, axis=1)[0]  # Get class index
    predicted_class_name = categories[predicted_class_index]  # Convert index to class name

    # ✅ Display Prediction Output (Same as in Kaggle)
    st.write("### Prediction Result:")
    st.success(f"Predicted Case → {predicted_class_name}")

   
