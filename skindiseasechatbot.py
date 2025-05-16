import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from dotenv import load_dotenv
import os
import tempfile
import google.generativeai as genai
from deep_translator import GoogleTranslator
import speech_recognition as sr
from pydub import AudioSegment
from langdetect import detect
from audio_recorder_streamlit import audio_recorder  # ✅ Browser-based voice input

# ✅ Load skin disease model
@st.cache_resource
def load_saved_model():
    return load_model("model.h5")

model = load_saved_model()

# ✅ Disease labels
categories = [
    'Atopic Dermatitis', 'Basal Cell Carcinoma', 'Benign Keratosis-like Lesions (BKL)', 
    'Eczema', 'Melanocytic Nevi', 'Melanoma', 'Psoriasis and Lichen Planus', 
    'Seborrheic Keratoses', 'Tinea Ringworm Candidiasis', 'Warts Molluscum'
]

# ✅ Load Gemini AI
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model_gemini = genai.GenerativeModel("gemini-1.5-flash")
chat = model_gemini.start_chat(history=[])

# ✅ Translator functions
def translate_to_english(text):
    return GoogleTranslator(source='auto', target='en').translate(text)

def translate_to_hindi(text):
    return GoogleTranslator(source='en', target='hi').translate(text)

# ✅ Gemini prompt logic
def get_response(input_text):
    try:
        prompt = f"""
        You are a medical assistant AI. When given a disease name, provide details in this format:
        
        **Condition Name:** <Definition>  
        **Symptoms:** <List of symptoms>  
        **Causes:** <Possible causes>  
        **Treatment Options:** <Common treatments>  
        **When to See a Doctor:** <Advice on seeking professional help>  

        If the input is not a disease but general text, respond naturally.

        The input is: "{input_text}"
        """
        response = chat.send_message(prompt, stream=True)
        return response
    except Exception as e:
        st.error(f"Error fetching response: {e}")
        return []

# ✅ App UI
st.title("🩺 MediBot - AI-Powered Medical Assistant")
st.write("Choose an input method: 📷 Image | 📝 Text | 🎙️ Voice")

option = st.radio("Select Input Method:", ("Text Input", "Voice Input", "Image Upload"))

# 1️⃣ TEXT INPUT
if option == "Text Input":
    user_input = st.text_input("Ask your question (in Hindi or English):", placeholder="e.g. पेट का इंफेक्शन क्या होता है?")
    submit = st.button("Get Answer")

    if submit and user_input:
        with st.spinner("Detecting language and processing..."):
            detected_lang = detect(user_input)
            translated_input = translate_to_english(user_input)

            response_chunks = get_response(translated_input)
            if response_chunks:
                response_en = "".join([chunk.text for chunk in response_chunks])
                response = translate_to_hindi(response_en) if detected_lang == "hi" else response_en
                st.write("### 🤖 Response:")
                st.info(response)
            else:
                st.warning("Unable to fetch response.")

# 2️⃣ VOICE INPUT
elif option == "Voice Input":
    st.write("🎙️ Record your medical query (Hindi or English):")
    audio_bytes = audio_recorder()

    if audio_bytes:
        st.audio(audio_bytes, format='audio/wav')

        with st.spinner("Processing audio..."):
            try:
                # Save and load audio
                temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                temp_audio.write(audio_bytes)
                temp_audio.close()

                recognizer = sr.Recognizer()
                with sr.AudioFile(temp_audio.name) as source:
                    audio_data = recognizer.record(source)
                    spoken_text = recognizer.recognize_google(audio_data)

                st.success(f"Recognized: {spoken_text}")
                detected_lang = detect(spoken_text)
                translated_input = translate_to_english(spoken_text)

                response_chunks = get_response(translated_input)
                if response_chunks:
                    response_en = "".join([chunk.text for chunk in response_chunks])
                    response = translate_to_hindi(response_en) if detected_lang == "hi" else response_en
                    st.write("### 🤖 Response:")
                    st.info(response)
                else:
                    st.warning("No response received.")
            except Exception as e:
                st.error(f"Audio processing failed: {e}")

# 3️⃣ IMAGE UPLOAD
elif option == "Image Upload":
    uploaded_file = st.file_uploader("Upload a skin image:", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        img = load_img(uploaded_file, target_size=(125, 125))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)
        predicted_class_index = np.argmax(pred, axis=1)[0]
        predicted_class_name = categories[predicted_class_index]

        st.success(f"Predicted Condition: {predicted_class_name}")

        with st.spinner("Getting medical info..."):
            response_chunks = get_response(predicted_class_name)
            if response_chunks:
                response_en = "".join([chunk.text for chunk in response_chunks])
                response_hi = translate_to_hindi(response_en)

                st.write("### 📗 Medical Info (English):")
                st.info(response_en)

                st.write("### 📘 जानकारी (Hindi):")
                st.info(response_hi)
            else:
                st.warning("Could not retrieve additional details.")
