import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Stock Image Describer", page_icon="🖼️", layout="centered")

# --- LOAD AI MODEL (CACHED) ---
# st.cache_resource ensures the model is only loaded into memory once when the app starts
@st.cache_resource
def load_model():
    model_id = "vikhyatk/moondream2"
    revision = "2024-08-26" # Using a stable revision
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    # Load model. We let PyTorch handle CPU mapping automatically.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        revision=revision
    )
    model.eval() # Set model to evaluation mode
    return tokenizer, model

with st.spinner("Loading AI Model into CPU Memory... (This takes a minute on startup)"):
    tokenizer, model = load_model()

# --- USER INTERFACE ---
st.title("🖼️ Stock Image Describer")
st.write("Upload a stock image to generate descriptions, alt-text, or tags using an AI model running on CPU.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
prompt = st.text_input("What do you want to know?", value="Write a detailed description of this stock image.")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Text", type="primary"):
        with st.spinner("Analyzing image and generating text..."):
            try:
                # 1. Encode the image
                enc_image = model.encode_image(image)
                
                # 2. Ask the model the question
                answer = model.answer_question(enc_image, prompt, tokenizer)
                
                # 3. Output the result
                st.success("Generation Complete!")
                st.write("### Result:")
                st.info(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")