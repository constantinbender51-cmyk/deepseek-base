from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import io
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="Moondream API")

# Define model ID
MODEL_ID = "vikhyatk/moondream2"
REVISION = "2024-08-26" # Pinned for stability

# Global variables for model and tokenizer
model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    print("Loading Moondream model... This may take a minute on the first boot.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    # We load standard CPU inference here
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True,
        revision=REVISION
    )
    model.eval()
    print("Model loaded successfully!")

@app.get("/")
def read_root():
    return {"status": "Moondream API is running. Send a POST request to /ask"}

@app.post("/ask")
async def ask_question(prompt: str = Form(...), image: UploadFile = File(...)):
    # Read the image file
    image_bytes = await image.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Process image and generate answer
    enc_image = model.encode_image(img)
    answer = model.answer_question(enc_image, prompt, tokenizer)
    
    return {"prompt": prompt, "answer": answer}