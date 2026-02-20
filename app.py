from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
import io
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="Moondream API")

MODEL_ID = "vikhyatk/moondream2"
REVISION = "2024-08-26" 

model = None
tokenizer = None

# 1. This function will run in the background
def load_ai_model():
    global model, tokenizer
    print("Starting background AI download/load... The web server is already live!")
    temp_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    temp_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True,
        revision=REVISION
    )
    temp_model.eval()
    
    # 2. Assign globals only when fully finished
    tokenizer = temp_tokenizer
    model = temp_model
    print("Moondream Model loaded successfully and is ready for questions!")

# 3. Start the background thread when FastAPI boots
@app.on_event("startup")
def startup_event():
    print("Booting FastAPI web server...")
    thread = threading.Thread(target=load_ai_model)
    thread.start()

# 4. Check status from your browser
@app.get("/")
def read_root():
    if model is None:
        return {"status": "Web server is online! However, the AI is still loading in the background. Please wait a minute."}
    return {"status": "Moondream API is fully loaded and ready to answer questions!"}

@app.post("/ask")
async def ask_question(prompt: str = Form(...), image: UploadFile = File(...)):
    # If the user tries to ask a question before it finishes loading
    if model is None:
        return JSONResponse(
            status_code=503, 
            content={"detail": "The AI model is still booting up. Please wait a minute and try again."}
        )

    # Process image and generate answer
    image_bytes = await image.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    enc_image = model.encode_image(img)
    answer = model.answer_question(enc_image, prompt, tokenizer)
    
    return {"prompt": prompt, "answer": answer}