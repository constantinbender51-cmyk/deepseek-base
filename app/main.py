from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import replicate
import os

app = FastAPI()

# Load Replicate API key from environment
REPLICATE_API_KEY = os.getenv("RKEY")
if not REPLICATE_API_KEY:
    raise ValueError("RKEY environment variable not set")

# Initialize Replicate client
client = replicate.Client(api_token=REPLICATE_API_KEY)

# Mount static files (for frontend)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/static")

# API Endpoint to interact with Replicate
@app.post("/predict")
async def predict(prompt: str):
    try:
        deployment = client.deployments.get("constantinbender51-cmyk/deepseek-base")
        prediction = deployment.predictions.create(input={"prompt": prompt})
        prediction.wait()
        return {"output": prediction.output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Simple HTML frontend (optional)
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
