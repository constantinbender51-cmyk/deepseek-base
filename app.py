import os
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Filter MVP")
templates = Jinja2Templates(directory="templates")

# Initialize DeepSeek Client (using OpenAI SDK)
client = AsyncOpenAI(
    api_key=os.getenv("DSKEY"),
    base_url="https://api.deepseek.com" # DeepSeek base URL
)

# RAM Memory (stores User prompts and LLM 1 original responses)
# Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
conversation_memory = [
    {"role": "system", "content": "You are a helpful assistant."}
]

class ChatRequest(BaseModel):
    user_input: str

@app.get("/")
async def serve_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    user_text = req.user_input
    
    # 1. Append User Input to memory
    conversation_memory.append({"role": "user", "content": user_text})
    
    # 2. Get LLM 1 (Original) Response
    # We wait for this to finish entirely (Delay acceptable for testing)
    llm1_response = await client.chat.completions.create(
        model="deepseek-chat",
        messages=conversation_memory,
        stream=False
    )
    
    original_text = llm1_response.choices[0].message.content
    
    # 3. Append LLM 1 original response to memory (so future prompts use it)
    conversation_memory.append({"role": "assistant", "content": original_text})
    
    # 4. Filter the Response (LLM 2) & Stream
    async def generate_filtered_stream():
        filter_prompt = [
            {"role": "system", "content": "You are a rigid filter. Your task is to extract ONLY the unemotional, objective factual content from the provided text. Strip away all emotions, opinions, filler words, and conversational fluff. Present only the cold, hard facts."},
            {"role": "user", "content": f"Filter this text: {original_text}"}
        ]
        
        # Call LLM 2 with streaming enabled
        stream = await client.chat.completions.create(
            model="deepseek-chat",
            messages=filter_prompt,
            stream=True
        )
        
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    return StreamingResponse(generate_filtered_stream(), media_type="text/plain")