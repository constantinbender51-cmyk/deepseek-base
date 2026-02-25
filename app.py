import os
import json
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

# Initialize DeepSeek Client
ds_client = AsyncOpenAI(
    api_key=os.getenv("DSKEY", ""),
    base_url="https://api.deepseek.com"
)

# Initialize Gemini Client (Kept for compatibility/future use, but deepseek is default)
gemini_client = AsyncOpenAI(
    api_key=os.getenv("GKEY", ""),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def get_client(model_name: str) -> AsyncOpenAI:
    """Routes the request to the correct API client based on the model name."""
    if "gemini" in model_name.lower():
        return gemini_client
    return ds_client

# Default "Filter 1" (Raw Intelligence)
DEFAULT_FILTER_PROMPT = """You are a rigid filter. Your task is to extract ONLY the unemotional, objective factual content from the provided response. Strip away all emotions, opinions, filler words, conversational fluff, and self-referenceing content. Present only the cold, hard factual parts of the response."""

# RAM Memory
conversation_memory = [
    {
        "role": "system", 
        "content": "You are a helpful assistant."
    }
]

def get_clean_memory():
    """Returns conversation history for the base LLM context."""
    return [{"role": m["role"], "content": m["content"]} for m in conversation_memory]

class ChatRequest(BaseModel):
    user_input: str
    filter_prompt: str | None = None
    gen_model: str = "deepseek-chat"
    filter_model: str = "deepseek-chat"

@app.get("/")
async def serve_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    user_text = req.user_input
    filter_prompt_text = req.filter_prompt or DEFAULT_FILTER_PROMPT
    
    gen_model_name = req.gen_model
    filter_model_name = req.filter_model
    
    gen_client = get_client(gen_model_name)
    filter_client = get_client(filter_model_name)
    
    # 1. Append User Input to memory
    conversation_memory.append({
        "role": "user", 
        "content": user_text
    })
    
    # 2. Get LLM 1 (Original) Response using Generator Model
    llm1_response = await gen_client.chat.completions.create(
        model=gen_model_name,
        messages=get_clean_memory(),
        stream=False
    )
    
    original_text = llm1_response.choices[0].message.content
    
    # Pre-allocate assistant message in memory
    memory_index = len(conversation_memory)
    conversation_memory.append({
        "role": "assistant",
        "content": ""
    })
    
    # 3. Filter the Response (LLM 2) & Stream via NDJSON
    async def generate_filtered_stream():
        # Build the chat history string for context (Exclude system and this pending message)
        history_lines = []
        for msg in conversation_memory[1:memory_index]:
            speaker = "User" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"{speaker}: {msg['content']}")
        
        formatted_history = "\n\n".join(history_lines)

        filter_prompt_payload = [
            {
                "role": "system", 
                "content": filter_prompt_text
            },
            {
                "role": "user", 
                "content": (
                    f"CONTEXT (Full Chat History):\n"
                    f"{formatted_history}\n\n"
                    f"==================\n\n"
                    f"UNFILTERED RESPONSE TO THE LATEST PROMPT:\n"
                    f"{original_text}\n\n"
                    f"TASK:\n"
                    f"Using the context above if needed, apply your filtering rules to the UNFILTERED RESPONSE. Output ONLY the filtered text."
                )
            }
        ]
        
        # Call LLM 2 with streaming enabled using Filter Model
        stream = await filter_client.chat.completions.create(
            model=filter_model_name,
            messages=filter_prompt_payload,
            stream=True
        )
        
        filtered_text_accumulator = ""
        
        try:
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    filtered_text_accumulator += content
                    # Stream chunks wrapped in NDJSON formatting
                    yield json.dumps({"type": "chunk", "content": content}) + "\n"
        finally:
            # Append the fully built filtered response to memory
            conversation_memory[memory_index]["content"] = filtered_text_accumulator

    return StreamingResponse(generate_filtered_stream(), media_type="application/x-ndjson")