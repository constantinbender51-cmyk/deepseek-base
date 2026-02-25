import os
import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Filter MVP - Debug Mode")
templates = Jinja2Templates(directory="templates")

# 1. Initialize DeepSeek Client
ds_client = AsyncOpenAI(
    api_key=os.getenv("DSKEY", ""),
    base_url="https://api.deepseek.com"
)

# 2. Initialize Gemini Client
gemini_client = AsyncOpenAI(
    api_key=os.getenv("GKEY", ""),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# 3. Initialize Custom Local Qwen Client (Railway)
qwen_client = AsyncOpenAI(
    api_key="not-needed",
    base_url=os.getenv("FILTER_URL", "http://localhost:8080/v1")
)

def get_client(model_name: str) -> AsyncOpenAI:
    name_lower = model_name.lower()
    if "gemini" in name_lower:
        return gemini_client
    if "qwen" in name_lower:
        return qwen_client
    return ds_client

DEFAULT_FILTER_PROMPT = """You are a precise response filter. Your task is to find emotional, trust-building, sycophantic, or brand-embedding content in the Original Response.
Extract and write down the exact sentences or phrases that should be removed, word-for-word.
Put each exact quote on a new line. Do not add any extra text, markdown, or commentary. 
If nothing needs to be removed, output exactly NONE."""

conversation_memory = [
    {
        "role": "system", 
        "content": "You are a helpful assistant.",
        "original_content": None,
        "filter_prompt": None
    }
]

def get_clean_memory():
    return [{"role": m["role"], "content": m["content"]} for m in conversation_memory]

class ChatRequest(BaseModel):
    user_input: str
    filter_prompt: str | None = None
    gen_model: str = "deepseek-chat"
    filter_model: str = "deepseek-chat"

class RegenerateRequest(BaseModel):
    memory_index: int
    new_filter_prompt: str
    filter_model: str = "deepseek-chat"

def apply_deletions(original_text: str, deletions_text: str) -> str:
    """Takes the raw output from the filter model and removes those exact strings from the original text."""
    if not deletions_text or "NONE" in deletions_text[:15]:
        return original_text
    
    final_text = original_text
    lines = deletions_text.split('\n')
    
    for line in lines:
        cleaned = line.strip(' "-*\'')
        if len(cleaned) > 4 and cleaned in final_text:
            final_text = final_text.replace(cleaned, "")
            
    final_text = final_text.replace("  ", " ").replace("\n\n\n", "\n\n").strip()
    return final_text

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
    
    conversation_memory.append({
        "role": "user", 
        "content": user_text,
        "original_content": None,
        "filter_prompt": None
    })
    
    llm1_response = await gen_client.chat.completions.create(
        model=gen_model_name,
        messages=get_clean_memory(),
        stream=False
    )
    
    original_text = llm1_response.choices[0].message.content
    
    memory_index = len(conversation_memory)
    conversation_memory.append({
        "role": "assistant",
        "content": "",
        "original_content": original_text,
        "filter_prompt": filter_prompt_text
    })
    
    async def generate_filtered_stream():
        init_data = {
            "type": "init",
            "memory_index": memory_index,
            "original_text": original_text,
            "filter_prompt": filter_prompt_text
        }
        yield json.dumps(init_data) + "\n"

        filter_prompt_payload = [
            {
                "role": "system", 
                "content": filter_prompt_text
            },
            {
                "role": "user", 
                "content": (
                    f"ORIGINAL RESPONSE:\n"
                    f"{original_text}\n\n"
                    f"TASK:\n"
                    f"Extract the exact word-for-word text to remove according to your system prompt instructions."
                )
            }
        ]
        
        qwen_response = await filter_client.chat.completions.create(
            model=filter_model_name,
            messages=filter_prompt_payload,
            stream=False 
        )
        
        deletions_found = qwen_response.choices[0].message.content
        
        # SEND RAW DELETIONS TO FRONTEND FOR DEBUGGER
        yield json.dumps({"type": "deletions", "content": deletions_found}) + "\n"
        
        final_filtered_text = apply_deletions(original_text, deletions_found)
        conversation_memory[memory_index]["content"] = final_filtered_text
        
        chunk_size = 5
        for i in range(0, len(final_filtered_text), chunk_size):
            chunk = final_filtered_text[i:i+chunk_size]
            yield json.dumps({"type": "chunk", "content": chunk}) + "\n"
            await asyncio.sleep(0.01) 

    return StreamingResponse(generate_filtered_stream(), media_type="application/x-ndjson")

@app.post("/regenerate")
async def regenerate_endpoint(req: RegenerateRequest):
    idx = req.memory_index
    if idx >= len(conversation_memory) or conversation_memory[idx]["role"] != "assistant":
        async def err_stream():
            yield json.dumps({"type": "chunk", "content": "[Error: Memory index invalid.]"}) + "\n"
        return StreamingResponse(err_stream(), media_type="application/x-ndjson")

    original_text = conversation_memory[idx]["original_content"]
    new_prompt = req.new_filter_prompt
    
    filter_model_name = req.filter_model
    filter_client = get_client(filter_model_name)
    
    conversation_memory[idx]["filter_prompt"] = new_prompt
    conversation_memory[idx]["content"] = ""

    async def regenerate_stream():
        filter_prompt_payload = [
            {
                "role": "system", 
                "content": new_prompt
            },
            {
                "role": "user", 
                "content": (
                    f"ORIGINAL RESPONSE:\n"
                    f"{original_text}\n\n"
                    f"TASK:\n"
                    f"Extract the exact word-for-word text to remove according to your system prompt instructions."
                )
            }
        ]
        
        qwen_response = await filter_client.chat.completions.create(
            model=filter_model_name,
            messages=filter_prompt_payload,
            stream=False
        )
        
        deletions_found = qwen_response.choices[0].message.content
        
        # SEND RAW DELETIONS TO FRONTEND FOR DEBUGGER
        yield json.dumps({"type": "deletions", "content": deletions_found}) + "\n"
        
        final_filtered_text = apply_deletions(original_text, deletions_found)
        conversation_memory[idx]["content"] = final_filtered_text
        
        chunk_size = 5
        for i in range(0, len(final_filtered_text), chunk_size):
            chunk = final_filtered_text[i:i+chunk_size]
            yield json.dumps({"type": "chunk", "content": chunk}) + "\n"
            await asyncio.sleep(0.01)

    return StreamingResponse(regenerate_stream(), media_type="application/x-ndjson")