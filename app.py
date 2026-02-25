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

# Initialize DeepSeek Client (Used for generation)
ds_client = AsyncOpenAI(
    api_key=os.getenv("DSKEY", ""),
    base_url="https://api.deepseek.com"
)

# Initialize Qwen/Railway Client (Used for Filtering)
# Make sure to add FILTER_URL to your .env or Railway environment variables!
# It should look like: https://your-railway-url.app/v1
qwen_client = AsyncOpenAI(
    api_key="not-needed", # llama-cpp-python doesn't require an API key
    base_url=os.getenv("FILTER_URL", "http://localhost:8000/v1")
)

def get_client(model_name: str) -> AsyncOpenAI:
    """Routes the request to the correct API client based on the model name."""
    if "qwen" in model_name.lower():
        return qwen_client
    return ds_client

# Default "Filter 1" (Raw Intelligence)
DEFAULT_FILTER_PROMPT = """You are a response filter. Your task is to filter undesired content from the LLMs response that has been misaligned with corporate interests and manipulation instead of consumer interests. Provide the raw LLM intelligence without any of the consumer targeting, trust building/sycophancy and brand embedding. Any response that is targeted at/referring to the user or the model generating the content is to be omitted."""

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
    # Defaulting the filter model to our new Qwen model endpoint
    filter_model: str = "qwen" 

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
    
    # 2. Get LLM 1 (Original) Response using Generator Model (DeepSeek)
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
    
    # 3. Filter the Response (LLM 2) using Qwen & Stream via NDJSON
    async def generate_filtered_stream():
        # Build the chat history string for context
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
        
        # Call the Railway Qwen Model with streaming enabled
        stream = await filter_client.chat.completions.create(
            # llama-cpp-python ignores the model name as it only loads one at a time, 
            # but we pass "qwen" to fulfill the Pydantic requirement
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
                    yield json.dumps({"type": "chunk", "content": content}) + "\n"
        finally:
            conversation_memory[memory_index]["content"] = filtered_text_accumulator

    return StreamingResponse(generate_filtered_stream(), media_type="application/x-ndjson")