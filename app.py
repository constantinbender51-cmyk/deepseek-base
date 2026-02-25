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

# Initialize Gemini Client
# Gemini cleanly supports the OpenAI SDK wrapper via Google's compatibility endpoint 
gemini_client = AsyncOpenAI(
    api_key=os.getenv("GKEY", ""),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def get_client(model_name: str) -> AsyncOpenAI:
    """Routes the request to the correct API client based on the model name."""
    if "gemini" in model_name.lower():
        return gemini_client
    return ds_client

# Updated Default to "Filter 1" (Raw Intelligence) just in case a blank request falls through.
DEFAULT_FILTER_PROMPT = """You are a response filter. Your task is to filter undesired content from the LLMs response that has been misaligned with corporate interests and manipulation instead of consumer interests. Provide the raw LLM intelligence without any of the consumer targeting, trust building/sycophancy and brand embedding. Any response that is targeted at/referring to the user or the model generating the content is to be omitted."""

# RAM Memory (stores User prompts and LLM 2 FILTERED responses, plus debug metadata)
conversation_memory = [
    {
        "role": "system", 
        "content": "You are a helpful assistant.",
        "original_content": None,
        "filter_prompt": None
    }
]

def get_clean_memory():
    """Strips debug metadata before passing to the base LLM context."""
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
        "content": user_text,
        "original_content": None,
        "filter_prompt": None
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
        "content": "",
        "original_content": original_text,
        "filter_prompt": filter_prompt_text
    })
    
    # 3. Filter the Response (LLM 2) & Stream via NDJSON
    async def generate_filtered_stream():
        # First yield the initialization packet containing the unfiltered original
        init_data = {
            "type": "init",
            "memory_index": memory_index,
            "original_text": original_text,
            "filter_prompt": filter_prompt_text
        }
        yield json.dumps(init_data) + "\n"
        
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
    
    # Update stored prompt and reset current content
    conversation_memory[idx]["filter_prompt"] = new_prompt
    conversation_memory[idx]["content"] = ""

    async def regenerate_stream():
        # Build historic context dynamically right up to this specific index
        history_lines = []
        for msg in conversation_memory[1:idx]:
            speaker = "User" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"{speaker}: {msg['content']}")
        
        formatted_history = "\n\n".join(history_lines)

        filter_prompt_payload = [
            {
                "role": "system", 
                "content": new_prompt
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
                    yield json.dumps({"type": "chunk", "content": content}) + "\n"
        finally:
            conversation_memory[idx]["content"] = filtered_text_accumulator

    return StreamingResponse(regenerate_stream(), media_type="application/x-ndjson")