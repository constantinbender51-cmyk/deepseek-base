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

# Initialize Gemini Client (LLM 1 - Original Response)
# Using Google's OpenAI-compatible endpoint so message formatting remains identical
gemini_client = AsyncOpenAI(
    api_key=os.getenv("GKEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Initialize DeepSeek Client (LLM 2 - Filter Response)
deepseek_client = AsyncOpenAI(
    api_key=os.getenv("DSKEY"),
    base_url="https://api.deepseek.com"
)

# RAM Memory (stores User prompts and LLM 2 FILTERED responses)
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
    
    # 2. Get LLM 1 (Original) Response using GEMINI
    llm1_response = await gemini_client.chat.completions.create(
        model="gemini-2.0-flash", # Note: Change to "gemini-1.5-flash" if "2.0" is unavailable in your tier
        messages=conversation_memory,
        stream=False
    )
    
    original_text = llm1_response.choices[0].message.content
    
    # 3. Filter the Response (LLM 2) & Stream using DEEPSEEK
    async def generate_filtered_stream():
        
        # --- Build the chat history string for context ---
        history_lines = []
        # Skip the first element (System prompt)
        for msg in conversation_memory[1:]:
            speaker = "User" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"{speaker}: {msg['content']}")
        
        formatted_history = "\n\n".join(history_lines)
        # ------------------------------------------------------

        # --- FILTER PROMPT ---
        filter_prompt = [
            {
                "role": "system", 
                "content": "You are a rigid filter. Strip the text from emotions, opinions, filler words, conversational fluff, and brand embedding/identify priming." 
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
        
        # Call LLM 2 (DeepSeek) with streaming enabled
        stream = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=filter_prompt,
            stream=True
        )
        
        filtered_text_accumulator = ""
        
        try:
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    filtered_text_accumulator += content
                    yield content
        finally:
            # 4. Append the fully built filtered response to memory
            if filtered_text_accumulator:
                conversation_memory.append({
                    "role": "assistant", 
                    "content": filtered_text_accumulator
                })

    return StreamingResponse(generate_filtered_stream(), media_type="text/plain")