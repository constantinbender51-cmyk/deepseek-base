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

# Initialize Gemini Client (Kept for compatibility)
gemini_client = AsyncOpenAI(
    api_key=os.getenv("GKEY", ""),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def get_client(model_name: str) -> AsyncOpenAI:
    """Routes the request to the correct API client based on the model name."""
    if "gemini" in model_name.lower():
        return gemini_client
    return ds_client

# RAM Memory (now starts empty, system instructions are injected dynamically)
conversation_memory = []

class ChatRequest(BaseModel):
    user_input: str
    mode: str = "system_instruction"
    gen_model: str = "deepseek-chat"
    filter_model: str = "deepseek-chat"

@app.get("/")
async def serve_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    mode = req.mode
    user_text = req.user_input
    gen_model = req.gen_model
    filter_model = req.filter_model
    
    gen_client = get_client(gen_model)
    filter_client = get_client(filter_model)
    
    # 1. Append Original User Input to memory
    conversation_memory.append({
        "role": "user", 
        "content": user_text
    })
    memory_index = len(conversation_memory)
    
    # Pre-allocate assistant message in memory
    conversation_memory.append({
        "role": "assistant",
        "content": ""
    })
    
    def build_messages(sys_prompt: str, override_user: str = None):
        """Constructs API payload history, optionally overriding the final user prompt (used for Representative mode)"""
        msgs = [{"role": "system", "content": sys_prompt}]
        for i, msg in enumerate(conversation_memory[:memory_index]):
            if i == memory_index - 1 and override_user:
                msgs.append({"role": "user", "content": override_user})
            else:
                msgs.append({"role": msg["role"], "content": msg["content"]})
        return msgs

    async def stream_response(client, model, msgs):
        """Helper to call LLM, stream via NDJSON, and save final response to RAM."""
        stream = await client.chat.completions.create(
            model=model, messages=msgs, stream=True
        )
        accumulated = ""
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                accumulated += content
                yield json.dumps({"type": "chunk", "content": content}) + "\n"
        conversation_memory[memory_index]["content"] = accumulated

    # === SYSTEM PROMPTS & DEFINITIONS ===
    UNFILTERED_SYS = "You are a helpful assistant."
    SYS_INST_SYS = "You are a helpful assistant. However, the user wishes a substantive, intelligent response."
    
    REV_SYS = "You are a strict revision assistant."
    REV_PROMPT = """Review the following response. Flag what isn't substantive or intelligent by outputting the exact original text verbatim, but wrap any unintelligent, filler, or non-substantive strings with a hardcoded red marker using HTML: <span style='color: red; background-color: #fee2e2; padding: 0 2px; border-radius: 4px; font-weight: bold;'>...</span>. 
Do not summarize, do not alter the good parts of the text, and do not add any introductory/concluding commentary.
    
Original Response:
{text}"""

    REP_SYS = "You are a prompt engineering intermediary."
    REP_PROMPT = """Rewrite the following user prompt to illicit a substantive and intelligent response from a flagship AI model. Output ONLY the rewritten prompt, without any quotes or extra conversational filler.
    
User Prompt:
{text}"""

    # === ROUTING PIPELINES BY MODE ===
    
    if mode == "unfiltered":
        return StreamingResponse(
            stream_response(gen_client, gen_model, build_messages(UNFILTERED_SYS)), 
            media_type="application/x-ndjson"
        )
        
    elif mode == "system_instruction":
        return StreamingResponse(
            stream_response(gen_client, gen_model, build_messages(SYS_INST_SYS)), 
            media_type="application/x-ndjson"
        )
        
    elif mode == "revision":
        # Step 1: Base Generation (Wait for completion)
        llm1_res = await gen_client.chat.completions.create(
            model=gen_model, messages=build_messages(SYS_INST_SYS), stream=False
        )
        original_text = llm1_res.choices[0].message.content
        
        # Step 2: Stream Revision Pass
        rev_msgs = [{"role": "system", "content": REV_SYS}, {"role": "user", "content": REV_PROMPT.format(text=original_text)}]
        return StreamingResponse(
            stream_response(filter_client, filter_model, rev_msgs), 
            media_type="application/x-ndjson"
        )
        
    elif mode == "representative":
        # Step 1: Intermediary Prompt Rewrite
        rep_msgs = [{"role": "system", "content": REP_SYS}, {"role": "user", "content": REP_PROMPT.format(text=user_text)}]
        llm0_res = await gen_client.chat.completions.create(
            model=gen_model, messages=rep_msgs, stream=False
        )
        rewritten_prompt = llm0_res.choices[0].message.content
        
        # Step 2: Generate Answer based on rewritten prompt
        return StreamingResponse(
            stream_response(gen_client, gen_model, build_messages(SYS_INST_SYS, override_user=rewritten_prompt)), 
            media_type="application/x-ndjson"
        )
        
    elif mode == "cumulative":
        # Step 1: Intermediary Prompt Rewrite
        rep_msgs = [{"role": "system", "content": REP_SYS}, {"role": "user", "content": REP_PROMPT.format(text=user_text)}]
        llm0_res = await gen_client.chat.completions.create(
            model=gen_model, messages=rep_msgs, stream=False
        )
        rewritten_prompt = llm0_res.choices[0].message.content

        # Step 2: Base Generation (Wait for completion)
        llm1_res = await gen_client.chat.completions.create(
            model=gen_model, messages=build_messages(SYS_INST_SYS, override_user=rewritten_prompt), stream=False
        )
        original_text = llm1_res.choices[0].message.content

        # Step 3: Stream Revision Pass (Highlight flaws)
        rev_msgs = [{"role": "system", "content": REV_SYS}, {"role": "user", "content": REV_PROMPT.format(text=original_text)}]
        return StreamingResponse(
            stream_response(filter_client, filter_model, rev_msgs), 
            media_type="application/x-ndjson"
        )
