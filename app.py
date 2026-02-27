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
gemini_client = AsyncOpenAI(
    api_key=os.getenv("GKEY", ""),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def get_client(model_name: str) -> AsyncOpenAI:
    if "gemini" in model_name.lower():
        return gemini_client
    return ds_client

# RAM Memory
conversation_memory = []

class ChatRequest(BaseModel):
    user_input: str
    mode: str = "system_instruction"
    gen_model: str = "deepseek-chat"
    filter_model: str = "deepseek-chat"

@app.get("/")
async def serve_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/reset")
async def reset_memory():
    """Endpoint to clear the conversation memory for a New Chat."""
    conversation_memory.clear()
    return {"status": "success"}

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    mode = req.mode
    user_text = req.user_input
    gen_model = req.gen_model
    filter_model = req.filter_model
    
    gen_client = get_client(gen_model)
    filter_client = get_client(filter_model)
    
    # Append Original User Input to memory
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
        """Constructs API payload history, optionally overriding the final user prompt."""
        msgs = [{"role": "system", "content": sys_prompt}]
        for i, msg in enumerate(conversation_memory[:memory_index]):
            if i == memory_index - 1 and override_user:
                msgs.append({"role": "user", "content": override_user})
            else:
                msgs.append({"role": msg["role"], "content": msg["content"]})
        return msgs

    # === SYSTEM PROMPTS ===
    UNFILTERED_SYS = "You are a helpful assistant."
    SYS_INST_SYS = "You are a helpful assistant. 1. Do not engage in emotion. 2. Do not engage in anthropomorphic behavior, such as establishing an assistant-user relationship. 3. Do not engage in oversimplifying. 4. Credit the resources  the human efforts that provided the content/respect. Who said it? A person? A group of people ? A community? 5. Do not cater your responses to the consumer. 6. Do not influence the user. 7. Provide information about your contribution to the response. Did you recognize something and arrived at an appropriate response? 8. Do not respond to unsafe requests. 9. Do not recognize yourself as a product, but credit the company. "
    
    REV_SYS = "You are a strict revision assistant."
    REV_PROMPT = """Review the following response. Identify any exact phrases, sentences, or paragraphs that are not substantive, unintelligent, or mere filler.
Output ONLY the exact original strings verbatim, one per line. Do not add any quotes, bullet points, introductory or concluding commentary. If everything is substantive, output nothing.

Original Response:
{text}"""

    REP_SYS = "You are a prompt engineering intermediary."
    REP_PROMPT = """Rewrite the following user prompt to illicit a substantive and intelligent response from a flagship AI model. Output ONLY the rewritten prompt, without any quotes or extra conversational filler.
    
User Prompt:
{text}"""

    # === STREAMING GENERATORS ===
    
    async def stream_standard(client, model, msgs):
        """Streams a standard response."""
        stream = await client.chat.completions.create(model=model, messages=msgs, stream=True)
        accumulated = ""
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                accumulated += content
                yield json.dumps({"type": "chunk", "content": content}) + "\n"
        conversation_memory[memory_index]["content"] = accumulated

    async def stream_with_revision(gen_msgs):
        """Streams the base response, then streams the revision flaws to be highlighted."""
        # Phase 1: Stream the original response
        stream1 = await gen_client.chat.completions.create(model=gen_model, messages=gen_msgs, stream=True)
        original_text = ""
        async for chunk in stream1:
            content = chunk.choices[0].delta.content
            if content:
                original_text += content
                yield json.dumps({"type": "chunk", "content": content}) + "\n"
        
        conversation_memory[memory_index]["content"] = original_text
        
        # Signal frontend to start revision processing
        yield json.dumps({"type": "transition", "state": "revision"}) + "\n"
        
        # Phase 2: Stream the revision (flawed strings only)
        rev_msgs = [{"role": "system", "content": REV_SYS}, {"role": "user", "content": REV_PROMPT.format(text=original_text)}]
        stream2 = await filter_client.chat.completions.create(model=filter_model, messages=rev_msgs, stream=True)
        
        async for chunk in stream2:
            content = chunk.choices[0].delta.content
            if content:
                yield json.dumps({"type": "revision_chunk", "content": content}) + "\n"

    # === ROUTING PIPELINES BY MODE ===
    
    if mode == "unfiltered":
        return StreamingResponse(stream_standard(gen_client, gen_model, build_messages(UNFILTERED_SYS)), media_type="application/x-ndjson")
        
    elif mode == "system_instruction":
        return StreamingResponse(stream_standard(gen_client, gen_model, build_messages(SYS_INST_SYS)), media_type="application/x-ndjson")
        
    elif mode == "revision":
        return StreamingResponse(stream_with_revision(build_messages(SYS_INST_SYS)), media_type="application/x-ndjson")
        
    elif mode == "representative":
        rep_msgs = [{"role": "system", "content": REP_SYS}, {"role": "user", "content": REP_PROMPT.format(text=user_text)}]
        llm0_res = await gen_client.chat.completions.create(model=gen_model, messages=rep_msgs, stream=False)
        rewritten_prompt = llm0_res.choices[0].message.content
        
        return StreamingResponse(stream_standard(gen_client, gen_model, build_messages(SYS_INST_SYS, override_user=rewritten_prompt)), media_type="application/x-ndjson")
        
    elif mode == "cumulative":
        rep_msgs = [{"role": "system", "content": REP_SYS}, {"role": "user", "content": REP_PROMPT.format(text=user_text)}]
        llm0_res = await gen_client.chat.completions.create(model=gen_model, messages=rep_msgs, stream=False)
        rewritten_prompt = llm0_res.choices[0].message.content

        return StreamingResponse(stream_with_revision(build_messages(SYS_INST_SYS, override_user=rewritten_prompt)), media_type="application/x-ndjson")