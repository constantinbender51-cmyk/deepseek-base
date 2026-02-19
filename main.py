import os
from flask import Flask, request, render_template_string
from openai import OpenAI

app = Flask(__name__)

# Initialize the client pointing to OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

# A great test for a pure base model. It will just continue the thought.
DEFAULT_PROMPT = '''The concept of time travel has fascinated humanity for centuries. However, the very first successful temporal displacement did not occur in a massive government laboratory, but rather'''

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Llama 405B Base Engine</title>
    <style>
        body { font-family: monospace; max-width: 800px; margin: 2rem auto; padding: 0 1rem; color: #000; }
        textarea { width: 100%; height: 150px; margin-bottom: 1rem; font-family: inherit; box-sizing: border-box; }
        button { padding: 0.5rem 1rem; font-family: inherit; cursor: pointer; background: #eee; border: 1px solid #000; }
        button:hover { background: #ddd; }
        .output-box { margin-top: 2rem; padding: 1rem; border: 1px solid #000; background: #f9f9f9; white-space: pre-wrap; word-wrap: break-word; }
        .error-box { margin-top: 2rem; padding: 1rem; border: 1px solid red; color: red; }
    </style>
</head>
<body>
    <h2>Llama 3.1 405B - Raw Base Engine</h2>
    <p>Model: <code>meta-llama/llama-3.1-405b</code> (Pure Base)</p>
    
    <form method="POST">
        <label for="prompt">Text to Complete:</label><br><br>
        <textarea id="prompt" name="prompt" required autofocus>{{ prompt if prompt else default_prompt }}</textarea><br>
        <button type="submit">Run Engine</button>
    </form>

    {% if response %}
    <div class="output-box"><strong>Continued Text:</strong><br><br>{{ response }}</div>
    {% endif %}
    
    {% if error %}
    <div class="error-box">{{ error }}</div>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prompt = ""
    response_text = ""
    error_text = ""

    if request.method == "POST":
        prompt = request.form.get("prompt", "")
        if prompt:
            try:
                # Use standard completions for the base engine
                response = client.completions.create(
                    model="meta-llama/llama-3.1-405b", # The pure Llama 405B base model string on OpenRouter
                    prompt=prompt,
                    max_tokens=300,  # Let it write a good paragraph or two
                    temperature=0.7, # Good balance of creativity and logic
                    extra_headers={
                        "HTTP-Referer": "https://your-app-url.com", # Optional for OpenRouter
                        "X-Title": "My Base Model App",             # Optional for OpenRouter
                    }
                )
                
                # Append the raw completion seamlessly to your prompt
                response_text = prompt + response.choices[0].text
                
            except Exception as e:
                error_text = f"API Error: {str(e)}"

    return render_template_string(
        HTML_TEMPLATE, 
        prompt=prompt, 
        default_prompt=DEFAULT_PROMPT, 
        response=response_text, 
        error=error_text
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)