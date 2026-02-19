import os
from flask import Flask, request, render_template_string
from openai import OpenAI

app = Flask(__name__)

# Initialize the client pointing to OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

# Moved the default prompt out here to prevent Quote-Escaping crashes!
DEFAULT_PROMPT = '''def fast_inverse_square_root(number):
    """
    Computes the fast inverse square root
    """'''

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DeepSeek Base Engine</title>
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
    <h2>DeepSeek Raw Base Engine (via OpenRouter)</h2>
    <p>Model: <code>deepseek/deepseek-coder</code> (Base Model)</p>
    
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
                    model="deepseek/deepseek-coder", 
                    prompt=prompt,
                    max_tokens=300,
                    temperature=0.6,
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
        default_prompt=DEFAULT_PROMPT, # Pass the safe string in here
        response=response_text, 
        error=error_text
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)