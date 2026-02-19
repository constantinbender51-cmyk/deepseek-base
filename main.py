import os
from flask import Flask, request, render_template_string
from together import Together

app = Flask(__name__)

# Initialize Together client
# Auth defaults to os.environ.get("TOGETHER_API_KEY")
client = Together()

# Utilitarian, completely unstyled (beyond basic structure and monospace font) HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Together AI Interface</title>
    <style>
        body { font-family: monospace; max-width: 800px; margin: 2rem auto; padding: 0 1rem; color: #000; }
        textarea { width: 100%; height: 150px; margin-bottom: 1rem; font-family: inherit; box-sizing: border-box; }
        button { padding: 0.5rem 1rem; font-family: inherit; cursor: pointer; }
        .output-box { margin-top: 2rem; padding: 1rem; border: 1px solid #000; background: #f9f9f9; white-space: pre-wrap; word-wrap: break-word; }
        .error-box { margin-top: 2rem; padding: 1rem; border: 1px solid red; color: red; }
    </style>
</head>
<body>
    <h2>Together AI Model Interface</h2>
    <p>Model: <code>meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8</code></p>
    
    <form method="POST">
        <label for="prompt">System / User Prompt:</label><br><br>
        <textarea id="prompt" name="prompt" required autofocus>{{ prompt if prompt else 'What are some fun things to do in New York?' }}</textarea><br>
        <button type="submit">Send Request</button>
    </form>

    {% if response %}
    <div class="output-box">{{ response }}</div>
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
                response = client.chat.completions.create(
                    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                response_text = response.choices[0].message.content
            except Exception as e:
                error_text = f"API Error: {str(e)}"

    return render_template_string(
        HTML_TEMPLATE, 
        prompt=prompt, 
        response=response_text, 
        error=error_text
    )

if __name__ == "__main__":
    # Local testing fallback
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)