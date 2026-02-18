import os
from flask import Flask, render_template, request, jsonify
import replicate

app = Flask(__name__)

# 1. Configuration
# Replicate SDK looks for 'REPLICATE_API_TOKEN', so we map your 'RKEY' to it.
if "RKEY" in os.environ:
    os.environ["REPLICATE_API_TOKEN"] = os.environ["RKEY"]

# The specific version hash provided
MODEL_VERSION = "deepseek-ai/deepseek-67b-base:0f2469607b150ffd428298a6bb57874f3657ab04fc980f7b5aa8fdad7bd6b46b"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        user_prompt = data.get('prompt', '')

        if not os.environ.get("REPLICATE_API_TOKEN"):
            return jsonify({"error": "RKEY environment variable not set."}), 500

        # 2. Call Replicate
        # deepseek-67b-base is a completion model. It completes the text you give it.
        output_iterator = replicate.run(
            MODEL_VERSION,
            input={
                "prompt": user_prompt,
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9
            }
        )

        # 3. Process Result
        # Replicate returns a generator (stream). We join it into a single string.
        result_text = "".join(list(output_iterator))
        
        return jsonify({"result": result_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)