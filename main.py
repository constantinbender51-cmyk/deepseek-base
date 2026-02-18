import os
import replicate
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Map the RKEY environment variable to the one Replicate expects
if "RKEY" in os.environ:
    os.environ["REPLICATE_API_TOKEN"] = os.environ["RKEY"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')

        # Define the input arguments
        input_args = {
            "prompt": prompt,
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9
        }

        full_response = ""
        
        # --- YOUR SNIPPET IMPLEMENTATION ---
        # We iterate through the stream and build the string
        for event in replicate.stream(
            "deepseek-ai/deepseek-67b-base:0f2469607b150ffd428298a6bb57874f3657ab04fc980f7b5aa8fdad7bd6b46b",
            input=input_args
        ):
            # Instead of print(event), we append to our string
            full_response += str(event)
        # -----------------------------------

        return jsonify({"result": full_response})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)