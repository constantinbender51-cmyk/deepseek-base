import os
import replicate
from flask import Flask, render_template, request, Response, stream_with_context

# Initialize Flask (it will automatically look in the 'templates' folder)
app = Flask(__name__)

if "RKEY" in os.environ:
    os.environ["REPLICATE_API_TOKEN"] = os.environ["RKEY"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json or {}
    prompt = data.get('prompt', '')
    
    try:
        max_tokens = int(data.get('max_tokens', 512))
        temperature = float(data.get('temperature', 0.7))
        top_p = float(data.get('top_p', 0.9))
    except (ValueError, TypeError):
        max_tokens = 512
        temperature = 0.7
        top_p = 0.9

    model_id = "deepseek-ai/deepseek-67b-base:0f2469607b150ffd428298a6bb57874f3657ab04fc980f7b5aa8fdad7bd6b46b"
    
    input_args = {
        "prompt": prompt,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    }

    def generate_stream():
        try:
            for event in replicate.stream(model_id, input=input_args):
                yield str(event)
        except Exception as e:
            yield f"\n[SERVER ERROR: {str(e)}]"

    return Response(stream_with_context(generate_stream()), mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
