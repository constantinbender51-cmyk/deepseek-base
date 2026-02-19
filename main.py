from flask import Flask, render_template, request, Response
import replicate

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', prompt='', prediction_text='')

@app.route('/predict', methods=['POST'])
def predict():
    prompt = request.form['prompt']
    
    def generate():
        try:
            output = replicate.run(
                "deepseek-ai/deepseek-67b-base:0f2469607b150ffd428298a6bb57874f3657ab04fc980f7b5aa8fdad7bd6b46b",
                input={"prompt": prompt}
            )
            
            for chunk in output:
                yield chunk
                
        except Exception as e:
            yield f"Error: {str(e)}"
    
    return Response(generate(), mimetype='text/plain')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)