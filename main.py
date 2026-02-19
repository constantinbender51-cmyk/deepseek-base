from flask import Flask, render_template, request
import replicate

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', 
                         prompt='', 
                         prediction_text='',
                         temp=0.7,
                         max_len=500,
                         top_p=0.9,
                         freq_pen=0,
                         pres_pen=0)

@app.route('/predict', methods=['POST'])
def predict():
    prompt = request.form['prompt']
    result = ""
    
    # Get parameters with defaults
    try:
        inputs = {
            "prompt": prompt,
            "temperature": float(request.form.get('temperature', 0.7)),
            "max_length": int(request.form.get('max_length', 500)),
            "top_p": float(request.form.get('top_p', 0.9)),
            "frequency_penalty": float(request.form.get('frequency_penalty', 0)),
            "presence_penalty": float(request.form.get('presence_penalty', 0))
        }
        
        output = replicate.run(
            "deepseek-ai/deepseek-67b-base:0f2469607b150ffd428298a6bb57874f3657ab04fc980f7b5aa8fdad7bd6b46b",
            input=inputs
        )
        
        for chunk in output:
            result += chunk
            
    except Exception as e:
        result = f"Error: {str(e)}"
    
    return render_template('index.html', 
                         prompt=prompt, 
                         prediction_text=result,
                         temp=request.form.get('temperature', 0.7),
                         max_len=request.form.get('max_length', 500),
                         top_p=request.form.get('top_p', 0.9),
                         freq_pen=request.form.get('frequency_penalty', 0),
                         pres_pen=request.form.get('presence_penalty', 0))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)