from flask import Flask, render_template, request
import replicate

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    result = ""
    try:
        output = replicate.run(
            "deepseek-ai/deepseek-67b-base:0f2469607b150ffd428298a6bb57874f3657ab04fc980f7b5aa8fdad7bd6b46b",
            input={"prompt": request.form['prompt']}
        )
        for chunk in output:
            result += chunk
    except Exception as e:
        result = str(e)
    
    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)