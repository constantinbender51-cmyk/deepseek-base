
from flask import Flask, render_template, request
import replicate

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        inputs = {
            "prompt": request.form['prompt']
        }
        outputs = replicate.run(
            "deepseek-ai/deepseek-67b-base:0f2469607b150ffd428298a6bb57874f3657ab04fc980f7b5aa8fdad7bd6b46b",
            input=inputs
        )
        for text in outputs:
            result = text
    return render_template('index.html', prediction_text='{}'.format(result))

 if __name__ == "__main__":
    app.run(debug=True)
