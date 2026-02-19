from flask import Flask, render_template, request
import replicate

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    result = ""  # Initialize empty string
    if request.method == 'POST':
        inputs = {
            "prompt": request.form['prompt'],
            # Add these parameters for better control
            "max_length": 500,  # Adjust as needed
            "temperature": 0.7,  # Controls randomness (0-1)
            "top_p": 0.9,  # Nucleus sampling
        }
        
        try:
            outputs = replicate.run(
                "deepseek-ai/deepseek-67b-base:0f2469607b150ffd428298a6bb57874f3657ab04fc980f7b5aa8fdad7bd6b46b",
                input=inputs
            )
            
            # Collect all chunks and join them
            for text in outputs:
                result += text  # Append each chunk instead of overwriting
                
        except Exception as e:
            result = f"Error: {str(e)}"
            
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)