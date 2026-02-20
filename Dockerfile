# Use Python 3.11
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch CPU directly to save 4GB+ of space
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the latest requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app
COPY . .

# Pre-download the model during the build so the app starts instantly
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('vikhyatk/moondream2', revision='2024-08-26'); AutoModelForCausalLM.from_pretrained('vikhyatk/moondream2', trust_remote_code=True, revision='2024-08-26')"

ENV PORT=8501
EXPOSE $PORT

CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0