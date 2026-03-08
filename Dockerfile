FROM python:3.12-slim

WORKDIR /app

# Install system dependencies that might be needed for hnswlib (used by chromadb) or sentence_transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app/ ./app/

EXPOSE 7860

# Run uvicorn pointing to the main.py inside the app module
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
