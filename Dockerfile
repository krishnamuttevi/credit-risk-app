# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# (Optional fix) Enforce scikit-learn version
# RUN pip install --no-cache-dir scikit-learn==1.4.2

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY model.joblib .
COPY static/ ./static/

# Create static directory if it doesn't exist
RUN mkdir -p static

# Expose correct port
EXPOSE 9000

# Start FastAPI server on port 9000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
