# Use MTG Essentia base image
FROM ghcr.io/mtg/essentia:latest

# Install ffmpeg and ensure pip is available
RUN apt-get update && \
    apt-get install -y ffmpeg python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Expose port 8080 (Cloud Run requirement)
EXPOSE 8080

# Run uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

