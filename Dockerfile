# Use MTG Essentia base image
FROM ghcr.io/mtg/essentia:latest

# Install ffmpeg and ensure pip is available
RUN apt-get update && \
    apt-get install -y ffmpeg python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /tmp && \
    chown -R appuser:appuser /app /tmp

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies as root (needed for system-wide install)
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip --root-user-action=ignore && \
    python3 -m pip install --no-cache-dir -r requirements.txt --root-user-action=ignore

# Copy application code
COPY main.py .

# Switch to non-root user
USER appuser

# Expose port 8080 (Cloud Run requirement)
EXPOSE 8080

# Run uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

