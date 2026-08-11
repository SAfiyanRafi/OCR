FROM python:3.11-slim

# Install system libraries for OpenCV and document image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libspatialindex-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Environment setup
ENV PYTHONPATH=/app

# Default CLI entrypoint
ENTRYPOINT ["python", "-m", "app.cli"]
CMD ["process", "--help"]
