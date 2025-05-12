# Use Python 3.9 as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir lxml-html-clean \
    && pip install --no-cache-dir "lxml[html_clean]" \
    && pip install --no-cache-dir trafilatura --upgrade

# Copy the application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV PYTHONPATH=/app
ENV GRADIO_ANALYTICS_ENABLED=false
ENV PYTHONASYNCIO=1

# Expose the port
EXPOSE 7860

# Run the application with asyncio support
CMD ["python", "-X", "dev", "-u", "app.py"] 