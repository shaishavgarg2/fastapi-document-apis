# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose the port that Railway will assign
EXPOSE $PORT

# Run with uvicorn directly
CMD ["uvicorn", "parameter_match_api:app", "--host", "0.0.0.0", "--port", "8000"]