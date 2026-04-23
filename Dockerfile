FROM python:3.10-slim

# Set working directory
WORKDIR /app

# System dependencies (needed for OpenCV / TensorFlow)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Set default port (Render overrides this)
ENV PORT=8080

# Expose port
EXPOSE 8080

# Start app using Gunicorn (dynamic port support)
CMD ["sh", "-c", "gunicorn main:app --bind 0.0.0.0:$PORT"]