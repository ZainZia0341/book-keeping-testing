# Use Python 3.9.0 as the base image
FROM python:3.9.0

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*  # Cleanup

# Copy only requirements.txt first (to optimize caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files (excluding files in .dockerignore)
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Set default command to run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
