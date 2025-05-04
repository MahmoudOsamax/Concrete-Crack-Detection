# Use a slim Python image as the base to reduce image size
FROM python:3.9-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage: Build the runtime image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy installed dependencies from the builder stage
COPY --from=builder /root/.local /root/.local

# Copy the application files, including the utils folder
COPY . .

# Ensure the Python path includes the user-installed packages
ENV PATH=/root/.local/bin:$PATH

# Install runtime system dependencies (e.g., for OpenCV and neural network support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Expose the port for Hugging Face Spaces
EXPOSE 7860

# Set environment variables for Flask and Hugging Face Spaces
ENV PORT=7860
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

# Run the app with Gunicorn
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:7860", "--timeout", "120", "app:app"]