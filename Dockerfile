# Step 1: Base Image - Use an official Python image
FROM python:3.10.12-slim-buster as builder

# Set working directory
WORKDIR /app

# Environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies required by OpenCV and other libraries
# (Pillow might need zlib, jpeg; OpenCV needs libgl1, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # For fonts if Pillow needs them (e.g., if arial.ttf is not packaged)
    # ttf-mscorefonts-installer \ 
    # fontconfig \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to reduce image size slightly
# Some torch/ultralytics dependencies can be large.
# Consider if a specific torch version without CUDA is needed if not using GPU in Docker.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create necessary directories and set permissions (if needed)
# These will be created inside the container.
# If you mount volumes from host, permissions on host matter more.
RUN mkdir -p /app/CKPT /app/uploads /app/detected_images \
    && chown -R www-data:www-data /app # Assuming uvicorn might run as non-root

# (Optional) If you have a specific font like arial.ttf you want to package:
# COPY arial.ttf /usr/share/fonts/truetype/
# RUN fc-cache -fv # Update font cache if you added fonts

# Expose the port the app runs on
EXPOSE 8000

# Default command to run the application using Uvicorn
# Use 0.0.0.0 to allow connections from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- Optional: Multi-stage build for smaller final image ---
# If you want a smaller final image, you can create a new stage
# and copy only the necessary artifacts.
# FROM python:3.9-slim-buster as final
# WORKDIR /app
# ENV PYTHONDONTWRITEBYTECODE 1
# ENV PYTHONUNBUFFERED 1

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender1 \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*


# Copy application code
# COPY . .

# RUN mkdir -p /app/CKPT /app/uploads /app/detected_images \
#     && chown -R www-data:www-data /app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]