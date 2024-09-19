# Use a base image with Python
FROM python:3.9-slim

# Set the working directory
WORKDIR .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

# Copy the requirements file
COPY app/requirements/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Specify the command to run your application
CMD ["python", "app/main.py"]

