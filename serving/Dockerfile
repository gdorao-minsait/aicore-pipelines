FROM python:3.12-slim
ENV LANG C.UTF-8

# Install the required packages for OpenCV
RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx \
    && apt-get install -y libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install requirements
RUN mkdir -p /app/src
COPY requirements.txt /app/src/requirements.txt
RUN pip3 install --no-cache-dir -r /app/src/requirements.txt

# Copy main script
COPY main.py /app/src/main.py

# Make scripts executable
RUN chgrp -R nogroup /app && \
    chmod -R 770 /app

# DEBUG: remove. This is to copy the model and avoid placing artifacts in the configuration
RUN mkdir -p /mnt/models
COPY yolo_model.pt /mnt/models/yolo_model.pt