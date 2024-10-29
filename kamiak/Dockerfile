# Start from the NVIDIA TensorFlow image version 23.07 with GPU support
FROM nvcr.io/nvidia/tensorflow:23.07-tf2-py3

# Set environment variable to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies, such as OpenCV (used by cv2)
RUN apt-get update && apt-get install -y \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages using pip
RUN pip install --no-cache-dir \
    opencv-python \
    numpy \
    scikit-learn \
    pandas \
    seaborn \
    matplotlib \
    scikit-image

# Set the default command for the container
CMD ["python"]
