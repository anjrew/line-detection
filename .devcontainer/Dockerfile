# Use the Python dev container base image
FROM mcr.microsoft.com/devcontainers/python

# Set the working directory in the container
WORKDIR /workspace

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
