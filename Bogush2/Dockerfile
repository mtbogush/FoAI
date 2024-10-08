# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Cython if needed
RUN pip install Cython==0.29.23

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install any required packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app

# Expose the port that the Flask app will run on
EXPOSE 5000

# Command to run the application
CMD ["python", "starter_notebook_phase_2.py"]
