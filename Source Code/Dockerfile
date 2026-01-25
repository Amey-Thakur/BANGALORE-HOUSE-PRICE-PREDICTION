# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
# Upgrade pip first to avoid legacy resolver issues
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Create specific directories for permissions (Hugging Face runs as user 1000)
RUN mkdir -p /app/templates && chmod -R 777 /app

# Expose the port that Hugging Face Spaces expects (7860)
EXPOSE 7860

# Define environment variable for Flask
ENV FLASK_APP=app.py

# Run gunicorn server on port 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
