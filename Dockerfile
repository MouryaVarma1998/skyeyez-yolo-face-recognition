# Base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the Python code
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose necessary ports (if any)
EXPOSE 8080

# Run the application
CMD ["python", "app.py"]
