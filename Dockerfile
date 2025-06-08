# Use the official Python image as base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app


# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]

