FROM python:3.10

# Install required system packages
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Set the working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . /app/

# Ensure /data directory exists
RUN mkdir -p /data

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "task:app", "--host", "0.0.0.0", "--port", "8000"]
