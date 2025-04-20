FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential

# Install Python dependencies from pinned list
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy whole project into the container
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Launch API using Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
