FROM python:3.14-slim

WORKDIR /app

# Copy dependency specs
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the core orchestration app
COPY . .

# Expose internal app port and metrics port
EXPOSE 8000
EXPOSE 9090

# Default command 
CMD ["python", "demos.py"]
