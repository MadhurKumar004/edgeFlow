FROM python:3.11-slim

WORKDIR /app

# Install curl for healthchecks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install backend runtime deps
COPY src/edgeflow/backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the entire source code
COPY src /app/src

# Set PYTHONPATH so python can find the edgeflow package
ENV PYTHONPATH=/app/src

EXPOSE 8000

# Health check endpoint with initial delay for startup
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

# Run the application
CMD ["uvicorn", "edgeflow.backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
