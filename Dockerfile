# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install uv (Python package manager)
RUN pip install --no-cache-dir uv

# Copy dependency files first (for better caching)
COPY pyproject.toml uv.lock README.md ./

# Install third-party dependencies only, so code changes don't bust this layer
RUN uv pip install --system -r pyproject.toml

# Copy application code
COPY . .

# Create necessary directories and run as an unprivileged user
RUN mkdir -p raw_data clean_data kaggle_data plots \
    && useradd --create-home app && chown -R app:app /app
USER app

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the data agent CLI. Swap for the HTTP server with:
#   CMD ["uvicorn", "--app-dir", "backend", "server:app", "--host", "0.0.0.0", "--port", "8017"]
CMD ["python", "backend/data_agent.py"]
