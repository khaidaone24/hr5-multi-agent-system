# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies including Rust (needed for uv)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv (CRITICAL for MCP)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Verify uv installation
RUN uv --version

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies via pip
RUN pip install --no-cache-dir -r requirements.txt

# Install MCP and postgres-mcp via uv (IMPORTANT)
RUN uv pip install --system mcp postgres-mcp

# Alternative: Install via pip if uv fails
# RUN pip install --no-cache-dir mcp postgres-mcp

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p uploads charts cache && \
    chmod 777 uploads charts cache

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Railway auto-assigns PORT, but default to 5000 for local dev
ENV PORT=5000

# Health check (Railway uses this)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port (Railway will override with $PORT)
EXPOSE ${PORT}

# Run the application
CMD ["python", "app.py"]
