FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/neuralmagic/guidellm"
LABEL org.opencontainers.image.description="GuideLLM Benchmark Container"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 guidellm

# Set working directory
WORKDIR /app

# Install GuideLLM
RUN pip install git+https://github.com/neuralmagic/guidellm.git

# Copy and set up the benchmark script
COPY run_benchmark.sh /app/
RUN chmod +x /app/run_benchmark.sh

# Set ownership to non-root user
RUN chown -R guidellm:guidellm /app

# Switch to non-root user
USER guidellm

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set the entrypoint
ENTRYPOINT ["/app/run_benchmark.sh"] 