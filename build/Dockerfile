FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/neuralmagic/guidellm"
LABEL org.opencontainers.image.description="GuideLLM Benchmark Container"

# Install dependencies and set up environment in a single layer
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && pip install git+https://github.com/neuralmagic/guidellm.git \
    && useradd -m -u 1000 guidellm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and set up the benchmark script
COPY build/run_benchmark.sh /app/

# Set ownership to non-root user
RUN chown -R guidellm:guidellm /app

# Switch to non-root user
USER guidellm

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set the entrypoint
ENTRYPOINT ["/app/run_benchmark.sh"]
