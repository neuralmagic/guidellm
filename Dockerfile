FROM --platform=linux/amd64 python:3.8-slim

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/guidellm/src/

RUN : \
    && apt-get update \
    # dependencies for building Python packages && cleaning up unused files
    && apt-get install -y \
        build-essential \
        libcurl4-openssl-dev \
        libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade \
        pip \
        setuptools


WORKDIR /app

# Install project dependencies
COPY ./ ./
RUN pip install -e .[dev,deepsparse,vllm]

