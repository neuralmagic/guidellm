FROM --platform=linux/amd64 python:3.8-slim

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/guidellm/src/

RUN : \
    && apt-get update \
    # dependencies for building Python packages && cleaning up unused files
    && apt-get install -y --no-insatll-recommend \
        build-essential \
        libcurl4-openssl-dev \
        libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Python dependencies
RUN pip install --upgrade pip setuptools
COPY requirements.txt ./
RUN pip install -r requirements.txt

WORKDIR /app/
