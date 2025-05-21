FROM registry.access.redhat.com/ubi9/python-312:9.5-1744198409

RUN pip install --upgrade pip && \
    pip install git+https://github.com/neuralmagic/guidellm.git@main

# Replace these env vars in the guidellm-job.yaml
ENV TARGET=http://localhost:8000/v1 \
    MODEL=neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16 \
    DATA_TYPE=emulated \
    DATA=prompt_tokens=512,generated_tokens=128

ENTRYPOINT ["guidellm"]
