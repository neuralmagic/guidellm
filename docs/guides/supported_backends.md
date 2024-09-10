# Supported Backends 


GuideLLM requires an OpenAI-compatible server to run evaluations. [vLLM](https://github.com/vllm-project/vllm) is recommended for this purpose; however, GuideLLM is compatible with many backend inference servers such as TGI, llama.cpp, and DeepSparse. 

## OpenAI/HTTP Backends

### Text Generation Inference 
[Text Generation Inference](https://github.com/huggingface/text-generation-inference) can be used with GuideLLM. To start a TGI server with a Llama 3.1 8B using docker, run the following command:
```bash
docker run --gpus 1 -ti --shm-size 1g --ipc=host --rm -p 8080:80 \
  -e MODEL_ID=https://huggingface.co/llhf/Meta-Llama-3.1-8B-Instruct \
  -e NUM_SHARD=1 \
  -e MAX_INPUT_TOKENS=4096 \
  -e MAX_TOTAL_TOKENS=6000 \
  -e HF_TOKEN=$(cat ~/.cache/huggingface/token) \
  ghcr.io/huggingface/text-generation-inference:2.2.0
```

For more information on starting a TGI server, see the [TGI Documentation](https://huggingface.co/docs/text-generation-inference/index).


### Llama.cpp
TBD

### DeepSparse
TBD



## Python Backends
TBD


## Contribute a new backend

We appreciate contributions to the code, examples, integrations, documentation, bug reports, and feature requests! Your feedback and involvement are crucial in helping GuideLLM grow and improve. Below are some ways you can get involved:

- [**DEVELOPING.md**](https://github.com/neuralmagic/guidellm/blob/main/DEVELOPING.md) - Development guide for setting up your environment and making contributions.
- [**CONTRIBUTING.md**](https://github.com/neuralmagic/guidellm/blob/main/CONTRIBUTING.md) - Guidelines for contributing to the project, including code standards, pull request processes, and more.
- [**CODE_OF_CONDUCT.md**](https://github.com/neuralmagic/guidellm/blob/main/CODE_OF_CONDUCT.md) - Our expectations for community behavior to ensure a welcoming and inclusive environment.
