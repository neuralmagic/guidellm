from typing import Any, AsyncGenerator, Dict, List, Optional

from loguru import logger
from vllm import LLM, CompletionOutput, SamplingParams

from guidellm.backend import Backend, GenerativeResponse
from guidellm.config import settings
from guidellm.core import TextGenerationRequest


@Backend.register(backend_type="vllm")
class VllmBackend(Backend):
    """
    An vLLM Backend implementation for the generative AI result.
    """

    def __init__(self, model: str = settings.llm_model, **request_args):
        _model = self._get_model(model)
        self._request_args: Dict[str, Any] = request_args
        self.llm = LLM(_model)

        super().__init__(type_="vllm", model=_model, target="not used")

        logger.info(f"vLLM Backend uses model '{self._model}'")

    def _get_model(self, model_from_cli: Optional[str] = None) -> str:
        """Provides the model by the next priority list:
        1. from function argument (comes from CLI)
        1. from environment variable
        2. `self.default_model` from `self.available_models`
        """

        if model_from_cli is not None:
            return model_from_cli
        elif settings.llm_model is not None:
            logger.info(
                "Using vLLM model from environment variable: " f"{settings.llm_model}"
            )
            return settings.llm_model
        else:
            logger.info(f"Using default vLLM model: {self.default_model}")
            return self.default_model

    async def make_request(
        self, request: TextGenerationRequest
    ) -> AsyncGenerator[GenerativeResponse, None]:
        """
        Make a request to the vLLM Python API client.

        :param request: The result request to submit.
        :type request: TextGenerationRequest
        :return: An iterator over the generative responses.
        :rtype: Iterator[GenerativeResponse]
        """

        logger.debug(f"Making request to vLLM backend with prompt: {request.prompt}")

        token_count = 0
        request_args = {
            **self._request_args,
            "inputs": [request.prompt],
            "sampling_params": SamplingParams(max_tokens=request.output_token_count),
        }

        final_response = GenerativeResponse(
            type_="final",
            prompt=request.prompt,
            prompt_token_count=request.prompt_token_count,
            output_token_count=token_count,
        )

        breakpoint()  # TODO: remove
        if not (result := self.llm.generate(**request_args)):
            yield final_response
            return

        try:
            generations: List[CompletionOutput] = result[0].outputs
        except IndexError:
            yield final_response
            return

        for generation in generations:
            if not (token := generation.text):
                yield GenerativeResponse(
                    type_="final",
                    prompt=request.prompt,
                    prompt_token_count=request.prompt_token_count,
                    output_token_count=token_count,
                )
                break
            else:
                token_count += 1
                yield GenerativeResponse(
                    type_="token_iter",
                    add_token=token,
                    prompt=request.prompt,
                    prompt_token_count=request.prompt_token_count,
                    output_token_count=token_count,
                )

    def available_models(self) -> List[str]:
        """
        Get the available models for the backend.

        ref: https://docs.vllm.ai/en/v0.4.1/models/supported_models.html

        :return: A list of available models.
        :rtype: List[str]
        """

        return [
            "mistralai/Mistral-7B-Instruct-v0.3",
            "meta-llama/Meta-Llama-3-8B-Instruct",
        ]

    def _token_count(self, text: str) -> int:
        token_count = len(text.split())
        logger.debug(f"Token count for text '{text}': {token_count}")
        return token_count
