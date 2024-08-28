from typing import Any, AsyncGenerator, Dict, List, Optional

from deepsparse import Pipeline, TextGeneration
from loguru import logger

from guidellm.backend import Backend, GenerativeResponse
from guidellm.config import settings
from guidellm.core import TextGenerationRequest


@Backend.register(backend_type="deepsparse")
class DeepsparseBackend(Backend):
    """
    An Deepsparse backend implementation for the generative AI result.
    """

    def __init__(self, model: Optional[str] = None, **request_args):
        self._request_args: Dict[str, Any] = request_args
        self.model: str = self._get_model(model)
        self.pipeline: Pipeline = TextGeneration(model=self.model)

    def _get_model(self, model_from_cli: Optional[str] = None) -> str:
        """Provides the model by the next priority list:
        1. from function argument (comes from CLI)
        1. from environment variable
        2. `self.default_model` from `self.available_models`
        """

        if model_from_cli is not None:
            return model_from_cli
        elif settings.deepsprase.model is not None:
            logger.info(
                "Using Deepsparse model from environment variable: "
                f"{settings.deepsprase.model}"
            )
            return settings.deepsprase.model
        else:
            logger.info(f"Using default Deepsparse model: {self.default_model}")
            return self.default_model

    async def make_request(
        self, request: TextGenerationRequest
    ) -> AsyncGenerator[GenerativeResponse, None]:
        """
        Make a request to the Deepsparse Python API client.

        :param request: The result request to submit.
        :type request: TextGenerationRequest
        :return: An iterator over the generative responses.
        :rtype: Iterator[GenerativeResponse]
        """

        logger.debug(
            f"Making request to Deepsparse backend with prompt: {request.prompt}"
        )

        token_count = 0
        request_args = {
            **self._request_args,
            "streaming": True,
            "max_new_tokens": request.output_token_count,
        }

        if not (output := self.pipeline(prompt=request.prompt, **request_args)):
            yield GenerativeResponse(
                type_="final",
                prompt=request.prompt,
                prompt_token_count=request.prompt_token_count,
                output_token_count=token_count,
            )
            return

        for generation in output.generations:
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

        :return: A list of available models.
        :rtype: List[str]
        """

        # WARNING: The default model from the documentation is defined here
        return ["hf:mgoin/TinyStories-33M-quant-deepsparse"]

    def _token_count(self, text: str) -> int:
        token_count = len(text.split())
        logger.debug(f"Token count for text '{text}': {token_count}")
        return token_count
