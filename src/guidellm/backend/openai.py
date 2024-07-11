import functools
import os
from typing import Any, Dict, Iterator, List, Optional

from loguru import logger
from openai import OpenAI, Stream
from openai.types import Completion
from transformers import AutoTokenizer

from guidellm.backend import Backend, BackendEngine, GenerativeResponse
from guidellm.core import TextGenerationRequest

__all__ = ["OpenAIBackend"]


@Backend.register(BackendEngine.OPENAI_SERVER)
class OpenAIBackend(Backend):
    """
    An OpenAI backend implementation for the generative AI result.

    :param target: The target URL string for the OpenAI server.
    :type target: str
    :param host: Optional host for the OpenAI server.
    :type host: Optional[str]
    :param port: Optional port for the OpenAI server.
    :type port: Optional[int]
    :param path: Optional path for the OpenAI server.
    :type path: Optional[str]
    :param model: The OpenAI model to use, defaults to the first available model.
    :type model: Optional[str]
    :param api_key: The OpenAI API key to use.
    :type api_key: Optional[str]
    :param request_args: Optional arguments for the OpenAI request.
    :type request_args: Dict[str, Any]
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        internal_callback_url: Optional[str] = None,
        model: Optional[str] = None,
        **request_args: Any,
    ):
        """
        Initialize an OpenAI Client
        """

        self.request_args = request_args

        if not (_api_key := (openai_api_key or os.getenv("OPENAI_API_KEY", None))):
            raise ValueError(
                "`OPENAI_API_KEY` environment variable "
                "or --openai-api-key CLI parameter "
                "must be specify for the OpenAI backend"
            )

        if not (
            _base_url := (internal_callback_url or os.getenv("OPENAI_BASE_URL", None))
        ):
            raise ValueError(
                "`OPENAI_BASE_URL` environment variable "
                "or --openai-base-url CLI parameter "
                "must be specify for the OpenAI backend"
            )
        self.openai_client = OpenAI(api_key=_api_key, base_url=_base_url)
        self.model = model or self.default_model

        logger.info(
            f"Initialized OpenAIBackend with callback url: {internal_callback_url} "
            f"and model: {self.model}"
        )

    def make_request(
        self, request: TextGenerationRequest
    ) -> Iterator[GenerativeResponse]:
        """
        Make a request to the OpenAI backend.

        :param request: The result request to submit.
        :type request: TextGenerationRequest
        :return: An iterator over the generative responses.
        :rtype: Iterator[GenerativeResponse]
        """

        logger.debug(f"Making request to OpenAI backend with prompt: {request.prompt}")

        # How many completions to generate for each prompt
        request_args: Dict = {"n": 1}

        if (num_gen_tokens := request.params.get("generated_tokens", None)) is not None:
            request_args.update(max_tokens=num_gen_tokens, stop=None)

        if self.request_args:
            request_args.update(self.request_args)

        response: Stream[Completion] = self.openai_client.completions.create(
            model=self.model,
            prompt=request.prompt,
            stream=True,
            **request_args,
        )

        for chunk in response:
            chunk_content: str = getattr(chunk, "content", "")

            if getattr(chunk, "stop", True) is True:
                logger.debug("Received final response from OpenAI backend")

                yield GenerativeResponse(
                    type_="final",
                    prompt=getattr(chunk, "prompt", request.prompt),
                    prompt_token_count=(
                        request.prompt_token_count or self._token_count(request.prompt)
                    ),
                    output_token_count=(
                        num_gen_tokens
                        if num_gen_tokens
                        else self._token_count(chunk_content)
                    ),
                )
            else:
                logger.debug("Received token from OpenAI backend")
                yield GenerativeResponse(type_="token_iter", add_token=chunk_content)

    def available_models(self) -> List[str]:
        """
        Get the available models for the backend.

        :return: A list of available models.
        :rtype: List[str]
        """

        models: list[str] = [
            model.id for model in self.openai_client.models.list().data
        ]
        logger.info(f"Available models: {models}")

        return models

    @property
    @functools.lru_cache(maxsize=1)
    def default_model(self) -> str:
        """
        Get the default model for the backend.

        :return: The default model.
        :rtype: str
        """

        if models := self.available_models():
            logger.info(f"Default model: {models[0]}")
            return models[0]

        logger.error("No models available.")
        raise ValueError("No models available.")

    def model_tokenizer(self, model: str) -> Optional[Any]:
        """
        Get the tokenizer for a model.

        :param model: The model to get the tokenizer for.
        :type model: str
        :return: The tokenizer for the model, or None if it cannot be created.
        :rtype: Optional[Any]
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(model)
            logger.info(f"Tokenizer created for model: {model}")
            return tokenizer
        except Exception as e:
            logger.warning(f"Could not create tokenizer for model {model}: {e}")
            return None

    def _token_count(self, text: str) -> int:
        token_count = len(text.split())
        logger.debug(f"Token count for text '{text}': {token_count}")
        return token_count
