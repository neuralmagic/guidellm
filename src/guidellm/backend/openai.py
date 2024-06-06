import openai
from typing import Iterator, List, Optional, Dict, Any
from urllib.parse import urlparse
from transformers import AutoTokenizer
from loguru import logger
from guidellm.backend import Backend, BackendTypes, GenerativeResponse
from guidellm.core.request import BenchmarkRequest

__all__ = ["OpenAIBackend"]


@Backend.register_backend(BackendTypes.OPENAI_SERVER)
class OpenAIBackend(Backend):
    """
    An OpenAI backend implementation for the generative AI benchmark.

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
    :param model_args: Additional model arguments for the request.
    :type model_args: Optional[Dict[str, Any]]
    """

    def __init__(
        self,
        target: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        path: Optional[str] = None,
        model: Optional[str] = None,
        **model_args,
    ):
        if target:
            parsed_url = urlparse(target)
            self.host = parsed_url.hostname
            self.port = parsed_url.port
            self.path = parsed_url.path
        else:
            self.host = host
            self.port = port
            self.path = path
        self.model = model
        self.model_args = model_args
        openai.api_key = model_args.get("api_key", None)
        logger.info(f"Initialized OpenAIBackend with model: {self.model}")

    def make_request(self, request: BenchmarkRequest) -> Iterator[GenerativeResponse]:
        """
        Make a request to the OpenAI backend.

        :param request: The benchmark request to submit.
        :type request: BenchmarkRequest
        :return: An iterator over the generative responses.
        :rtype: Iterator[GenerativeResponse]
        """
        logger.debug(f"Making request to OpenAI backend with prompt: {request.prompt}")
        response = openai.Completion.create(
            engine=self.model or self.default_model(),
            prompt=request.prompt,
            max_tokens=request.params.get("max_tokens", 100),
            n=request.params.get("n", 1),
            stop=request.params.get("stop", None),
            stream=True,
            **self.model_args,
        )

        for chunk in response:
            if chunk.get("choices"):
                choice = chunk["choices"][0]
                if choice.get("finish_reason") == "stop":
                    logger.debug("Received final response from OpenAI backend")
                    yield GenerativeResponse(
                        type_="final",
                        output=choice["text"],
                        prompt=request.prompt,
                        prompt_token_count=self._token_count(request.prompt),
                        output_token_count=self._token_count(choice["text"]),
                    )
                    break
                else:
                    logger.debug("Received token from OpenAI backend")
                    yield GenerativeResponse(
                        type_="token_iter", add_token=choice["text"]
                    )

    def available_models(self) -> List[str]:
        """
        Get the available models for the backend.

        :return: A list of available models.
        :rtype: List[str]
        """
        models = [model["id"] for model in openai.Engine.list()["data"]]
        logger.info(f"Available models: {models}")
        return models

    def default_model(self) -> str:
        """
        Get the default model for the backend.

        :return: The default model.
        :rtype: str
        """
        models = self.available_models()
        if models:
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
        """
        Count the number of tokens in a text.

        :param text: The text to tokenize.
        :type text: str
        :return: The number of tokens.
        :rtype: int
        """
        token_count = len(text.split())
        logger.debug(f"Token count for text '{text}': {token_count}")
        return token_count
