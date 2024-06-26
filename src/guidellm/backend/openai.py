from typing import Any, Iterator, List, Optional

import openai
from loguru import logger
from transformers import AutoTokenizer

from guidellm.backend import Backend, BackendType, GenerativeResponse
from guidellm.core.request import TextGenerationRequest

__all__ = ["OpenAIBackend"]


@Backend.register_backend(BackendType.OPENAI_SERVER)
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
        target: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        path: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **request_args,
    ):
        self.target = target
        self.model = model
        self.request_args = request_args

        if not self.target:
            if not host:
                raise ValueError("Host is required if target is not provided.")

            port_incl = f":{port}" if port else ""
            path_incl = path if path else ""
            self.target = f"http://{host}{port_incl}{path_incl}"

        openai.api_base = self.target
        openai.api_key = api_key

        if not model:
            self.model = self.default_model()

        logger.info(
            f"Initialized OpenAIBackend with target: {self.target} "
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
        num_gen_tokens = request.params.get("generated_tokens", None)
        request_args = {
            "n": 1,
        }

        if num_gen_tokens:
            request_args["max_tokens"] = num_gen_tokens
            request_args["stop"] = None

        if self.request_args:
            request_args.update(self.request_args)

        response = openai.Completion.create(
            engine=self.model,
            prompt=request.prompt,
            stream=True,
            **request_args,
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
                        prompt_token_count=(
                            request.token_count
                            if request.token_count
                            else self._token_count(request.prompt)
                        ),
                        output_token_count=(
                            num_gen_tokens
                            if num_gen_tokens
                            else self._token_count(choice["text"])
                        ),
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
        token_count = len(text.split())
        logger.debug(f"Token count for text '{text}': {token_count}")
        return token_count
