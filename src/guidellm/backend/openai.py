from typing import AsyncGenerator, Dict, List, Optional

from loguru import logger
from openai import AsyncOpenAI, OpenAI

from guidellm.backend.base import Backend, GenerativeResponse
from guidellm.config import settings
from guidellm.core import TextGenerationRequest

__all__ = ["OpenAIBackend"]


@Backend.register("openai_server")
class OpenAIBackend(Backend):
    """
    An OpenAI backend implementation for generative AI results.

    This class provides an interface to communicate with the
    OpenAI server for generating responses based on given prompts.

    :param openai_api_key: The API key for OpenAI.
        If not provided, it will default to the key from settings.
    :type openai_api_key: Optional[str]
    :param target: The target URL string for the OpenAI server.
    :type target: Optional[str]
    :param model: The OpenAI model to use, defaults to the first available model.
    :type model: Optional[str]
    :param request_args: Additional arguments for the OpenAI request.
    :type request_args: Dict[str, Any]
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        target: Optional[str] = None,
        model: Optional[str] = None,
        **request_args,
    ):
        self._request_args: Dict = request_args
        api_key: str = openai_api_key or settings.openai.api_key

        if not api_key:
            err = ValueError(
                "`GUIDELLM__OPENAI__API_KEY` environment variable or "
                "--openai-api-key CLI parameter must be specified for the "
                "OpenAI backend."
            )
            logger.error("{}", err)
            raise err

        base_url = target or settings.openai.base_url

        if not base_url:
            err = ValueError(
                "`GUIDELLM__OPENAI__BASE_URL` environment variable or "
                "target parameter must be specified for the OpenAI backend."
            )
            logger.error("{}", err)
            raise err

        self._async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model or self.default_model

        super().__init__(type_="openai_server", target=base_url, model=self._model)
        logger.info("OpenAI {} Backend listening on {}", self._model, base_url)

    async def make_request(
        self,
        request: TextGenerationRequest,
    ) -> AsyncGenerator[GenerativeResponse, None]:
        """
        Make a request to the OpenAI backend.

        This method sends a prompt to the OpenAI backend and streams
        the response tokens back.

        :param request: The text generation request to submit.
        :type request: TextGenerationRequest
        :yield: A stream of GenerativeResponse objects.
        :rtype: AsyncGenerator[GenerativeResponse, None]
        """

        logger.debug("Making request to OpenAI backend with prompt: {}", request.prompt)

        request_args: Dict = {
            "n": 1,  # Number of completions for each prompt
        }

        if request.output_token_count is not None:
            request_args.update(
                {
                    "max_tokens": request.output_token_count,
                    "stop": None,
                    "extra_body": {
                        "ignore_eos": True,
                    }
                }
            )
        elif settings.openai.max_gen_tokens and settings.openai.max_gen_tokens > 0:
            request_args.update(
                {
                    "max_tokens": settings.openai.max_gen_tokens,
                }
            )

        request_args.update(self._request_args)

        stream = await self._async_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": request.prompt},
            ],
            stream=True,
            **request_args,
        )

        token_count = 0
        async for chunk in stream:
            choice = chunk.choices[0]
            token = choice.delta.content or ""

            if choice.finish_reason is not None:
                yield GenerativeResponse(
                    type_="final",
                    prompt=request.prompt,
                    prompt_token_count=request.prompt_token_count,
                    output_token_count=token_count,
                )
                break

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

        This method queries the OpenAI API to retrieve a list of available models.

        :return: A list of available models.
        :rtype: List[str]
        :raises openai.OpenAIError: If an error occurs while retrieving models.
        """

        try:
            return [model.id for model in self._client.models.list().data]
        except Exception as error:
            logger.error("Failed to retrieve available models: {}", error)
            raise error

    def validate_connection(self):
        """
        Validate the connection to the OpenAI backend.

        This method checks that the OpenAI backend is reachable and
        the API key is valid.

        :raises openai.OpenAIError: If the connection is invalid.
        """

        try:
            self._client.models.list()
        except Exception as error:
            logger.error("Failed to validate OpenAI connection: {}", error)
            raise error
