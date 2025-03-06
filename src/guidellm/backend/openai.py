import base64
import json
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Union

import httpx
from loguru import logger
from PIL import Image

from guidellm.backend.backend import (
    Backend,
    StreamingRequestArgs,
    StreamingResponse,
    StreamingTextResponseStats,
)
from guidellm.config import settings

__all__ = ["OpenAIHTTPBackend"]


@Backend.register("openai_http")
class OpenAIHTTPBackend(Backend):
    """
    A HTTP-based backend implementation for requests to an OpenAI compatible server.
    For example, a vLLM server instance or requests to OpenAI's API.

    :param target: The target URL string for the OpenAI server. ex: http://0.0.0.0:8000
    :param model: The model to use for all requests on the target server.
        If none is provided, the first available model will be used.
    :param api_key: The API key to use for requests to the OpenAI server.
        If provided, adds an Authorization header with the value
        "Authorization: Bearer {api_key}".
        If not provided, no Authorization header is added.
    :param orginization: The organization to use for requests to the OpenAI server.
        For example, if set to "org_123", adds an OpenAI-Organization header with the
        value "OpenAI-Organization: org_123".
        If not provided, no OpenAI-Organization header is added.
    :param project: The project to use for requests to the OpenAI server.
        For example, if set to "project_123", adds an OpenAI-Project header with the
        value "OpenAI-Project: project_123".
        If not provided, no OpenAI-Project header is added.
    :param timeout: The timeout to use for requests to the OpenAI server.
        If not provided, the default timeout provided from settings is used.
    :param http2: If True, uses HTTP/2 for requests to the OpenAI server.
        Defaults to True.
    :param max_output_tokens: The maximum number of tokens to request for completions.
        If not provided, the default maximum tokens provided from settings is used.
    """

    def __init__(
        self,
        target: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        orginization: Optional[str] = None,
        project: Optional[str] = None,
        timeout: Optional[float] = None,
        http2: Optional[bool] = True,
        max_output_tokens: Optional[int] = None,
    ):
        super().__init__(type_="openai_http")
        self._target = target or settings.openai.base_url
        self._model = model

        api_key = api_key or settings.openai.api_key
        self.authorization = (
            f"Bearer {api_key}" if api_key else settings.openai.bearer_token
        )

        self.orginization = orginization or settings.openai.organization
        self.project = project or settings.openai.project
        self.timeout = timeout if timeout is not None else settings.request_timeout
        self.http2 = http2 if http2 is not None else settings.request_http2
        self.max_output_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else settings.openai.max_output_tokens
        )

    @property
    def target(self) -> str:
        """
        :return: The target URL string for the OpenAI server.
        """
        return self._target

    @property
    def model(self) -> Optional[str]:
        """
        :return: The model to use for all requests on the target server.
            If validate hasn't been called yet and no model was passed in,
            this will be None until validate is called to set the default.
        """
        return self._model

    def check_setup(self):
        """
        Check if the backend is setup correctly and can be used for requests.
        Specifically, if a model is not provided, it grabs the first available model.
        If no models are available, raises a ValueError.
        If a model is provided and not available, raises a ValueError.

        :raises ValueError: If no models or the provided model is not available.
        """
        models = self.available_models()
        if not models:
            raise ValueError(f"No models available for target: {self.target}")

        if not self.model:
            self._model = models[0]
        elif self.model not in models:
            raise ValueError(
                f"Model {self.model} not found in available models:"
                "{models} for target: {self.target}"
            )

    def available_models(self) -> List[str]:
        """
        Get the available models for the target server using the OpenAI models endpoint:
        /v1/models
        """
        target = f"{self.target}/v1/models"
        headers = self._headers()

        with httpx.Client(http2=self.http2, timeout=self.timeout) as client:
            response = client.get(target, headers=headers)
            response.raise_for_status()

            models = []

            for item in response.json()["data"]:
                models.append(item["id"])

            return models

    async def text_completions(  # type: ignore[override]
        self,
        prompt: Union[str, List[str]],
        id_: Optional[str] = None,
        prompt_token_count: Optional[int] = None,
        output_token_count: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[StreamingResponse, None]:
        """
        Generate text completions for the given prompt using the OpenAI
        completions endpoint: /v1/completions.

        :param prompt: The prompt (or list of prompts) to generate a completion for.
            If a list is supplied, these are concatenated and run through the model
            for a single prompt.
        :param id_: The unique identifier for the request, if any.
            Added to logging statements and the response for tracking purposes.
        :param prompt_token_count: The number of tokens measured in the prompt, if any.
            Returned in the response stats for later analysis, if applicable.
        :param output_token_count: If supplied, the number of tokens to enforce
            generation of for the output for this request.
        :param kwargs: Additional keyword arguments to pass with the request.
        :return: An async generator that yields StreamingResponse objects containing the
            response content. Will always start with a 'start' response,
            followed by 0 or more 'iter' responses, and ending with a 'final' response.
        """

        logger.debug("{} invocation with args: {}", self.__class__.__name__, locals())
        headers = self._headers()
        payload = self._completions_payload(
            orig_kwargs=kwargs,
            max_output_tokens=output_token_count,
            prompt=prompt,
        )

        try:
            async for resp in self._iterative_completions_request(
                type_="text",
                id_=id_,
                headers=headers,
                payload=payload,
                stats=StreamingTextResponseStats(
                    request_prompt_tokens=prompt_token_count,
                    request_output_tokens=output_token_count,
                ),
            ):
                yield resp
        except Exception as ex:
            logger.error(
                "{} request with headers: {} and payload: {} failed: {}",
                self.__class__.__name__,
                headers,
                payload,
                ex,
            )
            raise ex

    async def chat_completions(  # type: ignore[override]
        self,
        content: Union[
            str,
            List[Union[str, Dict[str, Union[str, Dict[str, str]]], Path, Image.Image]],
            Any,
        ],
        id_: Optional[str] = None,
        prompt_token_count: Optional[int] = None,
        output_token_count: Optional[int] = None,
        raw_content: bool = False,
        **kwargs,
    ) -> AsyncGenerator[StreamingResponse, None]:
        """
        Generate chat completions for the given content using the OpenAI
        chat completions endpoint: /v1/chat/completions.

        :param content: The content (or list of content) to generate a completion for.
            This supports any combination of text, images, and audio (model dependent).
            Supported text only request examples:
                content="Sample prompt", content=["Sample prompt", "Second prompt"],
                content=[{"type": "text", "value": "Sample prompt"}.
            Supported text and image request examples:
                content=["Describe the image", PIL.Image.open("image.jpg")],
                content=["Describe the image", Path("image.jpg")],
                content=["Describe the image", {"type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}].
            Supported text and audio request examples:
                content=["Transcribe the audio", Path("audio.wav")],
                content=["Transcribe the audio", {"type": "input_audio",
                "input_audio": {"data": f"{base64_bytes}", "format": "wav}].
            Additionally, if raw_content=True then the content is passed directly to the
            backend without any processing.
        :param id_: The unique identifier for the request, if any.
            Added to logging statements and the response for tracking purposes.
        :param prompt_token_count: The number of tokens measured in the prompt, if any.
            Returned in the response stats for later analysis, if applicable.
        :param output_token_count: If supplied, the number of tokens to enforce
            generation of for the output for this request.
        :param kwargs: Additional keyword arguments to pass with the request.
        :return: An async generator that yields StreamingResponse objects containing the
            response content. Will always start with a 'start' response,
            followed by 0 or more 'iter' responses, and ending with a 'final' response.
        """
        logger.debug("{} invocation with args: {}", self.__class__.__name__, locals())
        headers = self._headers()
        messages = (
            content if raw_content else OpenAIHTTPBackend._create_chat_messages(content)
        )
        payload = self._completions_payload(
            orig_kwargs=kwargs,
            max_output_tokens=output_token_count,
            messages=messages,
        )

        try:
            async for resp in self._iterative_completions_request(
                type_="chat",
                id_=id_,
                headers=headers,
                payload=payload,
                stats=StreamingTextResponseStats(
                    request_prompt_tokens=prompt_token_count,
                    request_output_tokens=output_token_count,
                ),
            ):
                yield resp
        except Exception as ex:
            logger.error(
                "{} request with headers: {} and payload: {} failed: {}",
                self.__class__.__name__,
                headers,
                payload,
                ex,
            )
            raise ex

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }

        if self.authorization:
            headers["Authorization"] = self.authorization

        if self.orginization:
            headers["OpenAI-Organization"] = self.orginization

        if self.project:
            headers["OpenAI-Project"] = self.project

        return headers

    def _completions_payload(
        self, orig_kwargs: Optional[Dict], max_output_tokens: Optional[int], **kwargs
    ) -> Dict:
        payload = orig_kwargs or {}
        payload.update(kwargs)
        payload["model"] = self.model
        payload["stream"] = True
        payload["stream_options"] = {
            "include_usage": True,
        }

        if max_output_tokens or self.max_output_tokens:
            logger.debug(
                "{} adding payload args for setting output_token_count: {}",
                self.__class__.__name__,
                max_output_tokens or self.max_output_tokens,
            )
            payload["max_tokens"] = max_output_tokens or self.max_output_tokens
            payload["max_completion_tokens"] = max_output_tokens

            if max_output_tokens:
                # only set stop and ignore_eos if max_output_tokens set at request level
                # otherwise the instance value is just the max to enforce we stay below
                payload["stop"] = None
                payload["ignore_eos"] = True

        return payload

    async def _iterative_completions_request(
        self,
        type_: Literal["text", "chat"],
        id_: Optional[str],
        headers: Dict,
        payload: Dict,
        stats: StreamingTextResponseStats,
    ) -> AsyncGenerator[StreamingResponse, None]:
        target = f"{self.target}/v1/"

        if type_ == "text":
            target += "completions"
        elif type_ == "chat":
            target += "chat/completions"
        else:
            raise ValueError(f"Unsupported type: {type_}")

        response = StreamingResponse(
            request_args=StreamingRequestArgs(
                target=target,
                headers=headers,
                payload=payload,
                timeout=self.timeout,
                http2=self.http2,
            ),
            stats=stats,
        )

        logger.info(
            "{} making request {} to OpenAI backend {} using http2: {} for "
            "timeout: {} with headers: {} and payload: {}",
            self.__class__.__name__,
            id_,
            target,
            self.http2,
            self.timeout,
            headers,
            payload,
        )
        yield response

        async with httpx.AsyncClient(http2=self.http2, timeout=self.timeout) as client:
            response.timings.request_start = time.time()

            async with client.stream(
                "POST", target, headers=headers, json=payload
            ) as stream:
                stream.raise_for_status()

                async for line in stream.aiter_lines():
                    logger.debug(
                        "{} request {} recieved iter response line: {}",
                        self.__class__.__name__,
                        id_,
                        line,
                    )

                    if not line or not line.startswith("data:"):
                        continue

                    iter_time = time.time()

                    if line.strip() == "data: [DONE]":
                        response.timings.request_end = iter_time
                        break

                    data = json.loads(line.strip()[len("data: ") :])
                    delta = OpenAIHTTPBackend._extract_completions_delta_content(
                        type_, data
                    )

                    if delta:
                        response.type_ = "iter"
                        response.timings.values.append(iter_time)
                        last_time = (
                            response.timings.values[-1]
                            if response.timings.values
                            else response.timings.request_start
                        )
                        response.timings.delta = iter_time - last_time
                        response.stats.response_stream_iterations += 1
                        response.delta = delta
                        response.content += delta
                        yield response

                    usage = OpenAIHTTPBackend._extract_completions_usage(data)
                    if usage:
                        response.stats.response_prompt_tokens = usage["prompt"]
                        response.stats.response_output_tokens = usage["output"]

        logger.info(
            "{} request {} with headers: {} and payload: {} completed with content: {}",
            self.__class__.__name__,
            id_,
            headers,
            payload,
            response.content,
        )
        response.type_ = "final"
        response.delta = None
        yield response

    @staticmethod
    def _create_chat_messages(
        content: Union[
            str,
            List[Union[str, Dict[str, Union[str, Dict[str, str]]], Path, Image.Image]],
            Any,
        ],
    ) -> List[Dict]:
        if isinstance(content, str):
            return [
                {
                    "role": "user",
                    "content": content,
                }
            ]

        if isinstance(content, list):
            resolved_content = []

            for item in content:
                if isinstance(item, Dict):
                    resolved_content.append(item)
                elif isinstance(item, str):
                    resolved_content.append({"type": "text", "text": item})
                elif isinstance(item, Image.Image) or (
                    isinstance(item, Path) and item.suffix.lower() in [".jpg", ".jpeg"]
                ):
                    image = item if isinstance(item, Image.Image) else Image.open(item)
                    encoded = base64.b64encode(image.tobytes()).decode("utf-8")
                    resolved_content.append(
                        {
                            "type": "image",
                            "image": {
                                "url": f"data:image/jpeg;base64,{encoded}",
                            },
                        }
                    )
                elif isinstance(item, Path) and item.suffix.lower() in [".wav"]:
                    encoded = base64.b64encode(item.read_bytes()).decode("utf-8")
                    resolved_content.append(
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": f"{encoded}",
                                "format": "wav",
                            },
                        }
                    )
                else:
                    raise ValueError(
                        f"Unsupported content item type: {item} in list: {content}"
                    )

            return [
                {
                    "role": "user",
                    "content": resolved_content,
                }
            ]

        raise ValueError(f"Unsupported content type: {content}")

    @staticmethod
    def _extract_completions_delta_content(
        type_: Literal["text", "chat"], data: Dict
    ) -> Optional[str]:
        if "choices" not in data or not data["choices"]:
            return None

        if type_ == "text":
            return data["choices"][0]["text"]

        if type_ == "chat":
            return data["choices"][0]["delta"]["content"]

        raise ValueError(f"Unsupported type: {type_}")

    @staticmethod
    def _extract_completions_usage(
        data: Dict,
    ) -> Optional[Dict[Literal["prompt", "output"], int]]:
        if "usage" not in data or not data["usage"]:
            return None

        return {
            "prompt": data["usage"]["prompt_tokens"],
            "output": data["usage"]["completion_tokens"],
        }
