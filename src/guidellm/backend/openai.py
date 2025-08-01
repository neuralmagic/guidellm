import base64
import copy
import json
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal, Optional, Union

import httpx
from PIL import Image
from pydantic import dataclasses

from guidellm.backend.backend import Backend
from guidellm.backend.objects import (
    GenerationRequest,
    GenerationRequestTimings,
    GenerationResponse,
)
from guidellm.scheduler import ScheduledRequestInfo

__all__ = [
    "CHAT_COMPLETIONS",
    "CHAT_COMPLETIONS_PATH",
    "MODELS",
    "TEXT_COMPLETIONS",
    "TEXT_COMPLETIONS_PATH",
    "OpenAIHTTPBackend",
]


TEXT_COMPLETIONS_PATH = "/v1/completions"
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

EndpointType = Literal["chat_completions", "models", "text_completions"]
CHAT_COMPLETIONS: EndpointType = "chat_completions"
MODELS: EndpointType = "models"
TEXT_COMPLETIONS: EndpointType = "text_completions"


@dataclasses.dataclass
class UsageStats:
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


@Backend.register("openai_http")
class OpenAIHTTPBackend(Backend):
    """
    A HTTP-based backend implementation for requests to an OpenAI compatible server.
    For example, a vLLM server instance or requests to OpenAI's API.
    """

    def __init__(
        self,
        target: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        timeout: Optional[float] = None,
        http2: Optional[bool] = True,
        follow_redirects: Optional[bool] = None,
        max_output_tokens: Optional[int] = None,
        extra_query: Optional[dict] = None,
        extra_body: Optional[dict] = None,
        remove_from_body: Optional[list[str]] = None,
        headers: Optional[dict] = None,
        verify: Optional[bool] = None,
    ):
        """
        Initialize the OpenAI HTTP backend with the target server and optional params.

        :param target: The target URL string for the OpenAI server. ex: http://0.0.0.0:8000
        :param model: The model to use for all requests on the target server.
            If none is provided, the first available model will be used.
        :param api_key: The API key to use for requests to the OpenAI server.
            If provided, adds an Authorization header with the value
            "Authorization: Bearer {api_key}".
            If not provided, no Authorization header is added.
        :param organization: The organization to use for requests to the OpenAI server.
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
        :param follow_redirects: If True, the HTTP client will follow redirect responses.
            If not provided, the default value from settings is used.
        :param max_output_tokens: The maximum number of tokens to request for completions.
            If not provided, the default maximum tokens provided from settings is used.
        :param extra_query: Query parameters to include in requests to the OpenAI server.
            If "chat_completions", "models", or "text_completions" are included as keys,
            the values of these keys will be used as the parameters for the respective
            endpoint.
            If not provided, no extra query parameters are added.
        :param extra_body: Body parameters to include in requests to the OpenAI server.
            If "chat_completions", "models", or "text_completions" are included as keys,
            the values of these keys will be included in the body for the respective
            endpoint.
            If not provided, no extra body parameters are added.
        :param remove_from_body: Parameters that should be removed from the body of each
            request.
            If not provided, no parameters are removed from the body.
        """
        super().__init__(type_="openai_http")
        if target.endswith(("/v1", "/v1/")):
            target = target[:-3]
        if target.endswith("/"):
            target = target[:-1]
        self.target = target
        self.model = model

        # Start with default headers based on other params
        default_headers: dict[str, str] = {}
        if api_key:
            default_headers["Authorization"] = (
                f"Bearer {api_key}" if not api_key.startswith("Bearer") else api_key
            )
        if organization:
            default_headers["OpenAI-Organization"] = organization
        if project:
            default_headers["OpenAI-Project"] = project

        # User-provided headers from kwargs or settings override defaults
        merged_headers = copy.deepcopy(default_headers)
        if headers:
            merged_headers.update(headers)

        # Remove headers with None values for backward compatibility and convenience
        self.headers = {k: v for k, v in merged_headers.items() if v is not None}

        self.timeout = timeout
        self.http2 = http2
        self.follow_redirects = follow_redirects
        self.verify = verify
        self.max_output_tokens = max_output_tokens
        self.extra_query = extra_query
        self.extra_body = extra_body
        self.remove_from_body = remove_from_body

        # processing variables
        self._in_process: bool = False
        self._async_client: Optional[httpx.AsyncClient] = None

    async def info(self) -> dict[str, Any]:
        """
        :return: The information about the backend.
        """
        return {
            "target": self.target,
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "timeout": self.timeout,
            "http2": self.http2,
            "follow_redirects": self.follow_redirects,
            "verify": self.verify,
            "headers": self.headers,
            "extra_query": self.extra_query,
            "extra_body": self.extra_body,
            "remove_from_body": self.remove_from_body,
            "text_completions_path": TEXT_COMPLETIONS_PATH,
            "chat_completions_path": CHAT_COMPLETIONS_PATH,
        }

    async def default_model(self) -> Optional[str]:
        """
        :return: The model name or identifier that this backend is
            configured to use by default for generation requests.
        """
        if self.model or not self._in_process:
            return self.model

        models = await self.available_models()
        return models[0] if models else None

    async def process_startup(self):
        """
        Initialize process-specific resources and connections.
        For this backend, ensures the async client is initialized.
        """
        if self._in_process:
            raise RuntimeError("Backend already started up for process.")

        self._async_client = httpx.AsyncClient(
            http2=self.http2,
            timeout=self.timeout,
            follow_redirects=self.follow_redirects,
            verify=self.verify,
        )
        self._in_process = True

    async def validate(self):
        """
        Validate backend configuration and readiness for request processing.
        Checks available models and sets a default if none is configured.
        """
        # TODO: implement validation logic with fallbacks on sending requests first to /health, next to models, and finally a dummy request to text completions

    async def process_shutdown(self):
        """
        Clean up process-specific resources and connections.
        For this backend, closes the async client if it exists.
        """
        if not self._in_process:
            raise RuntimeError("Backend not started up for process.")

        await self._async_client.aclose()
        self._async_client = None
        self._in_process = False

    async def resolve(
        self,
        request: GenerationRequest,
        request_info: ScheduledRequestInfo[GenerationRequestTimings],
        history: Optional[list[tuple[GenerationRequest, GenerationResponse]]] = None,
    ) -> AsyncIterator[
        tuple[GenerationResponse, ScheduledRequestInfo[GenerationRequestTimings]]
    ]:
        """
        Process a generation request and yield progressive responses.

        :param request: The generation request containing content and parameters.
        :param request_info: Request tracking information to be updated with
            timing and progress metadata during processing.
        :param history: Optional conversation history for multi-turn requests.
        :raises NotImplementedError: Multi-turn requests with history are not
            yet supported.
        :yields: Tuples of (response, updated_request_info) as the generation
            progresses. The final tuple contains the complete response.
        """
        if history is not None:
            raise NotImplementedError(
                "Multi-turn requests with conversation history are not yet supported"
            )

        response = GenerationResponse(
            request_id=request.request_id,
            request_args={
                "request_type": request.request_type,
                "output_token_count": request.constraints.get("max_output_tokens"),
                **request.params,
            },
            value="",
            request_prompt_tokens=request.stats.get("prompt_tokens"),
            request_output_tokens=request.constraints.get("output_tokens"),
        )
        request_info.request_timings = GenerationRequestTimings()
        request_info.request_timings.request_start = time.time()

        completion_method = (
            self.text_completions
            if request.request_type == "text_completions"
            else self.chat_completions
        )
        completion_kwargs = (
            {
                "prompt": request.content,
                "request_id": request.request_id,
                "output_token_count": request.constraints.get("max_output_tokens"),
                "stream_response": request.params.get("stream", True),
                **request.params,
            }
            if request.request_type == "text_completions"
            else {
                "content": request.content,
                "request_id": request.request_id,
                "output_token_count": request.constraints.get("max_output_tokens"),
                "stream_response": request.params.get("stream", True),
                **request.params,
            }
        )

        async for delta, usage_stats in completion_method(**completion_kwargs):
            if request_info.request_timings.request_start is None:
                request_info.request_timings.request_start = time.time()

            if delta is not None:
                if request_info.request_timings.first_iteration is None:
                    request_info.request_timings.first_iteration = time.time()
                response.value += delta
                response.delta = delta
                request_info.request_timings.last_iteration = time.time()
                response.iterations += 1

            if usage_stats is not None:
                request_info.request_timings.request_end = time.time()
                response.request_output_tokens = usage_stats.output_tokens
                response.request_prompt_tokens = usage_stats.prompt_tokens

            yield response, request_info

        if request_info.request_timings.request_end is None:
            request_info.request_timings.request_end = time.time()
        response.delta = None
        yield response, request_info

    async def text_completions(
        self,
        prompt: Union[str, list[str]],
        request_id: Optional[str],
        output_token_count: Optional[int] = None,
        stream_response: bool = True,
        **kwargs,
    ) -> AsyncIterator[tuple[Optional[str], Optional[UsageStats]]]:
        target = f"{self.target}{TEXT_COMPLETIONS_PATH}"
        headers = self._get_headers()
        params = self._get_params(TEXT_COMPLETIONS)
        body = self._get_body(
            endpoint_type=TEXT_COMPLETIONS,
            request_kwargs=kwargs,
            max_output_tokens=output_token_count,
            prompt=prompt,
        )
        yield None, None  # Initial yield for async iterator

        if stream_response:
            body.update({"stream": True, "stream_options": {"include_usage": True}})
            async with self._async_client.stream(
                "POST",
                target,
                headers=headers,
                params=params,
                json=body,
            ) as stream:
                stream.raise_for_status()
                async for line in stream.aiter_lines():
                    if not line or not line.strip().startswith("data:"):
                        continue
                    if line.strip() == "data: [DONE]":
                        break
                    data = json.loads(line.strip()[len("data: ") :])
                    yield (
                        self._get_completions_text_content(data),
                        self._get_completions_usage_stats(data),
                    )
        else:
            async with self._async_client.post(
                target,
                headers=headers,
                params=params,
                json=body,
            ) as response:
                response.raise_for_status()
                data = response.json()
                yield (
                    self._get_completions_text_content(data),
                    self._get_completions_usage_stats(data),
                )

    async def chat_completions(
        self,
        content: Union[
            str,
            list[Union[str, dict[str, Union[str, dict[str, str]]], Path, Image.Image]],
            Any,
        ],
        request_id: Optional[str] = None,
        output_token_count: Optional[int] = None,
        raw_content: bool = False,
        stream_response: bool = True,
        **kwargs,
    ) -> AsyncIterator[tuple[Optional[str], Optional[UsageStats]]]:
        target = f"{self.target}{CHAT_COMPLETIONS_PATH}"
        headers = self._get_headers()
        params = self._get_params(CHAT_COMPLETIONS)
        body = self._get_body(
            endpoint_type=CHAT_COMPLETIONS,
            request_kwargs=kwargs,
            max_output_tokens=output_token_count,
            messages=self._get_chat_messages(content) if not raw_content else content,
            **kwargs,
        )
        yield None, None  # Initial yield for async iterator

        if stream_response:
            body.update({"stream": True, "stream_options": {"include_usage": True}})
            async with self._async_client.stream(
                "POST",
                target,
                headers=headers,
                params=params,
                json=body,
            ) as stream:
                stream.raise_for_status()
                async for line in stream.aiter_lines():
                    if not line or not line.strip().startswith("data:"):
                        continue
                    if line.strip() == "data: [DONE]":
                        break
                    data = json.loads(line.strip()[len("data: ") :])
                    yield (
                        self._get_completions_text_content(data),
                        self._get_completions_usage_stats(data),
                    )
        else:
            async with self._async_client.post(
                target,
                headers=headers,
                params=params,
                json=body,
            ) as response:
                response.raise_for_status()
                data = response.json()
                yield (
                    self._get_completions_text_content(data),
                    self._get_completions_usage_stats(data),
                )

    def _get_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            **self.headers,
        }

    def _get_params(self, endpoint_type: EndpointType) -> dict[str, str]:
        params = self.extra_query or {}

        if any(
            endpoint in params
            for endpoint in [CHAT_COMPLETIONS, MODELS, TEXT_COMPLETIONS]
        ):
            params = params.get(endpoint_type, {})

        return copy.deepcopy(params)

    def _get_chat_messages(
        self,
        content: Union[
            str,
            list[Union[str, dict[str, Union[str, dict[str, str]]], Path, Image.Image]],
            Any,
        ],
    ) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"role": "user", "content": content}]

        if not isinstance(content, list):
            raise ValueError(f"Unsupported content type: {content}")

        resolved_content = []
        for item in content:
            if isinstance(item, dict):
                resolved_content.append(item)
            elif isinstance(item, str):
                resolved_content.append({"type": "text", "text": item})
            elif isinstance(item, (Image.Image, Path)):
                resolved_content.append(self._get_chat_message_media_item(item))
            else:
                raise ValueError(
                    f"Unsupported content item type: {item} in list: {content}"
                )

        return [{"role": "user", "content": resolved_content}]

    def _get_chat_message_media_item(
        self, item: Union[Path, Image.Image]
    ) -> dict[str, Any]:
        if isinstance(item, Image.Image):
            encoded = base64.b64encode(item.tobytes()).decode("utf-8")
            return {
                "type": "image",
                "image": {"url": f"data:image/jpeg;base64,{encoded}"},
            }

        if item.suffix.lower() in [".jpg", ".jpeg"]:
            image = Image.open(item)
            encoded = base64.b64encode(image.tobytes()).decode("utf-8")
            return {
                "type": "image",
                "image": {"url": f"data:image/jpeg;base64,{encoded}"},
            }

        if item.suffix.lower() == ".wav":
            encoded = base64.b64encode(item.read_bytes()).decode("utf-8")
            return {
                "type": "input_audio",
                "input_audio": {"data": encoded, "format": "wav"},
            }

        raise ValueError(f"Unsupported file type: {item.suffix}")

    def _get_body(
        self,
        endpoint_type: EndpointType,
        request_kwargs: Optional[dict[str, Any]],
        max_output_tokens: Optional[int] = None,
        **kwargs,
    ) -> dict[str, Any]:
        extra_body = self.extra_body or {}
        if any(ep in extra_body for ep in [CHAT_COMPLETIONS, MODELS, TEXT_COMPLETIONS]):
            extra_body = extra_body.get(endpoint_type, {})

        body = copy.deepcopy(extra_body)
        body.update(request_kwargs or {})
        body.update(kwargs)
        body["model"] = self.model

        if (max_tokens := max_output_tokens or self.max_output_tokens) is not None:
            body.update(
                {
                    "max_tokens": max_tokens,
                    "max_completion_tokens": max_tokens,
                }
            )
            # Set stop conditions only for request-level limits
            if max_output_tokens:
                body.update({"stop": None, "ignore_eos": True})

        if self.remove_from_body:
            for key in self.remove_from_body:
                body.pop(key, None)

        return body

    def _get_completions_text_content(self, data: dict) -> Optional[str]:
        if not data.get("choices"):
            return None

        choice = data["choices"][0]
        return choice.get("text") or choice.get("delta", {}).get("content")

    def _get_completions_usage_stats(self, data: dict) -> Optional[UsageStats]:
        if not data.get("usage"):
            return None

        return UsageStats(
            prompt_tokens=data["usage"].get("prompt_tokens"),
            output_tokens=data["usage"].get("completion_tokens"),
        )

    async def available_models(self) -> list[str]:
        """
        Get the available models for the target server using the OpenAI models endpoint:
        /v1/models
        """
        if not self._in_process or self._async_client is None:
            raise RuntimeError(
                "Backend not started up for process, cannot fetch models."
            )

        target = f"{self.target}/v1/models"
        headers = self._headers()
        params = self._params(MODELS)
        response = await self._async_client.get(target, headers=headers, params=params)
        response.raise_for_status()

        models = []

        for item in response.json()["data"]:
            models.append(item["id"])

        return models
