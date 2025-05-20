import asyncio
import json
import random
import time
import uuid
from typing import Optional, Union

import click
import uvicorn
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from lorem.text import TextLorem
from pydantic import BaseModel, Field

from guidellm.utils import split_text


def sample_number(
    mean: float,
    std_dev: float,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    integer: bool = False,
) -> Union[float, int]:
    sampled = mean if std_dev <= 0 else random.gauss(mean, std_dev)

    if minimum is not None and sampled < minimum:
        sampled = minimum

    if maximum is not None and sampled > maximum:
        sampled = maximum

    return int(sampled) if integer else sampled


class CompletionRequest(BaseModel):
    prompt: str
    model: str
    stream: Optional[bool] = False
    stream_options: Optional[dict] = Field(default_factory=dict)
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stop: Optional[list[str]] = None
    ignore_eos: Optional[bool] = None


class MockOpenAIRouter:
    def __init__(
        self,
        ttft_range: tuple[float, float] = (0, 0),
        itl_range: tuple[float, float] = (0, 0),
        output_token_range: tuple[float, float] = (256, 64),
    ):
        self.router = APIRouter()
        self.router.add_api_route("/v1/models", self.list_models, methods=["GET"])
        self.router.add_api_route("/v1/completions", self.completions, methods=["POST"])

        self.ttft_range = ttft_range
        self.itl_range = itl_range
        self.output_token_range = output_token_range
        self.lorem = TextLorem()

    def list_models(self):
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": f"model-{index}",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "user",
                    }
                    for index in range(5)
                ],
            }
        )

    async def completions(self, request: Request):
        req = await request.json()
        parsed = CompletionRequest(**req)
        include_usage = parsed.stream_options.get("include_usage", False)
        token_count = (
            parsed.max_tokens
            or parsed.max_completion_tokens
            or sample_number(*self.output_token_range, minimum=1, integer=True)
        )
        ttft = sample_number(*self.ttft_range, minimum=0)
        itl = sample_number(*self.itl_range, minimum=0)
        prompt_tokens = len(split_text(parsed.prompt))

        async def token_stream():
            if ttft > 0:
                await asyncio.sleep(ttft)

            for ind in range(token_count):
                token = self.lorem._word() if ind % 2 == 0 else " "
                response_dict = {
                    "id": str(uuid.uuid4()),
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": parsed.model,
                    "choices": [
                        {
                            "text": token if ind < token_count - 1 else ".",
                            "index": ind,
                            "logprobs": None,
                            "finish_reason": (
                                None if ind < token_count - 1 else "length"
                            ),
                        }
                    ],
                }
                yield f"data: {json.dumps(response_dict)}\n\n"
                if itl > 0:
                    await asyncio.sleep(itl)

            if include_usage:
                usage_dict = {
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": token_count,
                        "total_tokens": prompt_tokens + token_count,
                    }
                }
                yield f"data: {json.dumps(usage_dict)}\n\n"

            yield "data: [DONE]"

        if parsed.stream:
            return StreamingResponse(token_stream(), media_type="text/event-stream")

        lorem_text = " ".join([self.lorem._word() for _ in range(token_count // 2)]) + (
            "." if token_count % 2 == 1 else ""
        )
        response_dict = {
            "id": str(uuid.uuid4()),
            "object": "text_completion",
            "created": int(time.time()),
            "model": parsed.model,
            "choices": [
                {
                    "text": lorem_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length",
                }
            ],
        }
        if include_usage:
            response_dict["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": token_count,
                "total_tokens": prompt_tokens + token_count,
            }
        return JSONResponse(response_dict)


def create_app(
    ttft_range: tuple[float, float] = (0, 0),
    itl_range: tuple[float, float] = (0, 0),
    output_token_range: tuple[float, float] = (256, 64),
) -> FastAPI:
    app = FastAPI()
    mock_openai_router = MockOpenAIRouter()
    app.include_router(mock_openai_router.router)
    return app


@click.command()
@click.option("--host", default="0.0.0.0", help="Host to run the server on.")  # noqa: S104
@click.option("--port", default=8000, type=int, help="Port to run the server on.")
@click.option(
    "--ttft-range", default=(0, 0), type=(float, float), help="Tuple for TTFT range."
)
@click.option(
    "--itl-range", default=(0, 0), type=(float, float), help="Tuple for ITL range."
)
@click.option(
    "--output-token-range",
    default=(256, 64),
    type=(float, float),
    help="Tuple for output token range.",
)
def start_server(
    host: str = "0.0.0.0",  # noqa: S104
    port: int = 8000,
    ttft_range: tuple[float, float] = (0, 0),
    itl_range: tuple[float, float] = (0, 0),
    output_token_range: tuple[float, float] = (256, 64),
):
    """
    Start the server using uvicorn with the ability to launch multiple processes.
    """
    # Pass the module and app instance as a string for multi-process support
    uvicorn.run(
        "mock_openai_server:create_app",
        host=host,
        port=port,
        workers=8,
        factory=True,
    )


if __name__ == "__main__":
    start_server()
