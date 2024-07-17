"""
This module includes data models factories for openai 3-rd party package
"""

import random
import string
import time
import uuid
from typing import Generator

from openai.types import Completion, Model


def words(n: int = 1) -> Generator[str, None, None]:
    for _ in range(n):
        yield "".join(
            (random.choice(string.ascii_letters) for _ in range(random.randint(3, 10)))
        )


def openai_completion_factory(
    n: int = 3, **kwargs
) -> Generator[Completion, None, None]:
    """
    The factory that yields the openai Completion instance.
    """

    for i in range(1, n + 1):
        payload = {
            "id": str(uuid.uuid4()),
            "choices": [],
            "stop": False if i < n else True,
            "content": " ".join(words(random.randint(3, 10))) if i < n else "",
            "object": "text_completion",
            "model": "mock-model",
            "created": int(time.time()),
        }
        payload.update(kwargs)

        yield Completion(**payload)  # type: ignore


def openai_model_factory(n: int = 3) -> Generator[Model, None, None]:
    """
    The factory that yields the random openai Model instance.
    """
    for _ in range(n):
        yield Model(
            id=str(uuid.uuid4()),
            created=int(time.time()),
            object="model",
            owned_by="neuralmagic",
        )
