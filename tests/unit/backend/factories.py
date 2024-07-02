import functools
import random

from openai.types import Completion, CompletionChoice, Model
from polyfactory.factories.pydantic_factory import ModelFactory

__all__ = ["OpenAIModel", "OpenAICompletionChoice", "OpenAICompletion"]


class OpenAIModel(ModelFactory[Model]):
    """
    A model factory for Open AI Model representation.
    """

    pass


class OpenAICompletionChoice(ModelFactory[CompletionChoice]):
    """
    A model factory for Open AI Completion Choice representation.
    """

    finish_reason = "stop"


class OpenAICompletion(ModelFactory[Completion]):
    """
    A model factory for Open AI Completion representation.
    """

    choices = functools.partial(OpenAICompletionChoice.batch, random.randint(3, 5))
