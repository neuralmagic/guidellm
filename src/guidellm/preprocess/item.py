from collections.abc import Sequence
from typing import Generic, Optional, TypeVar

from pydantic import Field

from guidellm.objects.pydantic import StandardBaseModel

PromptT = TypeVar("PromptT")


class Item(StandardBaseModel, Generic[PromptT]):
    """
    Represents a single item in a dataset,
    containing a prompt and its associated metadata.
    """

    value: PromptT = Field(
        description="The prompt text or data for the item.",
        examples=[
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
        ],
    )
    prompt_tokens: Optional[int] = Field(
        default=None, gt=0, description="Number of tokens in the prompt"
    )
    output_tokens: Optional[int] = Field(
        default=None, gt=0, description="Number of tokens in the output"
    )


class ItemList(Sequence[Item[PromptT]]):
    """
    Represents a list of items, each containing a prompt and its metadata.
    """

    shared_prefix: Optional[PromptT]

    def __init__(self, *items: Item[PromptT], shared_prefix: Optional[PromptT] = None):
        self.shared_prefix = shared_prefix
        self._items = list(items)

    def __getitem__(self, key):
        return self._items[key]

    def __len__(self) -> int:
        return len(self._items)
