from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from datasets import Dataset, IterableDataset
from transformers import PreTrainedTokenizer


class RequestLoader:
    def __init__(
        self,
        dataset: Union[Dataset, IterableDataset],
        processor: Optional[Union[str, Path, PreTrainedTokenizer, Callable]],
        processor_args: Optional[Dict[str, Any]],
    ):
        pass
