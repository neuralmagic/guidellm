from typing import Any, Dict, List, Optional, Union

from datasets import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from guidellm.dataset.datasets import HFDatasetsCreator
from guidellm.dataset.file import FileDatasetCreator
from guidellm.dataset.in_memory import InMemoryDatasetCreator
from guidellm.dataset.synthetic import SyntheticDatasetCreator

__all__ = ["load_dataset"]


def load_dataset(
    data: Any,
    data_args: Optional[Dict[str, Any]],
    processor: PreTrainedTokenizerBase,
    split_pref_order: Optional[List[str]] = None,
) -> Union[Dataset, IterableDataset]:
    creators = [
        InMemoryDatasetCreator,
        SyntheticDatasetCreator,
        FileDatasetCreator,
        HFDatasetsCreator,
    ]

    for creator in creators:
        if creator.is_supported(data, data_args):
            return creator.create(data, data_args, processor, split_pref_order)

    raise ValueError(f"Unsupported data type: {type(data)} given for {data}. ")
