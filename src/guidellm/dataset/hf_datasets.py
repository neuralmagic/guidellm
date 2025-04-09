from pathlib import Path
from typing import Any, Dict, Optional, Union

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    get_dataset_config_info,
    load_dataset,
)
from transformers import PreTrainedTokenizerBase

from guidellm.dataset.creator import DatasetCreator

__all__ = ["HFDatasetsCreator"]


class HFDatasetsCreator(DatasetCreator):
    @classmethod
    def is_supported(cls, data: Any, data_args: Optional[Dict[str, Any]]) -> bool:
        if isinstance(
            data, (Dataset, DatasetDict, IterableDataset, IterableDatasetDict)
        ):
            # base type is supported
            return True

        if isinstance(data, (str, Path)) and (path := Path(data)).exists():
            # local folder or py file, assume supported
            return path.is_dir() or path.suffix == ".py"

        if isinstance(data, (str, Path)):
            try:
                # try to load dataset
                return get_dataset_config_info(data) is not None
            except:
                pass

        return False

    @classmethod
    def handle_create(
        cls,
        data: Any,
        data_args: Optional[Dict[str, Any]],
        processor: Optional[Union[str, Path, PreTrainedTokenizerBase]],
        processor_args: Optional[Dict[str, Any]],
        random_seed: int,
    ) -> Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]:
        if isinstance(data, (str, Path)):
            data = load_dataset(data, **(data_args or {}))
        elif data_args:
            raise ValueError(
                f"data_args should not be provided when data is a {type(data)}"
            )

        if isinstance(
            data, (Dataset, DatasetDict, IterableDataset, IterableDatasetDict)
        ):
            return data

        raise ValueError(f"Unsupported data type: {type(data)} given for {data}. ")
