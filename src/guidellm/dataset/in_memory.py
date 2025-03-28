from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
)
from transformers import PreTrainedTokenizerBase

from guidellm.dataset.creator import DatasetCreator

__all__ = ["InMemoryDatasetCreator"]


class InMemoryDatasetCreator(DatasetCreator):
    @classmethod
    def is_supported(cls, data: Any, data_args: Optional[Dict[str, Any]]) -> bool:
        return isinstance(data, Iterable) and not isinstance(data, str)

    @classmethod
    def handle_create(
        cls,
        data: Any,
        data_args: Optional[Dict[str, Any]],
        processor: Optional[Union[str, Path, PreTrainedTokenizerBase]],
        processor_args: Optional[Dict[str, Any]],
        random_seed: int,
    ) -> Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]:
        if not isinstance(data, Iterable):
            raise TypeError(
                f"Unsupported data format. Expected Iterable[Any], got {type(data)}"
            )

        if not data:
            raise ValueError("Data is empty")

        if isinstance(data, Dict):
            # assume data is a dictionary of columns and values: {"c1": ["i1", "i2"]}
            data_dict = cls.format_data_dict(data)
        elif isinstance(data[0], Dict):
            # assume data is a list of dictionaries: [{"c1": "i1"}, {"c1": "i2"}]
            data_dict = cls.format_data_iterable_dicts(data)
        else:
            # assume data is a list of items with no columns: ["i1", "i2"]
            data_dict = cls.format_data_iterable_values(data)

        return Dataset.from_dict(data_dict, **(data_args or {}))

    @classmethod
    def format_data_dict(cls, data: Dict[Any, Any]) -> Dict[str, Any]:
        if not isinstance(data, Dict):
            raise TypeError(
                f"Unsupported data format. Expected Dict[str, Iterable[Any]], "
                f"got {type(data)}"
            )

        if not all(
            isinstance(key, str) and isinstance(val, Iterable)
            for key, val in data.items()
        ):
            raise TypeError(
                "Unsupported data format. Expected Dict[str, Iterable[Any]], "
                f"got {type(data)}"
            )

        samples = len(list(data.values())[0])
        if not all(len(val) == samples for val in data.values()):
            raise ValueError(
                "Unsupported data format. Not all columns have the same number samples "
                f"for {data}"
            )

        return data

    @classmethod
    def format_data_iterable_dicts(
        cls, data: Iterable[Dict[Any, Any]]
    ) -> Dict[str, Any]:
        if not isinstance(data, Iterable):
            raise TypeError(
                f"Unsupported data format. Expected Iterable[Dict[str, Any]], "
                f"got {type(data)}"
            )

        if not all(isinstance(item, Dict) for item in data):
            raise TypeError(
                f"Unsupported data format. Expected Iterable[Dict[str, Any]], "
                f"got {type(data)}"
            )

        if not all(isinstance(key, str) for key in data[0]):
            raise TypeError(
                "Unsupported data format. Expected Dict[str, Any], "
                f"but one of the items had a non string column for {data}"
            )

        columns = list(data[0].keys())
        if not all(
            len(item) == len(columns) and all(key in item for key in columns)
            for item in data
        ):
            raise ValueError(
                "Unsupported data format. Not all items have the same columns "
                f"for {data}"
            )

        data_dict = {key: [] for key in columns}
        for item in data:
            for key, value in item.items():
                data_dict[key].append(value)

        return data_dict

    @classmethod
    def format_data_iterable_values(cls, data: Iterable[Any]) -> Dict[str, Any]:
        if not isinstance(data, Iterable):
            raise TypeError(
                f"Unsupported data format. Expected Iterable[Iterable[Any]], "
                f"got {type(data)}"
            )

        first_type = type(data[0])
        if not all(isinstance(item, first_type) for item in data):
            raise TypeError(
                f"Unsupported data format. Not all types are the same for {data}"
            )

        return {"data": list(data)}
