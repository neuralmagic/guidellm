import csv
import json
from typing import Optional, Union, List
from transformers import PreTrainedTokenizer
from loguru import logger
from guidellm.core.request import BenchmarkRequest
from guidellm.request.base import RequestGenerator
from guidellm.utils import PREFERRED_DATA_COLUMNS

__all__ = ["FileRequestGenerator"]


class FileRequestGenerator(RequestGenerator):
    """
    A request generator implementation for files.

    :param file_path: The path to the file containing the data.
    :type file_path: str
    :param tokenizer: The tokenizer instance or the name/config to use for tokenizing prompts.
    :type tokenizer: Union[str, PreTrainedTokenizer]
    :param mode: The generation mode, either 'async' or 'sync'.
    :type mode: str
    :param async_queue_size: The size of the request queue.
    :type async_queue_size: int
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
        mode: str = "async",
        async_queue_size: int = 50,
    ):
        super().__init__(tokenizer, mode, async_queue_size)
        self._file_path = file_path
        self._data = self._load_file()
        self._iterator = iter(self._data)

    def create_item(self) -> BenchmarkRequest:
        """
        Create a new benchmark request item from the data.

        :return: A new benchmark request.
        :rtype: BenchmarkRequest
        """
        try:
            data = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._data)
            data = next(self._iterator)

        token_count = (
            self.tokenizer(data)["input_ids"].shape[0] if self.tokenizer else None
        )
        request = BenchmarkRequest(prompt=data, token_count=token_count)
        logger.debug(f"Created new BenchmarkRequest: {request}")

        return request

    def _load_file(self) -> List[str]:
        if self._file_path.endswith(".txt"):
            data = self._load_text_file()
        elif self._file_path.endswith(".csv"):
            data = self._load_csv_file()
        elif self._file_path.endswith(".jsonl"):
            data = self._load_jsonl_file()
        elif self._file_path.endswith(".json"):
            data = self._load_json_file()
        else:
            raise ValueError("Unsupported file type")

        return [line.strip() for line in data if line and line.strip()]

    def _load_text_file(self) -> List[str]:
        with open(self._file_path, "r", encoding="utf-8") as file:
            data = file.readlines()

        return data

    def _load_csv_file(self) -> List[str]:
        data = []
        with open(self._file_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            columns = reader.fieldnames
            for row in reader:
                # convert the row to a dictionary
                obj = {col: row[col] for col in columns}
                data.append(obj)

        return self._extract_prompts(data)

    def _load_jsonl_file(self) -> List[str]:
        data = []
        with open(self._file_path, "r", encoding="utf-8") as file:
            for line in file:
                obj = json.loads(line)
                data.append(obj)

        return self._extract_prompts(data)

    def _load_json_file(self) -> List[str]:
        with open(self._file_path, "r", encoding="utf-8") as file:
            obj = json.load(file)
            data = None

            if isinstance(obj, list):
                data = obj
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, list):
                        data = value
                        break

            if data is None:
                raise ValueError(
                    f"Unsupported JSON structure, expected a list or a dictionary with a list. Given: {obj}"
                )

        return self._extract_prompts(data)

    def _extract_prompts(self, objects: List[dict]) -> List[str]:
        data = []
        for obj in objects:
            for col in PREFERRED_DATA_COLUMNS:
                if col in obj:
                    data.append(obj[col])
                    break
            else:
                data.append(next(iter(obj.values())))
        return data
