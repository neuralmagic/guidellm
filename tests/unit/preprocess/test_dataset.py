import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from guidellm.preprocess.dataset import (
    handle_ignore_strategy,
    handle_concatenate_strategy,
    handle_pad_strategy,
    process_dataset,
    push_dataset_to_hub,
    ShortPromptStrategy,
    STRATEGY_HANDLERS, save_dataset_to_file,
)


@pytest.fixture
def tokenizer_mock():
    tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
    tokenizer.encode.side_effect = lambda x: [1] * len(x)
    tokenizer.decode.side_effect = lambda x, *args, **kwargs: ''.join(str(item) for item in x)
    return tokenizer


@patch.dict(STRATEGY_HANDLERS, {ShortPromptStrategy.IGNORE: MagicMock(return_value="processed_prompt")})
@patch(f"{process_dataset.__module__}.guidellm_load_dataset")
@patch(f"{process_dataset.__module__}.check_load_processor")
@patch(f"{process_dataset.__module__}.Dataset")
@patch(f"{process_dataset.__module__}.IntegerRangeSampler")
def test_strategy_handler_called(
        mock_sampler, mock_dataset_class, mock_check_processor, mock_load_dataset, tokenizer_mock
):
    mock_handler = STRATEGY_HANDLERS[ShortPromptStrategy.IGNORE]
    mock_dataset = [{"prompt": "abc"}, {"prompt": "def"}]
    mock_load_dataset.return_value = (mock_dataset, {"prompt_column": "prompt"})
    mock_check_processor.return_value = tokenizer_mock
    mock_sampler.side_effect = lambda **kwargs: [10, 10]

    mock_dataset_obj = MagicMock(spec=Dataset)
    mock_dataset_class.from_list.return_value = mock_dataset_obj

    process_dataset("input", "output_dir/data.json", tokenizer_mock, short_prompt_strategy=ShortPromptStrategy.IGNORE)

    assert mock_handler.call_count == 2
    mock_load_dataset.assert_called_once()
    mock_check_processor.assert_called_once()


def test_handle_ignore_strategy_too_short(tokenizer_mock):
    result = handle_ignore_strategy("short", 10, tokenizer_mock)
    assert result is None
    tokenizer_mock.encode.assert_called_with("short")


def test_handle_ignore_strategy_sufficient_length(tokenizer_mock):
    result = handle_ignore_strategy("long prompt", 5, tokenizer_mock)
    assert result == "long prompt"
    tokenizer_mock.encode.assert_called_with("long prompt")


def test_handle_concatenate_strategy_enough_prompts(tokenizer_mock):
    dataset_iter = iter([{"prompt": "longer"}])
    result = handle_concatenate_strategy("short", 10, dataset_iter, "prompt", tokenizer_mock)
    assert result == "shortlonger"


def test_handle_concatenate_strategy_not_enough_prompts(tokenizer_mock):
    dataset_iter = iter([])
    result = handle_concatenate_strategy("short", 10, dataset_iter, "prompt", tokenizer_mock)
    assert result is None


def test_handle_pad_strategy(tokenizer_mock):
    result = handle_pad_strategy("short", 10, tokenizer_mock, "p")
    assert result == "shortppppp"


@patch("guidellm.preprocess.dataset.save_dataset_to_file")
@patch("guidellm.preprocess.dataset.Dataset")
@patch("guidellm.preprocess.dataset.guidellm_load_dataset")
@patch("guidellm.preprocess.dataset.check_load_processor")
@patch("guidellm.preprocess.dataset.IntegerRangeSampler")
def test_process_dataset_non_empty(
        mock_sampler,
        mock_check_processor,
        mock_load_dataset,
        mock_dataset_class,
        mock_save_to_file,
        tokenizer_mock,
):
    from guidellm.preprocess.dataset import process_dataset

    mock_dataset = [{"prompt": "Hello"}, {"prompt": "How are you?"}]
    mock_load_dataset.return_value = (mock_dataset, {"prompt_column": "prompt"})
    mock_check_processor.return_value = tokenizer_mock
    mock_sampler.side_effect = lambda **kwargs: [3, 3, 3]

    mock_dataset_obj = MagicMock(spec=Dataset)
    mock_dataset_class.from_list.return_value = mock_dataset_obj

    output_path = "output_dir/data.json"
    process_dataset("input", output_path, tokenizer_mock)

    mock_load_dataset.assert_called_once()
    mock_check_processor.assert_called_once()
    mock_dataset_class.from_list.assert_called_once()
    mock_save_to_file.assert_called_once_with(mock_dataset_obj, output_path)

    args, _ = mock_dataset_class.from_list.call_args
    processed_list = args[0]
    assert len(processed_list) == 2
    for item in processed_list:
        assert "prompt" in item
        assert "prompt_tokens_count" in item
        assert "output_tokens_count" in item
        assert len(tokenizer_mock.encode(item["prompt"])) <= 3


@patch(f"{process_dataset.__module__}.Dataset")
@patch(f"{process_dataset.__module__}.guidellm_load_dataset")
@patch(f"{process_dataset.__module__}.check_load_processor")
@patch(f"{process_dataset.__module__}.IntegerRangeSampler")
def test_process_dataset_empty_after_processing(
        mock_sampler, mock_check_processor, mock_load_dataset, mock_dataset_class, tokenizer_mock
):
    mock_dataset = [{"prompt": ""}]
    mock_load_dataset.return_value = (mock_dataset, {"prompt_column": "prompt"})
    mock_check_processor.return_value = tokenizer_mock
    mock_sampler.side_effect = lambda **kwargs: [10]

    process_dataset("input", "output_dir/data.json", tokenizer_mock)

    mock_load_dataset.assert_called_once()
    mock_check_processor.assert_called_once()
    mock_dataset_class.from_list.assert_not_called()


@patch(f"{process_dataset.__module__}.push_dataset_to_hub")
@patch(f"{process_dataset.__module__}.Dataset")
@patch(f"{process_dataset.__module__}.guidellm_load_dataset")
@patch(f"{process_dataset.__module__}.check_load_processor")
@patch(f"{process_dataset.__module__}.IntegerRangeSampler")
def test_process_dataset_push_to_hub_called(
        mock_sampler, mock_check_processor, mock_load_dataset, mock_dataset_class, mock_push, tokenizer_mock
):
    mock_dataset = [{"prompt": "abc"}]
    mock_load_dataset.return_value = (mock_dataset, {"prompt_column": "prompt"})
    mock_check_processor.return_value = tokenizer_mock
    mock_sampler.side_effect = lambda **kwargs: [3]

    mock_dataset_obj = MagicMock(spec=Dataset)
    mock_dataset_class.from_list.return_value = mock_dataset_obj

    process_dataset("input", "output_dir/data.json", tokenizer_mock, push_to_hub=True, hub_dataset_id="id123")
    mock_push.assert_called_once_with("id123", mock_dataset_obj)


@patch(f"{process_dataset.__module__}.push_dataset_to_hub")
@patch(f"{process_dataset.__module__}.Dataset")
@patch(f"{process_dataset.__module__}.guidellm_load_dataset")
@patch(f"{process_dataset.__module__}.check_load_processor")
@patch(f"{process_dataset.__module__}.IntegerRangeSampler")
def test_process_dataset_push_to_hub_not_called(
        mock_sampler, mock_check_processor, mock_load_dataset, mock_dataset_class, mock_push, tokenizer_mock
):
    mock_dataset = [{"prompt": "abc"}]
    mock_load_dataset.return_value = (mock_dataset, {"prompt_column": "prompt"})
    mock_check_processor.return_value = tokenizer_mock
    mock_sampler.side_effect = lambda **kwargs: [3]

    mock_dataset_obj = MagicMock(spec=Dataset)
    mock_dataset_class.from_list.return_value = mock_dataset_obj

    process_dataset("input", "output_dir/data.json", tokenizer_mock, push_to_hub=False)
    mock_push.assert_not_called()


def test_push_dataset_to_hub_success():
    os.environ["HF_TOKEN"] = "token"
    mock_dataset = MagicMock(spec=Dataset)
    push_dataset_to_hub("dataset_id", mock_dataset)
    mock_dataset.push_to_hub.assert_called_once_with("dataset_id", token="token")


def test_push_dataset_to_hub_error_no_env():
    if "HF_TOKEN" in os.environ:
        del os.environ["HF_TOKEN"]
    mock_dataset = MagicMock(spec=Dataset)
    with pytest.raises(ValueError, match="hub_dataset_id and HF_TOKEN"):
        push_dataset_to_hub("dataset_id", mock_dataset)


def test_push_dataset_to_hub_error_no_id():
    os.environ["HF_TOKEN"] = "token"
    mock_dataset = MagicMock(spec=Dataset)
    with pytest.raises(ValueError, match="hub_dataset_id and HF_TOKEN"):
        push_dataset_to_hub(None, mock_dataset)


@patch.object(Path, "mkdir")
def test_save_dataset_to_file_csv(mock_mkdir):
    mock_dataset = MagicMock(spec=Dataset)
    output_path = Path("some/path/output.csv")
    save_dataset_to_file(mock_dataset, output_path)
    mock_dataset.to_csv.assert_called_once_with(str(output_path))
    mock_mkdir.assert_called_once()


@patch.object(Path, "mkdir")
def test_save_dataset_to_file_json(mock_mkdir):
    mock_dataset = MagicMock(spec=Dataset)
    output_path = Path("some/path/output.json")
    save_dataset_to_file(mock_dataset, output_path)
    mock_dataset.to_json.assert_called_once_with(str(output_path))
    mock_mkdir.assert_called_once()


@patch.object(Path, "mkdir")
def test_save_dataset_to_file_parquet(mock_mkdir):
    mock_dataset = MagicMock(spec=Dataset)
    output_path = Path("some/path/output.parquet")
    save_dataset_to_file(mock_dataset, output_path)
    mock_dataset.to_parquet.assert_called_once_with(str(output_path))
    mock_mkdir.assert_called_once()


@patch.object(Path, "mkdir")
def test_save_dataset_to_file_unsupported_type(mock_mkdir):
    mock_dataset = MagicMock(spec=Dataset)
    output_path = Path("some/path/output.txt")
    with pytest.raises(ValueError, match=r"Unsupported file suffix '.txt'.*"):
        save_dataset_to_file(mock_dataset, output_path)
    mock_mkdir.assert_called_once()
