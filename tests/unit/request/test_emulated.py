import json
import tempfile
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pytest
from transformers import PreTrainedTokenizer  # type: ignore

from guidellm.core.request import TextGenerationRequest
from guidellm.request.emulated import (
    EmulatedConfig,
    EmulatedRequestGenerator,
    EndlessTokens,
)


@pytest.mark.smoke()
def test_emulated_config_construction():
    config = EmulatedConfig(
        prompt_tokens=10,
        prompt_tokens_variance=2,
        prompt_tokens_min=5,
        prompt_tokens_max=15,
        generated_tokens=20,
        generated_tokens_variance=4,
        generated_tokens_min=10,
        generated_tokens_max=30,
    )
    assert config.prompt_tokens == 10
    assert config.prompt_tokens_variance == 2
    assert config.prompt_tokens_min == 5
    assert config.prompt_tokens_max == 15
    assert config.generated_tokens == 20
    assert config.generated_tokens_variance == 4
    assert config.generated_tokens_min == 10
    assert config.generated_tokens_max == 30


@pytest.mark.smoke()
def test_emulated_config_create_dict():
    config_dict = {
        "prompt_tokens": 10,
        "prompt_tokens_variance": 2,
        "prompt_tokens_min": 5,
        "prompt_tokens_max": 15,
        "generated_tokens": 20,
        "generated_tokens_variance": 4,
        "generated_tokens_min": 10,
        "generated_tokens_max": 30,
    }
    config = EmulatedConfig.create_config(config_dict)
    assert config.prompt_tokens == 10
    assert config.prompt_tokens_variance == 2
    assert config.prompt_tokens_min == 5
    assert config.prompt_tokens_max == 15
    assert config.generated_tokens == 20
    assert config.generated_tokens_variance == 4
    assert config.generated_tokens_min == 10
    assert config.generated_tokens_max == 30


@pytest.mark.smoke()
@pytest.mark.parametrize(
    ("base", "variance", "min_tokens", "max_tokens", "expected_range"),
    [
        (10, 2, None, None, (1, 10 + 5 * 2)),
        (10, 2, 5, 15, (5, 15)),
        (10, None, 5, 15, (5, 15)),
        (10, 2, 1, None, (1, 10 + 5 * 2)),
    ],
)
def test_emulated_config_token_range(
    base: int,
    variance: int,
    min_tokens: int,
    max_tokens: int,
    expected_range: Tuple[int, int],
):
    assert (
        EmulatedConfig._token_range(base, variance, min_tokens, max_tokens)
        == expected_range
    )


@pytest.mark.smoke()
@pytest.mark.parametrize(
    ("base", "variance", "min_tokens", "max_tokens", "expected_range"),
    [
        (10, None, None, None, (10, 10)),
        (10, 5, None, None, (1, 10 + 5 * 2)),
        (10, 5, 5, 15, (5, 15)),
        (10, None, 5, 15, (5, 15)),
        (10, 5, 2, None, (2, 10 + 5 * 2)),
        (10, 5, None, 20, (1, 20)),
    ],
)
def test_emulated_config_sample_tokens(
    base: int,
    variance: int,
    min_tokens: int,
    max_tokens: int,
    expected_range: Tuple[int, int],
):
    rng = np.random.default_rng()

    for _ in range(100):
        token_count = EmulatedConfig._sample_tokens(
            base, variance, min_tokens, max_tokens, rng
        )
        assert token_count >= expected_range[0]
        assert token_count <= expected_range[1]


@pytest.mark.sanity()
def test_emulated_config_create():
    test_dict = {
        "prompt_tokens": 10,
        "prompt_tokens_variance": 2,
        "prompt_tokens_min": 5,
        "prompt_tokens_max": 15,
        "generated_tokens": 20,
        "generated_tokens_variance": 4,
        "generated_tokens_min": 10,
        "generated_tokens_max": 30,
    }
    compare_config = EmulatedConfig(**test_dict)

    # test dict
    test_config = EmulatedConfig.create_config(test_dict)
    assert (
        test_config == compare_config
    ), f"Dictionary creation failed: {test_config} != {compare_config}"

    # test json str
    test_config = EmulatedConfig.create_config(json.dumps(test_dict))
    assert (
        test_config == compare_config
    ), f"JSON string creation failed: {test_config} != {compare_config}"

    # test json file str path
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = Path(temp_dir) / "test.json"
        test_path.write_text(json.dumps(test_dict))
        test_config = EmulatedConfig.create_config(str(test_path))
        assert (
            test_config == compare_config
        ), f"JSON file path creation failed: {test_config} != {compare_config}"

    # test json file Path object
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = Path(temp_dir) / "test.json"
        test_path.write_text(json.dumps(test_dict))
        test_config = EmulatedConfig.create_config(test_path)
        assert (
            test_config == compare_config
        ), f"JSON file Path object creation failed: {test_config} != {compare_config}"

    # test key value string
    test_str = (
        f"prompt_tokens={test_dict['prompt_tokens']}, "
        f"prompt_tokens_variance={test_dict['prompt_tokens_variance']}, "
        f"prompt_tokens_min={test_dict['prompt_tokens_min']}, "
        f"prompt_tokens_max={test_dict['prompt_tokens_max']}, "
        f"generated_tokens={test_dict['generated_tokens']}, "
        f"generated_tokens_variance={test_dict['generated_tokens_variance']}, "
        f"generated_tokens_min={test_dict['generated_tokens_min']}, "
        f"generated_tokens_max={test_dict['generated_tokens_max']}"
    )
    test_config = EmulatedConfig.create_config(test_str)
    assert (
        test_config == compare_config
    ), f"Key value string creation failed: {test_config} != {compare_config}"


# EndlessTokens


@pytest.mark.smoke()
@pytest.mark.parametrize(
    ("data", "expected_words", "expected_indices"),
    [
        (
            "word1 word2  word3\nword4   word5",
            ["word1", "word2", "word3", "word4", "word5"],
            [0, 3],
        ),
        (
            "word1  word2\n  word3   word4\n word5",
            ["word1", "word2", "word3", "word4", "word5"],
            [0, 2, 4],
        ),
    ],
)
def test_endless_data_words_construction(data, expected_words, expected_indices):
    tokens = EndlessTokens(data)
    assert tokens == expected_words
    assert tokens.line_indices == expected_indices


@pytest.mark.smoke()
def test_endless_data_words_create_from_basic_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test.txt"
        file_path.write_text("word1 word2 word3\nword4 word5")

        tokens = EndlessTokens(file_path)
        assert tokens == ["word1", "word2", "word3", "word4", "word5"]
        assert tokens.line_indices == [0, 3]

        tokens = EndlessTokens(str(file_path))
        assert tokens == ["word1", "word2", "word3", "word4", "word5"]
        assert tokens.line_indices == [0, 3]


@pytest.mark.smoke()
@pytest.mark.parametrize(
    ("data", "start", "length", "expected_text"),
    [
        ("word1 word2 word3 word4", 0, 2, "word1 word2"),
        ("word1     word2\nword3   word4", 1, 2, "word2\nword3"),
        (
            "word1     word2\nword3   word4",
            1,
            6,
            "word2\nword3   word4 word1     word2\nword3",
        ),
    ],
)
def test_endless_data_words_create_text(data, start, length, expected_text):
    words = EndlessTokens(data)
    text = words.create_text(start, length)
    assert text == expected_text


# EmulatedRequestGenerator


@pytest.mark.smoke()
def test_emulated_request_generator_construction(mocker, mock_auto_tokenizer):
    mocker.patch(
        "guidellm.request.emulated.EmulatedConfig.create_config",
        return_value=EmulatedConfig(prompt_tokens=10),
    )
    mocker.patch(
        "guidellm.request.emulated.EndlessTokens",
        return_value=EndlessTokens("word1 word2"),
    )
    generator = EmulatedRequestGenerator(
        config="mock_config", tokenizer="mock-tokenizer", mode="sync"
    )
    assert isinstance(generator._config, EmulatedConfig)
    assert isinstance(generator._tokens, EndlessTokens)


@pytest.mark.smoke()
def test_emulated_request_generator_create_item(mocker):
    mocker.patch(
        "guidellm.request.emulated.EndlessTokens",
        return_value=EndlessTokens("word1 word2"),
    )
    mock_tokenizer = mocker.Mock(PreTrainedTokenizer)
    mock_tokenizer.tokenize.return_value = ["word1", "word2"]
    generator = EmulatedRequestGenerator(
        config={
            "prompt_tokens": 10,
        },
        tokenizer=mock_tokenizer,
        mode="sync",
    )
    item = generator.create_item()
    assert isinstance(item, TextGenerationRequest)


@pytest.mark.smoke()
def test_emulated_request_generator_sample_prompt(mocker, mock_auto_tokenizer):
    mocker.patch(
        "guidellm.request.emulated.EndlessTokens",
        return_value=EndlessTokens("word1 word2"),
    )
    generator = EmulatedRequestGenerator(
        config={"prompt_tokens": 3}, tokenizer="mock-tokenizer", mode="sync"
    )
    prompt = generator.sample_prompt(3)
    assert prompt == "word1 word2 word1"

    request = generator.create_item()
    assert request.prompt_token_count == 3


@pytest.mark.smoke()
def test_emulated_request_generator_random_seed(mocker, mock_auto_tokenizer):
    mocker.patch(
        "guidellm.request.emulated.EndlessTokens",
        return_value=EndlessTokens("word1 word2"),
    )

    rand_gen = EmulatedRequestGenerator(
        config={"prompt_tokens": 20, "prompt_tokens_variance": 10},
        tokenizer="mock-tokenizer",
        random_seed=42,
        mode="sync",
    )
    rand_gen_comp_pos = EmulatedRequestGenerator(
        config={"prompt_tokens": 20, "prompt_tokens_variance": 10},
        tokenizer="mock-tokenizer",
        random_seed=42,
        mode="sync",
    )
    rand_gen_comp_neg = EmulatedRequestGenerator(
        config={"prompt_tokens": 20, "prompt_tokens_variance": 10},
        tokenizer="mock-tokenizer",
        random_seed=43,
        mode="sync",
    )

    assert rand_gen.create_item().prompt == rand_gen_comp_pos.create_item().prompt
    assert rand_gen.create_item().prompt != rand_gen_comp_neg.create_item().prompt


@pytest.mark.regression()
@pytest.mark.parametrize(
    ("config_type", "config"),
    [
        ("dict", {"prompt_tokens": 10, "generated_tokens": 20}),
        ("dict", {"prompt_tokens": 10, "prompt_tokens_variance": 2}),
        (
            "dict",
            {
                "prompt_tokens": 10,
                "prompt_tokens_min": 5,
                "prompt_tokens_max": 15,
                "generated_tokens": 20,
            },
        ),
        ("json_str", json.dumps({"prompt_tokens": 10, "generated_tokens": 20})),
        ("key_value_str", "prompt_tokens=10, generated_tokens=20"),
        ("file_str", json.dumps({"prompt_tokens": 10, "generated_tokens": 20})),
        ("file_path", json.dumps({"prompt_tokens": 10, "generated_tokens": 20})),
    ],
)
def test_emulated_request_generator_lifecycle(
    mock_requests_pride_and_prejudice,
    mock_auto_tokenizer,
    config_type: str,
    config: Union[str, dict, Path],
):
    if config_type in ["dict", "json_str", "key_value_str"]:
        generator = EmulatedRequestGenerator(config, tokenizer="mock-tokenizer")
    elif config_type in ["file_str", "file_path"]:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.json"
            file_path.write_text(config)  # type: ignore
            generator = EmulatedRequestGenerator(
                str(file_path) if config_type == "file_str" else file_path,
                tokenizer="mock-tokenizer",
            )
    else:
        raise Exception

    for _ in range(5):
        request = generator.create_item()
        prompt_range = generator._config.prompt_tokens_range
        outputs_range = generator._config.output_tokens_range

        assert request.prompt_token_count >= prompt_range[0]  # type: ignore
        assert request.prompt_token_count <= prompt_range[1]  # type: ignore

        prompt_tokens = len(generator.tokenizer.tokenize(request.prompt))
        assert request.prompt_token_count == prompt_tokens

        if generator._config.generated_tokens:
            assert len(outputs_range) == 2
            assert request.output_token_count >= outputs_range[0]  # type: ignore
            assert request.output_token_count <= outputs_range[1]  # type: ignore
