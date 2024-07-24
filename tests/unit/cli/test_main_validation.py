import pytest

from guidellm.main import main


def test_task_without_data(mocker, default_main_kwargs):
    patch = mocker.patch("guidellm.backend.Backend.create")
    default_main_kwargs.update({"task": "can't be used without data"})
    with pytest.raises(NotImplementedError):
        getattr(main, "callback")(**default_main_kwargs)

    assert patch.call_count == 1


def test_invalid_data_type(mocker, default_main_kwargs):
    patch = mocker.patch("guidellm.backend.Backend.create")
    default_main_kwargs.update({"data_type": "invalid"})

    with pytest.raises(ValueError):
        getattr(main, "callback")(**default_main_kwargs)

    assert patch.call_count == 1


def test_invalid_rate_type(mocker, default_main_kwargs):
    patch = mocker.patch("guidellm.backend.Backend.create")
    file_request_generator_initialization_patch = mocker.patch(
        "guidellm.request.file.FileRequestGenerator.__init__",
        return_value=None,
    )
    default_main_kwargs.update({"rate_type": "invalid", "data_type": "file"})

    with pytest.raises(ValueError):
        getattr(main, "callback")(**default_main_kwargs)

    assert patch.call_count == 1
    assert file_request_generator_initialization_patch.call_count == 1
