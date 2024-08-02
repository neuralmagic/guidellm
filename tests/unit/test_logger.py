import pytest
from loguru import logger

from config import LoggingSettings
from guidellm.logger import configure_logger


@pytest.fixture(autouse=True)
def reset_logger():
    # Ensure logger is reset before each test
    logger.remove()
    yield
    logger.remove()


def test_default_logger_settings(capsys):
    configure_logger(config=LoggingSettings())

    # Default settings should log to console with INFO level and no file logging
    logger.info("Info message")
    logger.debug("Debug message")

    captured = capsys.readouterr()
    assert captured.out.count("Info message") == 1
    assert "Debug message" not in captured.out


def test_configure_logger_console_settings(capsys):
    # Test configuring the logger to change console log level
    config = LoggingSettings(console_log_level="DEBUG")
    configure_logger(config=config)
    logger.info("Info message")
    logger.debug("Debug message")

    captured = capsys.readouterr()
    assert captured.out.count("Info message") == 1
    assert captured.out.count("Debug message") == 1


def test_configure_logger_file_settings(tmp_path):
    # Test configuring the logger to log to a file
    log_file = tmp_path / "test.log"
    config = LoggingSettings(log_file=str(log_file), log_file_level="DEBUG")
    configure_logger(config=config)
    logger.info("Info message")
    logger.debug("Debug message")

    with open(log_file, "r") as f:
        log_contents = f.read()
    assert log_contents.count('"message": "Info message"') == 1
    assert log_contents.count('"message": "Debug message"') == 1


def test_configure_logger_console_and_file(capsys, tmp_path):
    # Test configuring the logger to change both console and file settings
    log_file = tmp_path / "test.log"
    config = LoggingSettings(
        console_log_level="ERROR", log_file=str(log_file), log_file_level="INFO"
    )
    configure_logger(config=config)
    logger.info("Info message")
    logger.error("Error message")

    captured = capsys.readouterr()
    assert "Info message" not in captured.out
    assert captured.out.count("Error message") == 1

    with open(log_file, "r") as f:
        log_contents = f.read()
    assert log_contents.count('"message": "Info message"') == 1
    assert log_contents.count('"message": "Error message"') == 1


def test_environment_variable_override(monkeypatch, capsys, tmp_path):
    configure_logger(
        config=LoggingSettings(
            console_log_level="ERROR",
            log_file=str(tmp_path / "env_test.log"),
            log_file_level="DEBUG",
        )
    )
    logger.info("Info message")
    logger.error("Error message")
    logger.debug("Debug message")

    captured = capsys.readouterr()
    assert "Info message" not in captured.out
    assert captured.out.count("Error message") == 1
    assert "Debug message" not in captured.out

    with open(tmp_path / "env_test.log", "r") as f:
        log_contents = f.read()
    assert log_contents.count('"message": "Error message"') == 1
    assert log_contents.count('"message": "Info message"') == 1
    assert log_contents.count('"message": "Debug message"') == 1


def test_logging_disabled(capsys):
    configure_logger(config=LoggingSettings(disabled=True))
    logger.info("Info message")
    logger.error("Error message")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
