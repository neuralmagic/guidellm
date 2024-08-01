import pytest

from guidellm.config.base import (
    Settings,
    Environment,
    LoggingSettings,
    OpenAISettings,
    ReportGenerationSettings,
)


@pytest.mark.unit
def test_default_settings():
    settings = Settings()
    assert settings.env == Environment.PROD
    assert settings.logging == LoggingSettings()
    assert settings.openai == OpenAISettings()
    assert (
        settings.report_generation.source
        == "https://guidellm.neuralmagic.com/local-report/index.html"
    )


@pytest.mark.unit
def test_settings_from_env_variables(monkeypatch):
    monkeypatch.setenv("GUIDELLM__env", "dev")
    monkeypatch.setenv("GUIDELLM__logging__disabled", "true")
    monkeypatch.setenv("GUIDELLM__OPENAI__API_KEY", "test_key")
    monkeypatch.setenv("GUIDELLM__OPENAI__BASE_URL", "http://test.url")
    monkeypatch.setenv("GUIDELLM__REPORT_GENERATION__SOURCE", "http://custom.url")

    settings = Settings()
    assert settings.env == Environment.DEV
    assert settings.logging.disabled is True
    assert settings.openai.api_key == "test_key"
    assert settings.openai.base_url == "http://test.url"
    assert settings.report_generation.source == "http://custom.url"


@pytest.mark.unit
def test_report_generation_default_source():
    settings = Settings(env=Environment.LOCAL)
    assert settings.report_generation.source == "tests/dummy/report.html"

    settings = Settings(env=Environment.DEV)
    assert (
        settings.report_generation.source
        == "https://dev.guidellm.neuralmagic.com/local-report/index.html"
    )

    settings = Settings(env=Environment.STAGING)
    assert (
        settings.report_generation.source
        == "https://staging.guidellm.neuralmagic.com/local-report/index.html"
    )

    settings = Settings(env=Environment.PROD)
    assert (
        settings.report_generation.source
        == "https://guidellm.neuralmagic.com/local-report/index.html"
    )


@pytest.mark.sanity
def test_logging_settings():
    logging_settings = LoggingSettings(
        disabled=True,
        console_log_level="DEBUG",
        log_file="app.log",
        log_file_level="ERROR",
    )
    assert logging_settings.disabled is True
    assert logging_settings.console_log_level == "DEBUG"
    assert logging_settings.log_file == "app.log"
    assert logging_settings.log_file_level == "ERROR"


@pytest.mark.sanity
def test_openai_settings():
    openai_settings = OpenAISettings(api_key="test_api_key", base_url="http://test.api")
    assert openai_settings.api_key == "test_api_key"
    assert openai_settings.base_url == "http://test.api"


@pytest.mark.sanity
def test_report_generation_settings():
    report_settings = ReportGenerationSettings(source="http://custom.report")
    assert report_settings.source == "http://custom.report"
