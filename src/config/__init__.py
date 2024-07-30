from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["settings"]


class LoggingSettings(BaseModel):
    disabled: bool = False
    clear_loggers: bool = True
    console_log_level: str = "INFO"
    log_file: Optional[str] = None
    log_file_level: Optional[str] = None


class OpenAISettings(BaseModel):

    # OpenAI API key.
    api_key: str = "invalid"

    # OpenAI-compatible server URL
    # NOTE: The default value is default address of llama.cpp web server
    base_url: str = "http://localhost:8080"


class Settings(BaseSettings):
    """
    All the settings are powered by pydantic_settings and could be
    populated from the .env file.

    The format to populate the settings is next

    ```sh
    export GUIDELLM__LOGGING__DISABLED=true
    export GUIDELLM__OPENAI__API_KEY=******
    ```

    """

    model_config = SettingsConfigDict(
        env_prefix="GUIDELLM",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    root_dir: Path

    # TODO: add to the DEVELOPING.md after
    # https://github.com/neuralmagic/guidellm/pull/17
    # is merged
    debug: bool = False

    logging: LoggingSettings = LoggingSettings()
    openai: OpenAISettings = OpenAISettings()


settings = Settings(
    # NOTE: hardcoded since should not be changed in a runtime
    root_dir=Path(__file__).parent.parent.parent,
)
