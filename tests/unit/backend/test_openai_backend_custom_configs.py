import pytest

from guidellm.backend import OpenAIHTTPBackend
from guidellm.config import settings


@pytest.mark.smoke
def test_openai_http_backend_default_initialization():
    backend = OpenAIHTTPBackend()
    assert backend.verify_ssl is True


@pytest.mark.smoke
def test_openai_http_backend_custom_ssl_verification():
    settings.openai.verify_ssl = False
    backend = OpenAIHTTPBackend()
    assert backend.verify_ssl is False
    # Reset the setting
    settings.openai.verify_ssl = True


@pytest.mark.smoke
def test_openai_http_backend_custom_headers_override():
    # Set a default api_key, which would normally create an Authorization header
    settings.openai.api_key = "default-api-key"

    # Set custom headers that override the default Authorization and add a new header
    openshift_token = "Bearer sha256~xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    override_headers = {
        "Authorization": openshift_token,
        "Custom-Header": "Custom-Value",
    }
    settings.openai.headers = override_headers

    # Initialize the backend
    backend = OpenAIHTTPBackend()

    # Check that the override headers are used
    assert backend.headers["Authorization"] == openshift_token
    assert backend.headers["Custom-Header"] == "Custom-Value"
    assert len(backend.headers) == 2

    # Reset the settings
    settings.openai.api_key = None
    settings.openai.headers = None
