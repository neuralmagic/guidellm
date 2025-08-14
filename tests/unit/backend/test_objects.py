"""
Unit tests for GenerationRequest, GenerationResponse, GenerationRequestTimings.
"""

import uuid

import pytest

from guidellm.backend.objects import (
    GenerationRequest,
    GenerationRequestTimings,
    GenerationResponse,
)


class TestGenerationRequest:
    """Test cases for GenerationRequest model."""

    @pytest.mark.smoke
    def test_generation_request_creation(self):
        """Test basic GenerationRequest creation.

        ### WRITTEN BY AI ###
        """
        request = GenerationRequest(content="test content")

        assert request.content == "test content"
        assert request.request_type == "text_completions"  # default
        assert isinstance(request.request_id, str)
        assert request.params == {}
        assert request.stats == {}
        assert request.constraints == {}

    @pytest.mark.smoke
    def test_generation_request_with_all_fields(self):
        """Test GenerationRequest creation with all fields.

        ### WRITTEN BY AI ###
        """
        request_id = "test-123"
        content = ["message1", "message2"]
        params = {"temperature": 0.7, "max_tokens": 100}
        stats = {"prompt_tokens": 50}
        constraints = {"output_tokens": 100}

        request = GenerationRequest(
            request_id=request_id,
            request_type="chat_completions",
            content=content,
            params=params,
            stats=stats,
            constraints=constraints,
        )

        assert request.request_id == request_id
        assert request.request_type == "chat_completions"
        assert request.content == content
        assert request.params == params
        assert request.stats == stats
        assert request.constraints == constraints

    @pytest.mark.sanity
    def test_generation_request_auto_id_generation(self):
        """Test that request_id is auto-generated if not provided.

        ### WRITTEN BY AI ###
        """
        request1 = GenerationRequest(content="test1")
        request2 = GenerationRequest(content="test2")

        assert request1.request_id != request2.request_id
        assert len(request1.request_id) > 0
        assert len(request2.request_id) > 0

        # Should be valid UUIDs
        uuid.UUID(request1.request_id)
        uuid.UUID(request2.request_id)

    @pytest.mark.sanity
    def test_generation_request_type_validation(self):
        """Test request_type field validation.

        ### WRITTEN BY AI ###
        """
        # Valid types
        request1 = GenerationRequest(content="test", request_type="text_completions")
        request2 = GenerationRequest(content="test", request_type="chat_completions")

        assert request1.request_type == "text_completions"
        assert request2.request_type == "chat_completions"

    @pytest.mark.regression
    def test_generation_request_content_types(self):
        """Test GenerationRequest with different content types.

        ### WRITTEN BY AI ###
        """
        # String content
        request1 = GenerationRequest(content="string content")
        assert request1.content == "string content"

        # List content
        request2 = GenerationRequest(content=["item1", "item2"])
        assert request2.content == ["item1", "item2"]

        # Dict content
        dict_content = {"role": "user", "content": "test"}
        request3 = GenerationRequest(content=dict_content)
        assert request3.content == dict_content


class TestGenerationResponse:
    """Test cases for GenerationResponse model."""

    @pytest.mark.smoke
    def test_generation_response_creation(self):
        """Test basic GenerationResponse creation.

        ### WRITTEN BY AI ###
        """
        request_id = "test-123"
        request_args = {"model": "gpt-3.5-turbo"}

        response = GenerationResponse(request_id=request_id, request_args=request_args)

        assert response.request_id == request_id
        assert response.request_args == request_args
        assert response.value is None
        assert response.delta is None
        assert response.iterations == 0
        assert response.request_prompt_tokens is None
        assert response.request_output_tokens is None
        assert response.response_prompt_tokens is None
        assert response.response_output_tokens is None

    @pytest.mark.smoke
    def test_generation_response_with_all_fields(self):
        """Test GenerationResponse creation with all fields.

        ### WRITTEN BY AI ###
        """
        response = GenerationResponse(
            request_id="test-123",
            request_args={"model": "gpt-4"},
            value="Generated text",
            delta="new text",
            iterations=5,
            request_prompt_tokens=50,
            request_output_tokens=100,
            response_prompt_tokens=55,
            response_output_tokens=95,
        )

        assert response.request_id == "test-123"
        assert response.request_args == {"model": "gpt-4"}
        assert response.value == "Generated text"
        assert response.delta == "new text"
        assert response.iterations == 5
        assert response.request_prompt_tokens == 50
        assert response.request_output_tokens == 100
        assert response.response_prompt_tokens == 55
        assert response.response_output_tokens == 95

    @pytest.mark.sanity
    def test_generation_response_prompt_tokens_property(self):
        """Test prompt_tokens property logic.

        ### WRITTEN BY AI ###
        """
        # When both are available, prefers response_prompt_tokens
        response1 = GenerationResponse(
            request_id="test",
            request_args={},
            request_prompt_tokens=50,
            response_prompt_tokens=55,
        )
        assert response1.prompt_tokens == 55

        # When only request_prompt_tokens is available
        response2 = GenerationResponse(
            request_id="test", request_args={}, request_prompt_tokens=50
        )
        assert response2.prompt_tokens == 50

        # When only response_prompt_tokens is available
        response3 = GenerationResponse(
            request_id="test", request_args={}, response_prompt_tokens=55
        )
        assert response3.prompt_tokens == 55

        # When neither is available
        response4 = GenerationResponse(request_id="test", request_args={})
        assert response4.prompt_tokens is None

    @pytest.mark.sanity
    def test_generation_response_output_tokens_property(self):
        """Test output_tokens property logic.

        ### WRITTEN BY AI ###
        """
        # When both are available, prefers response_output_tokens
        response1 = GenerationResponse(
            request_id="test",
            request_args={},
            request_output_tokens=100,
            response_output_tokens=95,
        )
        assert response1.output_tokens == 95

        # When only request_output_tokens is available
        response2 = GenerationResponse(
            request_id="test", request_args={}, request_output_tokens=100
        )
        assert response2.output_tokens == 100

        # When only response_output_tokens is available
        response3 = GenerationResponse(
            request_id="test", request_args={}, response_output_tokens=95
        )
        assert response3.output_tokens == 95

        # When neither is available
        response4 = GenerationResponse(request_id="test", request_args={})
        assert response4.output_tokens is None

    @pytest.mark.sanity
    def test_generation_response_total_tokens_property(self):
        """Test total_tokens property calculation.

        ### WRITTEN BY AI ###
        """
        # When both prompt and output tokens are available
        response1 = GenerationResponse(
            request_id="test",
            request_args={},
            response_prompt_tokens=50,
            response_output_tokens=100,
        )
        assert response1.total_tokens == 150

        # When one is missing
        response2 = GenerationResponse(
            request_id="test", request_args={}, response_prompt_tokens=50
        )
        assert response2.total_tokens is None

        # When both are missing
        response3 = GenerationResponse(request_id="test", request_args={})
        assert response3.total_tokens is None

    @pytest.mark.regression
    def test_generation_response_preferred_token_methods(self):
        """Test preferred_*_tokens methods.

        ### WRITTEN BY AI ###
        """
        response = GenerationResponse(
            request_id="test",
            request_args={},
            request_prompt_tokens=50,
            request_output_tokens=100,
            response_prompt_tokens=55,
            response_output_tokens=95,
        )

        # Test preferred_prompt_tokens
        assert response.preferred_prompt_tokens("request") == 50
        assert response.preferred_prompt_tokens("response") == 55

        # Test preferred_output_tokens
        assert response.preferred_output_tokens("request") == 100
        assert response.preferred_output_tokens("response") == 95

    @pytest.mark.regression
    def test_generation_response_preferred_tokens_fallback(self):
        """Test preferred_*_tokens methods with fallback logic.

        ### WRITTEN BY AI ###
        """
        # Only response tokens available
        response1 = GenerationResponse(
            request_id="test",
            request_args={},
            response_prompt_tokens=55,
            response_output_tokens=95,
        )

        assert response1.preferred_prompt_tokens("request") == 55  # Falls back
        assert response1.preferred_output_tokens("request") == 95  # Falls back

        # Only request tokens available
        response2 = GenerationResponse(
            request_id="test",
            request_args={},
            request_prompt_tokens=50,
            request_output_tokens=100,
        )

        assert response2.preferred_prompt_tokens("response") == 50  # Falls back
        assert response2.preferred_output_tokens("response") == 100  # Falls back


class TestGenerationRequestTimings:
    """Test cases for GenerationRequestTimings model."""

    @pytest.mark.smoke
    def test_generation_request_timings_creation(self):
        """Test basic GenerationRequestTimings creation.

        ### WRITTEN BY AI ###
        """
        timings = GenerationRequestTimings()

        assert timings.first_iteration is None
        assert timings.last_iteration is None

    @pytest.mark.smoke
    def test_generation_request_timings_with_fields(self):
        """Test GenerationRequestTimings creation with fields.

        ### WRITTEN BY AI ###
        """
        first_time = 1234567890.0
        last_time = 1234567895.0

        timings = GenerationRequestTimings(
            first_iteration=first_time, last_iteration=last_time
        )

        assert timings.first_iteration == first_time
        assert timings.last_iteration == last_time

    @pytest.mark.regression
    def test_generation_request_timings_fields_optional(self):
        """Test that all timing fields are optional.

        ### WRITTEN BY AI ###
        """
        # Should be able to create with no fields
        timings1 = GenerationRequestTimings()
        assert timings1.first_iteration is None
        assert timings1.last_iteration is None

        # Should be able to create with only one field
        timings2 = GenerationRequestTimings(first_iteration=123.0)
        assert timings2.first_iteration == 123.0
        assert timings2.last_iteration is None

        timings3 = GenerationRequestTimings(last_iteration=456.0)
        assert timings3.first_iteration is None
        assert timings3.last_iteration == 456.0
