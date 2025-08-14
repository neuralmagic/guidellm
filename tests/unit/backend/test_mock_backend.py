"""
Unit tests for the MockBackend implementation.

### WRITTEN BY AI ###
"""

import pytest

from guidellm.backend import Backend
from guidellm.backend.objects import GenerationRequest, GenerationRequestTimings
from guidellm.scheduler import ScheduledRequestInfo
from tests.unit.mock_backend import MockBackend


class TestMockBackend:
    """Test cases for MockBackend."""

    @pytest.mark.smoke
    def test_mock_backend_creation(self):
        """Test MockBackend can be created.

        ### WRITTEN BY AI ###
        """
        backend = MockBackend()
        assert backend.type_ == "mock"
        assert backend.model == "mock-model"
        assert backend.target == "mock-target"

    @pytest.mark.smoke
    def test_mock_backend_registration(self):
        """Test MockBackend is properly registered.

        ### WRITTEN BY AI ###
        """
        backend = Backend.create("mock")
        assert isinstance(backend, MockBackend)
        assert backend.type_ == "mock"

    @pytest.mark.smoke
    def test_mock_backend_info(self):
        """Test MockBackend info method.

        ### WRITTEN BY AI ###
        """
        backend = MockBackend(model="test-model", target="test-target")
        info = backend.info()

        assert info["type"] == "mock"
        assert info["model"] == "test-model"
        assert info["target"] == "test-target"

    @pytest.mark.sanity
    @pytest.mark.asyncio
    async def test_mock_backend_lifecycle(self):
        """Test MockBackend process lifecycle.

        ### WRITTEN BY AI ###
        """
        backend = MockBackend()

        # Test startup
        await backend.process_startup()
        assert backend._in_process is True

        # Test validation
        await backend.validate()  # Should not raise

        # Test default model
        model = await backend.default_model()
        assert model == "mock-model"

        # Test shutdown
        await backend.process_shutdown()
        assert backend._in_process is False

    @pytest.mark.sanity
    @pytest.mark.asyncio
    async def test_mock_backend_validate_not_started(self):
        """Test validation fails when backend not started.

        ### WRITTEN BY AI ###
        """
        backend = MockBackend()

        with pytest.raises(RuntimeError, match="Backend not started up"):
            await backend.validate()

    @pytest.mark.regression
    @pytest.mark.asyncio
    async def test_mock_backend_resolve(self):
        """Test MockBackend resolve method.

        ### WRITTEN BY AI ###
        """
        backend = MockBackend(iter_delay=0.001)  # Small delay for testing
        await backend.process_startup()

        try:
            request = GenerationRequest(
                request_id="test-id",
                content="Test prompt",
                constraints={"output_tokens": 3},
            )
            request_info = ScheduledRequestInfo(
                request_id="test-id",
                status="pending",
                scheduler_node_id=1,
                scheduler_process_id=1,
                scheduler_start_time=123.0,
                request_timings=GenerationRequestTimings(),
            )

            responses = []
            async for response, info in backend.resolve(request, request_info):
                responses.append((response, info))

            # Should get multiple responses (one per token + final)
            assert len(responses) >= 2

            # Check final response
            final_response = responses[-1][0]
            assert final_response.request_id == "test-id"
            assert final_response.iterations > 0
            assert len(final_response.value) > 0
            assert final_response.delta is None  # Final response has no delta

            # Check timing information
            final_info = responses[-1][1]
            assert final_info.request_timings.request_start is not None
            assert final_info.request_timings.request_end is not None
            assert final_info.request_timings.first_iteration is not None
            assert final_info.request_timings.last_iteration is not None

        finally:
            await backend.process_shutdown()

    @pytest.mark.regression
    @pytest.mark.asyncio
    async def test_mock_backend_resolve_not_started(self):
        """Test resolve fails when backend not started.

        ### WRITTEN BY AI ###
        """
        backend = MockBackend()

        request = GenerationRequest(content="test")
        request_info = ScheduledRequestInfo(
            request_id="test",
            status="pending",
            scheduler_node_id=1,
            scheduler_process_id=1,
            scheduler_start_time=123.0,
            request_timings=GenerationRequestTimings(),
        )

        with pytest.raises(RuntimeError, match="Backend not started up"):
            async for _ in backend.resolve(request, request_info):
                pass

    @pytest.mark.regression
    @pytest.mark.asyncio
    async def test_mock_backend_resolve_with_history(self):
        """Test resolve method raises error with conversation history.

        ### WRITTEN BY AI ###
        """
        backend = MockBackend()
        await backend.process_startup()

        try:
            request = GenerationRequest(content="test")
            request_info = ScheduledRequestInfo(
                request_id="test",
                status="pending",
                scheduler_node_id=1,
                scheduler_process_id=1,
                scheduler_start_time=123.0,
                request_timings=GenerationRequestTimings(),
            )
            history = [(request, None)]  # Mock history

            with pytest.raises(
                NotImplementedError, match="Multi-turn requests not supported"
            ):
                async for _ in backend.resolve(request, request_info, history):
                    pass
        finally:
            await backend.process_shutdown()

    @pytest.mark.regression
    def test_mock_backend_token_generation(self):
        """Test token generation methods.

        ### WRITTEN BY AI ###
        """
        # Test with specific token count
        tokens = MockBackend._get_tokens(5)
        assert len(tokens) == 5
        assert tokens[-1] == "."  # Should end with period

        # Test with None (random count)
        tokens_random = MockBackend._get_tokens(None)
        assert len(tokens_random) >= 8
        assert len(tokens_random) <= 512

        # Test prompt token estimation
        estimated = MockBackend._estimate_prompt_tokens("hello world test")
        assert estimated == 3  # Three words
