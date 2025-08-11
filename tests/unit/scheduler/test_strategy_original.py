import inspect
import math
import time
from abc import ABC

import pytest
from pydantic import ValidationError

from guidellm.scheduler import (
    AsyncConstantStrategy,
    AsyncPoissonStrategy,
    ConcurrentStrategy,
    ConstantRateRequestTimings,
    LastCompletionRequestTimings,
    NoDelayRequestTimings,
    PoissonRateRequestTimings,
    ScheduledRequestInfo,
    ScheduledRequestTimings,
    SchedulingStrategy,
    SynchronousStrategy,
    ThroughputStrategy,
)
from guidellm.scheduler.strategy import (
    _exponential_decay_fraction,
    _exponential_decay_tau,
)


class TestExponentialDecayHelpers:
    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("max_progress", "convergence", "expected_range"),
        [
            (1.0, 0.99, (0.21, 0.22)),
            (5.0, 0.99, (1.08, 1.09)),
            (10.0, 0.95, (3.33, 3.35)),
        ],
    )
    def test_exponential_decay_tau(self, max_progress, convergence, expected_range):
        """Test exponential decay tau calculation."""
        tau = _exponential_decay_tau(max_progress, convergence)
        assert expected_range[0] <= tau <= expected_range[1]
        expected_tau = max_progress / (-math.log(1 - convergence))
        assert tau == pytest.approx(expected_tau, rel=1e-10)

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        ("progress", "tau", "expected_min", "expected_max"),
        [
            (0.0, 1.0, 0.0, 0.0),  # No progress = 0
            (1.0, 1.0, 0.6, 0.7),  # 1 tau ≈ 63.2%
            (2.0, 1.0, 0.85, 0.87),  # 2 tau ≈ 86.5%
            (3.0, 1.0, 0.95, 0.96),  # 3 tau ≈ 95.0%
        ],
    )
    def test_exponential_decay_fraction(
        self, progress, tau, expected_min, expected_max
    ):
        """Test exponential decay fraction calculation."""
        fraction = _exponential_decay_fraction(progress, tau)
        assert expected_min <= fraction <= expected_max
        expected_fraction = 1 - math.exp(-progress / tau)
        assert fraction == pytest.approx(expected_fraction, rel=1e-10)

    @pytest.mark.smoke
    def test_exponential_decay_fraction_boundary_conditions(self):
        """Test boundary conditions for exponential decay fraction."""
        assert _exponential_decay_fraction(0.0, 1.0) == 0.0
        assert _exponential_decay_fraction(0.0, 10.0) == 0.0
        large_progress = 100.0
        fraction = _exponential_decay_fraction(large_progress, 1.0)
        assert fraction > 0.99999


class TestStrategyStringRepresentation:
    # TODO: move these tests into the TestStrategy classes associated with each specific strategy that is tested

    @pytest.mark.smoke
    def test_synchronous_strategy_str(self):
        """Test __str__ method for SynchronousStrategy."""
        strategy = SynchronousStrategy()
        result = str(strategy)
        assert result == "synchronous"

    @pytest.mark.smoke
    def test_concurrent_strategy_str(self):
        """Test __str__ method for ConcurrentStrategy."""
        strategy = ConcurrentStrategy(streams=4)
        result = str(strategy)
        assert result == "concurrent@4"

        strategy = ConcurrentStrategy(streams=1)
        result = str(strategy)
        assert result == "concurrent@1"

    @pytest.mark.smoke
    def test_throughput_strategy_str(self):
        """Test __str__ method for ThroughputStrategy."""
        strategy = ThroughputStrategy()
        result = str(strategy)
        assert result == "throughput"

        strategy = ThroughputStrategy(max_concurrency=10)
        result = str(strategy)
        assert result == "throughput"

    @pytest.mark.smoke
    def test_async_constant_strategy_str(self):
        """Test __str__ method for AsyncConstantStrategy."""
        strategy = AsyncConstantStrategy(rate=10.5)
        result = str(strategy)
        assert result == "constant@10.50"

        strategy = AsyncConstantStrategy(rate=1.0)
        result = str(strategy)
        assert result == "constant@1.00"

    @pytest.mark.smoke
    def test_async_poisson_strategy_str(self):
        """Test __str__ method for AsyncPoissonStrategy."""
        strategy = AsyncPoissonStrategy(rate=5.25)
        result = str(strategy)
        assert result == "poisson@5.25"

        strategy = AsyncPoissonStrategy(rate=2.0)
        result = str(strategy)
        assert result == "poisson@2.00"


class TestScheduledRequestTimings:
    @pytest.mark.smoke
    def test_is_abstract_base_class(self):
        """Test that ScheduledRequestTimings is an abstract base class."""
        assert issubclass(ScheduledRequestTimings, ABC)
        assert inspect.isabstract(ScheduledRequestTimings)

    @pytest.mark.smoke
    def test_abstract_methods_defined(self):
        """Test that the required abstract methods are defined."""
        abstract_methods = ScheduledRequestTimings.__abstractmethods__
        expected_methods = {"next_offset", "request_completed"}
        assert abstract_methods == expected_methods

        # Validate method signatures
        next_offset_method = getattr(ScheduledRequestTimings, "next_offset")
        assert callable(next_offset_method)
        
        request_completed_method = getattr(ScheduledRequestTimings, "request_completed")
        assert callable(request_completed_method)
        
        # Check signature parameters using inspect
        next_offset_sig = inspect.signature(next_offset_method)
        assert len(next_offset_sig.parameters) == 1  # only self
        assert next_offset_sig.return_annotation == float
        
        request_completed_sig = inspect.signature(request_completed_method)
        assert len(request_completed_sig.parameters) == 2  # self and request_info
        params = list(request_completed_sig.parameters.values())
        assert params[1].annotation == ScheduledRequestInfo

    @pytest.mark.sanity
    def test_invalid_implementation(self):
        """Test that invalid implementations raise TypeError."""

        class InvalidImplementation(ScheduledRequestTimings):
            pass  # Missing required abstract methods

        with pytest.raises(TypeError):
            InvalidImplementation()

    @pytest.mark.smoke
    def test_child_implementation(self):
        """Test that concrete implementations can be constructed."""
        # Test with a proper concrete implementation in this test scope
        class TestRequestTimings(ScheduledRequestTimings):
            def __init__(self):
                super().__init__()
                self.offset = 0.0
                
            def next_offset(self) -> float:
                self.offset += 1.0
                return self.offset
                
            def request_completed(self, request_info: ScheduledRequestInfo):
                pass
        
        timing = TestRequestTimings()
        assert isinstance(timing, ScheduledRequestTimings)
        
        # Test that the methods work correctly
        assert timing.next_offset() == 1.0
        assert timing.next_offset() == 2.0
        
        # Test request_completed doesn't raise
        mock_request = ScheduledRequestInfo(
            request_id="test",
            status="completed",
            scheduler_node_id=0,
            scheduler_process_id=0,
            scheduler_start_time=time.time(),
        )
        timing.request_completed(mock_request)  # Should not raise


class TestLastCompletionRequestTimings:
    @pytest.fixture(
        params=[
            {},
            {"offset": 10.0},
            {"startup_requests": 5, "startup_requests_delay": 0.5},
            {
                "offset": 0.0,
                "startup_requests": 0,
                "startup_requests_delay": 0.0,
            },
            {
                "offset": 2.5,
                "startup_requests": 3,
                "startup_requests_delay": 1.0,
            },
        ]
    )
    def valid_instances(self, request):
        """Creates various valid configurations of LastCompletionRequestTimings."""
        constructor_args = request.param
        instance = LastCompletionRequestTimings(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_initialization(
        self, valid_instances: tuple[LastCompletionRequestTimings, dict]
    ):
        """Test initialization with valid configurations."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, LastCompletionRequestTimings)

        for key, value in constructor_args.items():
            assert getattr(instance, key) == value

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("startup_requests", -1),
            ("startup_requests_delay", -0.5),
            ("offset", "invalid"),
            ("startup_requests", 1.5),
        ],
    )
    def test_invalid_initialization(self, field, value):
        """Test invalid initialization scenarios."""
        kwargs = {field: value}
        with pytest.raises(ValidationError):
            LastCompletionRequestTimings(**kwargs)

    @pytest.mark.smoke
    def test_lifecycle(
        self, valid_instances: tuple[LastCompletionRequestTimings, dict]
    ):
        """Test the complete lifecycle of next_offset and request_completed calls."""
        instance, constructor_args = valid_instances
        initial_offset = instance.offset
        startup_requests = constructor_args.get("startup_requests", 0)
        startup_delay = constructor_args.get("startup_requests_delay", 0.0)
        request_times = []

        for index in range(max(5, startup_requests + 2)):
            offset = instance.next_offset()
            assert isinstance(offset, (int, float))

            if index < startup_requests:
                expected_offset = initial_offset + (index + 1) * startup_delay
                assert offset == pytest.approx(expected_offset, abs=1e-5)

            completion_time = time.time() + offset
            request_times.append(completion_time)

            mock_request = ScheduledRequestInfo(
                request_id=f"test-{index}",
                status="completed",
                scheduler_node_id=0,
                scheduler_process_id=0,
                scheduler_start_time=time.time(),
            )
            mock_request.scheduler_timings.resolve_end = completion_time
            instance.request_completed(mock_request)

    @pytest.mark.smoke
    def test_marshalling(
        self, valid_instances: tuple[LastCompletionRequestTimings, dict]
    ):
        """Test marshalling to/from pydantic dict formats."""
        instance, constructor_args = valid_instances

        data = instance.model_dump()
        assert isinstance(data, dict)

        for key, value in constructor_args.items():
            assert data[key] == value

        reconstructed = LastCompletionRequestTimings.model_validate(data)
        assert isinstance(reconstructed, LastCompletionRequestTimings)

        for key, value in constructor_args.items():
            assert getattr(reconstructed, key) == value


class TestNoDelayRequestTimings:
    @pytest.fixture(
        params=[
            {},
            {"offset": 5.0},
            {"startup_duration": 1.0},
            {"startup_target_requests": 5},
            {"startup_convergence": 0.95},
            {
                "offset": 1.0,
                "startup_duration": 0.5,
                "startup_target_requests": 3,
                "startup_convergence": 0.9,
            },
        ]
    )
    def valid_instances(self, request):
        """Creates various valid configurations of NoDelayRequestTimings."""
        constructor_args = request.param
        instance = NoDelayRequestTimings(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_initialization(self, valid_instances: tuple[NoDelayRequestTimings, dict]):
        """Test initialization with valid configurations."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, NoDelayRequestTimings)

        for key, value in constructor_args.items():
            assert getattr(instance, key) == value

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("offset", -1.0),
            ("startup_duration", -1.0),
            ("startup_target_requests", 0),
            ("startup_target_requests", -1),
        ],
    )
    def test_invalid_initialization(self, field, value):
        """Test invalid initialization scenarios."""
        kwargs = {field: value}
        with pytest.raises(ValidationError):
            NoDelayRequestTimings(**kwargs)

    @pytest.mark.smoke
    def test_lifecycle(self, valid_instances: tuple[NoDelayRequestTimings, dict]):
        """Test the complete lifecycle of timing methods."""
        instance, constructor_args = valid_instances
        startup_duration = constructor_args.get("startup_duration", 0.0)
        startup_target_requests = constructor_args.get("startup_target_requests", 1.0)
        base_offset = constructor_args.get("offset", 0.0)

        offsets = []
        for i in range(10):
            offset = instance.next_offset()
            assert isinstance(offset, (int, float))
            assert offset >= base_offset  # Should never be less than base offset
            offsets.append(offset)

            mock_request = ScheduledRequestInfo(
                request_id=f"test-{i}",
                status="completed",
                scheduler_node_id=0,
                scheduler_process_id=0,
                scheduler_start_time=time.time(),
            )
            instance.request_completed(mock_request)

        # If startup_duration > 0, check that we converge to the base offset + startup_duration
        if startup_duration > 0:
            # Initial offsets should ramp up during startup
            early_offsets = offsets[:3]
            later_offsets = offsets[-3:]
            
            # Later offsets should be closer to the max (base_offset + startup_duration)
            max_offset = base_offset + startup_duration
            
            # Check that we're moving toward the maximum offset
            for later_offset in later_offsets:
                assert later_offset <= max_offset + 0.1  # Allow small tolerance
                
            # Check that later offsets are generally larger than early ones
            assert max(later_offsets) >= max(early_offsets)
        else:
            # With no startup duration, all offsets should be the base offset
            for offset in offsets:
                assert offset == pytest.approx(base_offset, abs=1e-5)

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances: tuple[NoDelayRequestTimings, dict]):
        """Test marshalling to/from pydantic dict formats."""
        instance, constructor_args = valid_instances

        data = instance.model_dump()
        assert isinstance(data, dict)

        for key, value in constructor_args.items():
            assert data[key] == value

        reconstructed = NoDelayRequestTimings.model_validate(data)
        assert isinstance(reconstructed, NoDelayRequestTimings)

        for key, value in constructor_args.items():
            assert getattr(reconstructed, key) == value


class TestConstantRateRequestTimings:
    @pytest.fixture(
        params=[
            {"rate": 1.0},
            {"rate": 5.0, "offset": 2.0},
            {"rate": 0.5, "startup_duration": 10.0},
            {"rate": 10.0, "startup_convergence": 0.95},
        ]
    )
    def valid_instances(self, request):
        """Creates various valid configurations of ConstantRateRequestTimings."""
        constructor_args = request.param
        instance = ConstantRateRequestTimings(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_initialization(
        self, valid_instances: tuple[ConstantRateRequestTimings, dict]
    ):
        """Test initialization with valid configurations."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, ConstantRateRequestTimings)

        for key, value in constructor_args.items():
            assert getattr(instance, key) == value

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("rate", 0),
            ("rate", -1.0),
            ("offset", -1.0),
            ("startup_duration", -1.0),
        ],
    )
    def test_invalid_initialization(self, field, value):
        """Test invalid initialization scenarios."""
        kwargs = {"rate": 1.0}
        kwargs[field] = value
        with pytest.raises(ValidationError):
            ConstantRateRequestTimings(**kwargs)

    @pytest.mark.smoke
    def test_constant_rate_behavior(
        self, valid_instances: tuple[ConstantRateRequestTimings, dict]
    ):
        """Test that requests are scheduled at constant intervals."""
        instance, constructor_args = valid_instances
        rate = constructor_args["rate"]
        expected_interval = 1.0 / rate
        startup_duration = constructor_args.get("startup_duration", 0.0)

        offsets = []
        for i in range(10):
            offset = instance.next_offset()
            offsets.append(offset)

            mock_request = ScheduledRequestInfo(
                request_id=f"test-{i}",
                status="completed",
                scheduler_node_id=0,
                scheduler_process_id=0,
                scheduler_start_time=time.time(),
            )
            instance.request_completed(mock_request)

        # Check that offsets are increasing
        for i in range(1, len(offsets)):
            assert offsets[i] >= offsets[i - 1]

        if startup_duration == 0:
            # Without startup, intervals should be exactly constant
            intervals = [offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)]
            if len(intervals) > 0:
                for interval in intervals:
                    assert abs(interval - expected_interval) < 0.01
        else:
            # With startup duration, we should converge to the expected interval
            # Later intervals should be closer to the expected interval
            if len(offsets) > 5:
                later_intervals = [
                    offsets[i + 1] - offsets[i] for i in range(len(offsets) - 3, len(offsets) - 1)
                ]
                # At least one later interval should be close to the expected interval
                assert any(abs(interval - expected_interval) < 0.1 for interval in later_intervals)

    @pytest.mark.smoke
    def test_marshalling(
        self, valid_instances: tuple[ConstantRateRequestTimings, dict]
    ):
        """Test marshalling to/from pydantic dict formats."""
        instance, constructor_args = valid_instances

        data = instance.model_dump()
        assert isinstance(data, dict)

        for key, value in constructor_args.items():
            assert data[key] == value

        reconstructed = ConstantRateRequestTimings.model_validate(data)
        assert isinstance(reconstructed, ConstantRateRequestTimings)

        for key, value in constructor_args.items():
            assert getattr(reconstructed, key) == value


class TestPoissonRateRequestTimings:
    @pytest.fixture(
        params=[
            {"rate": 1.0},
            {
                "rate": 5.0,
                "random_seed": 123,
                "offset": 1.0,
            },
            {
                "rate": 0.5,
                "startup_duration": 5.0,
            },
            {
                "rate": 2.0,
                "random_seed": 456,
                "offset": 2.0,
                "startup_duration": 7.5,
            },
        ]
    )
    def valid_instances(self, request):
        """Creates various valid configurations of PoissonRateRequestTimings."""
        constructor_args = request.param
        instance = PoissonRateRequestTimings(**constructor_args)
        return instance, constructor_args

    @pytest.mark.smoke
    def test_initialization(
        self, valid_instances: tuple[PoissonRateRequestTimings, dict]
    ):
        """Test initialization with valid configurations."""
        instance, constructor_args = valid_instances
        assert isinstance(instance, PoissonRateRequestTimings)

        for key, value in constructor_args.items():
            assert getattr(instance, key) == value

    @pytest.mark.smoke
    def test_lifecycle(self, valid_instances: tuple[PoissonRateRequestTimings, dict]):
        """Test that Poisson timing produces variable intervals."""
        instance, _ = valid_instances
        offsets = []
        instance._requests_count = 100

        # TODO: implement handling of test cases to check that the poisson distribution ramps up to the desired rate over a given startup period, if available
        # TODO: implement statistical checks to ensure the poisson distribution is within a reasonable confidence interval of the target

        for i in range(10):
            offset = instance.next_offset()
            offsets.append(offset)

            mock_request = ScheduledRequestInfo(
                request_id=f"test-{i}",
                status="completed",
                scheduler_node_id=0,
                scheduler_process_id=0,
                scheduler_start_time=time.time(),
            )
            instance.request_completed(mock_request)

        # Check that offsets are increasing (cumulative)
        for i in range(1, len(offsets)):
            assert offsets[i] >= offsets[i - 1]

        # Check that intervals vary (not constant)
        intervals = [offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)]
        if len(intervals) > 3:
            assert max(intervals) > min(intervals)

    @pytest.mark.smoke
    def test_marshalling(self, valid_instances: tuple[PoissonRateRequestTimings, dict]):
        """Test marshalling to/from pydantic dict formats."""
        instance, constructor_args = valid_instances

        data = instance.model_dump()
        assert isinstance(data, dict)

        for key, value in constructor_args.items():
            assert data[key] == value

        reconstructed = PoissonRateRequestTimings.model_validate(data)
        assert isinstance(reconstructed, PoissonRateRequestTimings)

        for key, value in constructor_args.items():
            assert getattr(reconstructed, key) == value


class TestSchedulingStrategy:
    @pytest.mark.smoke
    def test_base(self):
        """Test that base methods are defined in SchedulingStrategy."""
        # TODO: test that it extends from the desired class and implements what's needed for that
        expected_methods = {
            "processes_limit",
            "requests_limit",
            "create_request_timings",
        }

        strategy_methods = set(dir(SchedulingStrategy))
        for method in expected_methods:
            assert method in strategy_methods

        # TODO: use inspect to validate the expected properties and function signatures it is supposed to have

    @pytest.mark.sanity
    def test_invalid_implementation(self):
        """Test that invalid implementations raise NotImplementedError."""

        class InvalidStrategy(SchedulingStrategy):
            pass

        strategy = InvalidStrategy(type_="strategy")
        with pytest.raises(NotImplementedError):
            strategy.create_request_timings(0, 1, 1)

    @pytest.mark.smoke
    def test_concrete_implementation(self):
        """Test that concrete implementations can be constructed."""
        # TODO: create test class in this local scope to test rather than relying on another class and validate that it can be constructed and function
        sync_strategy = SynchronousStrategy()
        assert isinstance(sync_strategy, SchedulingStrategy)


class TestSynchronousStrategy:
    @pytest.mark.smoke
    def test_initialization(self):
        """Test initialization of SynchronousStrategy."""
        strategy = SynchronousStrategy()
        assert strategy.type_ == "synchronous"

    @pytest.mark.smoke
    def test_limits(self):
        """Test that SynchronousStrategy enforces proper limits."""
        strategy = SynchronousStrategy()
        assert strategy.processes_limit == 1
        assert strategy.requests_limit == 1

    @pytest.mark.smoke
    def test_create_timings_valid(self):
        """Test creating timings with valid parameters."""
        # TODO: ensure that it only works with the proper arguments, e.g. rank=0, world_size=1
        strategy = SynchronousStrategy()
        timing = strategy.create_request_timings(0, 1, 1)
        assert isinstance(timing, LastCompletionRequestTimings)

    @pytest.mark.sanity
    def test_create_timings_invalid(self):
        """Test that invalid parameters raise ValueError."""
        strategy = SynchronousStrategy()

        with pytest.raises(ValueError):
            strategy.create_request_timings(1, 1, 1)  # rank != 0

        with pytest.raises(ValueError):
            strategy.create_request_timings(0, 2, 1)  # world_size > 1


class TestConcurrentStrategy:
    # TODO: add in parameterized fixture to test all expected args and combinations like timings tests

    @pytest.mark.smoke
    def test_initialization(self):
        """Test initialization of ConcurrentStrategy."""
        # TODO: parameterize this with the fixture to run checks across all desired strategy instances
        strategy = ConcurrentStrategy(streams=3)
        assert strategy.type_ == "concurrent"
        assert strategy.streams == 3

    @pytest.mark.sanity
    def test_invalid_initialization(self):
        """Test invalid initialization."""
        # TODO: add parameterized tests for invalid initialization
        with pytest.raises(ValidationError):
            ConcurrentStrategy(streams=0)

        with pytest.raises(ValidationError):
            ConcurrentStrategy(streams=-1)

    @pytest.mark.smoke
    def test_limits(self):
        """Test that ConcurrentStrategy returns correct limits."""
        # TODO: parameterize this with the fixture to run checks across all desired strategy instances
        strategy = ConcurrentStrategy(streams=5)
        assert strategy.processes_limit == 5
        assert strategy.requests_limit == 5

    @pytest.mark.smoke
    def test_create_timings(self):
        """Test creating timings."""
        # TODO: parameterize this with the fixture along with parameterizations for the rank and world_size to check that it creates the correct timings specifically for startup_duration as defined in help docs in the code
        strategy = ConcurrentStrategy(streams=2)
        timing = strategy.create_request_timings(0, 1, 2)
        assert isinstance(timing, LastCompletionRequestTimings)

    @pytest.mark.sanity
    def test_create_timings_invalid(self):
        # TODO: generalize this so it is parameterized to check various invalid inputs for create request timings
        pass


class TestThroughputStrategy:
    # TODO: add in parameterized fixture to test all expected args and combinations like timings tests

    @pytest.mark.smoke
    def test_initialization(self):
        """Test initialization of ThroughputStrategy."""
        # TODO: add parameterized tests for initialization with the fixture
        strategy = ThroughputStrategy()
        assert strategy.type_ == "throughput"
        assert strategy.max_concurrency is None

    @pytest.mark.smoke
    def test_invalid_initialization(self):
        # TODO: add parameterized tests for invalid initialization
        with pytest.raises(ValidationError):
            ThroughputStrategy(max_concurrency=0)


    @pytest.mark.smoke
    def test_create_timings(self):
        """Test creating timings."""
        # TODO: parameterize with the fixture and args for rank and world_size to test the expected outputs of the timing instances for no startup duration and with startup duration
        strategy = ThroughputStrategy()
        timing = strategy.create_request_timings(0, 1, 1)
        assert isinstance(timing, NoDelayRequestTimings)


class TestAsyncConstantStrategy:
    # TODO: add parameterized tests for initialization with the fixture

    @pytest.mark.smoke
    def test_initialization(self):
        """Test initialization of AsyncConstantStrategy."""
        # TODO: add parameterized tests for initialization with the fixture
        strategy = AsyncConstantStrategy(rate=2.0)
        assert strategy.type_ == "constant"
        assert strategy.rate == 2.0

    @pytest.mark.sanity
    def test_invalid_initialization(self):
        """Test invalid initialization."""
        # TODO: change to parameterized tests for the invalid configurations
        with pytest.raises(ValidationError):
            AsyncConstantStrategy(rate=0)  # Must be > 0

        with pytest.raises(ValidationError):
            AsyncConstantStrategy(rate=-1.0)  # Must be > 0

    @pytest.mark.smoke
    def test_create_timings(self):
        """Test creating timings."""
        # TODO: change to parameterized tests with the fixture and args for rank and world_size to ensure desired logic for timings that are returned
        strategy = AsyncConstantStrategy(rate=1.0)
        timing = strategy.create_request_timings(0, 2, 1)
        assert isinstance(timing, ConstantRateRequestTimings)
        assert timing.rate == 0.5  # Distributed across 2 workers


class TestAsyncPoissonStrategy:
    # TODO: add parameterized tests for initialization with the fixture

    @pytest.mark.smoke
    def test_initialization(self):
        """Test initialization of AsyncPoissonStrategy."""
        # TODO: change to parameterized tests with the fixture
        strategy = AsyncPoissonStrategy(rate=3.0, random_seed=123)
        assert strategy.type_ == "poisson"
        assert strategy.rate == 3.0
        assert strategy.random_seed == 123

    @pytest.mark.smoke
    def test_invalid_initialization(self):
        # TODO: add parameterized tests for invalid initialization
        with pytest.raises(ValidationError):
            AsyncPoissonStrategy(rate=0)  # Must be > 0

    @pytest.mark.smoke
    def test_create_timings(self):
        """Test creating timings."""
        # TODO: parameterize with the fixture and args for rank and world_size to test the expected outputs of the timing instances for no startup duration and with startup duration
        strategy = AsyncPoissonStrategy(rate=2.0)
        timing = strategy.create_request_timings(0, 2, 1)
        assert isinstance(timing, PoissonRateRequestTimings)
        assert timing.rate == 1.0  # Distributed across 2 workers
