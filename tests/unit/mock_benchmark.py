from guidellm.benchmark import (
    GenerativeBenchmark,
)

__all__ = ["mock_generative_benchmark"]


def mock_generative_benchmark() -> GenerativeBenchmark:
    """Create a minimal mock benchmark for testing."""
    # Return a minimal GenerativeBenchmark that can be created from a JSON file instead
    # This avoids the complex constructor issues

    # Create a minimal valid benchmark data structure
    mock_data = {
        "type_": "generative_benchmark",
        "run_id": "fa4a92c1-9a1d-4c83-b237-83fcc7971bd3",
        "run_index": 0,
        "scheduler": {"strategy": "synchronous", "max_duration": 10.0},
        "benchmarker": {"profile": "synchronous"},
        "env_args": {},
        "extras": {},
        "run_stats": {
            "start_time": 1744728125.0772898,
            "end_time": 1744728135.8407037,
            "requests_made": {
                "successful": 6,
                "errored": 0,
                "incomplete": 1,
                "total": 7,
            },
        },
        "start_time": 1744728125.0772898,
        "end_time": 1744728135.8407037,
        "metrics": {
            "request_latency": {
                "successful": {
                    "count": 6,
                    "min": 1.0,
                    "max": 2.0,
                    "mean": 1.5,
                    "std": 0.5,
                }
            },
        },
        "request_totals": {"successful": 6, "errored": 0, "incomplete": 1, "total": 7},
        "requests": [],
    }

    # Parse from dict to create a proper GenerativeBenchmark instance
    return GenerativeBenchmark.model_validate(mock_data)
