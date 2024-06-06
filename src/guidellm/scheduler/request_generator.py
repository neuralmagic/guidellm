from typing import Iterator


class RequestGenerator:
    def __init__(
        self, data_generator: DataGenerator, backend: Backend, queue_size: int
    ):
        self.data_generator = data_generator
        self.backend = backend
        self.queue_size = queue_size

    def generate_requests(self) -> Iterator[BenchmarkRequest]:
        for _ in range(self.queue_size):
            data = next(self.data_generator)
            yield BenchmarkRequest(prompt=data)
