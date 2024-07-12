.PHONY: install
install:
	python -m pip install .


.PHONY: install.dev
install.dev:
	python -m pip install .[dev]


.PHONY: build
build:
	python setup.py sdist bdist_wheel

.PHONY: style
style:
	python -m ruff format src tests
	python -m isort src tests
	python -m flake8 src tests --max-line-length 88


.PHONY: types
types:
	python -m mypy --check-untyped-defs


.PHONY: quality
quality:
	python -m ruff check src tests
	python -m black --check src tests
	python -m isort --check src tests
	python -m mypy --check-untyped-defs



.PHONY: test
test:
	python -m pytest -s -vvv --cache-clear tests


.PHONY: test.unit
test.unit:
	python -m pytest tests/unit


.PHONY: test.integration
test.integration:
	python -m pytest tests/integration


.PHONY: test.e2e
test.e2e:
	python -m pytest tests/e2e



.PHONY: clean
clean:
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -r {} +
	rm -rf .mypy_cache
	rm -rf .pytest_cache
