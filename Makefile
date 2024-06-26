install:
	pip install -r requirements.txt

install-dev:
	pip install -e .[dev]

quality:
	ruff check src tests
	isort --check-only src tests
	flake8 src tests --max-line-length 88
	mypy src

style:
	ruff format src tests
	isort src tests
	flake8 src tests --max-line-length 88

test:
	python -m pytest -s -vvv --cache-clear tests/

build:
	python setup.py sdist bdist_wheel

clean:
	rm -rf __pycache__
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -r {} +
	rm -rf .mypy_cache
	rm -rf .pytest_cache

.PHONY: install install-dev quality style test test-unit test-integration test-e2e test-smoke test-sanity test-regression build clean
