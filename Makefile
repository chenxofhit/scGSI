.PHONY: help install clean test format lint

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make clean      - Remove temporary files and caches"
	@echo "  make test       - Run tests (if available)"
	@echo "  make format     - Format code with black"
	@echo "  make lint       - Lint code with flake8"

install:
	pip install -r requirements.txt
	pip install -e .

clean:
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -r {} + 2>/dev/null || true
	@echo "Cleaned Python cache files"

test:
	@echo "Tests not yet implemented"

format:
	black scGSI/ eva/ vis/ data/

lint:
	flake8 scGSI/ eva/ vis/ data/

