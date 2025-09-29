# BiasLens Makefile

.PHONY: help install test clean run dev

# Default target
help:
	@echo "BiasLens - Political Bias Analysis with RAG + ICL"
	@echo ""
	@echo "Available targets:"
	@echo "  install    - Install dependencies"
	@echo "  test       - Run unit tests"
	@echo "  test-verbose - Run tests with verbose output"
	@echo "  run        - Run the development server"
	@echo "  dev        - Run server in development mode"
	@echo "  clean      - Clean up temporary files"
	@echo "  help       - Show this help message"

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	pip install -r requirements-test.txt

# Run unit tests
test:
	@echo "Running unit tests..."
	pytest tests/ -v

# Run tests with verbose output
test-verbose:
	@echo "Running unit tests with verbose output..."
	pytest tests/ -v -s

# Run the development server
run:
	@echo "Starting BiasLens server..."
	python app/server.py

# Run server in development mode
dev:
	@echo "Starting BiasLens server in development mode..."
	uvicorn app.server:app --reload --host 0.0.0.0 --port 8000

# Clean up temporary files
clean:
	@echo "Cleaning up temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name "htmlcov" -delete
	find . -type f -name ".coverage" -delete
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
