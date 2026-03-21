.PHONY: install sync dev test lint format check migrate migrate-gen export-openapi docker-up docker-down

install:
	uv sync

sync:
	uv sync --frozen

dev:
	uvicorn baize.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest

lint:
	ruff check src tests

format:
	ruff format src tests

check:
	ruff check src tests
	ruff format --check src tests

migrate:
	alembic upgrade head

migrate-gen:
	alembic revision --autogenerate -m "$(msg)"

export-openapi:
	python -c "import json; from baize.main import app; print(json.dumps(app.openapi(), indent=2))" > openapi.json

docker-up:
	docker compose -f docker/docker-compose.yml up -d

docker-down:
	docker compose -f docker/docker-compose.yml down
