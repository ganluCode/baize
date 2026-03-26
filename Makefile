.PHONY: install sync dev test lint format check migrate migrate-gen export-openapi docker-up docker-down

install:
	uv sync

sync:
	uv sync --frozen

dev:
	uv run uvicorn baize.main:app --reload --host 0.0.0.0 --port 8001

test:
	uv run pytest

lint:
	ruff check src tests

format:
	ruff format src tests

check:
	ruff check src tests
	ruff format --check src tests

migrate:
	uv run alembic upgrade head

migrate-gen:
	uv run alembic revision --autogenerate -m "$(msg)"

export-openapi:
	mkdir -p docs
	DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/db \
	REDIS_URL=redis://localhost:6379/0 \
	ADMIN_API_KEY=dummy \
	SECRET_KEY=dummy \
	uv run python -c "import json; from baize.main import app; open('docs/openapi.json', 'w').write(json.dumps(app.openapi(), indent=2))"

docker-up:
	docker compose -f docker/docker-compose.yml up -d

docker-down:
	docker compose -f docker/docker-compose.yml down
