.PHONY: install sync dev test lint format check migrate migrate-gen export-openapi docker-up docker-down

# 加载 .env.local（优先）或 .env
ENV_FILE := $(shell if [ -f .env.local ]; then echo .env.local; elif [ -f .env ]; then echo .env; fi)
LOAD_ENV := $(if $(ENV_FILE),env $$(grep -v '^\#' $(ENV_FILE) | grep -v '^$$' | xargs))

install:
	uv sync

sync:
	uv sync --frozen

dev:
	$(LOAD_ENV) uv run python -m baize

test:
	$(LOAD_ENV) uv run pytest

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

check:
	uv run ruff check src tests
	uv run ruff format --check src tests

migrate:
	$(LOAD_ENV) uv run alembic upgrade head

migrate-gen:
	$(LOAD_ENV) uv run alembic revision --autogenerate -m "$(msg)"

export-openapi:
	mkdir -p docs
	$(LOAD_ENV) uv run python -c "import json; from baize.main import app; open('docs/baize.json', 'w').write(json.dumps(app.openapi(), indent=2, ensure_ascii=False))"
	@echo "Generated docs/baize.json"

docker-up:
	docker compose up -d

docker-down:
	docker compose down
