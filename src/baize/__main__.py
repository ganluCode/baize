"""Entry point for ``python -m baize`` or ``uv run python -m baize``."""

import uvicorn

from baize.core.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "baize.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
