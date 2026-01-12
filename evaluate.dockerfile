FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY src/ src/
COPY data/ data/

WORKDIR /
RUN uv sync --locked --no-cache

ENTRYPOINT ["uv", "run", "src/my_project/evaluate.py"]