# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md requirements.txt ./
COPY attention_algebra ./attention_algebra
COPY app.py ./
COPY assets ./assets
RUN pip install --no-cache-dir -e ".[app]"

FROM base AS test
COPY tests ./tests
RUN pip install --no-cache-dir -e ".[dev]"
CMD ["pytest"]

FROM base AS runtime
EXPOSE 7860
ENV HOST=0.0.0.0 PORT=7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/')"
CMD ["python", "app.py"]
