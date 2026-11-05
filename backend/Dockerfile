# syntax=docker/dockerfile:1
# --- build stage: resolve and compile dependencies into a venv ---------------------------------
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /build
# CPU-only torch first: the default wheel bundles CUDA libraries and adds several GB.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml .
COPY app app
RUN pip install .

# --- runtime stage: no compilers, no build cache, non-root --------------------------------------
FROM python:3.11-slim AS runtime
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models \
    UPLOAD_DIR=/data/uploads
RUN useradd --create-home --uid 10001 app \
 && mkdir -p /models /data/uploads \
 && chown -R app:app /models /data /home/app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY alembic.ini .
COPY migrations migrations
USER app
EXPOSE 8000
# The embedding and reranker models download on first start, so allow a long start period.
HEALTHCHECK --interval=15s --timeout=5s --start-period=240s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
