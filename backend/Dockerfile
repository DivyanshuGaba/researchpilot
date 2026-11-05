FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
COPY app app
RUN pip install --no-cache-dir -e .
COPY alembic.ini .
COPY migrations migrations
