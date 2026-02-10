FROM python:3.12-slim AS base

WORKDIR /app

# Install dependencies first for layer caching
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir .

# Copy source
COPY src/ src/

EXPOSE 8084

CMD ["uvicorn", "aratta.server:app", "--host", "0.0.0.0", "--port", "8084"]
