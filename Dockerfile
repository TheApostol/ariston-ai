FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY --from=builder /install /usr/local
COPY . .

# Non-root user for least-privilege execution
RUN useradd --no-create-home --shell /bin/false ariston \
    && mkdir -p /app/data /app/benchmarks \
    && chown -R ariston:ariston /app

USER ariston

# Expose internal app port and metrics port
EXPOSE 8000
EXPOSE 9090

# Gunicorn with uvicorn workers — see gunicorn.conf.py for tuning
CMD ["gunicorn", "app.main:app", "--config", "gunicorn.conf.py"]
