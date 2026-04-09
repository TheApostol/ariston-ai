"""
Gunicorn configuration for production deployment.

Run with: gunicorn app.main:app --config gunicorn.conf.py
"""

import multiprocessing

# ── Server socket ─────────────────────────────────────────────────────────────
bind = "0.0.0.0:8000"

# ── Worker processes ──────────────────────────────────────────────────────────
# uvicorn workers enable async FastAPI handlers
worker_class = "uvicorn.workers.UvicornWorker"
# (2 * CPU cores) + 1 is the standard recommendation
workers = multiprocessing.cpu_count() * 2 + 1

# ── Timeouts ──────────────────────────────────────────────────────────────────
timeout = 120          # seconds — LLM calls can be slow
graceful_timeout = 30  # seconds — drain in-flight requests on shutdown
keepalive = 5          # seconds — keep-alive for downstream proxies

# ── Logging ───────────────────────────────────────────────────────────────────
accesslog = "-"   # stdout
errorlog = "-"    # stdout
loglevel = "info"

# ── Process naming ────────────────────────────────────────────────────────────
proc_name = "ariston-ai"
