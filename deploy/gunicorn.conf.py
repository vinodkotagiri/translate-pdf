"""
deploy/gunicorn.conf.py
========================
Gunicorn production configuration.

Start:
    gunicorn wsgi:application -c deploy/gunicorn.conf.py
"""
import multiprocessing
import os

# ── Binding ───────────────────────────────────────────────────────────────────
bind    = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
backlog = 2048

# ── Workers ───────────────────────────────────────────────────────────────────
# For CPU-bound PDF work: use fewer workers with threading
# formula: (2 × CPU cores) + 1  but cap at 8 for memory
workers     = int(os.environ.get("GUNICORN_WORKERS", min((2 * multiprocessing.cpu_count()) + 1, 8)))
worker_class = "gthread"          # threaded workers (best for I/O + CPU mix)
threads      = int(os.environ.get("GUNICORN_THREADS", 4))
worker_connections = 1000

# ── Timeouts ──────────────────────────────────────────────────────────────────
timeout      = int(os.environ.get("GUNICORN_TIMEOUT", 600))   # 10 min (large PDFs)
keepalive    = 5
graceful_timeout = 60

# ── Logging ───────────────────────────────────────────────────────────────────
loglevel      = os.environ.get("LOG_LEVEL", "info").lower()
accesslog     = os.environ.get("ACCESS_LOG", "-")     # stdout
errorlog      = os.environ.get("ERROR_LOG",  "-")     # stdout
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)sµs'

# ── Process naming ────────────────────────────────────────────────────────────
proc_name = "pdf-translator"

# ── Server mechanics ──────────────────────────────────────────────────────────
preload_app   = True    # load app once before forking (saves memory)
max_requests  = 1000    # restart worker after N requests (prevents memory leaks)
max_requests_jitter = 50

# ── Security ──────────────────────────────────────────────────────────────────
limit_request_line        = 8190
limit_request_fields      = 200
limit_request_field_size  = 8190
forwarded_allow_ips       = os.environ.get("FORWARDED_ALLOW_IPS", "127.0.0.1")

# ── Hooks ─────────────────────────────────────────────────────────────────────
def on_starting(server):
    server.log.info("PDF Translator — Gunicorn starting")

def worker_exit(server, worker):
    server.log.info(f"Worker {worker.pid} exited")
