"""
wsgi.py — Production WSGI entrypoint for Gunicorn / uWSGI.

Run:
    gunicorn wsgi:application -c deploy/gunicorn.conf.py
"""
from app import create_app

application = create_app()

# Alias for compatibility
app = application
