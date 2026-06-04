"""
app/__init__.py
================
Flask application factory.
"""
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from flask import Flask, jsonify
from flask_cors import CORS

from config import get_config


def create_app(config_override=None) -> Flask:
    app = Flask(__name__)

    # Load config
    cfg = config_override or get_config()
    app.config.from_object(cfg)
    cfg.ensure_dirs()

    # CORS
    origins = app.config.get("CORS_ORIGINS", "*")
    CORS(app, origins=origins if origins != "*" else "*")

    # Logging
    _configure_logging(app)

    # Rate limiting
    if app.config.get("RATELIMIT_ENABLED", True):
        try:
            from flask_limiter import Limiter
            from flask_limiter.util import get_remote_address
            redis_url = app.config.get("REDIS_URL", "")
            storage_uri = redis_url if redis_url else "memory://"
            limiter = Limiter(
                get_remote_address,
                app=app,
                default_limits=[app.config.get("RATELIMIT_DEFAULT", "200/hour")],
                storage_uri=storage_uri,
            )
            app.extensions["limiter"] = limiter
        except ImportError:
            app.logger.warning("flask-limiter not installed — rate limiting disabled")

    # Register blueprints
    from app.api.routes import bp as api_bp
    app.register_blueprint(api_bp)

    # Global error handlers
    @app.errorhandler(400)
    def bad_request(e):
        return jsonify(error="Bad request", detail=str(e)), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify(error="Not found"), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify(error="Method not allowed"), 405

    @app.errorhandler(413)
    def too_large(e):
        mb = app.config.get("MAX_CONTENT_LENGTH", 0) // (1024 * 1024)
        return jsonify(error=f"File too large (max {mb} MB)"), 413

    @app.errorhandler(429)
    def too_many_requests(e):
        return jsonify(error="Rate limit exceeded. Please retry later."), 429

    @app.errorhandler(500)
    def internal_error(e):
        app.logger.error(f"Internal error: {e}", exc_info=True)
        return jsonify(error="Internal server error"), 500

    app.logger.info(f"PDF Translator API ready  [env={os.environ.get('FLASK_ENV','production')}]")
    return app


def _configure_logging(app: Flask):
    level = getattr(logging, app.config.get("LOG_LEVEL", "INFO"), logging.INFO)
    log_dir = app.config.get("LOG_DIR", Path("logs"))

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.setLevel(level)

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_dir / "app.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
        )
        fh.setFormatter(fmt)
        fh.setLevel(level)
        logging.getLogger().addHandler(fh)
    except Exception as e:
        app.logger.warning(f"Could not set up file logging: {e}")

    logging.getLogger().setLevel(level)
    logging.getLogger().addHandler(console)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
