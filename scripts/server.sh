#!/usr/bin/env bash
# scripts/server.sh — Production server management
# Usage: ./scripts/server.sh [start|stop|restart|status|logs|test]
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$APP_DIR/logs/gunicorn.pid"
LOG_FILE="$APP_DIR/logs/app.log"

cd "$APP_DIR"

# Load .env if present
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

PORT="${PORT:-5000}"
WORKERS="${GUNICORN_WORKERS:-4}"
THREADS="${GUNICORN_THREADS:-4}"

_green()  { echo -e "\033[32m$*\033[0m"; }
_red()    { echo -e "\033[31m$*\033[0m"; }
_yellow() { echo -e "\033[33m$*\033[0m"; }

cmd_start() {
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        _yellow "Already running (PID $(cat "$PID_FILE"))"
        return
    fi
    mkdir -p logs uploads outputs
    _green "Starting PDF Translator (workers=$WORKERS, threads=$THREADS, port=$PORT)..."
    exec gunicorn wsgi:application \
        -c deploy/gunicorn.conf.py \
        --pid "$PID_FILE" \
        --daemon \
        --access-logfile "$LOG_FILE" \
        --error-logfile  "$LOG_FILE"
    sleep 2
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        _green "Started  (PID $(cat "$PID_FILE"))"
    else
        _red "Failed to start — check $LOG_FILE"
        exit 1
    fi
}

cmd_stop() {
    if [[ -f "$PID_FILE" ]]; then
        _yellow "Stopping (PID $(cat "$PID_FILE"))..."
        kill -TERM "$(cat "$PID_FILE")" 2>/dev/null || true
        sleep 3
        rm -f "$PID_FILE"
        _green "Stopped."
    else
        _yellow "Not running."
    fi
}

cmd_status() {
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        _green "Running (PID $(cat "$PID_FILE"))"
        curl -sf "http://localhost:$PORT/health" | python3 -m json.tool || true
    else
        _red "Not running."
    fi
}

cmd_logs() {
    tail -f "$LOG_FILE"
}

cmd_test() {
    _green "Running test suite..."
    python3 -m pytest tests/ -v
}

cmd_docker_up() {
    _green "Starting full stack with Docker Compose..."
    docker compose up -d --build
    echo ""
    _green "Services:"
    docker compose ps
}

cmd_docker_down() {
    docker compose down
}

case "${1:-help}" in
    start)   cmd_start   ;;
    stop)    cmd_stop    ;;
    restart) cmd_stop; sleep 1; cmd_start ;;
    status)  cmd_status  ;;
    logs)    cmd_logs    ;;
    test)    cmd_test    ;;
    docker-up)   cmd_docker_up   ;;
    docker-down) cmd_docker_down ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|test|docker-up|docker-down}"
        exit 1
        ;;
esac
