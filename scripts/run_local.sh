#!/usr/bin/env bash
# Start / stop / status of Podcast Studio on this machine (background, logs to data/server.log).
#   scripts/run_local.sh start [PORT]   # default port 8000 (or next free one)
#   scripts/run_local.sh stop | status | restart
# Keys are read from .env. With DUMMY_ keys the app runs in demo mode.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/dev_env.sh
mkdir -p data
PIDFILE=data/server.pid
LOG=data/server.log

running() { [[ -f $PIDFILE ]] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; }

case "${1:-start}" in
  start)
    if running; then echo "Already running (pid $(cat $PIDFILE)) on port $(cat data/server.port)"; exit 0; fi
    PORT="${2:-8000}"
    while (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; do PORT=$((PORT+1)); done
    nohup .venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port "$PORT" --workers 1 >>"$LOG" 2>&1 &
    echo $! > $PIDFILE; echo "$PORT" > data/server.port
    for _ in $(seq 1 40); do curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null && break; sleep 0.5; done
    if curl -sf "http://127.0.0.1:$PORT/api/health"; then
      echo; echo "Podcast Studio is up:  http://$(hostname -f):$PORT   (local: http://127.0.0.1:$PORT)"
      echo "From your laptop:      ssh -N -L $PORT:127.0.0.1:$PORT $(whoami)@$(hostname -f)  then open http://localhost:$PORT"
    else
      echo "Server failed to start; last log lines:"; tail -20 "$LOG"; exit 1
    fi ;;
  stop)
    if running; then kill "$(cat $PIDFILE)"; rm -f $PIDFILE; echo "Stopped."; else echo "Not running."; fi ;;
  status)
    if running; then echo "Running (pid $(cat $PIDFILE)) on port $(cat data/server.port)"; curl -s "http://127.0.0.1:$(cat data/server.port)/api/health"; echo; else echo "Not running."; fi ;;
  restart)
    "$0" stop; sleep 1; "$0" start "${2:-$(cat data/server.port 2>/dev/null || echo 8000)}" ;;
  *) echo "usage: $0 start [PORT] | stop | status | restart"; exit 2 ;;
esac
