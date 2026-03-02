#!/bin/bash
set -e

echo "==> Starting Ollama server in background..."
ollama serve &
OLLAMA_PID=$!

echo "==> Waiting for Ollama to be ready..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "==> Ollama is up."
    break
  fi
  echo "    Attempt $i/30 — waiting..."
  sleep 2
done

echo "==> Pulling model: tinyllama..."
ollama pull tinyllama
echo "==> Model ready."

echo "==> Starting FastAPI..."
uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
