#!/bin/bash
set -e

echo "=== Ollama Model Initialization ==="
echo "Waiting for Ollama service to be ready..."

# Aspetta che Ollama sia disponibile
MAX_RETRIES=30
RETRY_COUNT=0
until ollama list > /dev/null 2>&1; do
  RETRY_COUNT=$((RETRY_COUNT + 1))
  if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
    echo "ERROR: Ollama service failed to start after $MAX_RETRIES attempts"
    exit 1
  fi
  echo "Waiting for Ollama... ($RETRY_COUNT/$MAX_RETRIES)"
  sleep 2
done

echo "✓ Ollama service is ready!"

# Lista dei modelli da scaricare
MODELS=(
  "qwen3-embedding:4b"
  # "qwen2.5:3b-instruct"
  "llama3.1:8b"
)

echo ""
echo "=== Pulling required models ==="

for model in "${MODELS[@]}"; do
  echo ""
  echo "Checking model: $model"

  # Verifica se il modello esiste già
  if ollama list | grep -q "$model"; then
    echo "✓ Model $model already exists, skipping..."
  else
    echo "⬇ Pulling model: $model"
    ollama pull "$model"
    echo "✓ Model $model pulled successfully!"
  fi
done

echo ""
echo "=== All models ready! ==="
echo "Available models:"
ollama list

echo ""
echo "✓ Ollama initialization complete!"
