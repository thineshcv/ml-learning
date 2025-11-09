#!/usr/bin/env bash
# Serve the MLflow model stored in mlruns
# Usage: ./serve_model.sh [port]
set -euo pipefail
MODEL_DIR="/home/gilfoyle/data/play/ml/ml-pipeline-demo/mlruns/551694781745399147/models/m-02e631ce987947dcbe8221456ba5dda1/artifacts"
PORT=${1:-1234}

echo "Serving model from: $MODEL_DIR"
mlflow models serve -m "file://$MODEL_DIR" -p "$PORT" --no-conda
