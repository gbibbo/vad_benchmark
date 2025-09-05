#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

# Activa entorno sin romper si conda no está inicializado
. ./activate_vad.sh 2>/dev/null || true
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

echo ">>> Python:"; python -V
echo ">>> Test instalación"; python test_installation.py

echo ">>> Eval CHiME CMF"
python -m scripts.run_evaluation --config configs/config_chime_cmf.yaml

echo ">>> Eval CHiME CMFV"
python -m scripts.run_evaluation --config configs/config_chime_cmfv.yaml

echo ">>> Resultados en: $REPO_DIR/results"
ls -lh "$REPO_DIR/results" || true
