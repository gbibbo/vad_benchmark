#!/usr/bin/env bash
set -euo pipefail

# -------- Config ----------
ENV_NAME="${ENV_NAME:-vad_benchmark_py310}"
PY_VER="${PY_VER:-3.10}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"  # CPU por defecto
# --------------------------

echo ">>> Creando/activando entorno conda: $ENV_NAME (Python $PY_VER)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi

# usa conda run para no depender de un 'conda activate' dentro del script
echo ">>> Actualizando pip"
conda run -n "$ENV_NAME" python -m pip install -U pip

echo ">>> Instalando PyTorch desde el índice oficial (${TORCH_INDEX_URL})"
conda run -n "$ENV_NAME" pip install --index-url "$TORCH_INDEX_URL" torch torchaudio torchvision

echo ">>> Instalando dependencias del proyecto"
# Nota: requirements.txt NO debe contener torch/torchaudio/torchvision
conda run -n "$ENV_NAME" pip install -r requirements.txt

# webrtcvad: usamos ruedas precompiladas (si ya está, no hace nada)
conda run -n "$ENV_NAME" python - <<'PY'
try:
    import webrtcvad  # puede venir como 'webrtcvad-wheels'
    print("webrtcvad OK:", webrtcvad.__name__)
except Exception as e:
    print("Instalando webrtcvad-wheels…", e)
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "webrtcvad-wheels>=2.0.10"])
PY

# Crear activador para el usuario
echo ">>> Creando activate_vad.sh"
cat > activate_vad.sh <<'BASH'
#!/usr/bin/env bash
conda activate ENV_NAME_PLACEHOLDER
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
echo "VAD-Benchmark environment activated: $(python -V)"
BASH
sed -i "s|ENV_NAME_PLACEHOLDER|$ENV_NAME|g" activate_vad.sh
chmod +x activate_vad.sh

# Resumen final
echo
echo "✅ Instalación completada (sin datasets). Siguientes pasos:"
echo "1) source activate_vad.sh"
echo "2) Opcional: enlaza datasets/modelos existentes con symlinks para evitar descargas."
echo "   - ln -s /ruta/antigua/models models"
echo "   - ln -s /ruta/antigua/datasets/chime datasets/chime"
echo "3) python test_installation.py"
echo "4) python scripts/run_evaluation.py --config configs/config_demo.yaml"
