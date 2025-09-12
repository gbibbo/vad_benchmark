#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# VAD Benchmark - Instalador con limpieza de audios a 48 kHz
# - Descarga datasets (si tu repositorio ya los descarga, se usarán esos pasos)
# - Luego elimina todos los .wav a 48 kHz y limpia symlinks rotos
#
# Uso:
#   ./install.sh            # descarga (si procede) + limpia 48 kHz
#   ./install.sh --reset    # borra datasets/chime antes de descargar + limpiar
#
# Variables opcionales:
#   DATA_ROOT=/ruta/personalizada   # por defecto: <repo>/datasets
# -----------------------------------------------------------------------------

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-"$ROOT_DIR/datasets"}"
CHIME_DIR="$DATA_ROOT/chime"

log() { printf "\n[%s] %s\n" "$(date +'%H:%M:%S')" "$*"; }

maybe_download() {
  # Si tu repo ya tiene un paso de descarga, se detecta y ejecuta.
  # Añade aquí tu flujo real si quieres forzar uno concreto.
  if [[ -x "$ROOT_DIR/scripts/download_chime.sh" ]]; then
    log "Descargando CHIME con scripts/download_chime.sh..."
    bash "$ROOT_DIR/scripts/download_chime.sh"
  elif [[ -f "$ROOT_DIR/scripts/download_chime.py" ]]; then
    log "Descargando CHIME con scripts/download_chime.py..."
    python "$ROOT_DIR/scripts/download_chime.py"
  elif [[ -f "$ROOT_DIR/install_datasets.sh" ]]; then
    log "Ejecutando install_datasets.sh..."
    bash "$ROOT_DIR/install_datasets.sh"
  else
    log "No se detectó script de descarga. Si ya tienes el dataset, sigo con limpieza."
  fi
}

reset_if_requested() {
  if [[ "${1:-}" == "--reset" ]]; then
    log "Eliminando $CHIME_DIR para reinstalar desde cero..."
    rm -rf "$CHIME_DIR"
  fi
}

prune_48khz() {
  log "Eliminando .wav a 48 kHz bajo $DATA_ROOT (mantener solo 16 kHz)..."
  python - <<'PY'
import os, sys, glob
from pathlib import Path
try:
    import soundfile as sf
except Exception as e:
    print(f"[ERROR] Necesito 'soundfile' (pysoundfile). Instálalo con: pip install soundfile")
    sys.exit(1)

DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent / "datasets")).resolve()
patterns = [str(DATA_ROOT / "**" / "*.wav")]
removed = kept = errs = 0
for pat in patterns:
    for p in glob.glob(pat, recursive=True):
        path = Path(p)
        try:
            info = sf.info(str(path))
            if info.samplerate == 48000:
                # Borrar enlace simbólico o archivo real
                path.unlink()
                removed += 1
            else:
                kept += 1
        except Exception as e:
            print(f"[WARN] No pude leer {path}: {e}")
            errs += 1

print(f"[OK] Borrados 48 kHz: {removed}  | Conservados: {kept}  | Avisos: {errs}")
PY

  # Limpiar symlinks rotos que hayan quedado
  log "Limpiando symlinks rotos..."
  find -L "$DATA_ROOT" -type l -xtype l -print -delete || true
}

main() {
  reset_if_requested "${1:-}"
  mkdir -p "$CHIME_DIR"
  maybe_download
  prune_48khz
  log "Instalación/limpieza finalizada."
}

main "$@"
