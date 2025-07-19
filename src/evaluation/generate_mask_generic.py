# src/evaluation/generate_mask_generic.py

"""
Genera archivos mask_Θ.csv para un modelo VAD y una lista de WAVs.

• Mantiene compatibilidad con Silero, WebRTC, Whisper y **añade AST**.
• Útil para tests rápidos fuera del pipeline principal.
"""

import os, time, argparse
from pathlib import Path
from importlib import import_module

import numpy as np
import pandas as pd
from tqdm import tqdm

# Rutas por defecto (se pueden sobreescribir con variables de entorno)
CHUNKS_DIR = Path(os.getenv("CHUNKS_DIR", "datasets/test_sample"))
CSV_LIST   = Path(os.getenv("CSV_LIST",  "datasets/test_sample/list.csv"))
THETAS     = [0.00, 0.10, 0.20, 0.30, 0.50, 0.80]

# Tabla de wrappers disponibles
WRAPPERS = {
    "silero":  "src.wrappers.vad_silero.SileroVADWrapper",
    "webrtc":  "src.wrappers.vad_webrtc.WebRTCVADWrapper",
    "whisper": "src.wrappers.vad_whisper_small.WhisperSmallVADWrapper",
    "ast":     "src.wrappers.vad_ast.ASTWrapper",          # ← añadido
}

# --------------------------------------------------------------------------- #
def load_wrapper(name: str):
    if name not in WRAPPERS:
        raise ValueError(f"Modelo '{name}' no soportado. Escoge entre: {list(WRAPPERS)}")
    mod_path, cls_name = WRAPPERS[name].rsplit(".", 1)
    return getattr(import_module(mod_path), cls_name)()

def build_wav(chunks_dir: Path, cid: str) -> Path:
    """Si el id no lleva '.wav' añade '.16kHz.wav'."""
    return chunks_dir / (cid if cid.lower().endswith(".wav") else f"{cid}.16kHz.wav")

def save_masks(mask_dict, out_root: Path):
    out_root.mkdir(parents=True, exist_ok=True)
    for thr, rows in mask_dict.items():
        (out_root / f"mask_{thr:.2f}.csv").write_text(
            "Filename,Speech\n" + "\n".join(f"{f},{v}" for f, v in rows)
        )

# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=WRAPPERS.keys())
    ap.add_argument("--audio", help="WAV único (ignora CSV_LIST si se pasa).")
    ap.add_argument("--out-root", help="Carpeta destino (default masks_<model>)")
    ap.add_argument("--thresholds", nargs="*", type=float, default=THETAS)
    args = ap.parse_args()

    wrapper = load_wrapper(args.model)
    out_root = Path(args.out_root or f"masks_{args.model}")

    # Selección de WAVs
    if args.audio:
        wavs = [Path(args.audio)]
    else:
        df   = pd.read_csv(CSV_LIST, header=None)
        wavs = [build_wav(CHUNKS_DIR, cid) for cid in (df[1] if 1 in df else df[0])]

    masks = {t: [] for t in args.thresholds}
    t0 = time.time()

    for wav in tqdm(wavs, desc="Inferencia"):
        if not wav.is_file():
            for t in masks: masks[t].append((wav.name, 0))
            continue
        prob = wrapper.infer(str(wav))[0].item()
        for t in masks: masks[t].append((wav.name, int(prob >= t)))

    save_masks(masks, out_root)
    print(f"✅ {len(wavs)} archivos procesados en {time.time()-t0:.1f}s → {out_root}")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()