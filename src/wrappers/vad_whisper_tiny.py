# src/wrappers/vad_whisper_tiny.py
"""
Whisper-tiny VAD wrapper: marca como "speech" cada segmento que Whisper
devuelve con timestamps.  El audio de entrada DEBE ser 16 kHz.
"""

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel


class WhisperTinyVADWrapper:                # ← Nombre específico para evitar conflictos
    """Binary VAD basado en Whisper-tiny."""

    def __init__(self, model_size: str = "tiny"):
        self.sr = 16_000
        print(f"[Whisper-Tiny] Inicializando modelo {model_size}")
        # ¡ARGUMENTO OBLIGATORIO!
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print(f"[Whisper-Tiny] Modelo {model_size} cargado exitosamente")

    # ------------------------------------------------------------------ #
    def infer(self, wav_path: str, frame_ms: int = 10) -> np.ndarray:
        wav, sr = sf.read(wav_path, dtype="float32")
        assert sr == self.sr, f"Input must be 16 kHz, got {sr}"

        # Duración para debug
        duration = len(wav) / sr
        print(f"Processing audio with duration {duration/60:02.0f}:{duration%60:04.1f}")

        segments, _ = self.model.transcribe(
            wav,
            language="en",
            vad_filter=False,
            beam_size=1,
        )

        n_frames = int(np.ceil(len(wav) / (sr * frame_ms / 1000)))
        out = np.zeros(n_frames, dtype=np.float32)
        for s in segments:
            start_f = int(s.start / (frame_ms / 1000))
            end_f   = int(s.end   / (frame_ms / 1000))
            out[start_f:end_f] = 1.0
        return out


# Compatibilidad con el wrapper original
WhisperVADWrapper = WhisperTinyVADWrapper

__all__ = ["WhisperTinyVADWrapper", "WhisperVADWrapper"]