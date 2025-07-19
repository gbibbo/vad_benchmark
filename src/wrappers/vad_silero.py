# src/wrappers/vad_silero.py
"""
Silero VAD wrapper – devuelve un vector de probabilidades 0-1
con paso de 32 ms (512 muestras) para un wav mono 16 kHz.
Compatible con generate_mask_generic.py
"""

from pathlib import Path
from typing import Union
import numpy as np
import torch
import soundfile as sf


class SileroVADProb:
    def __init__(self) -> None:
        self.model, _ = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            trust_repo=True,
            verbose=False,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        self.sr = 16_000
        self.win = 512  # nº de muestras que acepta el modelo

    def _frame_signal(self, wav: np.ndarray) -> np.ndarray:
        """Rellena con ceros y parte en frames de 512 muestras (N, 512)."""
        n = len(wav)
        pad = (-n) % self.win          # 0 … 511 muestras para completar
        if pad:
            wav = np.hstack([wav, np.zeros(pad, dtype=wav.dtype)])
        return wav.reshape(-1, self.win)

    def infer(self, wav_path: Union[str, Path]) -> np.ndarray:
        """Devuelve un vector de probabilidades ∈[0,1] por frame de 32 ms."""
        wav, sr = sf.read(str(wav_path), dtype="float32")
        if sr != self.sr:
            raise RuntimeError(f"esperaba {self.sr} Hz y llegó {sr}")
        if wav.ndim == 2:                       # estéreo → mono
            wav = wav.mean(axis=1, dtype=wav.dtype)

        frames = self._frame_signal(wav)        # (N, 512)
        with torch.no_grad():
            logits = []
            for fr in torch.from_numpy(frames).to(self.device):
                p = self.model(fr, self.sr).sigmoid()  # (1,)
                logits.append(p.item())
        return np.array(logits, dtype=np.float32)
