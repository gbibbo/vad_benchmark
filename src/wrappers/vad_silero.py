"""!
@file vad_silero.py
@brief Silero VAD wrapper – returns a vector of probabilities [0,1].

@details This wrapper provides a vector of speech probabilities with a step of 32 ms
         (512 samples) for a 16 kHz mono WAV file. It is compatible with
         scripts like `generate_mask_generic.py`.
"""

from pathlib import Path
from typing import Union
import numpy as np
import torch
import soundfile as sf


class SileroVADProb:
    """! @brief A wrapper for the Silero VAD model to get frame-wise speech probabilities. """
    def __init__(self) -> None:
        """! @brief Initializes the Silero VAD model and its configuration. """
        self.model, _ = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            trust_repo=True,
            verbose=False,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        self.sr = 16_000
        self.win = 512  # number of samples the model accepts

    def _frame_signal(self, wav: np.ndarray) -> np.ndarray:
        """!
        @brief Pads a waveform with zeros and splits it into frames of 512 samples.
        @param wav The input numpy array representing the waveform.
        @return A numpy array of shape (N, 512) containing the framed signal.
        """
        n = len(wav)
        pad = (-n) % self.win          # 0 to 511 samples needed for padding
        if pad:
            wav = np.hstack([wav, np.zeros(pad, dtype=wav.dtype)])
        return wav.reshape(-1, self.win)

    def infer(self, wav_path: Union[str, Path]) -> np.ndarray:
        """!
        @brief Performs inference on an audio file and returns frame-wise speech probabilities.
        @param wav_path Path to the input audio file.
        @return A numpy array of probabilities ∈[0,1] for each 32 ms frame.
        """
        wav, sr = sf.read(str(wav_path), dtype="float32")
        if sr != self.sr:
            raise RuntimeError(f"expected {self.sr} Hz but got {sr}")
        if wav.ndim == 2:                       # stereo -> mono
            wav = wav.mean(axis=1, dtype=wav.dtype)

        frames = self._frame_signal(wav)        # (N, 512)
        with torch.no_grad():
            logits = []
            for fr in torch.from_numpy(frames).to(self.device):
                p = self.model(fr, self.sr).sigmoid()  # (1,)
                logits.append(p.item())
        return np.array(logits, dtype=np.float32)