"""!
@file vad_whisper_tiny.py
@brief A VAD wrapper using the Whisper-tiny model.

@details This wrapper marks as "speech" every segment that Whisper identifies
         with timestamps. The input audio MUST be 16 kHz.
"""

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel


class WhisperTinyVADWrapper:
    """!
    @brief A binary VAD implementation based on the Whisper-tiny model.
    @note The class is named specifically to avoid conflicts with other potential wrappers.
    """

    def __init__(self, model_size: str = "tiny"):
        """!
        @brief Initializes the Whisper VAD wrapper.
        @param model_size The size of the Whisper model to load (e.g., "tiny").
                          This is a required argument for the underlying WhisperModel.
        """
        self.sr = 16_000
        print(f"[Whisper-Tiny] Initializing model {model_size}")
        # This argument is mandatory for WhisperModel
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print(f"[Whisper-Tiny] Model {model_size} loaded successfully")

    def infer(self, wav_path: str, frame_ms: int = 10) -> np.ndarray:
        """!
        @brief Performs VAD inference by generating a binary mask from Whisper's segments.
        @param wav_path Path to the 16 kHz input audio file.
        @param frame_ms The duration of each frame in milliseconds for the output mask.
        @return A numpy array of 0s and 1s, where 1 indicates speech.
        """
        wav, sr = sf.read(wav_path, dtype="float32")
        assert sr == self.sr, f"Input must be 16 kHz, got {sr}"

        # Duration for debugging purposes
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


# Alias for backwards compatibility with the original wrapper name
WhisperVADWrapper = WhisperTinyVADWrapper

__all__ = ["WhisperTinyVADWrapper", "WhisperVADWrapper"]