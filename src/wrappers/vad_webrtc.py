# src/wrappers/vad_webrtc.py
# src/vad_wrappers/vad_webrtc.py
import numpy as np, webrtcvad, soundfile as sf

class WebRTCVADWrapper:
    """WebRTC VAD (mode 3). Expects 16 kHz."""
    def __init__(self, mode: int = 3):
        self.vad = webrtcvad.Vad(mode)
        self.sr = 16_000

    def infer(self, wav_path: str, frame_ms: int = 10) -> np.ndarray:
        audio, sr = sf.read(wav_path, dtype="int16")
        assert sr == self.sr, "Input must be 16 kHz"
        frame_len = int(sr*frame_ms/1000)
        n_frames = int(np.ceil(len(audio)/frame_len))
        out = np.zeros(n_frames, dtype=float)
        for i in range(n_frames):
            seg = audio[i*frame_len:(i+1)*frame_len]
            if len(seg) < frame_len:
                seg = np.pad(seg, (0, frame_len-len(seg)))
            out[i] = float(self.vad.is_speech(seg.tobytes(), sr))
        return out
