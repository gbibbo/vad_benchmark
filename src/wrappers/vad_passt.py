from pathlib import Path
# src/wrappers/vad_passt.py

"""
PaSST VAD wrapper — versión silenciosa
"""

from pathlib import Path
import csv
import torch, torchaudio
import sys
from io import StringIO
import contextlib

# Suprimir outputs verbosos
@contextlib.contextmanager
def suppress_stdout():
    """Suprimir stdout temporalmente"""
    old_stdout = sys.stdout
    try:
        sys.stdout = StringIO()
        yield
    finally:
        sys.stdout = old_stdout

# Import con supresión
with suppress_stdout():
    from hear21passt.base import load_model

# ---------------------------------------------------------------------
# Mini-base para imitar los otros wrappers
# ---------------------------------------------------------------------
class BaseVADWrapper:
    def __init__(self, device: str = "cpu"):
        self.device = device

# ---------------------------------------------------------------------
# CSV de etiquetas
# ---------------------------------------------------------------------
LABELS_CSV = Path(__file__).resolve().parent.parent.parent / 'models' / 'metadata' / 'class_labels_indices.csv'

with open(LABELS_CSV) as f:
    reader = csv.DictReader(f)
    ALL_LABELS = [row["display_name"].strip('"') for row in reader]

# Etiquetas de voz/canto
SPEECH_LABELS = {
    "Speech", "Singing", "Male singing", "Female singing", "Child singing",
    "Male speech, man speaking", "Female speech, woman speaking",
    "Conversation", "Narration, monologue", "Child speech, kid speaking",
}
SPEECH_IDXS = [i for i, lbl in enumerate(ALL_LABELS) if lbl in SPEECH_LABELS]

# ---------------------------------------------------------------------
# Wrapper PaSST silencioso
# ---------------------------------------------------------------------
class PaSSTWrapper(BaseVADWrapper):
    target_sr = 32_000

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        # Cargar modelo con supresión completa
        with suppress_stdout():
            print("Loading PASST (silent mode)...")  # Solo esto se ve
            self.model = load_model(mode="logits").to(self.device).eval()
        self.sigmoid = torch.nn.Sigmoid()

    @torch.no_grad()
    def infer(self, wav_path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(wav_path)
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)

        wav = torch.clamp(wav.mean(0, keepdim=True), -1.0, 1.0).to(self.device)
        
        # Inferencia con supresión de debug prints
        with suppress_stdout():
            logits = self.model(wav)[0]
        
        probs = self.sigmoid(logits[SPEECH_IDXS])
        speech_prob = probs.max().item()

        return torch.tensor([speech_prob], dtype=torch.float32)
