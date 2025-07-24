"""!
@file vad_passt.py
@brief PaSST VAD wrapper - silent version.

@details This wrapper is designed to suppress the verbose output from the
         original PaSST model during loading and inference.
"""

from pathlib import Path
import csv
import torch, torchaudio
import sys
from io import StringIO
import contextlib

@contextlib.contextmanager
def suppress_stdout():
    """!
    @brief A context manager to temporarily suppress stdout.
    """
    old_stdout = sys.stdout
    try:
        sys.stdout = StringIO()
        yield
    finally:
        sys.stdout = old_stdout

# Import with suppression to avoid verbose model loading messages
with suppress_stdout():
    from hear21passt.base import load_model

# ---------------------------------------------------------------------
# Mini-base class to mimic other wrappers
# ---------------------------------------------------------------------
class BaseVADWrapper:
    """! @brief A minimal base class to align with the structure of other VAD wrappers. """
    def __init__(self, device: str = "cpu"):
        """!
        @brief Initializes the base wrapper.
        @param device The compute device to use ("cpu" or "cuda").
        """
        self.device = device

# ---------------------------------------------------------------------
# Label Definitions
# ---------------------------------------------------------------------
LABELS_CSV = Path(__file__).resolve().parent.parent.parent / 'models' / 'metadata' / 'class_labels_indices.csv'

with open(LABELS_CSV) as f:
    reader = csv.DictReader(f)
    ALL_LABELS = [row["display_name"].strip('"') for row in reader]

# Speech/singing related labels
SPEECH_LABELS = {
    "Speech", "Singing", "Male singing", "Female singing", "Child singing",
    "Male speech, man speaking", "Female speech, woman speaking",
    "Conversation", "Narration, monologue", "Child speech, kid speaking",
}
SPEECH_IDXS = [i for i, lbl in enumerate(ALL_LABELS) if lbl in SPEECH_LABELS]

# ---------------------------------------------------------------------
# Silent PaSST Wrapper
# ---------------------------------------------------------------------
class PaSSTWrapper(BaseVADWrapper):
    """!
    @brief A VAD wrapper for the PaSST model that suppresses its verbose output.
    """
    target_sr = 32_000

    def __init__(self, device: str = "cpu"):
        """!
        @brief Initializes the PaSST wrapper, loading the model silently.
        @param device The compute device to use ("cpu" or "cuda").
        """
        super().__init__(device)
        # Load the model with full output suppression.
        with suppress_stdout():
            print("Loading PASST (silent mode)...")
            self.model = load_model(mode="logits").to(self.device).eval()
        self.sigmoid = torch.nn.Sigmoid()

    @torch.no_grad()
    def infer(self, wav_path: str) -> torch.Tensor:
        """!
        @brief Performs VAD inference on a single audio file.
        
        @details It loads an audio file, resamples it if necessary, and then runs
                 the PaSST model to get a single probability score for the presence
                 of speech or singing.
                 
        @param wav_path Path to the input audio file.
        @return A torch.Tensor containing a single float value representing the speech probability.
        """
        wav, sr = torchaudio.load(wav_path)
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)

        wav = torch.clamp(wav.mean(0, keepdim=True), -1.0, 1.0).to(self.device)
        
        # Inference with suppression of debug prints.
        with suppress_stdout():
            logits = self.model(wav)[0]
        
        probs = self.sigmoid(logits[SPEECH_IDXS])
        speech_prob = probs.max().item()

        return torch.tensor([speech_prob], dtype=torch.float32)