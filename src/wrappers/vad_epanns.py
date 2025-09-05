# --- FORCE REPO MODELS PACKAGE (idempotent) ---
import sys, types
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
m = sys.modules.get('models')
if m is not None and not getattr(m, '__path__', None):
    # Había un models.py de terceros: eliminarlo del cache
    del sys.modules['models']
# Asegurar que 'models' apunta a <repo>/models como paquete
if 'models' not in sys.modules:
    pkg = types.ModuleType('models')
    pkg.__path__ = [str(_REPO_ROOT / 'models')]
    sys.modules['models'] = pkg
# --- END FORCE REPO MODELS PACKAGE ---
# --- E-PANNs import hygiene ---
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if 'models' in sys.modules and not getattr(sys.modules['models'], '__path__', None):
    del sys.modules['models']
# --- end hygiene ---
"""!
@file vad_epanns.py
@brief A VAD wrapper for the E-PANNs model, adapted from its original source code.
"""

from pathlib import Path
import numpy as np
import sys
import os
import csv
import torch
import librosa

# Add the E-PANNs project path to sys.path
project_root = str(Path(__file__).parent.parent.parent)
epanns_path = os.path.join(project_root, 'models', 'epanns')
if epanns_path not in sys.path:
    sys.path.insert(0, epanns_path)

# Import from the existing E-PANNs files
from models.epanns.models import Cnn14_pruned
from models.epanns.utils import move_data_to_device

# Manually import AudioModelInference components to avoid relative import issues
import torch
import numpy as np
import librosa

# --- EPANNS flexible constructor ---
import inspect
def _epanns_constructor_kwargs():
    base = dict(sample_rate=32000, window_size=1024, hop_size=320,
                mel_bins=64, fmin=50, fmax=14000, classes_num=527)
    # Valores por defecto usados por algunas variantes E-PANNs.
    p = dict(p1=0, p2=0, p3=0, p4=0, p5=0, p6=0,
             p7=0.5, p8=0.5, p9=0.5, p10=0.5, p11=0.5, p12=0.5)
    sig = inspect.signature(Cnn14_pruned.__init__)
    allowed = set(sig.parameters) - {'self'}
    kwargs = {k: v for k, v in {**base, **p}.items() if k in allowed}
    return kwargs

def _build_epanns_model():
    return Cnn14_pruned(**_epanns_constructor_kwargs())
# --- end flexible constructor ---


class AudioModelInference:
    """! @brief A simplified copy of the AudioModelInference class from the E-PANNs project. """
    LOGMEL_MEANS = np.float32([
        -14.050895, -13.107869, -13.1390915, -13.255364, -13.917199,
        -14.087848, -14.855916, -15.266642, -15.884036, -16.491768,
        -17.067415, -17.717588, -18.075916, -18.84405, -19.233824,
        -19.954256, -20.180824, -20.695705, -21.031914, -21.33451,
        -21.758745, -21.917028, -22.283598, -22.737364, -22.920172,
        -23.23437, -23.66509, -23.965239, -24.580393, -24.67597,
        -25.194445, -25.55243, -25.825129, -26.309643, -26.703104,
        -27.28697, -27.839067, -28.228388, -28.746237, -29.236507,
        -29.937782, -30.755503, -31.674414, -32.853516, -33.959763,
        -34.88149, -35.81145, -36.72929, -37.746593, -39.000496,
        -40.069244, -40.947514, -41.79767, -42.81981, -43.8541,
        -44.895683, -46.086784, -47.255924, -48.520145, -50.726765,
        -52.932228, -54.713795, -56.69902, -59.078354])
    LOGMEL_STDDEVS = np.float32([
        22.680508, 22.13264, 21.857653, 21.656355, 21.565693, 21.525793,
        21.450764, 21.377304, 21.338581, 21.3247, 21.289171, 21.221565,
        21.175856, 21.049534, 20.954664, 20.891844, 20.849905, 20.809206,
        20.71186, 20.726717, 20.72358, 20.655743, 20.650305, 20.579372,
        20.583157, 20.604849, 20.5452, 20.561695, 20.448244, 20.46753,
        20.433657, 20.412025, 20.47265, 20.456116, 20.487215, 20.387547,
        20.331848, 20.310328, 20.292257, 20.292326, 20.241796, 20.19396,
        20.23783, 20.564362, 21.075726, 21.332186, 21.508852, 21.644777,
        21.727905, 22.251642, 22.65972, 22.800117, 22.783764, 22.78581,
        22.86413, 22.948992, 23.12939, 23.180748, 23.03542, 23.131435,
        23.454556, 23.39839, 23.254364, 23.198978])

    def __init__(self, model, winsize=1024, stft_hopsize=512, samplerate=32000,
                 stft_window="hahn", n_mels=64, mel_fmin=50, mel_fmax=14000):
        self.model = model
        self.model.eval()
        self.winsize = winsize
        self.stft_hopsize = stft_hopsize
        self.stft_window = stft_window
        self.mel_filt = librosa.filters.mel(sr=samplerate, n_fft=winsize, n_mels=n_mels,
                                           fmin=mel_fmin, fmax=mel_fmax)

    def wav_to_logmel(self, wav_arr):
        stft_spec = np.abs(librosa.stft(y=wav_arr, n_fft=self.winsize,
                                       hop_length=self.stft_hopsize, center=True,
                                       window=self.stft_window, pad_mode="reflect")) ** 2
        mel_spec = np.dot(self.mel_filt, stft_spec).T
        logmel_spec = librosa.power_to_db(mel_spec, ref=1.0, amin=1e-10, top_db=None)
        logmel_spec -= self.LOGMEL_MEANS
        logmel_spec /= self.LOGMEL_STDDEVS
        return logmel_spec

    def __call__(self, wav_arr, top_k=None):
        audio = wav_arr[None, :]
        audio = move_data_to_device(audio, device='cpu')
        with torch.no_grad():
            preds = self.model(audio).to("cpu").numpy().squeeze(axis=0)
        return preds

# Reuse labels from PANNs
LABELS_CSV = Path(project_root) / "models" / "panns" / "class_labels_indices.csv"

with open(LABELS_CSV) as f:
    reader = csv.DictReader(f)
    AUDIOSET_LABELS = [row["display_name"].strip('"') for row in reader]

SPEECH_TAGS = {
    "Speech", "Singing", "Male singing", "Female singing", "Child singing",
    "Male speech, man speaking", "Female speech, woman speaking",
    "Conversation", "Narration, monologue", "Child speech, kid speaking"
}

SPEECH_INDICES = [i for i, label in enumerate(AUDIOSET_LABELS) if label in SPEECH_TAGS]

class EPANNsVADWrapper:
    """! @brief A VAD wrapper using an existing E-PANNs model. """
    
    def __init__(self, checkpoint="models/epanns/E-PANNs/models/checkpoint_closeto_.44.pt"):
        """!
        @brief Initializes the E-PANNs VAD wrapper.
        @param checkpoint Path to the E-PANNs model checkpoint file.
        """
        self.checkpoint = Path(checkpoint)
        
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")
            
        print(f"[E-PANNs] Loading: {self.checkpoint}")
        self._load_model()

    def _load_model(self):
        """! @brief Loads the E-PANNs model from the specified checkpoint. """
        model = _build_epanns_model()
        
        checkpoint = torch.load(self.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        
        self.audio_inference = AudioModelInference(model)
        print("[E-PANNs] Model loaded")

    def infer(self, wav_path: str) -> np.ndarray:
        """!
        @brief Performs VAD inference on an audio file.
        @param wav_path Path to the input audio file.
        @return np.ndarray An array of probabilities representing speech presence.
        """
        try:
            wav, sr = librosa.load(wav_path, sr=32000, mono=True)
            predictions = self.audio_inference(wav)
            
            speech_probs = predictions[SPEECH_INDICES]
            max_speech_prob = np.max(speech_probs)
            
            # Estimate frames (E-PANNs only provides a clip-wise prediction)
            num_frames = max(1, int(len(wav) / 320))
            frame_probs = np.full(num_frames, max_speech_prob, dtype=np.float32)
            
            print(f"[E-PANNs] {Path(wav_path).name} → {max_speech_prob:.3f}")
            return frame_probs
            
        except Exception as e:
            print(f"[E-PANNs] Error: {e}")
            return np.array([0.5], dtype=np.float32)

# Alias for compatibility
EPANNsVADProb = EPANNsVADWrapper