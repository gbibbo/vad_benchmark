"""!
@file vad_panns.py
@brief A VAD wrapper using the PANNs (Pre-trained Audio Neural Networks) model.

@note This is a corrected version of the wrapper.
"""
from pathlib import Path
import numpy as np
import sys
import os
from pathlib import Path

# Add necessary paths
project_root = str(Path(__file__).parent.parent.parent)
models_path = os.path.join(project_root, 'models')
if models_path not in sys.path:
    sys.path.insert(0, models_path)

# Add models path to sys.path if not already present
models_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
if models_path not in sys.path:
    sys.path.insert(0, models_path)

from models.panns.inference import run_panns_inference

# Set of labels from AudioSet that are considered as speech.
SPEECH_TAGS = {
    "Speech", "Singing", "Male singing", "Female singing", "Child singing",
    "Male speech, man speaking", "Female speech, woman speaking",
    "Conversation", "Narration, monologue", "Child speech, kid speaking"
}

class PANNsVADWrapper:
    """! @brief A VAD wrapper using the PANNs Cnn14_DecisionLevelAtt model. """
    
    def __init__(self, 
                 checkpoint="models/panns/Cnn14_DecisionLevelAtt",
                 model_type="Cnn14_DecisionLevelAtt"):
        """!
        @brief Initializes the PANNs VAD wrapper.
        @param checkpoint Path to the PANNs model checkpoint directory.
        @param model_type The specific type of the PANNs model.
        """
        self.checkpoint = Path(checkpoint)
        self.model_type = model_type
        
        # Verify that the checkpoint exists
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")
            
        print(f"[PANNs] Initialized with checkpoint: {self.checkpoint}")

    def infer(self, wav_path: str) -> np.ndarray:
        """!
        @brief Returns frame-by-frame probabilities of speech presence.
        
        @param wav_path Path to the input audio file.
        @return np.ndarray An array of probabilities (0-1) for each frame.
        """
        try:
            # Execute PANNs inference
            out = run_panns_inference(
                audio_path=wav_path,
                checkpoint_path=str(self.checkpoint),
                model_type=self.model_type
            )
            
            # Process results frame by frame
            probs = []
            
            for frame in out["sed"]:  # sed = Sound Event Detection
                # frame["predictions"] is a LIST of dicts: {class: str, prob: float}
                prob_speech = 0.0
                
                for prediction in frame["predictions"]:
                    if prediction["class"] in SPEECH_TAGS:
                        prob_speech = max(prob_speech, prediction["prob"])
                
                probs.append(prob_speech)
            
            return np.array(probs, dtype=float)
            
        except Exception as e:
            print(f"[PANNs] Error during inference: {e}")
            # Return an array with a single zero in case of error
            return np.array([0.0])

# Alias for compatibility with the dynamic loading system
PANNsVADProb = PANNsVADWrapper