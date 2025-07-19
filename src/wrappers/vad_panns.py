# src/wrappers/vad_panns.py - VERSIÓN CORREGIDA
from pathlib import Path
import numpy as np
import sys
import os
import sys
import os
from pathlib import Path

# Agregar paths necesarios
project_root = str(Path(__file__).parent.parent.parent)
models_path = os.path.join(project_root, 'models')
if models_path not in sys.path:
    sys.path.insert(0, models_path)



# Agregar ruta de modelos al path si no está
models_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
if models_path not in sys.path:
    sys.path.insert(0, models_path)

from models.panns.inference import run_panns_inference

# Etiquetas que cuentan como speech en AudioSet
SPEECH_TAGS = {
    "Speech", "Singing", "Male singing", "Female singing", "Child singing",
    "Male speech, man speaking", "Female speech, woman speaking",
    "Conversation", "Narration, monologue", "Child speech, kid speaking"
}

class PANNsVADWrapper:
    """VAD usando PANNs Cnn14_DecisionLevelAtt."""
    
    def __init__(self, 
                 checkpoint="models/panns/Cnn14_DecisionLevelAtt",
                 model_type="Cnn14_DecisionLevelAtt"):
        self.checkpoint = Path(checkpoint)
        self.model_type = model_type
        
        # Verificar que existe el checkpoint
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint no encontrado: {self.checkpoint}")
            
        print(f"[PANNs] Inicializado con checkpoint: {self.checkpoint}")

    def infer(self, wav_path: str) -> np.ndarray:
        """
        Devuelve probabilidades frame por frame de que haya speech.
        
        Returns:
            np.ndarray: Array de probabilidades (0-1) por frame
        """
        try:
            # Ejecutar inferencia PANNs
            out = run_panns_inference(
                audio_path=wav_path,
                checkpoint_path=str(self.checkpoint),
                model_type=self.model_type
            )
            
            # Procesar resultados frame por frame
            probs = []
            
            for frame in out["sed"]:  # sed = Sound Event Detection
                # frame["predictions"] es una LISTA de dict {class: str, prob: float}
                prob_speech = 0.0
                
                for prediction in frame["predictions"]:
                    if prediction["class"] in SPEECH_TAGS:
                        prob_speech = max(prob_speech, prediction["prob"])
                
                probs.append(prob_speech)
            
            return np.array(probs, dtype=float)
            
        except Exception as e:
            print(f"[PANNs] Error en inferencia: {e}")
            # Retornar array vacío en caso de error
            return np.array([0.0])

# Alias para compatibilidad con el sistema de carga dinámica
PANNsVADProb = PANNsVADWrapper