# src/wrappers/vad_ast.py

import torch
import torchaudio
from typing import Optional
from transformers import AutoModelForAudioClassification, AutoProcessor
from src.wrappers.base_wrapper import BaseVADWrapper


class ASTWrapper(BaseVADWrapper):
    """
    Audio Spectrogram Transformer (AST) wrapper compatible with run_evaluation.py.

    • Carga un checkpoint finetuneado en AudioSet.
    • Devuelve **una sola probabilidad de habla** por clip/ventana, igual que PANNs,
      para que el código de métricas y thresholds funcione sin cambios.
    """

    SPEECH_LABELS = {
        "Speech", "Singing", "Male singing", "Female singing", "Child singing",
        "Male speech, man speaking", "Female speech, woman speaking",
        "Conversation", "Narration, monologue", "Child speech, kid speaking"
    }

    def __init__(
        self,
        model_id: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        device: Optional[str] = None,
        target_sr: int = 16000,
    ) -> None:
        # IMPORTANTE: Llamar al constructor padre
        super().__init__(target_sr)
        
        # Usa GPU si hay; si no, CPU
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.target_sr = target_sr

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = (
            AutoModelForAudioClassification.from_pretrained(model_id)
            .to(self.device)
            .eval()
        )

        # Índices de las etiquetas de habla
        label2id = (
            getattr(self.model.config, "label2id", None)
            or getattr(self.processor.feature_extractor, "class_to_id", {})
        )
        self.speech_ids = [idx for lbl, idx in label2id.items() if lbl in self.SPEECH_LABELS]
        if not self.speech_ids:
            raise ValueError(
                "❌ Ninguna etiqueta de habla encontrada en el checkpoint AST; "
                "revisa SPEECH_LABELS o usa otro modelo."
            )

    @torch.no_grad()
    def infer(self, wav_path: str):
        """Devuelve tensor shape (1,) con la probabilidad de habla."""
        wav, sr = torchaudio.load(wav_path)
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)

        inputs = self.processor(
            wav.squeeze().numpy(), sampling_rate=self.target_sr, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logits = self.model(**inputs).logits.squeeze(0)       # (n_classes,)
        probs = torch.softmax(logits, dim=-1).cpu()

        # Fix: usar indexación segura y max() de PyTorch
        speech_probs = probs[self.speech_ids]  # Tensor with speech probabilities
        speech_prob = torch.max(speech_probs).item()  # float ∈ [0,1]
        return torch.tensor([speech_prob], dtype=torch.float32)