"""!
@file vad_ast.py
@brief A VAD wrapper for the Audio Spectrogram Transformer (AST) model.

@details This wrapper is designed to be compatible with the `run_evaluation.py` script.
         It loads a model checkpoint fine-tuned on the AudioSet dataset.
         It returns a single speech probability for the entire input clip/window,
         similar to PANNs-based wrappers, ensuring it works seamlessly with the
         existing metrics and thresholding logic.
"""

import torch
import torchaudio
from typing import Optional
from transformers import AutoModelForAudioClassification, AutoProcessor
from src.wrappers.base_wrapper import BaseVADWrapper


class ASTWrapper(BaseVADWrapper):
    """!
    @brief Audio Spectrogram Transformer (AST) wrapper class.
    """

    //!< Set of labels considered as speech or singing.
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
        """!
        @brief Initializes the AST wrapper.
        @param model_id The Hugging Face model ID for the AST model.
        @param device The compute device to use (e.g., 'cpu', 'cuda'). If None, it auto-detects GPU.
        @param target_sr The target sample rate required by the model.
        """
        # IMPORTANT: Call the parent constructor
        super().__init__(target_sr)
        
        # Use GPU if available; otherwise, CPU
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.target_sr = target_sr

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = (
            AutoModelForAudioClassification.from_pretrained(model_id)
            .to(self.device)
            .eval()
        )

        # Get the indices for speech-related labels
        label2id = (
            getattr(self.model.config, "label2id", None)
            or getattr(self.processor.feature_extractor, "class_to_id", {})
        )
        self.speech_ids = [idx for lbl, idx in label2id.items() if lbl in self.SPEECH_LABELS]
        if not self.speech_ids:
            raise ValueError(
                "❌ No speech labels found in the AST checkpoint; "
                "check SPEECH_LABELS or use another model."
            )

    @torch.no_grad()
    def infer(self, wav_path: str):
        """!
        @brief Performs VAD inference on an audio file.
        @param wav_path Path to the input audio file.
        @return A torch.Tensor of shape (1,) with the speech probability.
        """
        wav, sr = torchaudio.load(wav_path)
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)

        inputs = self.processor(
            wav.squeeze().numpy(), sampling_rate=self.target_sr, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logits = self.model(**inputs).logits.squeeze(0)       # (n_classes,)
        probs = torch.softmax(logits, dim=-1).cpu()

        # Fix: Use safe indexing and PyTorch's max() function
        speech_probs = probs[self.speech_ids]  # Tensor with speech probabilities
        speech_prob = torch.max(speech_probs).item()  # float in [0,1]
        return torch.tensor([speech_prob], dtype=torch.float32)