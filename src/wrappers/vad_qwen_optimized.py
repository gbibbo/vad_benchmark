"""
Qwen2-Audio VAD wrapper - Versión Corregida
FIX: Cambiar prompt de "human speech" a "someone talking"
"""

from pathlib import Path
import torch, torchaudio
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
import gc

class BaseVADWrapper:
    def __init__(self, device="auto"):
        self.device = device

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"

# MEJORA 1: Prompt corregido - usar "talking" en lugar de "speech"
IMPROVED_PROMPT = (
    "<|audio_bos|><|AUDIO|><|audio_eos|>"
    "Listen to this audio clip carefully. "
    "Does this audio contain someone talking? "
    "This includes conversation, singing, narration, or any human voice. "
    "Answer with only 'YES' if you hear someone talking, or 'NO' if you don't hear anyone talking."
)

class Qwen2AudioWrapper(BaseVADWrapper):
    target_sr = 16_000

    def __init__(self, device="auto"):
        super().__init__(device)
        self.model = None
        self.processor = None
        
        # Detectar la mejor GPU disponible
        if device == "auto" and torch.cuda.is_available():
            best_gpu = 0
            max_memory = 0
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                if props.total_memory > max_memory:
                    max_memory = props.total_memory
                    best_gpu = i
            
            self.device = f"cuda:{best_gpu}"
            gpu_name = torch.cuda.get_device_name(best_gpu)
            gpu_memory = max_memory / 1e9
            print(f"🎯 Usando mejor GPU: {gpu_name}")
            print(f"💾 Memoria disponible: {gpu_memory:.1f} GB")
            
            if gpu_memory < 10:
                print("⚠️ GPU tiene menos de 10GB, puede fallar")
        else:
            self.device = device if device != "auto" else "cpu"
        
        try:
            print("🔄 Cargando Qwen2-Audio...")
            
            self.processor = AutoProcessor.from_pretrained(
                MODEL_ID, 
                trust_remote_code=True
            )
            
            if "cuda" in self.device:
                print(f"🚀 Cargando modelo en {self.device}")
                
                self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    MODEL_ID,
                    torch_dtype=torch.float16,
                    device_map=self.device,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    attn_implementation="eager"
                ).eval()
                
                print("✅ Qwen2-Audio cargado exitosamente en GPU")
            else:
                raise RuntimeError("❌ Este modelo requiere GPU con al menos 10GB")
                
        except Exception as e:
            print(f"❌ Error cargando Qwen2-Audio: {e}")
            self.model = None
            self.processor = None

    @torch.no_grad()
    def infer(self, wav_path: str) -> torch.Tensor:
        if self.model is None or self.processor is None:
            print(f"⚠️ Modelo no disponible para {wav_path}")
            return torch.tensor([0.5], dtype=torch.float32)
        
        try:
            # MEJORA 2: Mejor procesamiento de audio
            wav, sr = torchaudio.load(wav_path)
            if sr != self.target_sr:
                wav = torchaudio.functional.resample(wav, sr, self.target_sr)

            # Mantener estéreo si existe, convertir a mono solo si es necesario
            if wav.shape[0] > 1:
                audio_np = wav.mean(0).cpu().numpy()  # Promedio de canales
            else:
                audio_np = wav.squeeze(0).cpu().numpy()

            # MEJORA 3: Normalización de audio
            audio_np = audio_np / (max(abs(audio_np)) + 1e-8)  # Normalizar a [-1, 1]

            # MEJORA 3b: Manejar tokens especiales correctamente según el método
            if hasattr(self.processor, 'apply_chat_template'):
                # Con apply_chat_template, los tokens de audio se manejan automáticamente
                clean_prompt = IMPROVED_PROMPT.replace("<|audio_bos|><|AUDIO|><|audio_eos|>", "").strip()
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio": audio_np},
                            {"type": "text", "text": clean_prompt}
                        ]
                    }
                ]
                
                text_input = self.processor.apply_chat_template(
                    conversation, 
                    add_generation_prompt=True, 
                    tokenize=False
                )
                inputs = self.processor(
                    text=text_input,
                    audio=audio_np,
                    sampling_rate=self.target_sr,
                    return_tensors="pt"
                )
            else:
                # Sin apply_chat_template, necesitamos los tokens especiales explícitamente
                inputs = self.processor(
                    text=IMPROVED_PROMPT,  # Incluye <|audio_bos|><|AUDIO|><|audio_eos|>
                    audio=audio_np,
                    sampling_rate=self.target_sr,
                    return_tensors="pt"
                )
            
            inputs = inputs.to(self.device)

            # MEJORA 4: Configuración de generación más robusta (basada en testing exitoso)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=10,       # Suficiente para YES/NO
                    do_sample=False,         # Determinístico para consistencia
                    num_beams=1,            # Sin beam search para ahorrar memoria
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )

            # Extraer respuesta
            response_ids = generated_ids[:, inputs.input_ids.shape[1]:]
            answer = self.processor.batch_decode(
                response_ids, 
                skip_special_tokens=True
            )[0].lower().strip()

            # Limpiar memoria
            del inputs, generated_ids, response_ids
            torch.cuda.empty_cache()
            gc.collect()

            # MEJORA 5: Parsing más robusto de respuesta
            answer_clean = answer.replace(".", "").replace(",", "").strip()
            
            # Buscar indicadores positivos y negativos
            positive_indicators = ["yes", "sí", "talking", "voice", "speaking", "human", "conversation"]
            negative_indicators = ["no", "not", "none", "silence", "quiet", "instrumental"]
            
            # Contar indicadores
            pos_count = sum(1 for word in positive_indicators if word in answer_clean)
            neg_count = sum(1 for word in negative_indicators if word in answer_clean)
            
            if pos_count > neg_count:
                is_speech = 1
                confidence = "high" if pos_count > 1 else "medium"
            elif neg_count > pos_count:
                is_speech = 0
                confidence = "high" if neg_count > 1 else "medium"
            else:
                # Si hay empate o no hay indicadores claros, ser conservador
                is_speech = 1 if "yes" in answer_clean else 0
                confidence = "low"
            
            print(f"🔍 Audio: {Path(wav_path).name} → '{answer}' → {is_speech} ({confidence})")
            
            return torch.tensor([is_speech], dtype=torch.float32)
            
        except Exception as e:
            print(f"❌ Error procesando {Path(wav_path).name}: {e}")
            return torch.tensor([0.5], dtype=torch.float32)

    def cleanup(self):
        """Liberar memoria del modelo cuando no se necesite más"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        torch.cuda.empty_cache()
        gc.collect()
        print("🧹 Memoria de Qwen2-Audio liberada")