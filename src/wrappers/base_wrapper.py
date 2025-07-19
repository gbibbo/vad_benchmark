"""
Base wrapper class for VAD models
"""

class BaseVADWrapper:
    """Base class for all VAD wrappers"""
    
    def __init__(self, device="cpu"):
        self.device = device
    
    def infer(self, wav_path: str):
        """
        Perform inference on audio file
        
        Args:
            wav_path: Path to audio file
            
        Returns:
            Tensor or array with VAD predictions
        """
        raise NotImplementedError("Subclasses must implement infer method")
