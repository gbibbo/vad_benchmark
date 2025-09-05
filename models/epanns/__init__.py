from .models import Cnn14_pruned

# Re-exporta utilidades que el wrapper puede importar desde 'models.epanns'
try:
    from .utils import move_data_to_device
except Exception:
    def move_data_to_device(*args, **kwargs):
        raise ImportError("Expected 'move_data_to_device' in models/epanns/utils.py")

# Opcional: otras utilidades frecuentes (si existen no fallará, si no, define stubs claros)
try:
    from .utils import interpolate  # si está disponible
except Exception:
    def interpolate(*args, **kwargs):
        raise ImportError("Expected 'interpolate' in models/epanns/utils.py")

try:
    from .utils import pad_frame_sequence  # si está disponible
except Exception:
    def pad_frame_sequence(*args, **kwargs):
        raise ImportError("Expected 'pad_frame_sequence' in models/epanns/utils.py")

__all__ = ["Cnn14_pruned", "move_data_to_device", "interpolate", "pad_frame_sequence"]
