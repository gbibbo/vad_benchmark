import os, glob, importlib.util

BASE = os.path.dirname(__file__)

def _load_cnn14_pruned():
    # Busca el archivo que define Cnn14_pruned en distintas ubicaciones probables
    patterns = [
        "E-PANNs/**/cnn14_pruned.py",
        "E-PANNs/**/cnn14*.py",
        "E-PANNs/**/cnn_pruned*.py",
        "E-PANNs/**/model*.py",
        "E-PANNs/**/models.py",
    ]
    for pat in patterns:
        for path in glob.glob(os.path.join(BASE, pat), recursive=True):
            spec = importlib.util.spec_from_file_location("epanns_dynamic", path)
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
            except Exception:
                continue
            if hasattr(mod, "Cnn14_pruned"):
                return mod.Cnn14_pruned
    raise ImportError(
        "Cnn14_pruned not found under 'models/epanns/E-PANNs/**'. "
        "Verifica que la carpeta E-PANNs copiada incluya el código fuente."
    )

Cnn14_pruned = _load_cnn14_pruned()
__all__ = ["Cnn14_pruned"]
