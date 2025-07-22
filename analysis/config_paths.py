"""
Configuración de rutas para análisis VAD
"""
import os
from pathlib import Path

# Ruta base del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
ANALYSIS_ROOT = PROJECT_ROOT / "analysis"

# Rutas de datos
DATA_ROOT = ANALYSIS_ROOT / "data"
RESULTS_CHIME = DATA_ROOT / "results_CHiME"
RESULTS_MUSAN = DATA_ROOT / "results_MUSAN"
GROUND_TRUTH_CHIME = DATA_ROOT / "ground_truth_chime"

# Rutas de figuras
FIGURES_ROOT = ANALYSIS_ROOT / "figures"
ORIGINAL_FIGURES = FIGURES_ROOT / "original_figures"

# Archivos CSV
RTF_DATA = DATA_ROOT / "rtf_data.csv"
RTF_BY_MODEL = DATA_ROOT / "rtf_by_model.csv"

print("Configuración de rutas cargada")
