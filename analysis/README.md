# VAD Analysis Scripts

Este directorio contiene scripts de análisis y resultados de evaluación para modelos VAD.

## Estructura

```
analysis/
├── config_paths.py          # Configuración de rutas
├── scripts/                 # Scripts de análisis
│   ├── analyze_vad_results.py      # Análisis principal de resultados
│   ├── analyze_vad_parameters.py   # Análisis parámetros vs rendimiento  
│   ├── compare_gt_old_new.py       # Comparación ground truth
│   └── generate_new_gt.py          # Generación ground truth
├── data/                    # Datos y resultados
│   ├── results_CHiME/              # Resultados evaluación CHiME
│   ├── results_MUSAN/              # Resultados evaluación MUSAN
│   ├── ground_truth_chime/         # Ground truth CHiME
│   └── *.csv                       # Datos RTF y métricas
└── figures/                 # Figuras generadas
    └── original_figures/           # Figuras originales
```

## Uso

```bash
cd analysis/scripts/
python analyze_vad_results.py
python analyze_vad_parameters.py
```
