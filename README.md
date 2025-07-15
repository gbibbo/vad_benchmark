# Privacy-Preserving VAD Benchmark

> **Project status:** Under construction  
> **Expected completion:** 18/07/2025 at 23:59 (UK time)

This repository contains code and resources for a benchmark of voice activity detection (VAD) with a privacy‑preserving focus in domestic environments. It is currently under development and will be finalized by the date above.

## Repository structure

- **src/**: Wrappers for various VAD models (AST, PANNs, PaSST, Silero, WebRTC, Whisper, Pengi)  
- **scripts/**: Evaluation and inference scripts (`run_evaluation.py`, `run_all_scenarios_unified_FIXED.py`, etc.)  
- **models/**: Checkpoints and pretrained model resources  
- **config/**: Configuration files (e.g., `config_all_scenarios.yaml`)  
- **data/**: Test data and ground truth for evaluation  
- **results/**: Generated masks, metrics, plots, and reports  
- **notebooks/**: Exploratory analysis and interactive visualizations  

## Next steps

1. Complete installation and usage documentation  
2. Add execution examples and test cases  
3. Publish preliminary results and comparative analysis  
4. Integrate tests on edge devices  
5. Review and clean up code before final release  

## Current progress

- [x] Basic wrappers implemented  
- [x] Core evaluation script (`run_evaluation.py`)  
- [ ] Script documentation  
- [ ] Usage examples  
- [ ] Prepare beta release  

---

> **Note:** This README is a work in progress and will be updated. The project will be completed by **18/07/2025 at 23:59 (UK time)**.  
