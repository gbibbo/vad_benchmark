# VAD-Benchmark: Voice Activity Detection Evaluation Framework

This repository contains the code and evaluation framework supporting the paper:

**"Privacy-Preserving Voice Activity Detection: Evaluating AI Model Performance on Domestic Audio"**  
*Gabriel Bibbo, Arshdeep Singh, Mark D. Plumbley*  
Centre for Vision Speech & Signal Processing (CVSSP), University of Surrey, UK  
Detection and Classification of Acoustic Scenes and Events (DCASE) 2025

## Quick Start

```bash
# 1. Install everything (creates environment + downloads model weights ~500MB)
./install.sh

# 2. Activate environment  
source activate_vad.sh

# 3. Test installation
python test_installation.py

# 4. Run demo evaluation
python scripts/run_evaluation.py --config configs/config_demo.yaml
```

## Reproducing Paper Results

To reproduce the exact results from the paper:

### 1. Obtain CHiME-Home Dataset
- Download the CHiME-Home audio chunks (4-second segments, 16kHz)
- Place audio files in: `datasets/chime/chunks/`

### 2. Run Paper Evaluations
```bash
# Human speech detection (CMF scenario) - Table results from paper
python scripts/run_evaluation.py --config configs/config_chime_cmf.yaml

# Broad vocal content detection (CMFV scenario)  
python scripts/run_evaluation.py --config configs/config_chime_cmfv.yaml

# Run all models on both scenarios
python scripts/run_all_scenarios.py --config configs/config_paper_full.yaml
```

### 3. Results Location
- Individual metrics: `results/metrics_[model].json`
- Comparison plots: `results/comparison_all_models.png`
- Performance logs: `results/evaluation_[timestamp].log`

**Ground truth annotations** are included in `ground_truth/chime/`

## Supported Models

The framework evaluates 8 VAD models from the paper across 4 architectural families:

| Family | Models | Paper Results (F1-Score) |
|--------|--------|-------------------------|
| **Lightweight VAD** | Silero, WebRTC | 0.806, 0.708 |
| **AudioSet Pre-trained** | PANNs, EPANNs, AST, PaSST | 0.848, 0.847, 0.860, 0.861 |
| **Speech Recognition** | Whisper-Tiny, Whisper-Small | 0.668, 0.654 |

*Results shown for CMF scenario (human speech detection)*

## Project Structure

```
vad_benchmark/
├── install.sh                    # Automatic installer
├── configs/                     # Evaluation configurations
│   ├── config_demo.yaml           # Demo with test data
│   ├── config_chime_cmf.yaml      # Paper: Human speech scenario
│   └── config_chime_cmfv.yaml     # Paper: Broad vocal content
├── ground_truth/               # Paper ground truth annotations  
│   └── chime/                 # CHiME-Home labels (CMF/CMFV)
├── src/wrappers/              # VAD model implementations
├── scripts/                   # Evaluation scripts
├── models/                    # Downloaded model weights
└── results/                   # Output metrics and plots
```

## System Requirements

- **Python**: 3.9+
- **Storage**: 2GB (models + dependencies)
- **Memory**: 4GB RAM recommended
- **OS**: Linux, macOS, Windows

The installer automatically handles all dependencies including PyTorch (CPU version for stability).

## Citation

```bibtex
@inproceedings{bibbo2025privacy,
  title={Privacy-Preserving Voice Activity Detection: Evaluating AI Model Performance on Domestic Audio},
  author={Bibbo, Gabriel and Singh, Arshdeep and Plumbley, Mark D.},
  booktitle={Detection and Classification of Acoustic Scenes and Events 2025},
  year={2025},
  address={Barcelona, Spain}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Repository**: https://github.com/gbibbo/vad_benchmark  
**Paper**: DCASE 2025 Conference Proceedings
