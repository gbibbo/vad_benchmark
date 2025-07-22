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

## CHiME-Home Dataset Setup

To reproduce the exact paper results, you need to download and set up the CHiME-Home dataset:

### 1. Download CHiME-Home Dataset

The CHiME-Home dataset is available from the CHiME Challenge website:

```bash
# Create dataset directory
mkdir -p datasets/chime/chunks

# Download CHiME-Home dataset
# Visit: https://www.chimehome.org/ 
# Or use the direct download link provided by CHiME organizers
# Extract audio files to: datasets/chime/chunks/

# Expected structure:
# datasets/chime/chunks/
# ├── CR_lounge_220110_0731.s0_chunk0.wav
# ├── CR_lounge_220110_0731.s0_chunk1.wav
# ├── ...
# └── [additional 4-second audio chunks at 16kHz]
```

### 2. Alternative: Use Download Script

If available, you can use the provided download script:

```bash
# Make download script executable
chmod +x download_chime.sh

# Download dataset automatically
./download_chime.sh

# Verify dataset structure
ls -la datasets/chime/chunks/ | head -10
```

### 3. Dataset Requirements

- **Format**: WAV files, 16kHz sample rate
- **Duration**: 4-second chunks
- **Size**: ~1946 files for full evaluation
- **Scenarios**: CMF (Children, Mother, Father) and CMFV (+ Visitors)
- **Ground Truth**: Included in `ground_truth/chime/cmf.csv` and `ground_truth/chime/cmfv.csv`

## Reproducing Paper Results

### 1. Run Paper Evaluations
```bash
# Human speech detection (CMF scenario) - Table results from paper
python scripts/run_evaluation.py --config configs/config_chime_cmf.yaml

# Broad vocal content detection (CMFV scenario)  
python scripts/run_evaluation.py --config configs/config_chime_cmfv.yaml

# Run all models on both scenarios
python scripts/run_all_scenarios.py --config configs/config_paper_full.yaml
```

### 2. Results Location
- Individual metrics: `results/metrics_[model].json`
- Comparison plots: `results/comparison_all_models.png`
- Performance logs: `results/evaluation_[timestamp].log`

**Ground truth annotations** are included in `ground_truth/chime/`

## Analysis Suite

This repository includes a comprehensive analysis suite for in-depth VAD performance evaluation:

### 1. Run Analysis Scripts

```bash
# Navigate to analysis directory
cd analysis/scripts/

# Run complete VAD performance analysis
python analyze_vad_results.py

# Run parameter count vs performance analysis  
python analyze_vad_parameters.py

# Compare ground truth versions (if needed)
python compare_gt_old_new.py
```

### 2. Generated Analysis Outputs

The analysis scripts generate publication-ready figures and metrics:

```
analysis/data/Figures/
├── f1_vs_threshold_comparison.png          # F1 score comparisons
├── accuracy_vs_threshold_comparison.png    # Accuracy comparisons
├── roc_curves_comparison.png               # ROC curve analysis
├── pr_curves_comparison.png                # Precision-Recall curves
├── performance_vs_speed_comparison.png     # F1 vs RTF scatter plots
├── parameter_count_vs_performance_*.png    # Model size vs performance
├── performance_summary_cmf.csv             # CMF scenario metrics
├── performance_summary_cmfv.csv            # CMFV scenario metrics
└── parameter_count_analysis.csv            # Efficiency analysis
```

### 3. Key Analysis Features

- **Comparative Analysis**: CMF vs CMFV scenario performance
- **Speed Analysis**: Real-Time Factor (RTF) vs F1-score relationships  
- **Efficiency Analysis**: Parameter count vs performance trade-offs
- **Threshold Analysis**: Performance across different VAD thresholds
- **ROC/PR Curves**: Detailed classification performance metrics

### 4. Analysis Results Summary

The analysis reveals key findings:

- **Best Overall Performance**: PaSST (F1=0.861) and AST (F1=0.860) for CMF
- **Most Efficient**: Silero (0.5M params) and WebRTC (0.01M params)
- **Speed Leaders**: WebRTC (RTF=0.002) and AST (RTF=0.039)
- **Scenario Differences**: CMFV generally easier than CMF for all models

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
├── analysis/                    # Analysis suite
│   ├── scripts/                   # Analysis scripts
│   ├── data/                     # Results and ground truth data
│   └── figures/                  # Generated plots and figures
├── ground_truth/               # Paper ground truth annotations  
│   └── chime/                 # CHiME-Home labels (CMF/CMFV)
├── datasets/                   # Dataset directory
│   └── chime/chunks/           # CHiME-Home audio files (download required)
├── src/wrappers/              # VAD model implementations
├── scripts/                   # Evaluation scripts
├── models/                    # Downloaded model weights
└── results/                   # Output metrics and plots
```

## System Requirements

- **Python**: 3.9+
- **Storage**: 2GB (models + dependencies)
- **Memory**: 4GB RAM recommended
- **OS**: Linux, macOS, Windows (WSL supported)

The installer automatically handles all dependencies including PyTorch (CPU version for stability).

## Troubleshooting

### Common Issues

1. **Installation fails on Windows/WSL**: The installer has been updated to handle cross-platform compatibility
2. **PaSST wrapper errors**: Path issues have been resolved in the latest version
3. **Missing CHiME dataset**: Follow the dataset setup instructions above
4. **Analysis scripts fail**: Ensure you're in the correct directory and have run evaluations first

### Getting Help

- Check the installation test: `python test_installation.py`
- Verify dataset structure: `ls datasets/chime/chunks/ | wc -l` (should show ~1946 files)
- Review evaluation logs in `results/evaluation_*.log`

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
