#!/bin/bash
# VAD-Benchmark Complete Installer
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_step() { echo -e "${BLUE}[STEP]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

ENV_NAME="vad_benchmark"
PYTHON_VERSION="3.9"

echo "🚀 VAD-Benchmark Complete Installation"
echo "======================================"

# Check requirements
print_step "Checking system requirements..."
if ! command -v conda &> /dev/null && ! command -v python3 &> /dev/null; then
    print_error "Neither conda nor python3 found"
    exit 1
fi
if ! command -v wget &> /dev/null; then
    print_error "wget required but not found"
    exit 1
fi
print_success "System requirements OK"

# Create environment
print_step "Creating conda environment: $ENV_NAME"
if command -v conda &> /dev/null; then
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
    eval "$(conda shell.bash hook)"
    conda activate $ENV_NAME
else
    python3 -m venv $ENV_NAME
    source $ENV_NAME/bin/activate
fi
print_success "Environment created"

# Install dependencies
print_step "Installing PyTorch with CUDA..."
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

print_step "Installing torchvision (PaSST-compatible)..."
pip install --no-cache-dir torchvision==0.22.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

print_step "Installing core dependencies..."
pip install numpy pandas scipy matplotlib seaborn scikit-learn librosa soundfile PyYAML tqdm webrtcvad faster-whisper transformers huggingface-hub datasets accelerate hear21passt torchlibrosa torchsummary einops

# Download models
print_step "Downloading model weights..."
mkdir -p models/{panns,epanns/E-PANNs/models,metadata}

if [[ ! -f "models/epanns/E-PANNs/models/checkpoint_closeto_.44.pt" ]]; then
    wget -O models/epanns/E-PANNs/models/checkpoint_closeto_.44.pt "https://zenodo.org/records/7939403/files/checkpoint_closeto_.44.pt?download=1"
fi

if [[ ! -f "models/panns/Cnn14_DecisionLevelAtt" ]]; then
    wget -O models/panns/Cnn14_DecisionLevelAtt "https://zenodo.org/records/3987831/files/Cnn14_DecisionLevelAtt_mAP=0.425.pth?download=1"
fi

if [[ ! -f "models/metadata/class_labels_indices.csv" ]]; then
    wget -O models/metadata/class_labels_indices.csv "https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/metadata/class_labels_indices.csv"
fi

# Create activation script
if command -v conda &> /dev/null; then
    cat > activate_vad.sh << EOL
#!/bin/bash
eval "\$(conda shell.bash hook)"
conda activate $ENV_NAME
echo "🚀 VAD-Benchmark environment activated"
EOL
else
    cat > activate_vad.sh << EOL
#!/bin/bash
source $ENV_NAME/bin/activate
echo "🚀 VAD-Benchmark environment activated"
EOL
fi
chmod +x activate_vad.sh

print_success "Installation complete!"
echo "🚀 Run: source activate_vad.sh"
