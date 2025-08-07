#!/bin/bash

# Setup script for CNN autoencoder training
echo "Setting up CNN Autoencoder training environment..."

# Create models directory
MODELS_DIR="/nevis/riverside/data/sc5303/models"
echo "Creating models directory: ${MODELS_DIR}"
mkdir -p "${MODELS_DIR}"

# Make scripts executable
echo "Making scripts executable..."
chmod +x run_full_training.sh
chmod +x setup_training.sh

# Check Python dependencies
echo "Checking Python environment..."
python -c "import torch, numpy, pandas, matplotlib, seaborn, sklearn, tqdm; print('All required packages available')"

if [ $? -eq 0 ]; then
    echo "✓ Python environment is ready"
else
    echo "✗ Some Python packages are missing. Please install required packages:"
    echo "  pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn tqdm"
fi

# Check data directory
DATA_DIR="/nevis/riverside/data/sc5303/sbnd/offline_ad/pi0/npy"
if [ -d "${DATA_DIR}" ]; then
    NUM_FILES=$(ls "${DATA_DIR}"/*.npy 2>/dev/null | wc -l)
    echo "✓ Data directory found with ${NUM_FILES} NPY files"
else
    echo "✗ Data directory not found: ${DATA_DIR}"
    echo "  Please check the data path in the training script"
fi

# Show available disk space
echo ""
echo "Available disk space in models directory:"
df -h "${MODELS_DIR}"

echo ""
echo "Setup complete! You can now run:"
echo "  ./run_full_training.sh"
echo ""
echo "Or run training manually with custom parameters:"
echo "  python train_cnn_autoencoder.py --help"
