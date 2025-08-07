# CNN Autoencoder Training Scripts

This directory contains all the scripts needed to train and analyze the CNN autoencoder for particle physics anomaly detection.

## Files

### Python Scripts
- `train_cnn_autoencoder.py` - Main training script (converted from Jupyter notebook)
- `analyze_test_results.py` - Comprehensive analysis script for trained models

### Shell Scripts
- `setup_training.sh` - Setup environment and check dependencies
- `quick_train.sh` - Interactive training with preset configurations
- `run_full_training.sh` - Batch script for full dataset training
- `safe_train.sh` - Safe training with CUDA error handling and CPU fallback

## Quick Start

1. **Setup environment:**
   ```bash
   ./setup_training.sh
   ```

2. **Run safe training (recommended for CUDA issues):**
   ```bash
   ./safe_train.sh
   ```

3. **Run interactive training:**
   ```bash
   ./quick_train.sh
   ```

4. **Or run specific configuration:**
   ```bash
   # Quick test (5 files, 20 epochs)
   python train_cnn_autoencoder.py --max_files 5 --num_epochs 20 --save_plots
   
   # Full dataset training
   python train_cnn_autoencoder.py --num_epochs 100 --batch_size 32 --save_plots
   ```

## Output Location

All models and results are saved to: `/nevis/riverside/data/sc5303/models/`

## Key Features

- **Full Dataset Support**: Handles all NPY files by default
- **GPU Support**: Automatic CUDA detection and usage
- **Test Files Saved**: Automatically saves test events list for later analysis
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Comprehensive Analysis**: Automatic generation of analysis plots and reports

## Usage Examples

```bash
# Quick test run
python train_cnn_autoencoder.py --max_files 5 --num_epochs 10 --save_plots

# Production training
python train_cnn_autoencoder.py --num_epochs 100 --batch_size 32 --num_workers 8 --save_plots

# Run analysis on saved results
python analyze_test_results.py --results_dir /path/to/results --output_plots
```

## Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, tqdm
