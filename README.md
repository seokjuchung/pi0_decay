# pi0_decay - CNN Autoencoder for Particle Physics Anomaly Detection

This repository contains code and resources for analyzing anomalous showers resulting from pi0 decay in offline anomaly detection studies. It includes a complete CNN autoencoder implementation converted from Jupyter notebook for production use.

pi0 decays are simulated using sbndcode. Take LArCV files as input.

## Overview

The project focuses on identifying and studying anomalous electromagnetic showers in particle physics data, specifically those originating from neutral pion (pi0) decay. The analysis is performed offline using custom algorithms and CNN-based autoencoders.

## CNN Autoencoder Implementation

### Files Overview

#### Main Scripts
- `train_cnn_autoencoder.py` - Main training script (converted from Jupyter notebook)
- `analyze_test_results.py` - Analysis script for trained models
- `setup_training.sh` - Setup script to prepare environment
- `quick_train.sh` - Interactive training script with preset configurations
- `run_full_training.sh` - Batch script for full dataset training

#### Models Storage
All trained models and results are saved to: `/nevis/riverside/data/sc5303/models/`

### Quick Start

1. **Setup environment:**
   ```bash
   ./setup_training.sh
   ```

2. **Run interactive training:**
   ```bash
   ./quick_train.sh
   ```

3. **Or run specific configuration:**
   ```bash
   # Quick test (5 files, 20 epochs)
   python train_cnn_autoencoder.py --max_files 5 --num_epochs 20 --save_plots
   
   # Full dataset training
   python train_cnn_autoencoder.py --num_epochs 100 --batch_size 32 --save_plots
   ```

### Training Parameters

#### Data Parameters
- `--data_path`: Path pattern for NPY data files (default: `/nevis/riverside/data/sc5303/sbnd/offline_ad/pi0/npy/*.npy`)
- `--max_points`: Maximum points per event (default: 10000)
- `--min_points`: Minimum points per event (default: 10)
- `--max_files`: Limit number of files for testing (default: None = all files)

#### Model Parameters
- `--latent_dim`: Latent dimension size (default: 128)
- `--batch_size`: Training batch size (default: 16)
- `--learning_rate`: Learning rate (default: 1e-3)
- `--num_epochs`: Number of training epochs (default: 100)

#### System Parameters
- `--device`: Device to use (auto, cpu, cuda) (default: auto)
- `--num_workers`: Data loader workers (default: 4)
- `--output_dir`: Output directory (default: `/nevis/riverside/data/sc5303/models/cnn_autoencoder_output`)
- `--save_plots`: Save training and analysis plots

### Output Files

After training, the following files are saved in the output directory:

#### Model Files
- `best_cnn_autoencoder.pth` - Best model checkpoint during training
- `final_cnn_autoencoder.pth` - Final model with complete results

#### Data Files
- `test_events.pkl` - Test events for later analysis
- `test_events_info.json` - Test dataset metadata
- `test_losses.npy` - Test reconstruction losses
- `config.json` - Training configuration

#### Analysis Files
- `training_history.png` - Training loss curves
- `event_size_distribution.png` - Data analysis plots
- `test_loss_analysis.png` - Test loss distribution
- `anomaly_detection_analysis.png` - Anomaly detection thresholds
- `event_size_analysis.png` - Event size vs loss correlation
- `training_analysis.png` - Training convergence analysis
- `analysis_summary.txt` - Comprehensive text summary

### Model Architecture

The CNN autoencoder uses:
- **Input**: Variable-length sequences of [x, y, z, energy] points, sorted by energy
- **Encoder**: 4 convolutional blocks with max pooling and dropout
- **Bottleneck**: Fully connected layers with configurable latent dimension
- **Decoder**: 4 transposed convolutional blocks with upsampling
- **Output**: Reconstructed input sequences

### Anomaly Detection

The trained model can detect anomalies using reconstruction loss:
- **Threshold**: 95th percentile of test losses (recommended)
- **Expected false positive rate**: ~5%
- **Usage**: Events with reconstruction loss > threshold are flagged as anomalous

### Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn, tqdm

### Notes

- Models are automatically saved to `/nevis/riverside/data/sc5303/models/`
- Training uses early stopping with patience=15 epochs
- Data is automatically normalized using StandardScaler
- Variable-length sequences are handled with dynamic padding and masking
- GPU training is automatically enabled if available

## License

Written by Copilot