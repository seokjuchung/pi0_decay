#!/usr/bin/env python3
"""
CNN Autoencoder for Particle Physics Anomaly Detection
Converted from Jupyter notebook to handle full dataset processing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import glob
import os
import pickle
from tqdm import tqdm
import time
import argparse
import json
from multiprocessing import Pool, cpu_count

# Set up argument parsing
parser = argparse.ArgumentParser(description='Train CNN Autoencoder for Anomaly Detection')
parser.add_argument('--data_path', type=str, 
                    default='/nevis/riverside/data/sc5303/sbnd/offline_ad/pi0/npy/*.npy',
                    help='Path pattern for NPY data files')
parser.add_argument('--output_dir', type=str, default='/nevis/riverside/data/sc5303/models/cnn_autoencoder_output',
                    help='Directory to save results')
parser.add_argument('--max_points', type=int, default=10000,
                    help='Maximum number of points per event')
parser.add_argument('--min_points', type=int, default=10,
                    help='Minimum number of points per event')
parser.add_argument('--max_files', type=int, default=None,
                    help='Limit number of files (None for all)')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=1e-3,
                    help='Learning rate')
parser.add_argument('--num_epochs', type=int, default=100,
                    help='Number of training epochs')
parser.add_argument('--latent_dim', type=int, default=128,
                    help='Latent dimension')
parser.add_argument('--device', type=str, default='auto',
                    help='Device to use (auto, cpu, cuda)')
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of data loader workers')
parser.add_argument('--save_plots', action='store_true',
                    help='Save training plots')

def setup_device(device_arg):
    """Setup compute device"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    
    return device

def setup_reproducibility():
    """Set random seeds for reproducibility"""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

def load_npy_file(file_path, config):
    """Load and process a single NPY file"""
    try:
        data = np.load(file_path)
        print(f"Loading {os.path.basename(file_path)}: shape {data.shape}")
        
        if data.shape[1] != 5:  # [event, x, y, z, energy]
            print(f"  Warning: Expected 5 columns, got {data.shape[1]}")
            return []
        
        # Group by event
        df = pd.DataFrame(data, columns=['event', 'x', 'y', 'z', 'energy'])
        events = []
        
        for event_id in df['event'].unique():
            event_data = df[df['event'] == event_id][['x', 'y', 'z', 'energy']].values
            
            # Filter by size
            if config['min_points'] <= len(event_data) <= config['max_points']:
                # Sort by energy (highest to lowest) - creates structure for CNN
                sorted_indices = np.argsort(-event_data[:, 3])
                sorted_event = event_data[sorted_indices]
                events.append(sorted_event)
        
        print(f"  Loaded {len(events)} valid events")
        return events
        
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return []

def load_all_data(file_paths, config):
    """Load all data files and return list of events"""
    print("Loading all data files...")
    all_events = []
    
    for file_path in tqdm(file_paths, desc="Loading files"):
        events = load_npy_file(file_path, config)
        all_events.extend(events)
    
    print(f"\nTotal events loaded: {len(all_events)}")
    
    # Analyze event sizes
    if all_events:
        event_sizes = [len(event) for event in all_events]
        print(f"Event size statistics:")
        print(f"  Min: {min(event_sizes)} points")
        print(f"  Max: {max(event_sizes)} points") 
        print(f"  Mean: {np.mean(event_sizes):.1f} points")
        print(f"  Median: {np.median(event_sizes):.1f} points")
        
        # Save size distribution plot
        if config.get('save_plots', False):
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.hist(event_sizes, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Number of Points per Event')
            plt.ylabel('Frequency')
            plt.title('Event Size Distribution')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 3, 2)
            plt.boxplot(event_sizes)
            plt.ylabel('Number of Points')
            plt.title('Event Size Box Plot')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 3, 3)
            plt.hist(event_sizes, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Number of Points per Event')
            plt.ylabel('Frequency (Log Scale)')
            plt.title('Event Size Distribution (Log Scale)')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(config['output_dir'], 'event_size_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    return all_events

class ParticleEventDataset(Dataset):
    """Dataset class for particle physics events"""
    
    def __init__(self, events, scaler=None, fit_scaler=True):
        self.events = events
        self.scaler = scaler
        
        if self.scaler is None and fit_scaler:
            # Fit scaler on all data points
            all_points = np.vstack(events)
            self.scaler = StandardScaler()
            self.scaler.fit(all_points)
            print(f"Fitted scaler on {len(all_points)} total points")
            print(f"Feature means: {self.scaler.mean_}")
            print(f"Feature stds: {self.scaler.scale_}")
        
        # Normalize events
        self.normalized_events = []
        for event in events:
            if self.scaler is not None:
                normalized_event = self.scaler.transform(event)
            else:
                normalized_event = event
            self.normalized_events.append(torch.FloatTensor(normalized_event))
    
    def __len__(self):
        return len(self.normalized_events)
    
    def __getitem__(self, idx):
        return self.normalized_events[idx]

def collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    # Pad sequences to the same length
    padded = pad_sequence(batch, batch_first=True, padding_value=0)
    return padded

class CNN1DAutoEncoder(nn.Module):
    """1D CNN Autoencoder for particle physics events"""
    
    def __init__(self, input_dim=4, latent_dim=128, max_seq_length=10000):
        super(CNN1DAutoEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.max_seq_length = max_seq_length
        
        # Encoder
        self.encoder = nn.Sequential(
            # First conv block
            nn.Conv1d(input_dim, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            
            # Second conv block
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            
            # Third conv block
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            
            # Fourth conv block
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(latent_dim // 4),
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(512 * (latent_dim // 4), latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(latent_dim, 512 * (latent_dim // 4)),
            nn.ReLU(inplace=True),
        )
        
        # Decoder  
        self.decoder = nn.Sequential(
            # First deconv block
            nn.ConvTranspose1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Dropout(0.1),
            
            # Second deconv block
            nn.ConvTranspose1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Dropout(0.1),
            
            # Third deconv block
            nn.ConvTranspose1d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Dropout(0.1),
            
            # Output layer
            nn.ConvTranspose1d(64, input_dim, kernel_size=7, padding=3),
        )
        
    def encode(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = x.transpose(1, 2)  # -> (batch_size, input_dim, seq_len)
        
        encoded = self.encoder(x)  # -> (batch_size, 512, latent_dim//4)
        
        # Flatten for bottleneck
        batch_size = encoded.size(0)
        encoded = encoded.view(batch_size, -1)
        
        # Pass through bottleneck
        bottleneck = self.bottleneck(encoded)
        
        return bottleneck
    
    def decode(self, bottleneck, target_length):
        # Reshape back to conv format
        batch_size = bottleneck.size(0)
        reshaped = bottleneck.view(batch_size, 512, self.latent_dim // 4)
        
        # Pass through decoder
        decoded = self.decoder(reshaped)  # -> (batch_size, input_dim, some_length)
        
        # Interpolate to target length
        if decoded.size(2) != target_length:
            decoded = F.interpolate(decoded, size=target_length, mode='linear', align_corners=False)
        
        decoded = decoded.transpose(1, 2)  # -> (batch_size, seq_len, input_dim)
        
        return decoded
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Encode
        bottleneck = self.encode(x)
        
        # Decode to original length
        reconstructed = self.decode(bottleneck, seq_len)
        
        return reconstructed

def create_padding_mask(x, padding_value=0.0):
    """Create mask for padded sequences"""
    mask = ~(x.abs().sum(dim=-1) == padding_value)
    return mask

def masked_mse_loss(output, target, mask=None):
    """MSE loss that ignores padded positions"""
    if mask is None:
        return F.mse_loss(output, target)
    
    # Expand mask to match feature dimensions
    mask = mask.unsqueeze(-1)
    
    # Apply mask
    masked_output = output * mask.float()
    masked_target = target * mask.float()
    
    # Calculate loss only on non-padded positions
    loss = F.mse_loss(masked_output, masked_target, reduction='none')
    
    # Sum over features and sequence, then average over valid positions
    loss = loss.sum() / mask.sum().float()
    
    return loss

def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Create mask for padded positions
        mask = create_padding_mask(batch)
        
        # Forward pass
        reconstructed = model(batch, mask=mask)
        
        # Calculate loss
        loss = masked_mse_loss(reconstructed, batch, mask)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({'Loss': f'{loss.item():.6f}'})
    
    return total_loss / num_batches

def validate_epoch(model, dataloader, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            batch = batch.to(device)
            
            # Create mask
            mask = create_padding_mask(batch)
            
            # Forward pass
            reconstructed = model(batch, mask=mask)
            
            # Calculate loss
            loss = masked_mse_loss(reconstructed, batch, mask)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def evaluate_model(model, dataloader, device, scaler=None):
    """Evaluate model and return reconstruction losses"""
    model.eval()
    all_losses = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch.to(device)
            mask = create_padding_mask(batch)
            
            # Get reconstructions
            reconstructed = model(batch, mask=mask)
            
            # Calculate per-sample losses
            for i in range(batch.size(0)):
                sample_mask = mask[i]
                orig = batch[i][sample_mask]
                recon = reconstructed[i][sample_mask]
                
                sample_loss = F.mse_loss(recon, orig).item()
                all_losses.append(sample_loss)
    
    return np.array(all_losses)

def save_training_plots(train_losses, val_losses, output_dir):
    """Save training history plots"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.semilogy(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.semilogy(val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training History (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    # Loss difference
    if len(train_losses) > 1:
        train_diff = np.diff(train_losses)
        val_diff = np.diff(val_losses)
        plt.plot(train_diff, label='Train Loss Change', color='blue', alpha=0.7)
        plt.plot(val_diff, label='Val Loss Change', color='red', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss Change')
        plt.title('Loss Change per Epoch')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    args = parser.parse_args()
    
    # Create output directory and ensure parent directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Also create a models directory structure if it doesn't exist
    models_base_dir = '/nevis/riverside/data/sc5303/models'
    os.makedirs(models_base_dir, exist_ok=True)
    print(f"Models will be saved to: {args.output_dir}")
    
    # Setup
    setup_reproducibility()
    device = setup_device(args.device)
    
    # Configuration
    config = {
        'data_path': args.data_path,
        'max_points': args.max_points,
        'min_points': args.min_points,
        'max_files': args.max_files,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'latent_dim': args.latent_dim,
        'test_split': 0.2,
        'val_split': 0.1,
        'num_workers': args.num_workers,
        'feature_dim': 4,
        'output_dir': args.output_dir,
        'save_plots': args.save_plots
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Save configuration
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Find available data files
    data_files = sorted(glob.glob(config['data_path']))
    if config['max_files']:
        data_files = data_files[:config['max_files']]
    
    print(f"\nFound {len(data_files)} data files:")
    for file in data_files:
        file_size = os.path.getsize(file) / (1024**2)  # MB
        print(f"  {os.path.basename(file)} ({file_size:.1f} MB)")
    
    # Load data
    print("\nStarting data loading...")
    start_time = time.time()
    all_events = load_all_data(data_files, config)
    load_time = time.time() - start_time
    print(f"Data loading completed in {load_time:.2f} seconds")
    
    if not all_events:
        print("No events loaded - exiting")
        return
    
    # Split data
    print("\nSplitting data...")
    train_events, temp_events = train_test_split(
        all_events, test_size=config['test_split'] + config['val_split'], 
        random_state=42
    )
    
    val_events, test_events = train_test_split(
        temp_events, test_size=config['test_split'] / (config['test_split'] + config['val_split']),
        random_state=42
    )
    
    print(f"Dataset splits:")
    print(f"  Train: {len(train_events)} events")
    print(f"  Validation: {len(val_events)} events")
    print(f"  Test: {len(test_events)} events")
    
    # Save test event information for later analysis
    test_info = {
        'num_test_events': len(test_events),
        'test_event_sizes': [len(event) for event in test_events],
        'test_split_seed': 42,
        'files_used': [os.path.basename(f) for f in data_files]
    }
    
    with open(os.path.join(args.output_dir, 'test_events_info.json'), 'w') as f:
        json.dump(test_info, f, indent=2)
    
    # Save test events for later evaluation
    with open(os.path.join(args.output_dir, 'test_events.pkl'), 'wb') as f:
        pickle.dump(test_events, f)
    
    print(f"Saved test event information to {args.output_dir}")
    
    # Create datasets
    train_dataset = ParticleEventDataset(train_events, fit_scaler=True)
    val_dataset = ParticleEventDataset(val_events, scaler=train_dataset.scaler, fit_scaler=False)
    test_dataset = ParticleEventDataset(test_events, scaler=train_dataset.scaler, fit_scaler=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], 
        shuffle=True, collate_fn=collate_fn, num_workers=config['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], 
        shuffle=False, collate_fn=collate_fn, num_workers=config['num_workers']
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'],
        shuffle=False, collate_fn=collate_fn, num_workers=config['num_workers']
    )
    
    print("Data loaders created successfully")
    
    # Create model
    model = CNN1DAutoEncoder(
        input_dim=config['feature_dim'],
        latent_dim=config['latent_dim'],
        max_seq_length=config['max_points']
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel created successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / (1024**2):.2f} MB")
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=5, verbose=True
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    
    train_losses = []
    val_losses = []
    
    training_start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        
        # Validation
        val_loss = validate_epoch(model, val_loader, device)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print results
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}")
        
        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
                'scaler': train_dataset.scaler,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, os.path.join(args.output_dir, 'best_cnn_autoencoder.pth'))
            
            print(f"New best model saved! Val Loss: {val_loss:.6f}")
        else:
            patience_counter += 1
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement!")
            break
        
        # Save intermediate results every 10 epochs
        if (epoch + 1) % 10 == 0:
            if config['save_plots']:
                save_training_plots(train_losses, val_losses, args.output_dir)
    
    total_training_time = time.time() - training_start_time
    print(f"\nTraining completed in {total_training_time:.2f} seconds ({total_training_time/60:.1f} minutes)")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Epochs trained: {len(train_losses)}")
    
    # Save final training plots
    if config['save_plots']:
        save_training_plots(train_losses, val_losses, args.output_dir)
    
    # Load best model for evaluation
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_cnn_autoencoder.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation on test set
    print("Evaluating on test set...")
    test_losses = evaluate_model(model, test_loader, device, scaler=train_dataset.scaler)
    
    print(f"\nTest Set Evaluation:")
    print(f"Number of test samples: {len(test_losses)}")
    print(f"Mean reconstruction loss: {test_losses.mean():.6f}")
    print(f"Std reconstruction loss: {test_losses.std():.6f}")
    print(f"Min reconstruction loss: {test_losses.min():.6f}")
    print(f"Max reconstruction loss: {test_losses.max():.6f}")
    
    # Save final model and results
    final_results = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'scaler': train_dataset.scaler,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_losses': test_losses.tolist(),
        'best_val_loss': best_val_loss,
        'test_loss_stats': {
            'mean': float(test_losses.mean()),
            'std': float(test_losses.std()),
            'min': float(test_losses.min()),
            'max': float(test_losses.max())
        },
        'training_time': total_training_time,
        'total_params': total_params,
        'anomaly_threshold_95': float(np.percentile(test_losses, 95))
    }
    
    torch.save(final_results, os.path.join(args.output_dir, 'final_cnn_autoencoder.pth'))
    
    # Save test losses separately for easy access
    np.save(os.path.join(args.output_dir, 'test_losses.npy'), test_losses)
    
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - best_cnn_autoencoder.pth: Best model checkpoint")
    print(f"  - final_cnn_autoencoder.pth: Final model with all results")
    print(f"  - test_events.pkl: Test events for later analysis")
    print(f"  - test_events_info.json: Test dataset information")
    print(f"  - test_losses.npy: Test reconstruction losses")
    print(f"  - config.json: Training configuration")
    if config['save_plots']:
        print(f"  - training_history.png: Training plots")
        print(f"  - event_size_distribution.png: Data analysis plots")
    
    # Anomaly detection summary
    threshold_95 = np.percentile(test_losses, 95)
    print(f"\nAnomaly Detection Summary:")
    print(f"  Recommended threshold (95th percentile): {threshold_95:.6f}")
    print(f"  Expected anomaly rate: ~5%")
    print(f"  Use this threshold for real-time anomaly detection")

if __name__ == "__main__":
    main()
