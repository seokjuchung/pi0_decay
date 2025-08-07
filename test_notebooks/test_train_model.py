#!/usr/bin/env python3
"""
Graph Autoencoder Training Script
Converted from test_train_model.ipynb
"""

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import pandas as pd
import os
import torch.nn as nn
import torch.optim as optim
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial

# Device setup
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using {torch.cuda.device_count()} GPU(s)")
    if torch.cuda.device_count() > 1:
        print("Multiple GPUs detected - will use DataParallel")
else:
    device = torch.device('cpu')
    print("CUDA not available, using CPU")


def create_graph_data(event_data):
    """Create graph data for each event - optimized version"""
    # Node features: x, y, z, energy coordinates
    if isinstance(event_data, pd.DataFrame):
        x = torch.tensor(event_data[['x', 'y', 'z', 'energy']].values, dtype=torch.float)
    else:
        # Assume it's already a numpy array with [x, y, z, energy] columns
        x = torch.tensor(event_data, dtype=torch.float)
    
    num_nodes = len(x)
    
    # Optimized edge creation using torch operations
    # Create fully connected graph more efficiently
    if num_nodes > 1:
        # Create all possible pairs more efficiently
        nodes = torch.arange(num_nodes)
        edge_index = torch.combinations(nodes, r=2, with_replacement=False)
        # Make it bidirectional
        edge_index = torch.cat([edge_index, edge_index.flip(1)], dim=0).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index)


class GraphAutoEncoder(nn.Module):
    """Graph Autoencoder model"""
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(GraphAutoEncoder, self).__init__()
        # Encoder
        self.encoder1 = GCNConv(input_dim, hidden_dim)
        self.encoder2 = GCNConv(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder1 = GCNConv(latent_dim, hidden_dim)
        self.decoder2 = GCNConv(hidden_dim, input_dim)
        
    def encode(self, x, edge_index, batch):
        x = torch.relu(self.encoder1(x, edge_index))
        x = self.encoder2(x, edge_index)
        return global_mean_pool(x, batch)  # Graph-level embedding
    
    def decode(self, z, edge_index, num_nodes):
        # Expand graph-level embedding to node-level
        x = z.repeat_interleave(num_nodes, dim=0)
        x = torch.relu(self.decoder1(x, edge_index))
        x = self.decoder2(x, edge_index)
        return x
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        z = self.encode(x, edge_index, batch)
        
        # Get number of nodes per graph for decoding
        num_nodes = torch.bincount(batch)
        reconstructed = self.decode(z, edge_index, num_nodes)
        
        return reconstructed


def load_and_create_events(file_list, max_nodes_per_event=1000):
    """Load multiple npy files and create graph events - optimized version"""
    all_events = []
    
    print(f"Processing {len(file_list)} files...")
    
    for i, file_path in enumerate(file_list):
        print(f"  Processing file {i+1}/{len(file_list)}: {os.path.basename(file_path)}")
            
        try:
            # Load data
            file_data = np.load(file_path)
            
            # Skip if file is empty or malformed
            if len(file_data) == 0:
                print(f"    Empty file, skipping")
                continue
                
            print(f"    Loaded {len(file_data)} data points")
            
            # Convert to DataFrame only once
            file_df = pd.DataFrame(file_data, columns=['event', 'x', 'y', 'z', 'energy'])
            
            # Check unique events
            unique_events = file_df['event'].unique()
            print(f"    Found {len(unique_events)} unique events")
            
            # Group by event more efficiently
            event_groups = file_df.groupby('event')
            
            file_events_created = 0
            for event_id, event_data in event_groups:
                event_size = len(event_data)
                # Skip events with too few or too many points
                if event_size <= 1:
                    print(f"      Event {event_id}: Skipped (only {event_size} point)")
                    continue
                if event_size > max_nodes_per_event:
                    print(f"      Event {event_id}: Skipped (too many points: {event_size})")
                    continue
                
                # Extract only the needed columns as numpy array
                event_features = event_data[['x', 'y', 'z', 'energy']].values
                all_events.append(create_graph_data(event_features))
                file_events_created += 1
                
            print(f"    Created {file_events_created} valid events from this file")
                
        except Exception as e:
            print(f"    Warning: Error processing file {file_path}: {e}")
            continue
    
    print(f"Total: Created {len(all_events)} graph events from {len(file_list)} files")
    return all_events


def process_single_file(file_path, max_nodes_per_event=1000):
    """Process a single file and return events - for parallel processing"""
    events = []
    try:
        file_data = np.load(file_path)
        if len(file_data) == 0:
            print(f"  File {os.path.basename(file_path)}: Empty file")
            return events
            
        print(f"  File {os.path.basename(file_path)}: Loaded {len(file_data)} data points")
        
        file_df = pd.DataFrame(file_data, columns=['event', 'x', 'y', 'z', 'energy'])
        unique_events = file_df['event'].unique()
        print(f"  File {os.path.basename(file_path)}: Found {len(unique_events)} unique events")
        
        event_groups = file_df.groupby('event')
        
        valid_events = 0
        for event_id, event_data in event_groups:
            event_size = len(event_data)
            if event_size <= 1:
                print(f"    Event {event_id}: Skipped (too few points: {event_size})")
                continue
            if event_size > max_nodes_per_event:
                print(f"    Event {event_id}: Skipped (too many points: {event_size})")
                continue
                
            event_features = event_data[['x', 'y', 'z', 'energy']].values
            events.append(create_graph_data(event_features))
            valid_events += 1
            
        print(f"  File {os.path.basename(file_path)}: Created {valid_events} valid events")
            
    except Exception as e:
        print(f"  Warning: Error processing file {file_path}: {e}")
    
    return events


def load_and_create_events_parallel(file_list, max_nodes_per_event=1000, n_processes=None):
    """Parallel version of load_and_create_events for faster processing"""
    if n_processes is None:
        n_processes = min(cpu_count(), len(file_list))
    
    print(f"Processing {len(file_list)} files using {n_processes} processes...")
    
    # Use partial to fix the max_nodes_per_event parameter
    process_func = partial(process_single_file, max_nodes_per_event=max_nodes_per_event)
    
    all_events = []
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_func, file_list)
        
        # Flatten the list of lists
        for file_events in results:
            all_events.extend(file_events)
    
    print(f"Created {len(all_events)} graph events from {len(file_list)} files")
    return all_events


def inspect_data_file(file_path):
    """Inspect a single data file to understand its structure"""
    try:
        print(f"\n=== Inspecting {os.path.basename(file_path)} ===")
        file_data = np.load(file_path)
        print(f"File shape: {file_data.shape}")
        print(f"File dtype: {file_data.dtype}")
        
        if len(file_data) > 0:
            print(f"First few rows:")
            print(file_data[:5])
            
            # Try to create DataFrame
            file_df = pd.DataFrame(file_data, columns=['event', 'x', 'y', 'z', 'energy'])
            print(f"\nDataFrame info:")
            print(f"Number of rows: {len(file_df)}")
            print(f"Columns: {file_df.columns.tolist()}")
            print(f"Event column unique values: {sorted(file_df['event'].unique())}")
            print(f"Event counts:")
            event_counts = file_df['event'].value_counts().sort_index()
            print(event_counts.head(10))
            
            # Check for events with multiple points
            multi_point_events = event_counts[event_counts > 1]
            print(f"\nEvents with multiple points: {len(multi_point_events)}")
            if len(multi_point_events) > 0:
                print(f"Largest event has {multi_point_events.max()} points")
        else:
            print("File is empty")
            
    except Exception as e:
        print(f"Error inspecting file: {e}")


def main():
    # Initialize model, loss, and optimizer
    model = GraphAutoEncoder(input_dim=4, hidden_dim=16, latent_dim=8)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Get all npy files for full training
    npy_files = glob.glob('/nevis/riverside/data/sc5303/sbnd/offline_ad/pi0/npy/*.npy')
    print(f"Found {len(npy_files)} npy files")
    
    # Inspect the first file to understand data structure
    if len(npy_files) > 0:
        inspect_data_file(npy_files[0])
    
    # Split files: 75% for train/val, 25% for test
    train_val_files, test_files = train_test_split(npy_files[:10], test_size=0.25, random_state=42)
    
    # Split train/val: 2:1 ratio (50% train, 25% val of total)
    train_files, val_files = train_test_split(train_val_files, test_size=0.33, random_state=42)
    
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")
    
    # Load training and validation data
    print("Loading training data...")
    # Use sequential version first for debugging
    train_events = load_and_create_events(train_files, max_nodes_per_event=10000)
    print("Loading validation data...")
    val_events = load_and_create_events(val_files, max_nodes_per_event=10000)
    
    print(f"Training events: {len(train_events)}")
    print(f"Validation events: {len(val_events)}")
    
    # Create data loaders
    train_loader = DataLoader(train_events, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_events, batch_size=32, shuffle=False)
    
    # Move model to device
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # Training loop
    num_epochs = 10  # Reduced for testing, change to 50 for full training
    print(f"Starting training for {num_epochs} epochs...")
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch.x)
            
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        # Validation
        model.eval()
        epoch_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch.x)
                epoch_val_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    print("Training completed!")
    
    # Plot training and validation loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save the loss data
    loss_data = {
        'epoch': list(range(1, num_epochs + 1)),
        'train_loss': train_losses,
        'val_loss': val_losses
    }
    loss_df = pd.DataFrame(loss_data)
    loss_df.to_csv('training_loss_curves.csv', index=False)
    print("Loss curves saved to 'training_loss_curves.csv'")
    
    # Save the trained model
    # Create the models directory if it doesn't exist
    model_save_dir = '/nevis/riverside/data/sc5303/models'
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Save the model state dict
    model_path = os.path.join(model_save_dir, 'graph_autoencoder.pth')
    if torch.cuda.device_count() > 1:
        # If using DataParallel, save the module state dict
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)
    
    print(f"Model saved to: {model_path}")
    
    # Also save the complete model configuration for easy loading
    model_config = {
        'input_dim': 4,
        'hidden_dim': 16,
        'latent_dim': 8,
        'num_epochs': num_epochs,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1]
    }
    
    config_path = os.path.join(model_save_dir, 'model_config.pt')
    torch.save(model_config, config_path)
    print(f"Model configuration saved to: {config_path}")


if __name__ == "__main__":
    main()