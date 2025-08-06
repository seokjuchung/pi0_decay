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
    """Create graph data for each event"""
    # Node features: x, y, z, energy coordinates
    x = torch.tensor(event_data[['x', 'y', 'z', 'energy']].values, dtype=torch.float)
    
    # Create edges (fully connected graph for simplicity)
    num_nodes = len(event_data)
    edge_index = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_index.append([i, j])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
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


def load_and_create_events(file_list):
    """Load multiple npy files and create graph events"""
    all_events = []
    
    for file_path in file_list:
        # Load data
        file_data = np.load(file_path)
        file_df = pd.DataFrame(file_data, columns=['event', 'x', 'y', 'z', 'energy'])
        
        # Create events for this file
        for event_id in file_df['event'].unique():
            event_data = file_df[file_df['event'] == event_id]
            if len(event_data) > 1:
                all_events.append(create_graph_data(event_data))
    
    return all_events


def main():
    # Initialize model, loss, and optimizer
    model = GraphAutoEncoder(input_dim=4, hidden_dim=16, latent_dim=8)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Get all npy files for full training
    npy_files = glob.glob('/nevis/riverside/data/sc5303/sbnd/offline_ad/pi0/npy/*.npy')
    print(f"Found {len(npy_files)} npy files")
    
    # Split files: 75% for train/val, 25% for test
    train_val_files, test_files = train_test_split(npy_files[:10], test_size=0.25, random_state=42)
    
    # Split train/val: 2:1 ratio (50% train, 25% val of total)
    train_files, val_files = train_test_split(train_val_files, test_size=0.33, random_state=42)
    
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")
    
    # Load training and validation data
    print("Loading training data...")
    train_events = load_and_create_events(train_files)
    print("Loading validation data...")
    val_events = load_and_create_events(val_files)
    
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