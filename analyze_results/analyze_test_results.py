#!/usr/bin/env python3
"""
Test Set Analysis and Anomaly Detection
Load saved test results and perform detailed analysis.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os
import argparse
from tqdm import tqdm

# Import the model class
from train_cnn_autoencoder import CNN1DAutoEncoder, create_padding_mask, masked_mse_loss

parser = argparse.ArgumentParser(description='Analyze saved test results')
parser.add_argument('--results_dir', type=str, default='./cnn_autoencoder_output',
                    help='Directory containing saved results')
parser.add_argument('--output_plots', action='store_true',
                    help='Generate and save analysis plots')

def load_results(results_dir):
    """Load all saved results"""
    results = {}
    
    # Load final model results
    final_path = os.path.join(results_dir, 'final_cnn_autoencoder.pth')
    if os.path.exists(final_path):
        results['final_model'] = torch.load(final_path, map_location='cpu')
        print(f"Loaded final model results from {final_path}")
    
    # Load test events
    test_events_path = os.path.join(results_dir, 'test_events.pkl')
    if os.path.exists(test_events_path):
        with open(test_events_path, 'rb') as f:
            results['test_events'] = pickle.load(f)
        print(f"Loaded {len(results['test_events'])} test events")
    
    # Load test event info
    test_info_path = os.path.join(results_dir, 'test_events_info.json')
    if os.path.exists(test_info_path):
        with open(test_info_path, 'r') as f:
            results['test_info'] = json.load(f)
        print(f"Loaded test events info")
    
    # Load test losses
    test_losses_path = os.path.join(results_dir, 'test_losses.npy')
    if os.path.exists(test_losses_path):
        results['test_losses'] = np.load(test_losses_path)
        print(f"Loaded {len(results['test_losses'])} test loss values")
    
    # Load config
    config_path = os.path.join(results_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            results['config'] = json.load(f)
        print(f"Loaded configuration")
    
    return results

def analyze_test_losses(test_losses, output_plots=False, results_dir=None):
    """Analyze test reconstruction losses"""
    print("\n" + "="*50)
    print("TEST LOSSES ANALYSIS")
    print("="*50)
    
    print(f"Number of test samples: {len(test_losses)}")
    print(f"Mean reconstruction loss: {test_losses.mean():.6f}")
    print(f"Std reconstruction loss: {test_losses.std():.6f}")
    print(f"Min reconstruction loss: {test_losses.min():.6f}")
    print(f"Max reconstruction loss: {test_losses.max():.6f}")
    print(f"Median reconstruction loss: {np.median(test_losses):.6f}")
    
    # Percentile analysis
    percentiles = [50, 90, 95, 99, 99.5, 99.9]
    print(f"\nPercentile Analysis:")
    for p in percentiles:
        value = np.percentile(test_losses, p)
        print(f"  {p:4.1f}th percentile: {value:.6f}")
    
    if output_plots and results_dir:
        # Create comprehensive loss analysis plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Histogram
        axes[0, 0].hist(test_losses, bins=100, alpha=0.7, edgecolor='black', density=True)
        axes[0, 0].axvline(test_losses.mean(), color='red', linestyle='--', label='Mean')
        axes[0, 0].axvline(np.median(test_losses), color='green', linestyle='--', label='Median')
        axes[0, 0].set_xlabel('Reconstruction Loss')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Test Loss Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Log scale histogram
        axes[0, 1].hist(test_losses, bins=100, alpha=0.7, edgecolor='black', density=True)
        axes[0, 1].set_xlabel('Reconstruction Loss')
        axes[0, 1].set_ylabel('Density (Log Scale)')
        axes[0, 1].set_title('Test Loss Distribution (Log Scale)')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Box plot
        axes[0, 2].boxplot(test_losses, vert=True)
        axes[0, 2].set_ylabel('Reconstruction Loss')
        axes[0, 2].set_title('Test Loss Box Plot')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Q-Q plot against normal distribution
        from scipy import stats
        stats.probplot(test_losses, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot vs Normal Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cumulative distribution
        sorted_losses = np.sort(test_losses)
        cumulative = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)
        axes[1, 1].plot(sorted_losses, cumulative, linewidth=2)
        axes[1, 1].set_xlabel('Reconstruction Loss')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_title('Cumulative Distribution Function')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Mark important percentiles
        for p in [90, 95, 99]:
            value = np.percentile(test_losses, p)
            axes[1, 1].axvline(value, color='red', linestyle='--', alpha=0.7)
            axes[1, 1].text(value, p/100, f'{p}%', rotation=90, va='bottom')
        
        # Loss rank plot
        ranks = np.arange(1, len(sorted_losses) + 1)
        axes[1, 2].plot(ranks, sorted_losses, linewidth=1)
        axes[1, 2].set_xlabel('Sample Rank')
        axes[1, 2].set_ylabel('Reconstruction Loss')
        axes[1, 2].set_title('Loss by Rank (Sorted)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'test_loss_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()

def anomaly_detection_analysis(test_losses, output_plots=False, results_dir=None):
    """Perform anomaly detection analysis"""
    print("\n" + "="*50)
    print("ANOMALY DETECTION ANALYSIS")
    print("="*50)
    
    # Different threshold strategies
    strategies = {
        'percentile_90': np.percentile(test_losses, 90),
        'percentile_95': np.percentile(test_losses, 95),
        'percentile_99': np.percentile(test_losses, 99),
        'mean_plus_2std': test_losses.mean() + 2 * test_losses.std(),
        'mean_plus_3std': test_losses.mean() + 3 * test_losses.std(),
        'median_plus_mad': np.median(test_losses) + 2.5 * np.median(np.abs(test_losses - np.median(test_losses)))
    }
    
    results = {}
    for strategy, threshold in strategies.items():
        anomaly_mask = test_losses > threshold
        n_anomalies = np.sum(anomaly_mask)
        anomaly_rate = n_anomalies / len(test_losses) * 100
        
        results[strategy] = {
            'threshold': threshold,
            'n_anomalies': n_anomalies,
            'anomaly_rate': anomaly_rate,
            'mask': anomaly_mask
        }
        
        print(f"{strategy:20s}: threshold={threshold:.6f}, anomalies={n_anomalies:4d} ({anomaly_rate:5.1f}%)")
    
    # Recommended strategy
    print(f"\nRecommended Strategy: percentile_95")
    print(f"  Threshold: {results['percentile_95']['threshold']:.6f}")
    print(f"  Expected false positive rate: ~5%")
    print(f"  Actual anomaly rate: {results['percentile_95']['anomaly_rate']:.1f}%")
    
    if output_plots and results_dir:
        # Anomaly detection visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss distribution with thresholds
        axes[0, 0].hist(test_losses, bins=100, alpha=0.7, edgecolor='black', density=True)
        colors = ['red', 'orange', 'purple', 'green', 'blue', 'brown']
        for i, (strategy, result) in enumerate(results.items()):
            if i < len(colors):
                axes[0, 0].axvline(result['threshold'], color=colors[i], linestyle='--', 
                                  alpha=0.8, label=strategy)
        axes[0, 0].set_xlabel('Reconstruction Loss')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Loss Distribution with Anomaly Thresholds')
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Anomaly rates comparison
        strategies_list = list(results.keys())
        rates = [results[s]['anomaly_rate'] for s in strategies_list]
        
        bars = axes[0, 1].bar(range(len(strategies_list)), rates, alpha=0.7, 
                             color='skyblue', edgecolor='black')
        axes[0, 1].set_xticks(range(len(strategies_list)))
        axes[0, 1].set_xticklabels([s.replace('_', '\n') for s in strategies_list], 
                                  rotation=45, ha='right', fontsize=8)
        axes[0, 1].set_ylabel('Anomaly Rate (%)')
        axes[0, 1].set_title('Anomaly Detection Rates by Strategy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # ROC-like curve
        thresholds_range = np.linspace(test_losses.min(), test_losses.max(), 1000)
        false_positive_rates = []
        
        for thresh in thresholds_range:
            fp_rate = np.sum(test_losses > thresh) / len(test_losses)
            false_positive_rates.append(fp_rate)
        
        axes[1, 0].plot(thresholds_range, false_positive_rates, linewidth=2)
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('False Positive Rate')
        axes[1, 0].set_title('Threshold vs False Positive Rate')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Mark common thresholds
        for strategy in ['percentile_90', 'percentile_95', 'percentile_99']:
            thresh = results[strategy]['threshold']
            fp_rate = results[strategy]['anomaly_rate'] / 100
            axes[1, 0].scatter(thresh, fp_rate, s=100, zorder=5)
            axes[1, 0].annotate(strategy.split('_')[1], (thresh, fp_rate), 
                               xytext=(5, 5), textcoords='offset points')
        
        # Threshold sensitivity analysis
        percentile_range = np.arange(80, 99.9, 0.1)
        anomaly_rates_sens = []
        
        for p in percentile_range:
            thresh = np.percentile(test_losses, p)
            rate = np.sum(test_losses > thresh) / len(test_losses) * 100
            anomaly_rates_sens.append(rate)
        
        axes[1, 1].plot(percentile_range, anomaly_rates_sens, linewidth=2)
        axes[1, 1].set_xlabel('Threshold Percentile')
        axes[1, 1].set_ylabel('Anomaly Rate (%)')
        axes[1, 1].set_title('Sensitivity Analysis: Percentile Thresholds')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Mark recommended thresholds
        for p in [90, 95, 99]:
            idx = np.argmin(np.abs(percentile_range - p))
            axes[1, 1].scatter(p, anomaly_rates_sens[idx], color='red', s=100, zorder=5)
            axes[1, 1].annotate(f'{p}%', (p, anomaly_rates_sens[idx]), 
                               xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'anomaly_detection_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    return results

def event_size_vs_loss_analysis(test_events, test_losses, output_plots=False, results_dir=None):
    """Analyze relationship between event size and reconstruction loss"""
    print("\n" + "="*50)
    print("EVENT SIZE vs RECONSTRUCTION LOSS ANALYSIS")
    print("="*50)
    
    event_sizes = [len(event) for event in test_events]
    
    # Calculate correlation
    correlation = np.corrcoef(event_sizes, test_losses)[0, 1]
    print(f"Correlation between event size and loss: {correlation:.4f}")
    
    # Size bins analysis
    size_bins = np.percentile(event_sizes, [0, 25, 50, 75, 90, 95, 100])
    print(f"\nEvent size distribution:")
    for i in range(len(size_bins)):
        print(f"  {[0, 25, 50, 75, 90, 95, 100][i]:3d}th percentile: {size_bins[i]:6.0f} points")
    
    # Binned analysis
    n_bins = 10
    size_bins_analysis = np.percentile(event_sizes, np.linspace(0, 100, n_bins+1))
    bin_means = []
    bin_stds = []
    bin_counts = []
    
    print(f"\nBinned Analysis (by event size):")
    for i in range(n_bins):
        mask = (np.array(event_sizes) >= size_bins_analysis[i]) & (np.array(event_sizes) < size_bins_analysis[i+1])
        if i == n_bins-1:  # Include upper bound for last bin
            mask = (np.array(event_sizes) >= size_bins_analysis[i]) & (np.array(event_sizes) <= size_bins_analysis[i+1])
        
        if np.sum(mask) > 0:
            bin_losses = test_losses[mask]
            bin_means.append(bin_losses.mean())
            bin_stds.append(bin_losses.std())
            bin_counts.append(np.sum(mask))
            
            print(f"  Size [{size_bins_analysis[i]:4.0f}, {size_bins_analysis[i+1]:4.0f}]: "
                  f"count={np.sum(mask):4d}, loss_mean={bin_losses.mean():.6f}, "
                  f"loss_std={bin_losses.std():.6f}")
    
    if output_plots and results_dir:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Scatter plot
        axes[0, 0].scatter(event_sizes, test_losses, alpha=0.5, s=1)
        axes[0, 0].set_xlabel('Event Size (Number of Points)')
        axes[0, 0].set_ylabel('Reconstruction Loss')
        axes[0, 0].set_title(f'Event Size vs Reconstruction Loss\n(Correlation: {correlation:.4f})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(event_sizes, test_losses, 1)
        p = np.poly1d(z)
        axes[0, 0].plot(sorted(event_sizes), p(sorted(event_sizes)), "r--", alpha=0.8, linewidth=2)
        
        # Hexbin plot for density
        hb = axes[0, 1].hexbin(event_sizes, test_losses, gridsize=30, cmap='Blues')
        axes[0, 1].set_xlabel('Event Size (Number of Points)')
        axes[0, 1].set_ylabel('Reconstruction Loss')
        axes[0, 1].set_title('Event Size vs Loss (Density)')
        plt.colorbar(hb, ax=axes[0, 1], label='Count')
        
        # Binned means with error bars
        bin_centers = [(size_bins_analysis[i] + size_bins_analysis[i+1])/2 for i in range(len(bin_means))]
        axes[1, 0].errorbar(bin_centers, bin_means, yerr=bin_stds, 
                           fmt='o-', capsize=5, capthick=2, linewidth=2)
        axes[1, 0].set_xlabel('Event Size (Bin Centers)')
        axes[1, 0].set_ylabel('Mean Reconstruction Loss')
        axes[1, 0].set_title('Binned Analysis: Mean Loss by Event Size')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Distribution comparison for small vs large events
        median_size = np.median(event_sizes)
        small_events_mask = np.array(event_sizes) < median_size
        large_events_mask = np.array(event_sizes) >= median_size
        
        small_losses = test_losses[small_events_mask]
        large_losses = test_losses[large_events_mask]
        
        axes[1, 1].hist(small_losses, bins=50, alpha=0.7, label=f'Small Events (<{median_size:.0f} points)', 
                       density=True, color='blue')
        axes[1, 1].hist(large_losses, bins=50, alpha=0.7, label=f'Large Events (>={median_size:.0f} points)', 
                       density=True, color='red')
        axes[1, 1].set_xlabel('Reconstruction Loss')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Loss Distribution: Small vs Large Events')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'event_size_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nSmall events (< {median_size:.0f} points): mean_loss = {small_losses.mean():.6f}")
        print(f"Large events (>= {median_size:.0f} points): mean_loss = {large_losses.mean():.6f}")
        
    return correlation

def training_analysis(results, output_plots=False, results_dir=None):
    """Analyze training history"""
    if 'final_model' not in results or 'train_losses' not in results['final_model']:
        print("No training history available")
        return
    
    print("\n" + "="*50)
    print("TRAINING ANALYSIS")
    print("="*50)
    
    train_losses = results['final_model']['train_losses']
    val_losses = results['final_model']['val_losses']
    
    print(f"Total epochs trained: {len(train_losses)}")
    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Final val loss: {val_losses[-1]:.6f}")
    print(f"Best val loss: {min(val_losses):.6f} (epoch {np.argmin(val_losses) + 1})")
    print(f"Training time: {results['final_model'].get('training_time', 'N/A')} seconds")
    
    # Convergence analysis
    if len(train_losses) > 10:
        last_10_train = train_losses[-10:]
        last_10_val = val_losses[-10:]
        
        train_trend = np.polyfit(range(10), last_10_train, 1)[0]
        val_trend = np.polyfit(range(10), last_10_val, 1)[0]
        
        print(f"\nConvergence Analysis (last 10 epochs):")
        print(f"  Train loss trend: {train_trend:.8f}/epoch ({'decreasing' if train_trend < 0 else 'increasing'})")
        print(f"  Val loss trend: {val_trend:.8f}/epoch ({'decreasing' if val_trend < 0 else 'increasing'})")
    
    if output_plots and results_dir:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training history
        epochs = range(1, len(train_losses) + 1)
        axes[0, 0].plot(epochs, train_losses, label='Train Loss', color='blue', linewidth=2)
        axes[0, 0].plot(epochs, val_losses, label='Validation Loss', color='red', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training History')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Log scale
        axes[0, 1].semilogy(epochs, train_losses, label='Train Loss', color='blue', linewidth=2)
        axes[0, 1].semilogy(epochs, val_losses, label='Validation Loss', color='red', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss (Log Scale)')
        axes[0, 1].set_title('Training History (Log Scale)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Loss difference per epoch
        if len(train_losses) > 1:
            train_diff = np.diff(train_losses)
            val_diff = np.diff(val_losses)
            axes[1, 0].plot(epochs[1:], train_diff, label='Train Loss Change', color='blue', alpha=0.7)
            axes[1, 0].plot(epochs[1:], val_diff, label='Val Loss Change', color='red', alpha=0.7)
            axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss Change')
            axes[1, 0].set_title('Loss Change per Epoch')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Validation gap
        val_gap = np.array(val_losses) - np.array(train_losses)
        axes[1, 1].plot(epochs, val_gap, color='purple', linewidth=2)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Validation - Train Loss')
        axes[1, 1].set_title('Validation Gap (Overfitting Indicator)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'training_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()

def create_summary_report(results, results_dir):
    """Create a comprehensive summary report"""
    report_path = os.path.join(results_dir, 'analysis_summary.txt')
    
    with open(report_path, 'w') as f:
        f.write("CNN AUTOENCODER ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        # Model info
        if 'final_model' in results:
            f.write("MODEL INFORMATION:\n")
            f.write(f"  Architecture: CNN 1D Autoencoder\n")
            f.write(f"  Total parameters: {results['final_model'].get('total_params', 'N/A'):,}\n")
            f.write(f"  Latent dimension: {results['final_model']['config']['latent_dim']}\n")
            f.write(f"  Max points: {results['final_model']['config']['max_points']}\n")
            f.write(f"  Training time: {results['final_model'].get('training_time', 'N/A'):.1f} seconds\n\n")
        
        # Dataset info
        if 'test_info' in results:
            f.write("DATASET INFORMATION:\n")
            f.write(f"  Total test events: {results['test_info']['num_test_events']}\n")
            f.write(f"  Files used: {len(results['test_info']['files_used'])}\n")
            f.write(f"  Test event sizes: {min(results['test_info']['test_event_sizes'])}-{max(results['test_info']['test_event_sizes'])} points\n\n")
        
        # Performance
        if 'test_losses' in results:
            test_losses = results['test_losses']
            f.write("PERFORMANCE METRICS:\n")
            f.write(f"  Mean test loss: {test_losses.mean():.6f}\n")
            f.write(f"  Std test loss: {test_losses.std():.6f}\n")
            f.write(f"  95th percentile: {np.percentile(test_losses, 95):.6f}\n\n")
        
        # Anomaly detection
        if 'final_model' in results and 'anomaly_threshold_95' in results['final_model']:
            f.write("ANOMALY DETECTION:\n")
            f.write(f"  Recommended threshold: {results['final_model']['anomaly_threshold_95']:.6f}\n")
            f.write(f"  Expected false positive rate: ~5%\n")
            f.write(f"  Threshold strategy: 95th percentile\n\n")
        
        f.write("FILES GENERATED:\n")
        files = ['test_loss_analysis.png', 'anomaly_detection_analysis.png', 
                'event_size_analysis.png', 'training_analysis.png']
        for file in files:
            if os.path.exists(os.path.join(results_dir, file)):
                f.write(f"  ✓ {file}\n")
        
        f.write(f"\nReport generated: {pd.Timestamp.now()}\n")
    
    print(f"\nSummary report saved to: {report_path}")

def main():
    args = parser.parse_args()
    
    print(f"Loading results from: {args.results_dir}")
    
    # Load all results
    results = load_results(args.results_dir)
    
    if not results:
        print("No results found!")
        return
    
    # Analyze test losses
    if 'test_losses' in results:
        analyze_test_losses(results['test_losses'], args.output_plots, args.results_dir)
        
        # Anomaly detection analysis
        anomaly_results = anomaly_detection_analysis(results['test_losses'], args.output_plots, args.results_dir)
    
    # Event size analysis
    if 'test_events' in results and 'test_losses' in results:
        correlation = event_size_vs_loss_analysis(results['test_events'], results['test_losses'], 
                                                 args.output_plots, args.results_dir)
    
    # Training analysis
    training_analysis(results, args.output_plots, args.results_dir)
    
    # Create summary report
    create_summary_report(results, args.results_dir)
    
    print(f"\nAnalysis complete! Check {args.results_dir} for all results.")

if __name__ == "__main__":
    main()
