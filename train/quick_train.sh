#!/bin/bash

# Quick training script with predefined configurations
echo "CNN Autoencoder Quick Training Script"
echo "====================================="

# Get timestamp for unique output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_DIR="/nevis/riverside/data/sc5303/models"

# Create base directory if it doesn't exist
mkdir -p "${BASE_DIR}"

# Configuration options
echo "Select training configuration:"
echo "1) Quick test (5 files, 20 epochs) - for testing"
echo "2) Medium training (50 files, 50 epochs) - for development"
echo "3) Full dataset (all files, 100 epochs) - for production"
echo "4) Custom parameters"

read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "Running quick test configuration..."
        OUTPUT_DIR="${BASE_DIR}/cnn_test_${TIMESTAMP}"
        python train_cnn_autoencoder.py \
            --output_dir "${OUTPUT_DIR}" \
            --max_files 5 \
            --num_epochs 20 \
            --batch_size 16 \
            --save_plots
        ;;
    2)
        echo "Running medium training configuration..."
        OUTPUT_DIR="${BASE_DIR}/cnn_medium_${TIMESTAMP}"
        python train_cnn_autoencoder.py \
            --output_dir "${OUTPUT_DIR}" \
            --max_files 50 \
            --num_epochs 50 \
            --batch_size 32 \
            --save_plots
        ;;
    3)
        echo "Running full dataset training..."
        OUTPUT_DIR="${BASE_DIR}/cnn_full_${TIMESTAMP}"
        python train_cnn_autoencoder.py \
            --output_dir "${OUTPUT_DIR}" \
            --num_epochs 100 \
            --batch_size 32 \
            --num_workers 8 \
            --save_plots
        ;;
    4)
        echo "Custom training parameters:"
        read -p "Max files (or 'all' for no limit): " max_files
        read -p "Number of epochs: " epochs
        read -p "Batch size: " batch_size
        read -p "Learning rate: " lr
        
        OUTPUT_DIR="${BASE_DIR}/cnn_custom_${TIMESTAMP}"
        
        if [ "$max_files" = "all" ]; then
            python train_cnn_autoencoder.py \
                --output_dir "${OUTPUT_DIR}" \
                --num_epochs ${epochs} \
                --batch_size ${batch_size} \
                --learning_rate ${lr} \
                --save_plots
        else
            python train_cnn_autoencoder.py \
                --output_dir "${OUTPUT_DIR}" \
                --max_files ${max_files} \
                --num_epochs ${epochs} \
                --batch_size ${batch_size} \
                --learning_rate ${lr} \
                --save_plots
        fi
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

TRAINING_EXIT_CODE=$?

if [ ${TRAINING_EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "Training completed successfully!"
    echo "Results saved to: ${OUTPUT_DIR}"
    echo ""
    echo "Running analysis..."
    python analyze_test_results.py --results_dir "${OUTPUT_DIR}" --output_plots
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "Analysis completed!"
        echo "Check ${OUTPUT_DIR} for all results and plots."
    else
        echo "Analysis failed, but training results are saved."
    fi
else
    echo "Training failed!"
fi
