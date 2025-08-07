#!/bin/bash

# Safe CNN Training Script
# This script includes fallbacks for CUDA issues

echo "Safe CNN Autoencoder Training"
echo "============================="

# Get timestamp for unique output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_DIR="/nevis/riverside/data/sc5303/models"
OUTPUT_DIR="${BASE_DIR}/cnn_safe_${TIMESTAMP}"

# Create base directory if it doesn't exist
mkdir -p "${BASE_DIR}"

echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Training configuration
MAX_FILES=5  # Start with small test
NUM_EPOCHS=10
BATCH_SIZE=8  # Smaller batch size to reduce memory usage
MEMORY_FRACTION=0.6  # Use less GPU memory

echo "Starting safe training with reduced parameters..."
echo "Max files: ${MAX_FILES}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "GPU memory fraction: ${MEMORY_FRACTION}"
echo ""

# Try GPU training first
echo "Attempting GPU training..."
python train_cnn_autoencoder.py \
    --output_dir "${OUTPUT_DIR}" \
    --max_files ${MAX_FILES} \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --memory_fraction ${MEMORY_FRACTION} \
    --device cuda \
    --debug \
    --save_plots

GPU_EXIT_CODE=$?

if [ ${GPU_EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "GPU training completed successfully!"
    echo "Results saved to: ${OUTPUT_DIR}"
    
    # Run analysis
    echo "Running analysis..."
    python analyze_test_results.py --results_dir "${OUTPUT_DIR}" --output_plots
    
else
    echo ""
    echo "GPU training failed! Attempting CPU fallback..."
    
    # Try CPU training as fallback
    OUTPUT_DIR_CPU="${BASE_DIR}/cnn_cpu_${TIMESTAMP}"
    
    python train_cnn_autoencoder.py \
        --output_dir "${OUTPUT_DIR_CPU}" \
        --max_files ${MAX_FILES} \
        --num_epochs ${NUM_EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --device cpu \
        --save_plots
    
    CPU_EXIT_CODE=$?
    
    if [ ${CPU_EXIT_CODE} -eq 0 ]; then
        echo ""
        echo "CPU training completed successfully!"
        echo "Results saved to: ${OUTPUT_DIR_CPU}"
        
        # Run analysis
        echo "Running analysis..."
        python analyze_test_results.py --results_dir "${OUTPUT_DIR_CPU}" --output_plots
        
    else
        echo "Both GPU and CPU training failed!"
        echo ""
        echo "Debugging suggestions:"
        echo "1. Check data integrity: Verify NPY files are not corrupted"
        echo "2. Reduce memory usage: Try smaller batch size or max_points"
        echo "3. Check PyTorch installation: Ensure CUDA compatibility"
        echo "4. Monitor system resources: Check available RAM and GPU memory"
        exit 1
    fi
fi

echo ""
echo "Safe training script completed."
