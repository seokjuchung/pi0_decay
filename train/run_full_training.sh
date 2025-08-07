#!/bin/bash

# CNN Autoencoder Training Script
# Run this to train on the full dataset

echo "Starting CNN Autoencoder Training on Full Dataset"
echo "=================================================="

# Set up environment
export PYTHONPATH="${PYTHONPATH}:."

# Create output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/nevis/riverside/data/sc5303/models/cnn_autoencoder_results_${TIMESTAMP}"

echo "Output directory: ${OUTPUT_DIR}"

# Training parameters for full dataset
PYTHON_CMD="python train_cnn_autoencoder.py \
    --data_path '/nevis/riverside/data/sc5303/sbnd/offline_ad/pi0/npy/*.npy' \
    --output_dir ${OUTPUT_DIR} \
    --max_points 10000 \
    --min_points 10 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --num_epochs 100 \
    --latent_dim 128 \
    --device auto \
    --num_workers 8 \
    --save_plots"

echo "Running training command:"
echo "${PYTHON_CMD}"
echo ""

# Run training
eval ${PYTHON_CMD}

TRAINING_EXIT_CODE=$?

if [ ${TRAINING_EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "Training completed successfully!"
    echo "Running analysis..."
    
    # Run analysis
    python analyze_test_results.py \
        --results_dir ${OUTPUT_DIR} \
        --output_plots
    
    ANALYSIS_EXIT_CODE=$?
    
    if [ ${ANALYSIS_EXIT_CODE} -eq 0 ]; then
        echo ""
        echo "Analysis completed successfully!"
        echo "Results saved in: ${OUTPUT_DIR}"
        echo ""
        echo "Generated files:"
        echo "  - best_cnn_autoencoder.pth: Best model checkpoint"
        echo "  - final_cnn_autoencoder.pth: Final model with all results"
        echo "  - test_events.pkl: Test events for later analysis"
        echo "  - test_events_info.json: Test dataset information"
        echo "  - test_losses.npy: Test reconstruction losses"
        echo "  - config.json: Training configuration"
        echo "  - training_history.png: Training plots"
        echo "  - event_size_distribution.png: Data analysis plots"
        echo "  - test_loss_analysis.png: Test loss analysis"
        echo "  - anomaly_detection_analysis.png: Anomaly detection plots"
        echo "  - event_size_analysis.png: Event size vs loss analysis"
        echo "  - training_analysis.png: Training convergence analysis"
        echo "  - analysis_summary.txt: Comprehensive summary report"
        echo ""
        echo "Use the saved test_events.pkl and models for further anomaly detection!"
        
    else
        echo "Analysis failed with exit code: ${ANALYSIS_EXIT_CODE}"
    fi
    
else
    echo "Training failed with exit code: ${TRAINING_EXIT_CODE}"
fi

echo "Script completed."
