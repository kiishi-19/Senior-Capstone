#!/bin/bash

# Create necessary directories
#mkdir -p /home/ubuntu/GHIDS2/Senior-Capstone/output/features
#mkdir -p /home/ubuntu/GHIDS2/Senior-Capstone/output/models

# Process SCAP files
#echo "Step 1: Processing SCAP files..."
#uv run ML/process_scap_files.py \
    #/home/ubuntu/GHIDS2/Senior-Capstone/Data/CB-DS/NORMAL \
    #--output_dir /home/ubuntu/GHIDS2/Senior-Capstone/output/features \
    #--num_cores 4

# Prepare dataset
#echo "Step 2: Preparing dataset..."
##uv run ML/prepare_dataset.py \
    #/home/ubuntu/GHIDS2/Senior-Capstone/output/features \
    #--scaler_file /home/ubuntu/GHIDS2/Senior-Capstone/output/scaler.pkl \
    #--output_dir /home/ubuntu/GHIDS2/Senior-Capstone/output/scaled_features

# Concatenate CSV files
#echo "Step 3: Concatenating CSV files..."
#uv run ML/concatenate_csv.py \
    #/home/ubuntu/GHIDS2/Senior-Capstone/output/scaled_features \
    #/home/ubuntu/GHIDS2/Senior-Capstone/output/scaled_features.pkl

# Train autoencoder
echo "Step 4: Training autoencoder..."
uv run ML/train_autoencoder.py \
    /home/ubuntu/GHIDS2/Senior-Capstone/output/scaled_features.pkl \
    --model_output_file /home/ubuntu/GHIDS2/Senior-Capstone/output/models/autoencoder_model.pth \
    --epochs 120 \
    --hidden_dims 64 32 16 \
    --bottleneck_dim 8 \
    --activation relu \
    --n_splits 4 \
    --batch_size 64 \
    --dropout_prob 0.2 \
    --learning_rate 1e-4

# Check if models directory is populated
echo "Step 5: Verifying models directory..."
if [ "$(ls -A /home/ubuntu/GHIDS2/Senior-Capstone/output/models)" ]; then
    echo "Training pipeline completed successfully!"
else
    echo "Error: Models directory is empty. Training pipeline failed." >&2
    exit 1
fi
