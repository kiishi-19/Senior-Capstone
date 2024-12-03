#!/bin/bash

# Set project root directory
PROJECT_ROOT="/home/ubuntu/GHIDS2/Senior-Capstone"

# Create necessary directories
mkdir -p "$PROJECT_ROOT/output/features"
mkdir -p "$PROJECT_ROOT/output/scaled_features"
mkdir -p "$PROJECT_ROOT/output/models"

# Step 1: Process SCAP files
echo "Step 1: Processing SCAP files..."
uv run ML/process_scap_files.py \
    "$PROJECT_ROOT/Data/CB-DS/NORMAL" \
    --output_dir "$PROJECT_ROOT/output/features" \
    --num_cores 8

if [ $? -ne 0 ]; then
    echo "Error: SCAP file processing failed." >&2
    exit 1
fi

# Step 2: Prepare dataset
echo "Step 2: Preparing dataset..."
uv run ML/prepare_dataset.py \
    "$PROJECT_ROOT/output/features" \
    --scaler_file "$PROJECT_ROOT/output/scaler.pkl" \
    --output_dir "$PROJECT_ROOT/output/scaled_features"

if [ $? -ne 0 ]; then
    echo "Error: Dataset preparation failed." >&2
    exit 1
fi

# Step 3: Concatenate CSV files
echo "Step 3: Concatenating CSV files..."
uv run ML/concatenate_csv.py \
    "$PROJECT_ROOT/output/scaled_features" \
    "$PROJECT_ROOT/output/scaled_features.pkl"

if [ $? -ne 0 ]; then
    echo "Error: CSV concatenation failed." >&2
    exit 1
fi

# Step 4: Train autoencoder
echo "Step 4: Training autoencoder..."
uv run ML/train_autoencoder.py \
    "$PROJECT_ROOT/output/scaled_features.pkl" \
    --model_output_file "$PROJECT_ROOT/output/models/autoencoder_model.pth" \
    --epochs 120 \
    --hidden_dims 64 32 16 \
    --bottleneck_dim 8 \
    --activation relu \
    --n_splits 4 \
    --batch_size 64 \
    --dropout_prob 0.2

if [ $? -ne 0 ]; then
    echo "Error: Autoencoder training failed." >&2
    exit 1
fi

# Step 5: Verify models directory
echo "Step 5: Verifying models directory..."
if [ "$(ls -A "$PROJECT_ROOT/output/models")" ]; then
    echo "Training pipeline completed successfully!"
else
    echo "Error: Models directory is empty. Training pipeline failed." >&2
    exit 1
fi
