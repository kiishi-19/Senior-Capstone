#!/bin/bash

# Check if the user provided the SCAP files directory
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_scap_files>"
    exit 1
fi

# Set project root directory (update this if needed)
PROJECT_ROOT="/home/ubuntu/GHIDS2/Senior-Capstone"

# Get the SCAP files directory from the first argument
SCAP_FILES_DIR="$1"

# Create necessary directories
mkdir -p "$PROJECT_ROOT/output/inference_results/scaled_features"

# Step 1: Process SCAP files
echo "Step 1: Processing SCAP files from $SCAP_FILES_DIR..."
uv run inference/inference_process_scapfiles.py \
    "$SCAP_FILES_DIR" \
    "$PROJECT_ROOT/output/features/known_syscalls.pkl" \
    "$PROJECT_ROOT/output/features/known_arguments.pkl" \
    --output_dir "$PROJECT_ROOT/output/inference_results"

if [ $? -ne 0 ]; then
    echo "Error: SCAP file processing failed." >&2
    exit 1
fi

# Step 2: Prepare dataset for inference
echo "Step 2: Preparing dataset for inference..."
uv run inference/inference_prepare_dataset.py \
    "$PROJECT_ROOT/output/inference_results" \
    --scaler_file "$PROJECT_ROOT/output/scaler.pkl" \
    --output_dir "$PROJECT_ROOT/output/inference_results/scaled_features"

if [ $? -ne 0 ]; then
    echo "Error: Dataset preparation failed." >&2
    exit 1
fi

# Step 3: Run inference
echo "Step 3: Running inference..."
uv run inference/inference_model.py \
    "$PROJECT_ROOT/output/inference_results/scaled_features" \
    --model_paths "$PROJECT_ROOT/output/models/autoencoder_model_fold1.pth" \
    "$PROJECT_ROOT/output/models/autoencoder_model_fold2.pth" \
    "$PROJECT_ROOT/output/models/autoencoder_model_fold3.pth" \
    "$PROJECT_ROOT/output/models/autoencoder_model_fold4.pth" \
    --batch_size 64 \
    --hidden_dims 64 32 16 \
    --bottleneck_dim 8 \
    --activation relu \
    --dropout_prob 0.2 \
    --device cuda

if [ $? -ne 0 ]; then
    echo "Error: Inference failed." >&2
    exit 1
fi

# Step 4: Verify inference results
echo "Step 4: Verifying inference results..."
if [ "$(ls -A "$PROJECT_ROOT/output/inference_results/scaled_features")" ]; then
    echo "Inference pipeline completed successfully!"
else
    echo "Error: Inference results directory is empty. Inference pipeline failed." >&2
    exit 1
fi
