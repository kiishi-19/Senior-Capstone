#!/bin/bash

# Set project root directory and paths
PROJECT_ROOT="/home/ubuntu/GHIDS2/Senior-Capstone"
INPUT_DIR="$PROJECT_ROOT/output/inference_results/scaled_features"
OUTPUT_DIR="$PROJECT_ROOT/output/json"
PYTHON_SCRIPT="$PROJECT_ROOT/dashboard/pkl.py" 
SUBSTRING="anomaly_scores.pkl"

# Paths to known syscalls and arguments
KNOWN_ARGUMENTS_PKL="$PROJECT_ROOT/output/features/known_arguments.pkl"
KNOWN_SYSCALLS_PKL="$PROJECT_ROOT/output/features/known_syscalls.pkl"

# Create the output directory if it doesn't exist
echo "Creating output directory..."
mkdir -p "$OUTPUT_DIR"

# Execute the Python script with uv
echo "Running Python script to process files..."
uv run "$PYTHON_SCRIPT" --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR" --substring "$SUBSTRING"

# Check if the Python script ran successfully
if [ $? -ne 0 ]; then
    echo "Error: Python script execution failed." >&2
    exit 1
fi

echo "Processing completed. JSON files are saved in $OUTPUT_DIR."

# Run the Streamlit dashboard with additional arguments
streamlit run dashboard/visual2.py -- --json_dir "$OUTPUT_DIR" --batch_size 100 \
    --known_syscalls "$KNOWN_SYSCALLS_PKL" --known_arguments "$KNOWN_ARGUMENTS_PKL"
