# GHIDS2 Senior Capstone Project

## Overview

This project involves processing SCAP files and training an autoencoder model for inference tasks. Below are the steps to train the model and run inference.

## Training Process

### Step 1: Process SCAP Files

To process the SCAP files, run the following command:

`uv run ML/process_scap_files.py /home/ubuntu/GHIDS2/Senior-Capstone/Data/CB-DS/NORMAL --output_dir /home/ubuntu/GHIDS2/Senior-Capstone/output/features --num_cores 4`

### Step 2: Prepare the Dataset

Next, prepare the dataset by scaling the features:

`uv run ML/prepare_dataset.py /home/ubuntu/GHIDS2/Senior-Capstone/output/features --scaler_file output/scaler.pkl --output_file output/scaled_features.pkl`

### Step 3: Concatenate CSV Files

Run the following command to concatenate the processed CSV files:

`uv run ML/concatenate_csv.py`

### Step 4: Train the Autoencoder

Finally, train the autoencoder model with the following command:

`uv run ML/train_autoencoder.py /home/ubuntu/GHIDS2/Senior-Capstone/output/scaled_features.pkl \
    --model_output_file output/models/autoencoder_model.pth \
    --epochs 120 \
    --hidden_dims 64 32 16 \
    --bottleneck_dim 8 \
    --activation relu \
    --n_splits 4 \
    --batch_size 64 \
    --dropout_prob 0.2 \
    

### Step 5: Run the Training Pipeline

You can also run the entire training pipeline using the following script:

`./ML/train_pipeline.sh`

## Inference Process

To run inference on new SCAP files, use the following command:

`./inference/inference_pipeline.sh <path_to_scap_files>`

Replace `<path_to_scap_files>` with the actual path to your SCAP files.

## Notes

- Ensure that all required dependencies are installed and configured properly before running the commands.
- Adjust the parameters in the commands as necessary to fit your specific use case.

For any issues or questions, please refer to the project documentation or contact the project maintainers.