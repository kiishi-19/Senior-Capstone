

uv run ML/multiprocessing_scap_files.py /home/ubuntu/GHIDS2/Senior-Capstone/Data/CB-DS/NORMAL --output output/features.parquet 

uv run ML/train_autoencoder.py /home/ubuntu/GHIDS2/Senior-Capstone/output/scaled_features.pkl --model_output_file /home/ubuntu/GHIDS2/Senior-Capstone/output/autoencoder_model.pth --epochs 120 --hidden_dims 2 2 --bottleneck_dim 1 --activation sigmoid --n_splits 4

uv run ML/prepare_dataset.py /home/ubuntu/GHIDS2/Senior-Capstone/output/features --scaler_file output/scaler.pkl --output_file output/scaled_features.pkl