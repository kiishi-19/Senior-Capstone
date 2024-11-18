

uv run ML/multiprocessing_scap_files.py /home/ubuntu/GHIDS2/Senior-Capstone/Data/CB-DS/NORMAL --output output/features.parquet 

uv run ML/train_autoencoder.py /home/ubuntu/GHIDS2/Senior-Capstone/output/scaled_features.pkl --model_output_file /home/ubuntu/GHIDS2/Senior-Capstone/output/autoencoder_model.pth --epochs 120 --hidden_dims 2 2 --bottleneck_dim 1 --activation sigmoid --n_splits 4

uv run ML/prepare_dataset.py /home/ubuntu/GHIDS2/Senior-Capstone/output/features --scaler_file output/scaler.pkl --output_file output/scaled_features.pkl


python ML/train_autoencoder.py /home/ubuntu/GHIDS2/Senior-Capstone/output/scaled_features     --model_output_file output/autoencoder_model.pth     --epochs 120     --hidden_dims 2 2     --bottleneck_dim 1     --activation sigmoid     --n_splits 4     --batch_size 64


uv run inference/inference_model.py /home/ubuntu/GHIDS2/Senior-Capstone/inference/data/scaled_features \
    --model_paths /home/ubuntu/GHIDS2/Senior-Capstone/inference/data/models/autoencoder_model_fold4.pth \
    --batch_size 64 \
    --hidden_dims 2 2 \
    --bottleneck_dim 1 \
    --activation sigmoid



uv run inference/visualize/visualize_anomalies.py /home/ubuntu/GHIDS2/Senior-Capstone/inference/data/scaled_features --threshold 90


 streamlit run /home/ubuntu/GHIDS2/Senior-Capstone/inference/visualize/visualize_dashboard.py