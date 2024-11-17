import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available!")
    print("PyTorch can see and use the GPU.")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
else:
    print("CUDA is not available. PyTorch cannot access the GPU.")
