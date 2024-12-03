import os
import numpy as np
import matplotlib.pyplot as plt

def load_reconstruction_errors(directory):
    """
    Load reconstruction error files from a given directory.
    
    Parameters:
    - directory (str): Path to the directory containing .npy files.
    
    Returns:
    - dict: A dictionary with filenames as keys and numpy arrays of reconstruction errors as values.
    """
    error_data = {}
    for file in os.listdir(directory):
        if file.endswith("_scores.npy"):
            file_path = os.path.join(directory, file)
            error_data[file] = np.load(file_path)
    return error_data

def plot_error_distributions(error_data, output_path=None):
    """
    Plot reconstruction error distributions and save as a PNG file if output_path is provided.
    
    Parameters:
    - error_data (dict): A dictionary with filenames as keys and error arrays as values.
    - output_path (str, optional): Path to save the plot as a PNG file.
    """
    plt.figure(figsize=(12, 6))
    for filename, errors in error_data.items():
        plt.hist(errors, bins=50, alpha=0.5, label=filename)
    
    plt.title("Reconstruction Error Distributions")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

def main(directory, save_plot=False):
    """
    Main function to load and plot reconstruction error distributions.
    
    Parameters:
    - directory (str): Path to the directory containing .npy files.
    - save_plot (bool): Whether to save the plot as a PNG file.
    """
    print(f"Loading reconstruction errors from directory: {directory}")
    error_data = load_reconstruction_errors(directory)
    
    if not error_data:
        print("No reconstruction error files found in the directory.")
        return
    
    print(f"Loaded {len(error_data)} error files.")
    
    # Define output file path if saving the plot
    output_path = os.path.join(directory, "reconstruction_error_distributions.png") if save_plot else None
    plot_error_distributions(error_data, output_path)

# Example usage
directory = "/home/ubuntu/GHIDS2/Senior-Capstone/output/inference_results/scaled_features"  # Replace with your actual path
main(directory, save_plot=True)  # Set save_plot=True to save the plot as a PNG
