import re
import numpy as np

def calculate_thresholds_for_gammas(log_file_path, gamma_range=np.arange(0.2, 2.1, 0.2)):
    """
    Calculate thresholds for a range of gamma values based on training losses in the log file.

    Parameters:
    - log_file_path (str): Path to the log file containing training losses.
    - gamma_range (numpy.ndarray): Range of gamma values to test.

    Returns:
    - dict: A dictionary where keys are gamma values, and values are thresholds.
    """
    training_losses = []

    # Regular expression to extract training loss lines
    loss_pattern = re.compile(r'Training loss at batch \d+: ([\d.]+)')

    # Parse the log file to extract training losses
    with open(log_file_path, 'r') as file:
        for line in file:
            match = loss_pattern.search(line)
            if match:
                loss = float(match.group(1))
                training_losses.append(loss)

    if not training_losses:
        raise ValueError("No training losses found in the log file.")

    # Find the maximum training loss
    max_loss = max(training_losses)
    print("max_loss:",max_loss)

    # Calculate thresholds for each gamma value
    thresholds = {gamma: gamma * max_loss for gamma in gamma_range}

    return thresholds

# Example usage
log_file_path = '/home/ubuntu/GHIDS2/Senior-Capstone/output/train_autoencoder.log'  # Path to your log file

# Define the range of gamma values
gamma_range = np.arange(0.2, 2.1, 0.1)  # From 0.2 to 2 in steps of 0.1

# Calculate thresholds
thresholds = calculate_thresholds_for_gammas(log_file_path, gamma_range)

# Print the thresholds
for gamma, threshold in thresholds.items():
    print(f"Gamma: {gamma:.1f}, Threshold: {threshold:.4f}")
    
