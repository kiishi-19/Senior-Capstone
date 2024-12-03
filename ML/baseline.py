import re

def calculate_threshold_from_training_loss(log_file_path, gamma=1.5):
    """
    Calculate the anomaly detection threshold based on training losses in the log file.

    Parameters:
    - log_file_path (str): Path to the log file containing training losses.
    - gamma (float): Scaling factor for the threshold.

    Returns:
    - float: Calculated threshold.
    """
    training_losses = []

    # Regular expression to extract training loss lines
    loss_pattern = re.compile(r'Training loss at batch \d+: ([\d.]+)')

    with open(log_file_path, 'r') as file:
        for line in file:
            match = loss_pattern.search(line)
            if match:
                # Extract the training loss as a float
                loss = float(match.group(1))
                training_losses.append(loss)

    if not training_losses:
        raise ValueError("No training losses found in the log file.")

    # Find the maximum training loss
    max_loss = max(training_losses)

    # Calculate the threshold
    threshold = gamma * max_loss
    return max_loss
    return threshold

# Example usage
log_file_path = 'Senior-Capstone/output/train_autoencoder.log'  # Path to your log file
threshold = calculate_threshold_from_training_loss(log_file_path, gamma=1.5)
print(f"Calculated Threshold: {threshold}")


