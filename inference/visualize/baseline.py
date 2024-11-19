import numpy as np
import matplotlib.pyplot as plt

# Load saved MSE scores
training_mse_scores = np.load("/home/ubuntu/GHIDS2/Senior-Capstone/inference/data/train/scaled_features.pkl_fold1_scores.npy")

# Plot the distribution of MSE scores
plt.figure(figsize=(8, 6))
plt.hist(training_mse_scores, bins=50, alpha=0.75)
plt.title("Training Set Reconstruction Errors")
plt.xlabel("MSE")
plt.ylabel("Frequency")
plt.savefig("/home/ubuntu/GHIDS2/Senior-Capstone/inference/data/train/reconstruction_errors_distribution.png")
plt.show()

# Calculate the mean and standard deviation of MSE scores
mean_error = np.mean(training_mse_scores)
std_error = np.std(training_mse_scores)

# Define the threshold (e.g., 3 standard deviations above the mean)
threshold = mean_error + 3 * std_error
print(threshold)
