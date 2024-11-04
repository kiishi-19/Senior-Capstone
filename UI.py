import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Step 1: Load the CSV file
def load_data(file_path):
    """
    Load data from the given CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Step 2: Process the data (cleaning and normalization)
def process_data(data):
    """
    Process data by handling missing values, normalizing, and preparing it for analysis.
    """
    print("Checking for missing values...")
    print(data.isnull().sum())
    data.fillna(method='ffill', inplace=True)  # Fill missing values forward
    
    # Exclude non-numeric columns like 'Timestamp'
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    return numeric_data

# Step 3: Apply Isolation Forest for Anomaly Detection
def detect_anomalies(data, contamination=0.01):
    """
    Detect anomalies using Isolation Forest algorithm.
    """
    model = IsolationForest(contamination=contamination, random_state=42)
    print("Training the Isolation Forest model...")
    model.fit(data)
    predictions = model.predict(data)
    anomalies = data[predictions == -1]
    print(f"Anomalies detected: {len(anomalies)}")
    return anomalies, predictions

# Step 4: Save anomalies to a CSV file
def save_anomalies(anomalies, file_name='anomalies.csv'):
    anomalies.to_csv(file_name, index=False)
    print(f"Anomalies saved to {file_name}")

# Step 5: Visualize data and anomalies
def visualize_data(data, predictions, anomaly_column):
    plt.figure(figsize=(10, 6))
    plt.scatter(data.index, data[anomaly_column], color='blue', label='Normal Data')
    anomalies = data[predictions == -1]
    plt.scatter(anomalies.index, anomalies[anomaly_column], color='red', label='Anomalies')
    plt.title("IoT Device Metrics and Detected Anomalies")
    plt.xlabel("Index")
    plt.ylabel(anomaly_column)
    plt.legend()
    plt.show()

def main():
    file_path = 'C:\\Users\\splic\\Downloads\\device_metrics.csv'  # Your actual file path
    data = load_data(file_path)
    if data is None:
        return
    
    numeric_data = process_data(data)
    
    # Use the numeric data to detect anomalies
    anomalies, predictions = detect_anomalies(numeric_data)
    
    if not anomalies.empty:
        save_anomalies(anomalies)
        print("Anomalies detected! Check anomalies.csv for details.")
    else:
        print("No anomalies detected.")
    
    # Visualize the data using one of the numeric columns, e.g., 'CPU_Usage (%)'
    visualize_data(numeric_data, predictions, anomaly_column='CPU_Usage (%)')

if __name__ == "__main__":
    main()
