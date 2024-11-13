import pandas as pd

# Load the log data
log_file = r"C:\Users\splic\Downloads\stuff.log"  # Path to  log file path

# Read the log file with a pipe '|' delimiter
df = pd.read_csv(log_file, delimiter="|", header=None, names=["Timestamp", "EventType", "ProcessID", "Resource"])

# Ensure the Timestamp is numeric
df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")

# Save the parsed data to a CSV file
output_file = "output.csv"  # Specify the output file name
df.to_csv(output_file, index=False)

# Print the parsed DataFrame for verification
print(df)
