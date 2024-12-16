import pandas as pd

# Replace 'your_file.pkl' with the path to your .pkl file
file_path = '/home/ubuntu/GHIDS2/Senior-Capstone/output/features/known_syscalls.pkl'

# Load the pickle file into a DataFrame
df = pd.read_pickle(file_path)

# Print the DataFrame
print(df)
