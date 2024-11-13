from pathlib import Path
import sys
import pandas as pd
from src.ssg import SystemStateGraph

def load_syscall_data(file_path: str) -> pd.DataFrame:
    """Load syscall data from a file into a DataFrame."""
    df = pd.read_csv(file_path)
    # Ensure required columns exist
    required_cols = ['timestamp', 'syscall', 'args', 'return_value']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    return df

def main():
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    # Load your scap file (adjust path as needed)
    file_path = "path/to/your/scap.csv"  # Replace with your actual file path
    df = load_syscall_data(file_path)

    # Create SSG and extract features
    ssg = SystemStateGraph()
    ssg.build_from_syscalls(df)
    features = ssg.extract_features()

    # Print all features nicely formatted
    print("\nSystem State Graph Features:")
    print("-" * 40)
    for feature_name, value in features.items():
        print(f"{feature_name:25} : {value}")

if __name__ == "__main__":
    main()