import sys
from pathlib import Path
import pandas as pd
import networkx as nx

# Add the src directory to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.preprocessing import extract_syscalls, KNOWN_SYSCALLS, KNOWN_ARGUMENTS
from src.ssg import SystemStateGraph

def test_preprocessing():
    """Test syscall extraction and preprocessing functionality."""
    print("\n=== Testing Preprocessing ===")
    
    # Get path relative to project root
    scap_file = project_root / "data" / "1.scap"
    
    # Ensure file exists
    if not scap_file.exists():
        print(f"Error: File {scap_file} not found")
        return None
        
    # Extract syscalls
    df = extract_syscalls(scap_file)
    
    # Print basic statistics
    print("\nPreprocessing Statistics:")
    print(f"Total time windows: {len(df)}")
    print(f"Time range: {df['time_window'].min()} to {df['time_window'].max()}")
    
    # Print known syscalls and arguments
    print(f"\nUnique syscalls found: {len(KNOWN_SYSCALLS)}")
    print("Sample syscalls:", list(KNOWN_SYSCALLS)[:5])
    
    print(f"\nUnique argument patterns: {len(KNOWN_ARGUMENTS)}")
    print("Sample arguments:", list(KNOWN_ARGUMENTS)[:5])
    
    return df

def test_ssg(df):
    """Test System State Graph creation and feature extraction."""
    if df is None:
        print("Error: No DataFrame provided for SSG testing")
        return
        
    print("\n=== Testing SSG Creation ===")
    
    # Create SSG
    ssg = SystemStateGraph()
    
    # Update graph with all time windows
    ssg.update_graph(df)
    
    # Extract features
    features = ssg.extract_features()
    
    # Print graph statistics
    print("\nGraph Statistics:")
    print(f"Nodes: {features['num_nodes']}")
    print(f"Edges: {features['num_edges']}")
    print(f"Average degree: {features['avg_degree']:.2f}")
    print(f"Graph density: {features['graph_density']:.4f}")
    print(f"Connected components: {features['num_connected_components']}")
    print(f"Clustering coefficient: {features['clustering_coefficient']:.4f}")
    print(f"Graph diameter: {features['diameter']}")
    
    # Print syscall statistics
    print("\nTop 5 Syscalls by Frequency:")
    sorted_syscalls = sorted(
        features['syscall_node_counts'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:5]
    for syscall, count in sorted_syscalls:
        print(f"{syscall}: {count}")
    
    # Print anomaly detection metrics
    print("\nAnomaly Detection Metrics:")
    print(f"Unseen syscall influence: {features['unseen_syscall_influence']:.4f}")
    print(f"Unseen argument influence: {features['unseen_argument_influence']:.4f}")
    
    if features['frequency_increase']:
        print("\nSyscalls with Unusual Frequency:")
        for syscall, count in features['frequency_increase'].items():
            print(f"{syscall}: {count}")
    
    # Save visualization
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    ssg.visualize(str(output_dir / "system_state_graph.png"))
    print(f"\nGraph visualization saved to: {output_dir}/system_state_graph.png")

def main():
    """Run all tests."""
    print("Starting syscall processing tests...")
    
    # Test preprocessing
    df = test_preprocessing()
    
    # Test SSG
    test_ssg(df)
    
    print("\nTests completed.")

if __name__ == "__main__":
    main()