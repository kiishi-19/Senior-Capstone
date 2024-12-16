import logging
import pandas as pd
import os
import sys
from pathlib import Path
# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from src.ssg import SystemStateGraph

# Configure logging for the test
logging.basicConfig(filename='test_ssg.log', level=logging.INFO,
                    format='%(asctime)s %(message)s')

def test_system_state_graph():
    try:
        # Create a sample DataFrame to simulate syscall data
        data = {
            'time_window': ['2024-12-02 19:56:00', '2024-12-02 19:56:00'],
            'timestamp': [['2024-12-02 19:56:01', '2024-12-02 19:56:02'], ['2024-12-02 19:56:03', '2024-12-02 19:56:04']],
            'syscall': [['open', 'close'], ['read', 'write']],
            'arguments': [[['file1', 'mode1'], ['file2', 'mode2']], [['file3', 'size1'], ['file4', 'size2']]]
        }
        df_window = pd.DataFrame(data)

        # Initialize the SystemStateGraph
        ssg = SystemStateGraph()

        # Update the graph with the sample DataFrame
        ssg.update_graph(df_window)

        # Extract features from the SSG
        features = ssg.extract_features()

        # Log the features for verification
        logging.info(f"Extracted features: {features}")

        # Assertions to verify expected behavior
        assert features['num_nodes'] > 0, "Expected at least one node in the graph."
        assert features['num_edges'] > 0, "Expected at least one edge in the graph."
        assert 'avg_degree' in features, "Expected avg_degree in features."
        assert 'num_connected_components' in features, "Expected num_connected_components in features."
        assert 'clustering_coefficient' in features, "Expected clustering_coefficient in features."

        logging.info("All tests passed successfully.")

    except Exception as e:
        logging.error(f"Error during SSG test: {e}")

if __name__ == "__main__":
    test_system_state_graph()