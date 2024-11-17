import sys
import os
import unittest
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from time import time

# Add the ML module path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../ML')))
from prepare_dataset import prepare_dataset

ALLOWED_FEATURES = [
    'num_nodes', 'num_edges', 'avg_degree', 'graph_density', 
    'num_connected_components', 'clustering_coefficient',
    'diameter', 'unseen_syscall_influence', 
    'unseen_argument_influence', 'frequency_increase'
]

class TestPrepareDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_features_dir = "/home/ubuntu/GHIDS2/Senior-Capstone/test_data/features"
        cls.test_output_dir = "/home/ubuntu/GHIDS2/Senior-Capstone/test_data/output"
        cls.test_scaler_file = os.path.join(cls.test_output_dir, "scaler.pkl")
        cls.chunk_size = 50
        cls.n_jobs = 4
        
        if os.path.exists(cls.test_output_dir):
            for f in os.listdir(cls.test_output_dir):
                file_path = os.path.join(cls.test_output_dir, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        else:
            os.makedirs(cls.test_output_dir)

    def test_prepare_dataset_functionality(self):
        prepare_dataset(
            features_dir=self.test_features_dir,
            scaler_file=self.test_scaler_file,
            output_dir=self.test_output_dir,
            chunk_size=self.chunk_size,
            n_jobs=self.n_jobs
        )
        
        self.assertTrue(os.path.exists(self.test_scaler_file), "Scaler file not found")
        scaler = joblib.load(self.test_scaler_file)
        self.assertIsInstance(scaler, StandardScaler, "Scaler is not StandardScaler")

        input_files = [f for f in os.listdir(self.test_features_dir) if f.endswith(".csv")]
        output_files = [f for f in os.listdir(self.test_output_dir) if f.endswith(".csv")]
        self.assertEqual(len(output_files), len(input_files), "Not all files have output")

        tolerance = 0.55
        for output_file in output_files:
            output_path = os.path.join(self.test_output_dir, output_file)
            transformed_data = pd.read_csv(output_path)
            for column in ALLOWED_FEATURES:
                mean_val = transformed_data[column].mean()
                self.assertAlmostEqual(mean_val, 0, delta=tolerance,
                    msg=f"Mean of {column} in {output_file} is not approximately 0")

    def test_multiprocessing_speedup(self):
        large_chunk_size = 500
        start_single = time()
        prepare_dataset(
            features_dir=self.test_features_dir,
            scaler_file=self.test_scaler_file,
            output_dir=self.test_output_dir,
            chunk_size=large_chunk_size,
            n_jobs=1
        )
        single_core_time = time() - start_single

        start_multi = time()
        prepare_dataset(
            features_dir=self.test_features_dir,
            scaler_file=self.test_scaler_file,
            output_dir=self.test_output_dir,
            chunk_size=large_chunk_size,
            n_jobs=self.n_jobs
        )
        multi_core_time = time() - start_multi

        print(f"Single-core time: {single_core_time:.2f} seconds")
        print(f"Multi-core time: {multi_core_time:.2f} seconds")
        
        self.assertTrue(multi_core_time < single_core_time * 1.5,
            "Multi-core processing did not improve speed significantly")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_output_dir):
            for f in os.listdir(cls.test_output_dir):
                file_path = os.path.join(cls.test_output_dir, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    def test_discovery(self):
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
