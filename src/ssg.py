import networkx as nx
from typing import List, Dict, Any
import pandas as pd
from collections import Counter
import numpy as np
from scipy.stats import entropy
from src.preprocessing import KNOWN_SYSCALLS, KNOWN_ARGUMENTS

class SystemStateGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def create_window_graph(self, syscalls: List[str], arguments: List[List[str]], timestamps: List[float]) -> nx.DiGraph:
        """Create a graph for a single time window."""
        window_graph = nx.DiGraph()

        # Process each syscall and its arguments
        for syscall, args, timestamp in zip(syscalls, arguments, timestamps):
            # Add syscall node with timestamp
            syscall_node = f"syscall_{syscall}"
            window_graph.add_node(syscall_node, type='syscall', timestamp=timestamp)

            # Process arguments
            for arg in args:
                arg_node = f"arg_{arg}"
                window_graph.add_node(arg_node, type='argument')
                window_graph.add_edge(syscall_node, arg_node)

        return window_graph

    def update_graph(self, df_window: pd.DataFrame):
        """Update the main graph with a new time window."""
        for _, row in df_window.iterrows():
            time_window = row['time_window']
            syscalls = row['syscall']
            arguments = row['arguments']
            timestamps = row['timestamp']

            # Create graph for this window
            window_graph = self.create_window_graph(syscalls, arguments, timestamps)

            # Add time window information
            for node in window_graph.nodes():
                window_graph.nodes[node]['time'] = time_window

            # Merge with main graph
            self.graph = nx.compose(self.graph, window_graph)

    def extract_features(self) -> Dict[str, Any]:
        """Extract comprehensive features from the system state graph."""
        features = {}

        # Graph Metrics
        features['num_nodes'] = self.graph.number_of_nodes()
        features['num_edges'] = self.graph.number_of_edges()

        if features['num_nodes'] > 0:
            features['avg_degree'] = (2.0 * features['num_edges']) / features['num_nodes']
            features['graph_density'] = nx.density(self.graph)
        else:
            features['avg_degree'] = 0.0
            features['graph_density'] = 0.0

        # Connected Components
        undirected_graph = self.graph.to_undirected()
        features['num_connected_components'] = nx.number_connected_components(undirected_graph)

        # Clustering Coefficient
        if features['num_nodes'] > 1:
            features['clustering_coefficient'] = nx.average_clustering(undirected_graph)
        else:
            features['clustering_coefficient'] = 0.0

        # Node-Based Features
        syscall_nodes = [n for n, attr in self.graph.nodes(data=True)
                         if attr.get('type') == 'syscall']
        arg_nodes = [n for n, attr in self.graph.nodes(data=True)
                     if attr.get('type') == 'argument']

        # Count unique syscalls and arguments
        syscall_counts = Counter(node.split('_', 1)[1] for node in syscall_nodes)
        arg_counts = Counter(node.split('_', 1)[1] for node in arg_nodes)

        features['syscall_node_counts'] = dict(syscall_counts)
        features['argument_node_counts'] = dict(arg_counts)

        # Path-Based Features
        components = list(nx.connected_components(undirected_graph))

        if components:
            diameters = []
            for component in components:
                subgraph = undirected_graph.subgraph(component)
                if len(subgraph) > 1:
                    try:
                        diameters.append(nx.diameter(subgraph))
                    except nx.exception.NetworkXError:
                        continue
            features['diameter'] = max(diameters) if diameters else 0
        else:
            features['diameter'] = 0

        # Anomaly Detection Features
        syscall_names = set(syscall_counts.keys())
        unseen_syscalls = syscall_names - KNOWN_SYSCALLS

        # Calculate influence of unseen syscalls
        total_syscalls = sum(syscall_counts.values())
        unseen_syscall_count = sum(syscall_counts[s] for s in unseen_syscalls)

        features['unseen_syscall_influence'] = (
            unseen_syscall_count / total_syscalls if total_syscalls > 0 else 0.0
        )

        # Calculate influence of unseen arguments
        arg_tokens = set(arg_counts.keys())
        unseen_args = arg_tokens - KNOWN_ARGUMENTS

        total_args = sum(arg_counts.values())
        unseen_args_count = sum(arg_counts[a] for a in unseen_args)

        features['unseen_argument_influence'] = (
            unseen_args_count / total_args if total_args > 0 else 0.0
        )

        # Calculate frequency increase compared to baseline
        BASELINE_THRESHOLD = 5
        if syscall_counts:
            avg_frequency = sum(syscall_counts.values()) / len(syscall_counts)
            max_increase = max(
                count / avg_frequency if avg_frequency > 0 else 0
                for count in syscall_counts.values()
            )
            features['frequency_increase'] = max_increase > BASELINE_THRESHOLD
        else:
            features['frequency_increase'] = False

        # In-Degree and Out-Degree Centrality
        if features['num_nodes'] > 0:
            features['avg_in_degree'] = sum(dict(self.graph.in_degree()).values()) / features['num_nodes']
            features['avg_out_degree'] = sum(dict(self.graph.out_degree()).values()) / features['num_nodes']

        # Sequence Entropy
        features['sequence_entropy'] = entropy(list(syscall_counts.values())) if syscall_counts else 0.0

        # Inter-Arrival Times
        timestamps = [attr['timestamp'] for _, attr in self.graph.nodes(data=True) if 'timestamp' in attr]
        if timestamps:
            inter_arrival_times = np.diff(sorted(timestamps))
            features['mean_inter_arrival_time'] = np.mean(inter_arrival_times)
            features['std_inter_arrival_time'] = np.std(inter_arrival_times)
        else:
            features['mean_inter_arrival_time'] = 0.0
            features['std_inter_arrival_time'] = 0.0

        # Transition Entropy
        syscall_names = [n.split('_', 1)[1] for n in syscall_nodes]
        transitions = Counter(zip(syscall_names[:-1], syscall_names[1:]))
        features['transition_entropy'] = entropy(list(transitions.values())) if transitions else 0.0

        return features
