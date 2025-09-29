"""
Data generation utilities for network flow anomaly detection.

This module provides functions to generate synthetic network flow metadata
that represents typical features found in encrypted traffic analysis.
"""

import numpy as np
import pandas as pd


def generate_network_flow_data(n_samples=10000, anomaly_rate=0.05, random_state=42):
    """
    Generate synthetic network flow metadata for anomaly detection.
    
    Features represent typical flow characteristics that can be extracted
    from encrypted traffic without deep packet inspection.
    
    Args:
        n_samples (int): Total number of samples to generate
        anomaly_rate (float): Proportion of anomalous samples (0-1)
        random_state (int): Random seed for reproducibility
    
    Returns:
        pd.DataFrame: DataFrame with flow features and labels
    """
    np.random.seed(random_state)
    
    # Normal traffic patterns
    normal_samples = int(n_samples * (1 - anomaly_rate))
    anomaly_samples = n_samples - normal_samples
    
    # Generate normal traffic features
    normal_data = {
        'flow_duration': np.random.exponential(scale=30, size=normal_samples),
        'total_fwd_packets': np.random.poisson(lam=50, size=normal_samples),
        'total_bwd_packets': np.random.poisson(lam=45, size=normal_samples),
        'total_length_fwd_packets': np.random.normal(loc=1500, scale=500, size=normal_samples),
        'total_length_bwd_packets': np.random.normal(loc=800, scale=300, size=normal_samples),
        'fwd_packet_length_max': np.random.normal(loc=1460, scale=100, size=normal_samples),
        'fwd_packet_length_min': np.random.normal(loc=60, scale=20, size=normal_samples),
        'fwd_packet_length_mean': np.random.normal(loc=300, scale=100, size=normal_samples),
        'bwd_packet_length_max': np.random.normal(loc=1460, scale=150, size=normal_samples),
        'bwd_packet_length_min': np.random.normal(loc=60, scale=15, size=normal_samples),
        'flow_bytes_per_sec': np.random.gamma(shape=2, scale=1000, size=normal_samples),
        'flow_packets_per_sec': np.random.gamma(shape=2, scale=5, size=normal_samples),
        'flow_iat_mean': np.random.exponential(scale=0.5, size=normal_samples),
        'flow_iat_std': np.random.exponential(scale=0.2, size=normal_samples),
        'fwd_iat_mean': np.random.exponential(scale=0.6, size=normal_samples),
        'bwd_iat_mean': np.random.exponential(scale=0.4, size=normal_samples),
    }
    
    # Generate anomalous traffic features (different distributions)
    anomaly_data = {
        'flow_duration': np.random.exponential(scale=100, size=anomaly_samples),
        'total_fwd_packets': np.random.poisson(lam=200, size=anomaly_samples),
        'total_bwd_packets': np.random.poisson(lam=5, size=anomaly_samples),
        'total_length_fwd_packets': np.random.normal(loc=5000, scale=1000, size=anomaly_samples),
        'total_length_bwd_packets': np.random.normal(loc=200, scale=100, size=anomaly_samples),
        'fwd_packet_length_max': np.random.normal(loc=1460, scale=50, size=anomaly_samples),
        'fwd_packet_length_min': np.random.normal(loc=1400, scale=30, size=anomaly_samples),
        'fwd_packet_length_mean': np.random.normal(loc=1400, scale=50, size=anomaly_samples),
        'bwd_packet_length_max': np.random.normal(loc=100, scale=50, size=anomaly_samples),
        'bwd_packet_length_min': np.random.normal(loc=60, scale=10, size=anomaly_samples),
        'flow_bytes_per_sec': np.random.gamma(shape=5, scale=2000, size=anomaly_samples),
        'flow_packets_per_sec': np.random.gamma(shape=3, scale=20, size=anomaly_samples),
        'flow_iat_mean': np.random.exponential(scale=0.1, size=anomaly_samples),
        'flow_iat_std': np.random.exponential(scale=0.05, size=anomaly_samples),
        'fwd_iat_mean': np.random.exponential(scale=0.1, size=anomaly_samples),
        'bwd_iat_mean': np.random.exponential(scale=2.0, size=anomaly_samples),
    }
    
    # Combine normal and anomaly data
    data = {}
    for feature in normal_data.keys():
        data[feature] = np.concatenate([normal_data[feature], anomaly_data[feature]])
    
    # Create labels (0 = normal, 1 = anomaly)
    labels = np.concatenate([np.zeros(normal_samples), np.ones(anomaly_samples)])
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df['label'] = labels
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df


def get_feature_descriptions():
    """
    Get descriptions of network flow features.
    
    Returns:
        dict: Mapping of feature names to descriptions
    """
    return {
        'flow_duration': 'Duration of the network flow in seconds',
        'total_fwd_packets': 'Total number of forward packets',
        'total_bwd_packets': 'Total number of backward packets',
        'total_length_fwd_packets': 'Total length of forward packets in bytes',
        'total_length_bwd_packets': 'Total length of backward packets in bytes',
        'fwd_packet_length_max': 'Maximum length of forward packets',
        'fwd_packet_length_min': 'Minimum length of forward packets',
        'fwd_packet_length_mean': 'Mean length of forward packets',
        'bwd_packet_length_max': 'Maximum length of backward packets',
        'bwd_packet_length_min': 'Minimum length of backward packets',
        'flow_bytes_per_sec': 'Flow rate in bytes per second',
        'flow_packets_per_sec': 'Flow rate in packets per second',
        'flow_iat_mean': 'Mean inter-arrival time between packets',
        'flow_iat_std': 'Standard deviation of inter-arrival times',
        'fwd_iat_mean': 'Mean inter-arrival time for forward packets',
        'bwd_iat_mean': 'Mean inter-arrival time for backward packets',
    }