"""
Example usage of the network anomaly detection system.

This script demonstrates how to use the modules to create, train,
and evaluate anomaly detection models with SHAP explanations.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_generator import generate_network_flow_data
from models import XGBoostAnomalyDetector, RandomForestAnomalyDetector, compare_models
from visualization import plot_feature_distributions, plot_model_comparison, plot_roc_curves


def main():
    """Main example execution."""
    print("=== Network Anomaly Detection Example ===")
    
    # Generate synthetic data
    print("\n1. Generating synthetic network flow data...")
    df = generate_network_flow_data(n_samples=5000, anomaly_rate=0.05)
    print(f"Generated {len(df)} samples with {df['label'].mean():.3f} anomaly rate")
    
    # Prepare data
    print("\n2. Preparing data...")
    X = df.drop('label', axis=1)
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=X.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=X.columns
    )
    
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    
    # Train models
    print("\n3. Training models...")
    
    # XGBoost model
    print("Training XGBoost...")
    xgb_model = XGBoostAnomalyDetector()
    xgb_model.fit(X_train_scaled, y_train)
    print(f"XGBoost training time: {xgb_model.training_time:.3f}s")
    
    # Random Forest model
    print("Training Random Forest...")
    rf_model = RandomForestAnomalyDetector()
    rf_model.fit(X_train_scaled, y_train)
    print(f"Random Forest training time: {rf_model.training_time:.3f}s")
    
    # Evaluate models
    print("\n4. Evaluating models...")
    
    # XGBoost evaluation
    xgb_results = xgb_model.evaluate(X_test_scaled, y_test)
    print(f"XGBoost ROC AUC: {xgb_results['roc_auc']:.4f}")
    
    # Random Forest evaluation
    rf_results = rf_model.evaluate(X_test_scaled, y_test)
    print(f"Random Forest ROC AUC: {rf_results['roc_auc']:.4f}")
    
    # Compare models
    print("\n5. Comparing models...")
    comparison_results = compare_models(
        [xgb_model, rf_model], 
        X_test_scaled, 
        y_test, 
        ['XGBoost', 'Random Forest']
    )
    print("\nPerformance Comparison:")
    print(comparison_results.to_string(index=False))
    
    # SHAP explanations
    print("\n6. Generating SHAP explanations...")
    sample_size = min(100, len(X_test_scaled))
    X_sample = X_test_scaled.sample(n=sample_size, random_state=42)
    
    try:
        xgb_shap = xgb_model.explain(X_sample)
        print(f"XGBoost SHAP values shape: {xgb_shap.shape}")
        
        rf_shap = rf_model.explain(X_sample)
        print(f"Random Forest SHAP values shape: {rf_shap.shape}")
        
        # Feature importance from SHAP
        xgb_importance = np.mean(np.abs(xgb_shap), axis=0)
        rf_importance = np.mean(np.abs(rf_shap), axis=0)
        
        print("\nTop 5 most important features (XGBoost SHAP):")
        feature_importance = pd.DataFrame({
            'Feature': X_sample.columns,
            'Importance': xgb_importance
        }).sort_values('Importance', ascending=False)
        print(feature_importance.head().to_string(index=False))
        
    except Exception as e:
        print(f"Error generating SHAP explanations: {e}")
    
    # Real-time performance analysis
    print("\n7. Real-time performance analysis...")
    target_latency_ms = 100
    print(f"Target latency: <{target_latency_ms}ms per sample")
    
    for _, row in comparison_results.iterrows():
        model_name = row['Model']
        total_time = row['Total_Time_ms']
        feasible = "✓" if total_time < target_latency_ms else "✗"
        print(f"{model_name}: {total_time:.2f}ms {feasible}")
    
    print("\n=== Example completed successfully! ===")
    print("For full interactive analysis, run the Jupyter notebook:")
    print("jupyter notebook notebooks/network_anomaly_detection.ipynb")


if __name__ == "__main__":
    main()