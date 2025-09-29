"""
Visualization utilities for network anomaly detection analysis.

This module provides functions for creating plots and visualizations
to analyze model performance and SHAP explanations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
import shap


def plot_feature_distributions(df, features, figsize=(15, 10)):
    """
    Plot feature distributions comparing normal vs anomalous traffic.
    
    Args:
        df (pd.DataFrame): Dataset with features and labels
        features (list): List of features to plot
        figsize (tuple): Figure size
    """
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, feature in enumerate(features):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        
        # Separate normal and anomalous data
        normal_data = df[df['label'] == 0][feature]
        anomaly_data = df[df['label'] == 1][feature]
        
        # Plot histograms
        ax.hist(normal_data, alpha=0.7, label='Normal', bins=50, density=True, color='blue')
        ax.hist(anomaly_data, alpha=0.7, label='Anomaly', bins=50, density=True, color='red')
        
        ax.set_xlabel(feature.replace('_', ' ').title())
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution of {feature.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_features, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_correlation_matrix(df, figsize=(12, 10)):
    """
    Plot correlation matrix of features.
    
    Args:
        df (pd.DataFrame): Dataset with features
        figsize (tuple): Figure size
    """
    # Calculate correlation matrix (excluding label column)
    features_df = df.drop('label', axis=1) if 'label' in df.columns else df
    correlation_matrix = features_df.corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    return plt.gcf()


def plot_model_comparison(results_df, figsize=(15, 6)):
    """
    Plot model comparison results.
    
    Args:
        results_df (pd.DataFrame): Results from compare_models function
        figsize (tuple): Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Performance metrics comparison
    metrics = ['Training_Time_s', 'Inference_Time_ms', 'ROC_AUC']
    n_models = len(results_df)
    x = np.arange(len(metrics))
    width = 0.35
    
    # Plot bars for each model
    for i, (_, row) in enumerate(results_df.iterrows()):
        values = [row[metric] for metric in metrics]
        offset = (i - (n_models - 1) / 2) * width / n_models
        axes[0].bar(x + offset, values, width/n_models, label=row['Model'], alpha=0.8)
    
    axes[0].set_xlabel('Metrics')
    axes[0].set_ylabel('Values')
    axes[0].set_title('Performance Metrics Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.replace('_', ' ') for m in metrics])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Total time comparison
    models = results_df['Model']
    total_times = results_df['Total_Time_ms']
    
    bars = axes[1].bar(models, total_times, alpha=0.8, color=['blue', 'orange'])
    axes[1].set_xlabel('Models')
    axes[1].set_ylabel('Total Time (ms)')
    axes[1].set_title('Total Processing Time per Sample')
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, total_times):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig


def plot_roc_curves(models, X_test, y_test, model_names=None, figsize=(10, 8)):
    """
    Plot ROC curves for multiple models.
    
    Args:
        models (list): List of trained models
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        model_names (list): Names for the models
        figsize (tuple): Figure size
    """
    if model_names is None:
        model_names = [f"Model_{i+1}" for i in range(len(models))]
    
    plt.figure(figsize=figsize)
    
    for model, name in zip(models, model_names):
        # Get predictions
        probabilities = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        auc = model.evaluate(X_test, y_test)['roc_auc']
        
        # Plot ROC curve
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc:.3f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


def plot_shap_summary(model, X_sample, model_name="Model", figsize=(10, 8)):
    """
    Plot SHAP summary plot for a model.
    
    Args:
        model: Trained model with SHAP explainer
        X_sample (pd.DataFrame): Sample data for SHAP analysis
        model_name (str): Name of the model
        figsize (tuple): Figure size
    """
    shap_values = model.explain(X_sample)
    
    plt.figure(figsize=figsize)
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title(f'{model_name} SHAP Summary Plot')
    plt.tight_layout()
    return plt.gcf()


def plot_shap_importance(models, X_sample, model_names=None, figsize=(15, 8)):
    """
    Plot SHAP feature importance comparison for multiple models.
    
    Args:
        models (list): List of trained models
        X_sample (pd.DataFrame): Sample data for SHAP analysis
        model_names (list): Names for the models
        figsize (tuple): Figure size
    """
    if model_names is None:
        model_names = [f"Model_{i+1}" for i in range(len(models))]
    
    fig, axes = plt.subplots(1, len(models), figsize=figsize)
    if len(models) == 1:
        axes = [axes]
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        shap_values = model.explain(X_sample)
        
        # Calculate mean absolute SHAP values
        importance = np.mean(np.abs(shap_values), axis=0)
        feature_names = X_sample.columns
        
        # Sort features by importance
        sorted_idx = np.argsort(importance)
        
        axes[i].barh(range(len(feature_names)), importance[sorted_idx])
        axes[i].set_yticks(range(len(feature_names)))
        axes[i].set_yticklabels([feature_names[idx] for idx in sorted_idx])
        axes[i].set_xlabel('Mean |SHAP Value|')
        axes[i].set_title(f'{name} SHAP Importance')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_individual_explanation(model, X_sample, sample_idx, model_name="Model", figsize=(12, 6)):
    """
    Plot individual prediction explanation using SHAP waterfall plot.
    
    Args:
        model: Trained model with SHAP explainer
        X_sample (pd.DataFrame): Sample data
        sample_idx (int): Index of sample to explain
        model_name (str): Name of the model
        figsize (tuple): Figure size
    """
    shap_values = model.explain(X_sample)
    
    plt.figure(figsize=figsize)
    
    # Get expected value
    expected_value = model.explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1]  # Positive class for binary classification
    
    # Create SHAP explanation object
    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=expected_value,
        data=X_sample.iloc[sample_idx],
        feature_names=X_sample.columns.tolist()
    )
    
    shap.waterfall_plot(explanation, show=False)
    plt.title(f'{model_name} - Individual Prediction Explanation')
    plt.tight_layout()
    return plt.gcf()