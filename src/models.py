"""
Machine learning models for network anomaly detection.

This module provides wrapper classes for XGBoost and Random Forest models
with integrated SHAP explanations for interpretable anomaly detection.
"""

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import xgboost as xgb
import shap


class AnomalyDetector:
    """Base class for anomaly detection models."""
    
    def __init__(self):
        self.model = None
        self.explainer = None
        self.is_trained = False
        self.training_time = 0
        
    def fit(self, X, y):
        """Train the model."""
        raise NotImplementedError
        
    def predict(self, X):
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)
    
    def explain(self, X, sample_size=None):
        """Generate SHAP explanations."""
        if not self.is_trained:
            raise ValueError("Model must be trained before generating explanations")
        if self.explainer is None:
            raise ValueError("SHAP explainer not initialized")
            
        if sample_size is not None and len(X) > sample_size:
            X = X.sample(n=sample_size, random_state=42)
            
        return self.explainer.shap_values(X)
    
    def evaluate(self, X, y):
        """Evaluate model performance."""
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)[:, 1]
        
        return {
            'classification_report': classification_report(y, predictions),
            'roc_auc': roc_auc_score(y, probabilities),
            'roc_curve': roc_curve(y, probabilities)
        }


class XGBoostAnomalyDetector(AnomalyDetector):
    """XGBoost-based anomaly detector with SHAP explanations."""
    
    def __init__(self, **kwargs):
        super().__init__()
        
        # Default parameters optimized for anomaly detection
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        default_params.update(kwargs)
        
        self.model = xgb.XGBClassifier(**default_params)
    
    def fit(self, X, y):
        """Train the XGBoost model."""
        start_time = time.time()
        
        # Calculate scale_pos_weight for class imbalance
        pos_weight = len(y[y == 0]) / len(y[y == 1]) if len(y[y == 1]) > 0 else 1
        self.model.set_params(scale_pos_weight=pos_weight)
        
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        
        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        self.is_trained = True
        
        return self


class RandomForestAnomalyDetector(AnomalyDetector):
    """Random Forest-based anomaly detector with SHAP explanations."""
    
    def __init__(self, **kwargs):
        super().__init__()
        
        # Default parameters optimized for anomaly detection
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'class_weight': 'balanced'
        }
        default_params.update(kwargs)
        
        self.model = RandomForestClassifier(**default_params)
    
    def fit(self, X, y):
        """Train the Random Forest model."""
        start_time = time.time()
        
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        
        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        self.is_trained = True
        
        return self
    
    def explain(self, X, sample_size=None):
        """Generate SHAP explanations for Random Forest."""
        if not self.is_trained:
            raise ValueError("Model must be trained before generating explanations")
        if self.explainer is None:
            raise ValueError("SHAP explainer not initialized")
            
        if sample_size is not None and len(X) > sample_size:
            X = X.sample(n=sample_size, random_state=42)
            
        shap_values = self.explainer.shap_values(X)
        
        # For binary classification, return positive class SHAP values
        if isinstance(shap_values, list):
            return shap_values[1]
        return shap_values


def compare_models(models, X_test, y_test, model_names=None):
    """
    Compare multiple anomaly detection models.
    
    Args:
        models (list): List of trained model instances
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        model_names (list): Names for the models
    
    Returns:
        pd.DataFrame: Comparison results
    """
    if model_names is None:
        model_names = [f"Model_{i+1}" for i in range(len(models))]
    
    results = []
    
    for model, name in zip(models, model_names):
        # Performance metrics
        start_time = time.time()
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        inference_time = time.time() - start_time
        
        # Calculate AUC
        auc = roc_auc_score(y_test, probabilities)
        
        # SHAP timing (sample-based)
        sample_size = min(100, len(X_test))
        X_sample = X_test.sample(n=sample_size, random_state=42)
        start_time = time.time()
        try:
            _ = model.explain(X_sample)
            shap_time = time.time() - start_time
        except Exception:
            shap_time = np.nan
        
        results.append({
            'Model': name,
            'Training_Time_s': model.training_time,
            'Inference_Time_ms': (inference_time / len(X_test)) * 1000,
            'SHAP_Time_ms': (shap_time / sample_size) * 1000 if not np.isnan(shap_time) else np.nan,
            'ROC_AUC': auc,
            'Total_Time_ms': ((inference_time / len(X_test)) + (shap_time / sample_size if not np.isnan(shap_time) else 0)) * 1000
        })
    
    return pd.DataFrame(results)