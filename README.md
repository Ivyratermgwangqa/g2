# Real-time Anomaly Detection in Encrypted Network Traffic

This project develops and compares machine learning models (XGBoost and Random Forest) for real-time anomaly detection in encrypted network traffic using flow metadata. By integrating TreeExplainer SHAP, the models provide fast and interpretable explanations that enhance transparency and trust in cybersecurity environments.

## 🎯 Project Overview

The project addresses the critical need for efficient explainable models in cybersecurity environments where timely and understandable decisions are essential for effective threat detection. The solution focuses on:

- **Real-time Performance**: Models optimized for low-latency inference suitable for live network monitoring
- **Interpretability**: SHAP (SHapley Additive exPlanations) integration for transparent decision-making
- **Encrypted Traffic Analysis**: Feature extraction from flow metadata without deep packet inspection
- **Model Comparison**: Comprehensive evaluation of XGBoost vs Random Forest approaches

## 🚀 Features

- **Synthetic Data Generation**: Realistic network flow metadata generation for testing and development
- **Dual Model Implementation**: XGBoost and Random Forest with optimized hyperparameters
- **SHAP Integration**: TreeExplainer for fast, interpretable explanations
- **Performance Benchmarking**: Training time, inference time, and explanation generation metrics
- **Comprehensive Visualization**: ROC curves, feature importance, SHAP plots, and model comparisons
- **Modular Architecture**: Clean, reusable code structure for easy integration

## 📁 Project Structure

```
g2/
├── notebooks/
│   └── network_anomaly_detection.ipynb    # Main analysis notebook
├── src/
│   ├── __init__.py
│   ├── data_generator.py                   # Synthetic data generation
│   ├── models.py                          # Model implementations
│   └── visualization.py                   # Plotting utilities
├── data/                                  # Data directory (for real datasets)
├── models/                                # Saved models directory
├── utils/                                 # Additional utilities
├── example_usage.py                       # Quick start example
├── requirements.txt                       # Python dependencies
└── README.md                             # This file
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ivyratermgwangqa/g2.git
   cd g2
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

### Option 1: Run the Example Script
```bash
python example_usage.py
```

### Option 2: Interactive Jupyter Notebook
```bash
jupyter notebook notebooks/network_anomaly_detection.ipynb
```

### Option 3: Use the API
```python
from src.data_generator import generate_network_flow_data
from src.models import XGBoostAnomalyDetector, RandomForestAnomalyDetector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate data
df = generate_network_flow_data(n_samples=10000, anomaly_rate=0.05)
X, y = df.drop('label', axis=1), df['label']

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
xgb_model = XGBoostAnomalyDetector()
xgb_model.fit(X_train_scaled, y_train)

# Make predictions
predictions = xgb_model.predict(X_test_scaled)
probabilities = xgb_model.predict_proba(X_test_scaled)

# Generate explanations
shap_values = xgb_model.explain(X_test_scaled)
```

## 📊 Key Results

Based on synthetic network flow data analysis:

### Model Performance
- **XGBoost**: Typically achieves ROC AUC > 0.95 with fast inference
- **Random Forest**: Comparable accuracy with slightly different feature importance patterns
- **Real-time Feasibility**: Both models achieve <100ms total processing time per sample

### Feature Importance
The most important features for anomaly detection typically include:
- Flow bytes per second
- Flow packets per second
- Packet length statistics (mean, max, min)
- Inter-arrival time patterns

### SHAP Explanations
- **Fast Generation**: TreeExplainer provides explanations in <10ms per sample
- **Interpretable**: Clear feature contribution visualization
- **Actionable**: Security teams can understand why specific flows are flagged

## 🔬 Network Flow Features

The system analyzes the following flow metadata features:

| Feature | Description |
|---------|-------------|
| `flow_duration` | Duration of the network flow in seconds |
| `total_fwd_packets` | Total number of forward packets |
| `total_bwd_packets` | Total number of backward packets |
| `total_length_fwd_packets` | Total length of forward packets in bytes |
| `total_length_bwd_packets` | Total length of backward packets in bytes |
| `fwd_packet_length_*` | Forward packet length statistics (max, min, mean) |
| `bwd_packet_length_*` | Backward packet length statistics (max, min) |
| `flow_bytes_per_sec` | Flow rate in bytes per second |
| `flow_packets_per_sec` | Flow rate in packets per second |
| `flow_iat_*` | Inter-arrival time statistics (mean, std) |
| `fwd_iat_mean` | Mean inter-arrival time for forward packets |
| `bwd_iat_mean` | Mean inter-arrival time for backward packets |

## 🎯 Use Cases

### Cybersecurity Applications
- **Network Monitoring**: Real-time detection of suspicious traffic patterns
- **Threat Hunting**: Identifying potential security incidents in historical data
- **Incident Response**: Understanding attack patterns through interpretable explanations
- **Compliance**: Providing auditable AI decisions for regulatory requirements

### Technical Applications
- **Model Development**: Baseline implementation for network anomaly detection
- **Research**: SHAP explanations for understanding model behavior
- **Integration**: Modular components for larger security platforms
- **Benchmarking**: Performance comparison framework for new algorithms

## 📈 Performance Benchmarks

Typical performance metrics on synthetic data (10,000 samples, 5% anomaly rate):

| Metric | XGBoost | Random Forest |
|--------|---------|---------------|
| Training Time | ~2-3 seconds | ~1-2 seconds |
| Inference Time per Sample | ~0.1-0.5 ms | ~0.2-0.8 ms |
| SHAP Time per Sample | ~2-5 ms | ~3-8 ms |
| ROC AUC | 0.95+ | 0.94+ |
| Total Time per Sample | <10 ms | <15 ms |

*Performance may vary based on hardware and dataset characteristics.*

## 🔧 Customization

### Adding New Features
Extend the `generate_network_flow_data()` function in `src/data_generator.py` to include additional flow metadata features.

### Model Hyperparameters
Modify default parameters in the model constructors in `src/models.py`:

```python
# XGBoost customization
xgb_model = XGBoostAnomalyDetector(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05
)

# Random Forest customization
rf_model = RandomForestAnomalyDetector(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10
)
```

### Real Data Integration
Replace the synthetic data generation with your own data loading:

```python
# Instead of generate_network_flow_data()
df = pd.read_csv('your_network_data.csv')
# Ensure your data has the expected feature columns and a 'label' column
```

## 📚 Dependencies

- **pandas** (2.0.3): Data manipulation and analysis
- **numpy** (1.24.3): Numerical computing
- **scikit-learn** (1.3.0): Machine learning tools
- **xgboost** (1.7.6): Gradient boosting framework
- **shap** (0.42.1): Model explanations
- **matplotlib** (3.7.2): Plotting library
- **seaborn** (0.12.2): Statistical visualization
- **jupyter** (1.0.0): Interactive notebooks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is provided as-is for educational and research purposes. Please ensure compliance with relevant cybersecurity and data protection regulations when using with real network data.

## 🔗 Citation

If you use this project in your research, please cite:

```bibtex
@software{network_anomaly_detection_g2,
  title={Real-time Anomaly Detection in Encrypted Network Traffic with SHAP Explanations},
  author={G2 Project},
  year={2024},
  url={https://github.com/Ivyratermgwangqa/g2}
}
```

## 📞 Support

For questions, issues, or contributions, please open an issue on GitHub or contact the development team.

---

**Note**: This project uses synthetic data for demonstration purposes. For production deployment, ensure proper validation with real network traffic data and consider privacy and security implications.