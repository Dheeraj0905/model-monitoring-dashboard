# ML Model Monitoring Dashboard

A comprehensive web-based dashboard for monitoring and evaluating machine learning models with advanced analytics and explainability features.

## 🚀 Features

### Core Features

- **📁 Model Upload**: Upload and validate trained ML models (.pkl files)
- **📊 Dataset Upload**: Upload test datasets with target columns
- **📋 Schema Definition**: Define input feature schemas manually
- **🎲 Data Generation**: Generate synthetic test data with various distributions
- **⚡ Performance Testing**: Test model latency and throughput
- **📈 Results & Analytics**: Comprehensive performance metrics and visualizations
- **🔍 SHAP Explainability**: Model interpretability and feature importance analysis

### Supported Models

- Classification models (Random Forest, SVM, Logistic Regression, etc.)
- Regression models (Linear Regression, Random Forest Regressor, etc.)
- Any scikit-learn compatible model

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/Mangun10/model-monitoring-dashboard.git
cd model-monitoring-dashboard
```

2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Running the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 📖 How to Use

### 1. Model Upload

- Navigate to "📁 Model Upload"
- Upload your trained model (.pkl file)
- Select model type (Classification or Regression)

### 2. Dataset Upload

- Go to "📊 Dataset Upload"
- Upload your test dataset (CSV format)
- Select the target column for evaluation

### 3. Schema Definition (Optional)

- Visit "📋 Schema Definition"
- Define feature names, types, and requirements
- Save schema for data generation

### 4. Data Generation (Optional)

- Go to "🎲 Data Generation"
- Set generation parameters (samples, distribution, noise)
- Generate synthetic test data

### 5. Performance Testing

- Navigate to "⚡ Performance Testing"
- Configure test parameters (iterations, batch size)
- Run latency and throughput tests

### 6. Results & Analytics

- Visit "📈 Results & Analytics"
- View comprehensive performance metrics
- Analyze classification/regression results

### 7. SHAP Explainability

- Go to "🔍 SHAP Explainability"
- Generate feature importance analysis
- Understand model predictions

## 📊 Metrics Supported

### Classification

- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Classification Report
- Prediction Distribution

### Regression

- R² Score, RMSE, MAE, MAPE
- Predicted vs Actual scatter plots
- Residual analysis
- Error distribution

## 🧪 Example Files

The `example_client/` directory contains:

- Sample models (`*.pkl`)
- Test datasets (`*.csv`)
- Model creation scripts

## 📁 Project Structure

```
model-monitoring-dashboard/
├── app.py                    # Main Streamlit application
├── model_utils.py           # Model utilities and metrics
├── requirements.txt         # Python dependencies
├── example_client/          # Example models and datasets
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## 🔧 Requirements

- Python 3.8+
- Streamlit >= 1.28.0
- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- plotly >= 5.15.0

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🐛 Known Issues

- SHAP explanations use simplified correlation-based importance (install SHAP library for full functionality)
- Performance testing results may vary based on system resources
- Large datasets may impact UI responsiveness

## 📧 Support

For questions or issues, please open a GitHub issue or contact the maintainers.
