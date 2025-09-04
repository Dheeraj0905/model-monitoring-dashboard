# Project Brief: Unified ML Model Monitoring Dashboard with API-Driven Testing

## Overview

This project is a unified dashboard for **monitoring, evaluating, and explaining ML models in production-like settings**, built with **Streamlit**, **scikit-learn**, and **SHAP**. It addresses the pain points of manual schema entry, lack of model observability, and limited drift/explainability support. The dashboard supports both **interactive UI** and **API-driven automatic testing**, making it ideal for both educational and prototyping purposes, and for integration into CI/CD systems.

---

## Features & Components

### 1. **Model Upload & Registration**
- **UI:** Users can upload their trained `.pkl` model via the dashboard.
- **API:** Users (or scripts) can POST their model to the dashboard endpoint for automated evaluation.
- **Schema Handling:**
  - **Automatic:** The dashboard attempts to introspect the model for input schema (feature names/types) using scikit-learn attributes.
  - **Fallback:** If introspection fails, it uses sensible defaults (e.g., N float features).
  - **Optional:** The API allows sending a schema or sample input for more accurate testing.

### 2. **Synthetic Data Generation**
- **Configurable:** Users can set parameters (number of samples, features, distribution) for synthetic data.
- **Automated:** For API calls, the dashboard automatically generates synthetic data matching the inferred or provided schema.
- **Robust:** Supports float/int/categorical features, and custom distributions.

### 3. **Automated Model Testing**
- **Prediction:** The model is tested on synthetic data.
- **Performance Metrics:** Throughput (preds/sec), latency (ms), error rate, and optional accuracy if simulated labels are available.
- **Drift Simulation:** Optionally simulates data drift by generating data with shifted distributions and observing model output changes.
- **Explainability:** Integrates SHAP to provide feature importance and local/global explanation visualizations.

### 4. **Dashboard Visualization**
- **Metrics:** Real-time and historical graphs for latency, throughput, error rate, accuracy, drift metrics, resource usage.
- **Explainability:** SHAP summary plots, force plots for sample predictions, feature importance charts.
- **Drift Monitoring:** Visualizes changes in prediction distributions when data drift is simulated.
- **Logs & Reports:** Users can view, download, and share metric summaries and visualizations.

### 5. **API Endpoint for Automated Testing**
- **Usage:** CI/CD scripts or other applications can POST a `.pkl` model (and optional schema/sample) to the dashboard API.
- **Function:** The dashboard runs the full suite of synthetic tests, updates the UI, and returns a summary response.
- **Integration:** Enables hands-free, standardized model validation in automated workflows.

### 6. **Historical Tracking**
- **Session Logs:** Stores results of all model tests for audit and comparison.
- **Comparative Analysis:** Users can compare models, track drift over time, and analyze explainability metrics across versions.

### 7. **Extensibility**
- Modular structure supports addition of new metrics, model types, drift tests, and explainability methods.

---

## Technical Details & File Structure

```
model-monitoring-dashboard/
├── app.py                    # Main Streamlit app entry point
├── api_handler.py            # Handles API requests (model uploads, testing triggers)
├── model_utils.py            # Model loading, introspection, and schema inference utilities
├── synthetic_data.py         # Synthetic data generation logic
├── metrics.py                # Performance and drift metric calculation
├── explainability.py         # SHAP integration and explainability plots
├── storage.py                # Session logs, results storage, historical tracking
├── requirements.txt          # Python package dependencies
├── README.md                 # Project documentation and usage instructions
├── static/
│   └── *.css                 # Custom dashboard styling (optional)
├── tests/
│   └── test_*.py             # Unit/integration tests for modules
└── example_client/
    └── client_api_example.py # Example script for API interaction (CI/CD integration)
```

---

## How a User Would Use This Project

### **A. Interactive Dashboard Usage**

1. **Start the dashboard:**  
   `streamlit run app.py`
2. **Upload Model:**  
   Use the UI to upload a `.pkl` file.
3. **(Optional) Upload Schema/Sample Input:**  
   For more accurate testing, upload a sample CSV or JSON schema.
4. **Configure Synthetic Data:**  
   Set number of samples, features, distributions in the UI.
5. **Run Tests:**  
   Click to run synthetic predictions; view real-time performance metrics, drift simulation, and SHAP explainability results.
6. **Analyze & Download Results:**  
   Download logs, metric summaries, and visualizations for reporting or further analysis.

### **B. API-Driven Automated Testing (CI/CD Integration)**

1. **Train and save your model as `.pkl`.**
2. **Send API request:**  
   Use `example_client/client_api_example.py` or your own script to POST the model to the dashboard endpoint.
3. **Dashboard runs full evaluation:**  
   Synthetic data is auto-generated, metrics and explainability are computed.
4. **Dashboard UI updates:**  
   Results visible for manual review; API response returns summary for automation.
5. **Integrate in workflows:**  
   Use this in CI/CD to validate models before deployment, trigger alerts on performance drift, etc.

---

## Detailed Feature Explanations

### **1. Model Upload & Introspection**
- **User:** Uploads `.pkl` file.
- **System:** Loads model, attempts to extract required input columns/types using `feature_names_in_`, pipeline steps, or fallback defaults.

### **2. Synthetic Data Generation**
- **User:** Sets data generation parameters or lets system auto-generate.
- **System:** Creates random (or structured) data matching expected schema, handling types and distributions.

### **3. Automated Testing & Metric Calculation**
- **User:** Triggers tests via UI or API.
- **System:** Measures prediction latency, throughput, error rate, and (if possible) accuracy. 
- **Drift Simulation:** System modifies data distribution (mean/variance shifts) and observes model output changes.

### **4. Explainability (SHAP Integration)**
- **User:** Views SHAP plots showing feature impact on predictions.
- **System:** Computes local/global SHAP values and presents visualizations.

### **5. Visualization & Reporting**
- **User:** Interactively explores charts, metrics, comparisons, and downloads results.
- **System:** Stores logs for historical analysis and audit trail.

### **6. API Endpoint**
- **User:** Sends `.pkl` (and optional schema) via POST request.
- **System:** Runs all above steps, updates dashboard, returns summary response.

### **7. Historical & Comparative Analysis**
- **User:** Reviews previous model tests and compares performance/drift/explainability over time.
- **System:** Maintains persistent logs and enables comparison.

---

## Example Usage Scenarios

- **Data Scientist:** Checks model robustness to drift and explains predictions before deployment.
- **MLOps Engineer:** Integrates dashboard in CI/CD pipeline for automated validation of new models.
- **Educator:** Demonstrates model monitoring, drift, and explainability concepts to students.
- **Auditor:** Reviews historical logs and explainability outputs for compliance.

---

## Getting Started

1. **Clone the repo:**  
   `git clone https://github.com/DURGAKALYAN27/model-monitoring-demo`
2. **Install dependencies:**  
   `pip install -r requirements.txt`
3. **Run the dashboard:**  
   `streamlit run app.py`
4. **(For API use) Run example client script:**  
   `python example_client/client_api_example.py`

---

## Extending the Project

- Add support for more ML frameworks (XGBoost, LightGBM, PyTorch, etc.).
- Enhance drift detection with advanced statistical methods.
- Integrate alerting (email, webhook) for automated anomaly triggers.
- Expand explainability beyond SHAP (LIME, integrated gradients).

---

## Contact & Contributions

- Open issues/feature requests via GitHub.
- Contributions welcome via PR!
