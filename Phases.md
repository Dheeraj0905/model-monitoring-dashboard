# Project Description: ML Model Monitoring Dashboard in Two Phases

---

## **Overview**

This project aims to build a robust, user-friendly dashboard for monitoring and evaluating machine learning models (.pkl files), focusing on both manual and automated workflows. The dashboard will be developed in two main phases to ensure modularity and extensibility:

---

## **Phase 1: Interactive Dashboard (Manual Workflow)**

### **Goal**
Provide a Streamlit dashboard where users can:
- Upload their trained model (.pkl file).
- Manually define the input schema (feature names, types, etc.).
- Generate synthetic data based on the schema.
- Run automated tests and visualize performance, drift, and explainability metrics.

### **Features**

1. **Model Upload**
   - Drag-and-drop UI for .pkl files.
   - Loads model using pickle; validates compatibility.

2. **Schema Definition**
   - Intuitive UI for users to specify feature names, types (float, int, categorical), and optional ranges/classes.
   - Option to upload a sample CSV for schema auto-detection.

3. **Synthetic Data Generation**
   - Users set data parameters (number of samples, distributions).
   - Data generated according to schema and user specifications.

4. **Automated Testing**
   - Runs predictions using the uploaded model on synthetic data.
   - Measures latency, throughput, error rate, and (if possible) accuracy.
   - Simulates data drift (e.g., mean/variance shift) and visualizes impact on predictions.

5. **Explainability**
   - Integrates SHAP to show feature importances and sample explanations.

6. **Visualization**
   - Real-time and historical metrics, SHAP plots, drift analysis.
   - Download/export results.

7. **Session Management**
   - Keeps logs of test runs for manual review and comparison.

### **User Workflow**

1. Start Streamlit app.
2. Upload `.pkl` model.
3. Define schema via UI or upload CSV sample.
4. Configure synthetic data generation.
5. Run tests and view results.
6. Download metrics and visualizations as needed.

---

## **Phase 2: API-Driven Automation**

### **Goal**
Extend the dashboard with API endpoints allowing direct programmatic interaction:
- Users (or CI/CD pipelines) can POST a model and schema to the dashboard.
- Dashboard performs all tests and returns results via API and updates the UI.

### **Features**

1. **API Endpoint**
   - `/run_tests`: Accepts POST requests with model file and optional schema/sample input.
   - Triggers the same logic as manual dashboard (synthetic data, testing, metrics, explainability).
   - Returns JSON summary of results (metrics, errors, drift info, SHAP values).

2. **Integration with Dashboard UI**
   - API-triggered test results update the dashboard for real-time review.

3. **Extensibility**
   - API can be extended to support more model formats, advanced drift detection, alerting, etc.

### **User/Developer Workflow**

1. Train and save a model as `.pkl`.
2. Use a client script or CI/CD job to POST the model (and schema, if available) to the dashboard API.
3. Dashboard runs all tests and updates metrics in the UI.
4. Developer retrieves metrics from the API response or reviews them on the dashboard.

---

## **Technical Details**

### **Core Stack**
- **Python 3.8+**
- **Streamlit** (main dashboard UI)
- **scikit-learn** (model/pipeline compatibility, metrics)
- **SHAP** (explainability)
- **pandas/numpy** (data handling)
- **Flask or FastAPI** (for API endpoints; can be integrated into Streamlit or as a separate service)

### **File Structure**

```
model-monitoring-dashboard/
├── app.py                    # Streamlit dashboard entry point (Phase 1 UI)
├── schema_utils.py           # Schema definition, parsing, validation
├── synthetic_data.py         # Synthetic data generation logic
├── model_utils.py            # Model loading, prediction, introspection
├── metrics.py                # Metric calculation (latency, throughput, etc.)
├── drift.py                  # Data drift simulation and analysis
├── explainability.py         # SHAP analysis and visualization
├── api_handler.py            # API endpoints for automated testing (Phase 2)
├── storage.py                # Session management and logging
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation
├── tests/
│   └── test_*.py             # Unit/integration tests
└── example_client/
    └── client_api_example.py # Example script for API usage (Phase 2)
```

### **Key Modules Explained**

- **app.py**: Main Streamlit UI for Phase 1 features.
- **schema_utils.py**: Handles user schema entry, validation, and parsing from CSV/sample files.
- **synthetic_data.py**: Generates synthetic test data according to schema/distribution.
- **model_utils.py**: Loads model, tries to introspect feature requirements, runs predictions.
- **metrics.py**: Calculates performance metrics (latency, throughput, error rate).
- **drift.py**: Modifies data distribution to simulate drift and measures impact.
- **explainability.py**: Runs SHAP analysis and creates plots.
- **api_handler.py**: Defines and manages the API endpoint(s) for Phase 2.
- **storage.py**: Logs all test runs, stores session results/history.

---

## **How to Build & Extend**

### **Phase 1 Steps**

1. Scaffold the file structure.
2. Implement Streamlit UI for model upload and schema definition.
3. Add synthetic data generation and integrate with model prediction logic.
4. Add metrics, drift simulation, and SHAP explainability.
5. Build session management and visualization.
6. Test with various models and schemas.

### **Phase 2 Steps**

1. Implement API handler (`api_handler.py`) using Flask/FastAPI or Streamlit’s experimental API support.
2. Integrate API logic with existing dashboard modules (synthetic data, metrics, explainability).
3. Build example client script for automated POST requests.
4. Test integration by calling API from client or CI/CD pipeline.

---

## **Usage Scenarios**

- **Manual Testing**: Data scientists manually upload models and define schemas, then run and visualize tests.
- **Automated Validation**: MLOps engineers integrate API calls into CI/CD for hands-free model evaluation and monitoring.
- **Educational Demos**: Teachers/students use the dashboard to explore model behavior, drift, and explainability concepts interactively.

---

## **Extensibility & Future Work**

- Support more ML frameworks (XGBoost, LightGBM, PyTorch).
- Add advanced drift detection/statistical tests.
- Integrate real-time alerting via API/webhooks.
- Enhance explainability with LIME, integrated gradients, etc.
- Add role-based access and multi-user support.

---

## **Documentation & Getting Started**

- **README.md** covers installation, usage for both phases, and example API calls.
- **Requirements.txt** lists all dependencies.
- **Example Client** folder provides ready-to-use scripts for Phase 2.

---

**Ready to start? Scaffold the file structure, build Phase 1 modules and UI, then extend with API logic for Phase 2!**
